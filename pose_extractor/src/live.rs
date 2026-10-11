//! ライブ手話認識デモ(設計: docs/live-demo-design.md)。
//!
//! フレームの入力元(カメラ / 動画ファイル)から 1 枚ずつ受け取り、Palm+Hand 推論で
//! 126 次元特徴にして貯め、区切りが来たら「手が写っている最長区間 + 余白 → 10 枚等間隔」で
//! 選んで認識モデルに渡し、日本語ラベルで結果を表示する。
//!
//! # 構造(L2 = 自動区切りを後から足しやすいように分けてある)
//!
//! ```text
//! FrameSource ──▶ Recognizer.observe ──▶ FrameBuffer ──▶ plan_selection ──▶ Recognizer.predict ──▶ LiveUi
//!   (カメラ/動画)   (Palm+Hand 推論)      (時刻つき)       (純粋関数)          (Predictor)            (表示)
//!                                              ▲
//!                       KeySource ──▶ Segmenter(いつ録画を始め/止めるか)
//! ```
//!
//! - L1(今回): `ManualSegmenter` = Enter で開始、もう一度 Enter で停止
//! - L2(未実装): `Segmenter` を実装する状態機械(手の出入りで自動)を足すだけ。
//!   `Boundary::Start { pre_roll_secs }` で直前の何秒を録画に含めるかを返せる
//!
//! # 守っていること
//!
//! - **カメラ映像・特徴量はディスクに一切書かない**(メモリ上のバッファのみ。ユーザー指定の制約)
//! - モデルと ONNX セッションは起動時に 1 回だけ読み込む

use crate::{
    downsample_indices, infer_frame_feature, load_palm_anchors, load_model_preprocess,
    longest_hit_span, probe_video, select_frame_mode, select_palm_roi, select_video,
    sign_span_frames, span_indices, FrameSelectMode, PalmRoiMode, PersonDetector, DICT_FEATURE_DIM,
    HAND_MODEL, PALM_MODEL, PERSON_MODEL, RAW_JSL_DIR, SIGN_SPAN,
};
use anyhow::{Context, Result};
use dialoguer::{theme::ColorfulTheme, Input, Select};
use ndarray::Array3;
use ort::session::Session;
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Instant;
use transformer_burn::config::SEQ_LEN;
use transformer_burn::recognition::Predictor;

/// ライブ用の既定モデル(center-square + サイン区間の dict で学習済み。preprocess.json は無い)
const DEFAULT_MODEL_DIR: &str = "../transformer_burn/models/rec_demo_center_signspan";

/// カメラの取り込みサイズ。学習データ(1620x1080)と同じにするため、
/// 1920x1080 で取って中央を 1620 幅に切る(座標は幅・高さで正規化され、手首の画像内座標が
/// 効く特徴なので、縦横比と CenterSquare の位置を学習時に揃える必要がある)
const CAMERA_CAPTURE_W: u32 = 1920;
const CAMERA_CAPTURE_H: u32 = 1080;
const CAMERA_CROP_W: u32 = 1620;
const CAMERA_FPS: u32 = 30;
/// avfoundation に要求する入力ピクセル形式。None なら ffmpeg(デバイス)の既定に任せる。
/// 実機で「pixel format not supported」と出たら、ここに "uyvy422" や "nv12" を入れる
const CAMERA_INPUT_PIXEL_FORMAT: Option<&str> = None;

/// サイン区間の瞬断(手が一瞬見えない)を橋渡しする長さ。学習時は 1 サンプル(約 0.23 秒)。
/// ライブは処理フレームが細かいので、サンプル数でなく**秒で揃える**(設計書 §3.2)
const SPAN_GAP_SECS: f64 = 0.23;
/// サイン区間の前後に付ける余白。学習時は step/2(約 0.1 秒)
const SPAN_MARGIN_SECS: f64 = 0.10;
/// 待機中に保持しておく直近のフレーム長(L2 の「記録開始前の余白」用。L1 では使わない)
const PRE_ROLL_MAX_SECS: f64 = 0.5;
/// top-1 の確率がこれ未満なら「自信なし」と添える(補助表示。高確信度の誤りは止められない)
const LOW_CONFIDENCE: f32 = 0.5;
/// 結果に表示する候補数
const SHOW_TOPK: usize = 3;

// ===== 入力元の抽象 =====

/// 入力元から届いた 1 フレーム
struct SourceFrame {
    frame: Array3<u8>,
    /// 入力開始からの秒数(カメラ: 受信時刻 / 動画: フレーム番号 ÷ fps)
    t: f64,
}

/// フレームの入力元。カメラと動画ファイルを同じ形で扱う(動画ならカメラ無しで全経路を試せる)
trait FrameSource {
    /// 次のフレームを待って返す。Ok(None) = 入力の終わり(動画の末尾)。
    /// カメラは終わらない(異常終了は Err)
    fn next_frame(&mut self) -> Result<Option<SourceFrame>>;
}

/// ffmpeg の rawvideo 出力から 1 フレーム分を読む。Ok(None) = 末尾
fn read_one_frame(stdout: &mut impl Read, frame_size: usize) -> Result<Option<Vec<u8>>> {
    let mut buf = vec![0u8; frame_size];
    match stdout.read_exact(&mut buf) {
        Ok(()) => Ok(Some(buf)),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// 動画ファイルを「カメラの代わり」に流す入力元。ffmpeg をデコーダとして使い、
/// 呼ばれるたびに 1 フレームずつ読む(パイプなので速く回すとデコード速度で進む)。
struct VideoFileSource {
    child: Child,
    stdout: ChildStdout,
    frame_size: usize,
    width: usize,
    height: usize,
    fps: f64,
    /// 次に読むフレーム番号
    next_idx: usize,
    /// N フレームに 1 枚だけ使う(1 = 全部)。実カメラの処理 fps(約 8 fps)を模擬するため
    stride: usize,
}

impl VideoFileSource {
    fn open(path: &PathBuf, stride: usize) -> Result<Self> {
        let info = probe_video(path)?;
        let mut child = Command::new("ffmpeg")
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(path)
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .spawn()
            .context("Failed to spawn ffmpeg")?;
        let stdout = child.stdout.take().context("ffmpeg stdout not piped")?;
        Ok(Self {
            child,
            stdout,
            frame_size: (info.width * info.height * 3) as usize,
            width: info.width as usize,
            height: info.height as usize,
            fps: if info.fps > 0.0 { info.fps } else { 30.0 },
            next_idx: 0,
            stride: stride.max(1),
        })
    }
}

impl FrameSource for VideoFileSource {
    fn next_frame(&mut self) -> Result<Option<SourceFrame>> {
        loop {
            let Some(buf) = read_one_frame(&mut self.stdout, self.frame_size)? else {
                return Ok(None);
            };
            let idx = self.next_idx;
            self.next_idx += 1;
            if !idx.is_multiple_of(self.stride) {
                continue; // 使わないフレームは読み捨て(ストリームを進めるため読む必要はある)
            }
            let frame = Array3::from_shape_vec((self.height, self.width, 3), buf)?;
            return Ok(Some(SourceFrame {
                frame,
                t: idx as f64 / self.fps,
            }));
        }
    }
}

impl Drop for VideoFileSource {
    // 途中で打ち切っても ffmpeg を残さない
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// カメラ入力で ffmpeg(avfoundation)に渡す引数。実行はしない純粋関数(テスト用に分離)。
/// `device` は avfoundation のデバイス番号("0" = FaceTime HD カメラ)。
/// ":none" は音声を取らない指定。
fn camera_ffmpeg_args(device: &str) -> Vec<String> {
    // -nostdin: ffmpeg は既定で端末の標準入力を読み(q で終了する等)、端末を raw モードにする。
    // 放置すると Enter / q が ffmpeg に奪われ、こちらのキー入力が届かなくなる
    let mut args: Vec<String> = [
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "avfoundation",
    ]
        .iter()
        .map(|s| s.to_string())
        .collect();
    args.extend(["-framerate".into(), CAMERA_FPS.to_string()]);
    args.extend([
        "-video_size".into(),
        format!("{}x{}", CAMERA_CAPTURE_W, CAMERA_CAPTURE_H),
    ]);
    if let Some(fmt) = CAMERA_INPUT_PIXEL_FORMAT {
        args.extend(["-pixel_format".into(), fmt.to_string()]);
    }
    args.extend(["-i".into(), format!("{}:none", device)]);
    // 中央を 1620x1080 に切る(crop の x,y を省略すると中央)。学習データと同じ寸法にする
    args.extend([
        "-vf".into(),
        format!("crop={}:{}", CAMERA_CROP_W, CAMERA_CAPTURE_H),
    ]);
    // -fps_mode passthrough: 届いたフレームをそのまま流す。指定しないと(出力が一定フレームレート扱いになり)
    // avfoundation の時刻とずれて、1 枚目のフレームだけが延々と複製される(実機で確認)
    args.extend(["-fps_mode".into(), "passthrough".into()]);
    args.extend(
        ["-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
            .iter()
            .map(|s| s.to_string()),
    );
    args
}

/// 読み取りスレッドとメインスレッドで共有する「最新フレーム置き場」
struct LatestSlot {
    /// 受信した総フレーム数(取りこぼし数の計算用)
    seq: u64,
    /// まだ誰も取っていない最新フレーム。新しいのが来たら**上書き**される(古いのは捨てる)
    frame: Option<Array3<u8>>,
    stamp: Instant,
    /// ffmpeg の出力が終わったら true(異常終了)
    closed: bool,
}

/// Mutex のロック。別スレッドが panic した後でも中身を取れるようにする(毒状態を無視)
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// 読み取りスレッドを起こして、`stdout` から 1 フレームずつ最新置き場へ書き込む。
/// カメラ(ffmpeg の rawvideo 出力)専用だが、`Read` なら何でも渡せる(テストでは byte 列を流す)。
///
/// Rust メモ: `move` = このクロージャが stdout / shared の所有権をスレッドへ持っていく。
/// shared は Arc なので呼び出し側が clone() して渡し、元のハンドルは呼び出し側にも残る
fn spawn_frame_reader(
    mut stdout: impl Read + Send + 'static,
    width: usize,
    height: usize,
    shared: Arc<(Mutex<LatestSlot>, Condvar)>,
) -> std::thread::JoinHandle<()> {
    let frame_size = width * height * 3;
    std::thread::spawn(move || {
        let (slot, cv) = &*shared;
        while let Ok(Some(buf)) = read_one_frame(&mut stdout, frame_size) {
            let Ok(frame) = Array3::from_shape_vec((height, width, 3), buf) else {
                break;
            };
            let mut s = lock(slot);
            s.seq += 1;
            s.frame = Some(frame); // 未取得の古いフレームがあれば、ここで捨てられる
            s.stamp = Instant::now();
            cv.notify_one();
        }
        // EOF / 読み取りエラー = ffmpeg が終了した
        lock(slot).closed = true;
        cv.notify_one();
    })
}

/// Mac のカメラ(ffmpeg avfoundation)を入力元にする。
///
/// 1 フレームは 1620x1080x3 = 約 5.2MB で、30fps だと約 157MB/s が流れる。推論はそれより
/// 遅い(約 8fps 想定)ので、**読み取りスレッドがパイプを読み続けて最新の 1 枚だけ保持**し、
/// メインスレッドは推論が終わるたびに最新を取る。追いつかない分は捨てるので遅延が溜まらない。
///
/// Rust メモ: 2 つのスレッドで同じデータを触るので `Arc<(Mutex<..>, Condvar)>` を使う。
/// - `Arc` = 参照カウント付きの共有所有(スレッドごとに clone して持つ)
/// - `Mutex` = 同時に 1 スレッドだけが中身を触れる錠
/// - `Condvar` = 「新しいフレームが来たよ」と待っている側を起こす仕組み
struct CameraSource {
    child: Child,
    shared: Arc<(Mutex<LatestSlot>, Condvar)>,
    reader: Option<std::thread::JoinHandle<()>>,
    start: Instant,
    last_seq: u64,
    /// 推論が追いつかず捨てたフレーム数(終了時に表示する)
    dropped: u64,
}

impl CameraSource {
    /// 注意: ここで**カメラが起動する**(初回はターミナルアプリにカメラ権限のダイアログが出る)
    fn open(device: &str) -> Result<Self> {
        let mut child = Command::new("ffmpeg")
            .args(camera_ffmpeg_args(device))
            // 端末の標準入力を引き継がせない(-nostdin と二重で、キー入力を奪われないようにする)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to spawn ffmpeg (camera)")?;
        let stdout = child.stdout.take().context("ffmpeg stdout not piped")?;
        let (width, height) = (CAMERA_CROP_W as usize, CAMERA_CAPTURE_H as usize);
        let start = Instant::now();
        let shared = Arc::new((
            Mutex::new(LatestSlot {
                seq: 0,
                frame: None,
                stamp: start,
                closed: false,
            }),
            Condvar::new(),
        ));

        let reader = spawn_frame_reader(stdout, width, height, Arc::clone(&shared));

        Ok(Self {
            child,
            shared,
            reader: Some(reader),
            start,
            last_seq: 0,
            dropped: 0,
        })
    }

    /// ffmpeg が止まったときの理由(stderr)を取り出す
    fn failure_reason(&mut self) -> String {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let mut msg = String::new();
        if let Some(mut err) = self.child.stderr.take() {
            let _ = err.read_to_string(&mut msg);
        }
        let msg = msg.trim().to_string();
        if msg.is_empty() {
            "(ffmpeg からのメッセージなし)".into()
        } else {
            msg
        }
    }

    fn dropped_frames(&self) -> u64 {
        self.dropped
    }
}

impl FrameSource for CameraSource {
    fn next_frame(&mut self) -> Result<Option<SourceFrame>> {
        let shared = Arc::clone(&self.shared);
        let (slot, cv) = &*shared;
        let mut s = lock(slot);
        // 新しいフレームが来るか、ffmpeg が終わるまで寝る
        while s.frame.is_none() && !s.closed {
            s = cv.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        if let Some(frame) = s.frame.take() {
            self.dropped += s.seq - self.last_seq - 1;
            self.last_seq = s.seq;
            let t = s.stamp.duration_since(self.start).as_secs_f64();
            return Ok(Some(SourceFrame { frame, t }));
        }
        drop(s); // failure_reason が self を借りるので、先に錠を手放す
        anyhow::bail!(
            "カメラ入力が止まりました(権限・デバイス番号・解像度を確認してください):\n{}",
            self.failure_reason()
        )
    }
}

impl Drop for CameraSource {
    fn drop(&mut self) {
        let _ = self.child.kill(); // ffmpeg を止めるとスレッドの read が EOF で抜ける
        let _ = self.child.wait();
        if let Some(h) = self.reader.take() {
            let _ = h.join();
        }
    }
}

// ===== キー入力 =====

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Key {
    /// Enter(録画の開始/停止)
    Toggle,
    /// q + Enter(終了)
    Quit,
}

/// 押されたキーを(待たずに)取り出す
trait KeySource {
    fn poll(&mut self) -> Option<Key>;
}

/// 標準入力を別スレッドで読み、行が来るたびに通知する。
/// 推論ループを止めずに Enter を拾うための仕組み(端末を raw モードにしなくて済む)
struct StdinKeys {
    rx: Receiver<Key>,
}

impl StdinKeys {
    fn spawn() -> Self {
        let (tx, rx) = mpsc::channel();
        // `move` でスレッドへ tx を渡す。スレッドは入力が閉じるまで残る
        // (終了時にプロセスごと消えるので join しない)
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut line = String::new();
            loop {
                line.clear();
                match stdin.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let key = if line.trim().eq_ignore_ascii_case("q") {
                            Key::Quit
                        } else {
                            Key::Toggle
                        };
                        if tx.send(key).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        Self { rx }
    }
}

impl KeySource for StdinKeys {
    fn poll(&mut self) -> Option<Key> {
        match self.rx.try_recv() {
            Ok(k) => Some(k),
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => None,
        }
    }
}

/// あらかじめ決めたキーを順に返す(動画ファイル入力・テスト用)
struct ScriptedKeys {
    queue: VecDeque<Key>,
}

impl KeySource for ScriptedKeys {
    fn poll(&mut self) -> Option<Key> {
        self.queue.pop_front()
    }
}

// ===== 区切り(いつ録画を始め/止めるか)=====

/// 録画の境界。L1 はキー、L2 は手の出入りの状態機械がこれを返す
#[derive(Debug, Clone, Copy, PartialEq)]
enum Boundary {
    /// 録画開始。直前の `pre_roll_secs` 秒ぶんのフレームも録画に含める(0 なら含めない)
    Start { pre_roll_secs: f64 },
    /// 録画停止 → 認識
    Stop,
}

trait Segmenter {
    /// Toggle キーが押されたとき
    fn on_toggle(&mut self, recording: bool) -> Option<Boundary>;
    /// フレームを 1 枚処理するたびに呼ばれる(L2 の自動区切りがここで手の有無を見る)
    fn on_frame(&mut self, _frame: &LiveFrame, _recording: bool) -> Option<Boundary> {
        None
    }
}

/// L1: Enter で開始、もう一度 Enter で停止
struct ManualSegmenter;

impl Segmenter for ManualSegmenter {
    fn on_toggle(&mut self, recording: bool) -> Option<Boundary> {
        Some(if recording {
            Boundary::Stop
        } else {
            Boundary::Start { pre_roll_secs: 0.0 }
        })
    }
}

// ===== バッファと 10 枚の選択(純粋な部分)=====

/// 処理済みの 1 フレーム(特徴量だけ。映像そのものは持たない)
#[derive(Debug, Clone, Copy)]
struct LiveFrame {
    t: f64,
    feature: [f32; DICT_FEATURE_DIM],
    left: bool,
    right: bool,
}

impl LiveFrame {
    fn has_hand(&self) -> bool {
        self.left || self.right
    }
}

/// 時刻つきフレームのバッファ(メモリ上のみ)
#[derive(Default)]
struct FrameBuffer {
    frames: Vec<LiveFrame>,
}

impl FrameBuffer {
    fn push(&mut self, f: LiveFrame) {
        self.frames.push(f);
    }

    fn clear(&mut self) {
        self.frames.clear();
    }

    /// 最新フレームから `secs` 秒より古いものを捨てる(待機中のリングバッファ動作)
    fn keep_last_secs(&mut self, secs: f64) {
        if let Some(last) = self.frames.last() {
            let cutoff = last.t - secs;
            self.frames.retain(|f| f.t >= cutoff);
        }
    }

    fn take(&mut self) -> Vec<LiveFrame> {
        std::mem::take(&mut self.frames)
    }
}

/// 処理した時刻列から処理 fps を推定する(2 枚未満なら 0)
fn processed_fps(times: &[f64]) -> f64 {
    match (times.first(), times.last()) {
        (Some(first), Some(last)) if times.len() >= 2 && last > first => {
            (times.len() - 1) as f64 / (last - first)
        }
        _ => 0.0,
    }
}

/// バッファから使う `target` 枚の選び方。
#[derive(Debug, Clone, PartialEq)]
struct Selection {
    /// バッファ内の添字(長さ target)
    indices: Vec<usize>,
    /// 使った区間(両端含む)。等間隔モードなら None
    span: Option<(usize, usize)>,
    /// 推定した処理 fps
    fps: f64,
}

/// バッファ(時刻 `times`・そのフレームで手が見えたか `hits`)から、認識に使う `target` 枚を選ぶ。
/// 手が 1 度も見えなければ None(推論しない。何も写っていないのに 51 語のどれかを答えさせない)。
///
/// - `Uniform`: バッファ全体から等間隔(`downsample_indices`)
/// - `SignSpan`: 手あり最長区間(瞬断は秒換算で橋渡し)+ 前後の余白 → 区間内で等間隔。
///   学習時(build-dict の `detect_sign_span` 以降)と**同じ純粋関数**
///   (`longest_hit_span` / `sign_span_frames` / `span_indices`)を使う。
///   違うのは「サンプル数」でなく「秒」で瞬断・余白を決める点だけ(設計書 §3.2)
// [TEST向き] 純粋関数。区間の橋渡し・余白・短区間の拡張・手なし
fn plan_selection(
    times: &[f64],
    hits: &[bool],
    select: FrameSelectMode,
    target: usize,
) -> Option<Selection> {
    let n = times.len();
    if n == 0 || n != hits.len() || target == 0 || !hits.iter().any(|h| *h) {
        return None;
    }
    let fps = processed_fps(times);
    match select {
        FrameSelectMode::Uniform => Some(Selection {
            indices: downsample_indices(n, target),
            span: None,
            fps,
        }),
        FrameSelectMode::SignSpan { .. } => {
            // 学習時の samples/max_gap の値(16 / 1)はサンプル数基準なので使わず、秒から換算する
            let gap = ((SPAN_GAP_SECS * fps).round() as usize).max(1);
            let margin = (SPAN_MARGIN_SECS * fps).round() as usize;
            let (s, e) = longest_hit_span(hits, gap)?;
            // 余白は span を先に広げて渡す(sign_span_frames は step=1 なら余白 0 で、
            // 短い区間を target 枚ぶんまで広げる処理とクランプだけを担う)
            let widened = (s.saturating_sub(margin), (e + margin).min(n - 1));
            let (start, end) = sign_span_frames(widened, 1, n, target)?;
            Some(Selection {
                indices: span_indices(start, end, target),
                span: Some((start, end)),
                fps,
            })
        }
    }
}

// ===== 認識器(推論の抽象)=====

/// 1 フレームの観測結果
struct Observation {
    feature: [f32; DICT_FEATURE_DIM],
    left: bool,
    right: bool,
}

/// 「フレーム → 特徴」と「特徴列 → タグ」。実体は ONNX + Predictor、テストでは差し替える
trait Recognizer {
    fn observe(&mut self, frame: &Array3<u8>) -> Result<Observation>;
    fn predict(&self, flat: &[f32], frames: usize, k: usize) -> Vec<(String, f32)>;
}

/// 実際の認識器。ONNX セッションと Predictor を**起動時に 1 回だけ**読み込んで使い回す
struct Engine {
    palm: Session,
    hand: Session,
    anchors: Vec<[f32; 2]>,
    roi: PalmRoiMode,
    person: Option<PersonDetector>,
    predictor: Predictor,
    /// 環境変数 LIVE_DEBUG_DUMP_DIR 指定時のみ Some(既定は None = 何も書き出さない)
    dump: Option<DebugDump>,
}

impl Engine {
    fn load(model_dir: &Path, roi: PalmRoiMode) -> Result<Self> {
        let palm = Session::builder()?
            .commit_from_file(PALM_MODEL)
            .with_context(|| format!("failed to load palm model: {}", PALM_MODEL))?;
        let hand = Session::builder()?
            .commit_from_file(HAND_MODEL)
            .with_context(|| format!("failed to load hand model: {}", HAND_MODEL))?;
        let person = if matches!(roi, PalmRoiMode::Person { .. }) {
            Some(PersonDetector::load(PERSON_MODEL)?)
        } else {
            None
        };
        // Box<dyn Error> は anyhow に直接変換できないので文字列化して持ち上げる
        let predictor = Predictor::load(model_dir)
            .map_err(|e| anyhow::anyhow!("認識モデルの読み込みに失敗: {}", e))?;
        Ok(Self {
            palm,
            hand,
            anchors: load_palm_anchors(),
            roi,
            person,
            predictor,
            dump: None,
        })
    }

    /// デバッグ書き出しを有効にする(live_wizard が環境変数から作って渡す)
    fn enable_debug_dump(&mut self, dump: DebugDump) {
        self.dump = Some(dump);
    }

    fn debug_dump_summary(&self) -> Option<String> {
        self.dump.as_ref().map(|d| d.summary())
    }
}

impl Recognizer for Engine {
    fn observe(&mut self, frame: &Array3<u8>) -> Result<Observation> {
        let h = frame.shape()[0] as f32;
        let w = frame.shape()[1] as f32;
        let (feature, left, right) = infer_frame_feature(
            frame,
            &mut self.palm,
            &mut self.hand,
            &self.anchors,
            w,
            h,
            self.roi,
            self.person.as_mut(),
        )?;
        // デバッグ書き出し(既定オフ)。間引き対象のフレームだけ、Palm の生スコア
        // (閾値 0.5 で切る前の最大値)を測ってファイル名に入れる
        if self.dump.as_mut().is_some_and(|d| d.should_dump()) {
            let palm_score = palm_max_score(&mut self.palm, frame, self.roi).ok();
            if let Some(d) = self.dump.as_mut() {
                d.write(frame, left, right, palm_score);
            }
        }
        Ok(Observation {
            feature,
            left,
            right,
        })
    }

    fn predict(&self, flat: &[f32], frames: usize, k: usize) -> Vec<(String, f32)> {
        self.predictor.predict_topk(flat, frames, k)
    }
}

// ===== デバッグ書き出し(オプトイン)=====

/// 書き出しを有効にする環境変数。未設定なら**何も書かない**(カメラ映像を保存しない方針の既定)
const DEBUG_DUMP_ENV: &str = "LIVE_DEBUG_DUMP_DIR";
/// N フレームに 1 枚(処理したフレームの数え方。先頭の 1 枚目を含む)
const DEBUG_DUMP_EVERY: u64 = 30;
/// 書き出す最大枚数
const DEBUG_DUMP_MAX: usize = 20;

/// 書き出し先として拒否するディレクトリ名(パスのどこかの要素がこれ/これで始まる場合)。
/// 学習データ・辞書・モデルに誤って映像を混ぜないための安全弁
fn dump_dir_violation(path: &Path) -> Option<String> {
    for c in path.components() {
        let name = c.as_os_str().to_string_lossy();
        if name == "raw_jsl" || name == "models" || name.starts_with("pose_dict") {
            return Some(format!("'{}' を含むパス", name));
        }
    }
    None
}

/// 相対パスを絶対化し、`.` `..` を字面で畳む(まだ存在しないディレクトリも判定できるように)
fn lexical_absolute(path: &Path) -> PathBuf {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    let mut out = PathBuf::new();
    for c in abs.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// 書き出し先を検証して作る。拒否: raw_jsl / models / pose_dict* を含むパス(シンボリックリンクの
/// 解決後も確認する)、既存の非ディレクトリ
fn prepare_dump_dir(raw: &str) -> Result<PathBuf> {
    let dir = lexical_absolute(Path::new(raw.trim()));
    if let Some(why) = dump_dir_violation(&dir) {
        anyhow::bail!(
            "{} の指定先 {} は使えません({})。学習データ・辞書・モデルの置き場を避けた別のディレクトリを指定してください",
            DEBUG_DUMP_ENV,
            dir.display(),
            why
        );
    }
    if dir.exists() && !dir.is_dir() {
        anyhow::bail!("{} はディレクトリではありません: {}", DEBUG_DUMP_ENV, dir.display());
    }
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("書き出し先を作れません: {}", dir.display()))?;
    // シンボリックリンク経由で禁止領域を指していないか(作成後に実体を確認)
    let real = dir.canonicalize()?;
    if let Some(why) = dump_dir_violation(&real) {
        anyhow::bail!(
            "{} の指定先 {} は実体が {} で、使えません({})",
            DEBUG_DUMP_ENV,
            dir.display(),
            real.display(),
            why
        );
    }
    Ok(real)
}

/// 推論に渡した Array3 を PNG で間引いて書き出す(ユーザーが実機で実行し、保存画像を目視するため)
struct DebugDump {
    dir: PathBuf,
    /// 処理したフレームの数
    seen: u64,
    written: usize,
}

impl DebugDump {
    /// 環境変数が未設定/空なら Ok(None)。設定されていて危ない/作れないなら Err
    fn from_env() -> Result<Option<Self>> {
        match std::env::var(DEBUG_DUMP_ENV) {
            Ok(v) if !v.trim().is_empty() => Ok(Some(Self::new(&v)?)),
            _ => Ok(None),
        }
    }

    fn new(raw_dir: &str) -> Result<Self> {
        Ok(Self {
            dir: prepare_dump_dir(raw_dir)?,
            seen: 0,
            written: 0,
        })
    }

    /// 次に処理するフレームを書き出すか。呼ぶたびにフレーム数を 1 進める
    fn should_dump(&mut self) -> bool {
        let take = self.written < DEBUG_DUMP_MAX && self.seen.is_multiple_of(DEBUG_DUMP_EVERY);
        self.seen += 1;
        take
    }

    /// 失敗しても認識は止めない(警告だけ)
    fn write(&mut self, frame: &Array3<u8>, left: bool, right: bool, palm_score: Option<f32>) {
        let (h, w) = (frame.shape()[0] as u32, frame.shape()[1] as u32);
        let Some(img) = image::RgbImage::from_raw(w, h, frame.iter().copied().collect()) else {
            eprintln!("警告: デバッグ書き出し用の画像を作れませんでした");
            return;
        };
        let hands = match (left, right) {
            (true, true) => "LR",
            (true, false) => "L",
            (false, true) => "R",
            (false, false) => "none",
        };
        let score = palm_score.map_or("na".to_string(), |s| format!("{:.2}", s));
        let name = format!(
            "live_{:02}_frame{:05}_hand-{}_palm{}.png",
            self.written,
            self.seen - 1,
            hands,
            score
        );
        let path = self.dir.join(name);
        match img.save(&path) {
            Ok(()) => self.written += 1,
            Err(e) => eprintln!("警告: 書き出し失敗 {}: {}", path.display(), e),
        }
    }

    fn summary(&self) -> String {
        format!("{} 枚を {} に書き出しました", self.written, self.dir.display())
    }
}

/// Palm の最大スコア(閾値 0.5 で切る前の sigmoid 値)。ROI は `roi` に従う。診断専用
fn palm_max_score(palm: &mut Session, frame: &Array3<u8>, roi: PalmRoiMode) -> Result<f32> {
    let (h, w) = (frame.shape()[0] as u32, frame.shape()[1] as u32);
    let mut best = 0.0f32;
    for r in crate::palm_rois(roi, w, h) {
        let pre = crate::preprocess_for_palm_roi(frame, r)?;
        let v = ort::value::Value::from_array((pre.shape.clone(), pre.data.clone()))?;
        let out = palm.run(ort::inputs!["input_1" => v])?;
        let (_, scores) = out["Identity_1"].try_extract_tensor::<f32>()?;
        for &raw in scores {
            let s = 1.0 / (1.0 + (-(raw as f64)).exp());
            best = best.max(s as f32);
        }
    }
    Ok(best)
}

// ===== 表示 =====

/// 1 回の認識結果
#[derive(Debug, Clone)]
struct LiveResult {
    /// 録画バッファのフレーム数
    buffered: usize,
    duration_secs: f64,
    selection: Selection,
    /// 選ばれた 10 枚のうち手が見えていた枚数
    selected_with_hand: usize,
    ranked: Vec<(String, f32)>,
}

/// 待機/録画中の 1 行表示の内容
struct StatusLine {
    recording: bool,
    rec_secs: f64,
    left: bool,
    right: bool,
    fps: f64,
}

trait LiveUi {
    fn status(&mut self, s: &StatusLine);
    fn message(&mut self, msg: &str);
    fn result(&mut self, r: &LiveResult);
}

/// words.tsv の (romaji, label_ja) を表示用に引く
struct Labels {
    /// words.tsv の並び順
    rows: Vec<(String, String)>,
}

impl Labels {
    fn label_ja(&self, tag: &str) -> Option<&str> {
        self.rows
            .iter()
            .find(|(r, _)| r == tag)
            .map(|(_, l)| l.as_str())
    }

    /// 日本語ラベル(無ければタグ名そのまま)
    fn display(&self, tag: &str) -> String {
        self.label_ja(tag).unwrap_or(tag).to_string()
    }
}

/// words.tsv の本文を読む。1 行目のヘッダ(word_id で始まる)は飛ばす。
/// 読み取り専用(`progress::load_words` は無ければ作る挙動なので使わない)
// [TEST向き] 純粋なパース
fn parse_words_tsv(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with("word_id"))
        .filter_map(|l| {
            let cols: Vec<&str> = l.split('\t').collect();
            if cols.len() >= 3 && !cols[1].is_empty() {
                Some((cols[1].to_string(), cols[2].to_string()))
            } else {
                None
            }
        })
        .collect()
}

fn load_labels() -> Labels {
    let path = Path::new(RAW_JSL_DIR).join("words.tsv");
    match std::fs::read_to_string(&path) {
        Ok(text) => Labels {
            rows: parse_words_tsv(&text),
        },
        Err(e) => {
            eprintln!(
                "警告: {} を読めないため日本語ラベル無しで表示します: {}",
                path.display(),
                e
            );
            Labels { rows: Vec::new() }
        }
    }
}

/// 結果 1 件を表示用の文字列にする。
/// 自信あり: `→ ありがとう (arigatou) 0.83   候補: おねがいします 0.10 / すみません 0.04`
/// 自信なし: `→ (自信なし) わたし 0.41 / あなた 0.33 / これ 0.12`
// [TEST向き] 純粋な整形。自信なし閾値の境界と候補が 1 件のとき
fn format_result(ranked: &[(String, f32)], labels: &Labels) -> String {
    let Some((top_tag, top_p)) = ranked.first() else {
        return "→ (結果なし)".into();
    };
    let show = |(tag, p): &(String, f32)| format!("{} {:.2}", labels.display(tag), p);
    if *top_p < LOW_CONFIDENCE {
        let all: Vec<String> = ranked.iter().take(SHOW_TOPK).map(show).collect();
        format!("→ (自信なし) {}", all.join(" / "))
    } else {
        let rest: Vec<String> = ranked.iter().skip(1).take(SHOW_TOPK - 1).map(show).collect();
        let head = format!("→ {} ({}) {:.2}", labels.display(top_tag), top_tag, top_p);
        if rest.is_empty() {
            head
        } else {
            format!("{}   候補: {}", head, rest.join(" / "))
        }
    }
}

/// 端末に出す UI。待機/録画中の行は `\r` で同じ行を書き換える
struct TerminalUi {
    labels: Labels,
}

impl TerminalUi {
    /// 状態行を消してから出力する(書き換え中の行に重ならないように)
    fn clear_line() {
        print!("\r\x1b[2K");
    }
}

impl LiveUi for TerminalUi {
    fn status(&mut self, s: &StatusLine) {
        let hands = match (s.left, s.right) {
            (true, true) => "LR",
            (true, false) => "L ",
            (false, true) => "R ",
            (false, false) => "--",
        };
        let state = if s.recording {
            format!("記録中 {:.1}s", s.rec_secs)
        } else {
            "待機中".to_string()
        };
        Self::clear_line();
        print!("[{}] 手: {}   処理 {:.1}fps", state, hands, s.fps);
        let _ = std::io::stdout().flush();
    }

    fn message(&mut self, msg: &str) {
        Self::clear_line();
        println!("{}", msg);
    }

    fn result(&mut self, r: &LiveResult) {
        Self::clear_line();
        println!("{}", format_result(&r.ranked, &self.labels));
        // 何を根拠に選んだかの補足(見ている人向けというより動作確認用)
        let span = match r.selection.span {
            Some((s, e)) => format!("区間 {}..={}", s, e),
            None => "全体から等間隔".to_string(),
        };
        println!(
            "   (録画 {} フレーム {:.1}s・処理 {:.1}fps・{}・手あり {}/{} 枚)",
            r.buffered,
            r.duration_secs,
            r.selection.fps,
            span,
            r.selected_with_hand,
            r.selection.indices.len()
        );
    }
}

// ===== メインループ =====

struct Settings {
    select: FrameSelectMode,
    /// 認識に使うフレーム数(学習時の SEQ_LEN)
    frames: usize,
    topk: usize,
}

/// 直近の処理 fps を測る(状態行の表示用)
struct RateMeter {
    recent: VecDeque<f64>,
}

impl RateMeter {
    fn new() -> Self {
        Self {
            recent: VecDeque::new(),
        }
    }

    fn tick(&mut self, t: f64) -> f64 {
        self.recent.push_back(t);
        while self.recent.len() > 16 {
            self.recent.pop_front();
        }
        let v: Vec<f64> = self.recent.iter().copied().collect();
        processed_fps(&v)
    }
}

/// 録画状態とバッファを持って、1 フレームずつ進める本体
struct LiveLoop<'a> {
    rec: &'a mut dyn Recognizer,
    ui: &'a mut dyn LiveUi,
    settings: &'a Settings,
    buf: FrameBuffer,
    recording: bool,
    rec_start_t: f64,
    results: Vec<LiveResult>,
}

impl LiveLoop<'_> {
    fn apply(&mut self, b: Boundary) {
        match (b, self.recording) {
            (Boundary::Start { pre_roll_secs }, false) => {
                if pre_roll_secs > 0.0 {
                    self.buf.keep_last_secs(pre_roll_secs);
                } else {
                    self.buf.clear();
                }
                self.rec_start_t = self.buf.frames.first().map(|f| f.t).unwrap_or(0.0);
                self.recording = true;
            }
            (Boundary::Stop, true) => self.finalize(),
            // 録画中の Start / 待機中の Stop は無視
            _ => {}
        }
    }

    /// 録画を止めて、10 枚を選び、推論して、結果を出す
    fn finalize(&mut self) {
        self.recording = false;
        let frames = self.buf.take();
        let times: Vec<f64> = frames.iter().map(|f| f.t).collect();
        let hits: Vec<bool> = frames.iter().map(|f| f.has_hand()).collect();
        let Some(selection) =
            plan_selection(&times, &hits, self.settings.select, self.settings.frames)
        else {
            self.ui.message("手が検出されませんでした(認識せず)。もう一度どうぞ");
            return;
        };
        if frames.len() < self.settings.frames {
            self.ui.message(&format!(
                "注意: 録画が {} フレームしかなく、{} 枚に足りないため同じフレームを重複して使います",
                frames.len(),
                self.settings.frames
            ));
        }
        let flat: Vec<f32> = selection
            .indices
            .iter()
            .flat_map(|&i| frames[i].feature.iter().copied())
            .collect();
        let selected_with_hand = selection
            .indices
            .iter()
            .filter(|&&i| frames[i].has_hand())
            .count();
        let ranked = self
            .rec
            .predict(&flat, self.settings.frames, self.settings.topk);
        let result = LiveResult {
            buffered: frames.len(),
            duration_secs: times.last().unwrap_or(&0.0) - times.first().unwrap_or(&0.0),
            selection,
            selected_with_hand,
            ranked,
        };
        self.ui.result(&result);
        self.results.push(result);
    }
}

/// 入力元が尽きる(動画の末尾)か、q が押されるまで回す。返り値は出した結果の一覧。
/// 動画の末尾で録画中なら、そこで停止して認識する。
fn run_session(
    rec: &mut dyn Recognizer,
    source: &mut dyn FrameSource,
    keys: &mut dyn KeySource,
    segmenter: &mut dyn Segmenter,
    ui: &mut dyn LiveUi,
    settings: &Settings,
) -> Result<Vec<LiveResult>> {
    let mut lp = LiveLoop {
        rec,
        ui,
        settings,
        buf: FrameBuffer::default(),
        recording: false,
        rec_start_t: 0.0,
        results: Vec::new(),
    };
    let mut meter = RateMeter::new();
    loop {
        while let Some(key) = keys.poll() {
            match key {
                Key::Quit => return Ok(lp.results),
                Key::Toggle => {
                    if let Some(b) = segmenter.on_toggle(lp.recording) {
                        lp.apply(b);
                    }
                }
            }
        }
        let Some(sf) = source.next_frame()? else {
            // 入力の終わり。録画中なら区切って認識する
            if lp.recording {
                lp.finalize();
            }
            return Ok(lp.results);
        };
        let obs = lp.rec.observe(&sf.frame)?;
        let frame = LiveFrame {
            t: sf.t,
            feature: obs.feature,
            left: obs.left,
            right: obs.right,
        };
        let fps = meter.tick(sf.t);
        lp.buf.push(frame);
        if !lp.recording {
            lp.buf.keep_last_secs(PRE_ROLL_MAX_SECS);
        }
        if let Some(b) = segmenter.on_frame(&frame, lp.recording) {
            lp.apply(b);
        }
        let rec_secs = if lp.recording {
            sf.t - lp.rec_start_t
        } else {
            0.0
        };
        lp.ui.status(&StatusLine {
            recording: lp.recording,
            rec_secs,
            left: obs.left,
            right: obs.right,
            fps,
        });
    }
}

// ===== ウィザード =====

/// メニュー「ライブ認識デモ」。TTY 必須(対話入力)。
pub fn live_wizard(theme: &ColorfulTheme) -> Result<()> {
    // 1. フレームの入力元
    let src_idx = Select::with_theme(theme)
        .with_prompt("フレームの入力元")
        .items(&[
            "カメラ(Mac 内蔵カメラ。Enter で録画開始/停止)",
            "動画ファイル(カメラの代わり。再生開始と同時に録画し、終端で認識)",
        ])
        .default(0)
        .interact()?;

    // 2. 認識モデル(既定: デモ用)
    let model_dir: String = Input::with_theme(theme)
        .with_prompt("認識モデルのディレクトリ")
        .default(DEFAULT_MODEL_DIR.into())
        .interact_text()?;
    let model_dir = PathBuf::from(model_dir.trim());

    // 3. 前処理(ROI・フレーム選択・フレーム数)。モデルの preprocess.json が最優先
    let (roi, select, frames) = match load_model_preprocess(&model_dir)? {
        Some(p) => {
            eprintln!(
                "モデルの preprocess.json に合わせます: palm_roi={} / frame_select={}",
                p.roi.label(),
                p.select.label()
            );
            (p.roi, p.select, p.frames.unwrap_or(SEQ_LEN))
        }
        None => {
            eprintln!(
                "警告: {} に preprocess.json がありません。\n\
                 \x20     ライブ用の既定(中央正方クロップ + サイン区間)を使うか、手動で選んでください。\n\
                 \x20     学習に使った dict の作り方と一致しないと精度が落ちます",
                model_dir.display()
            );
            let idx = Select::with_theme(theme)
                .with_prompt("前処理")
                .items(&[
                    "中央正方クロップ + サイン区間(デモ用モデル rec_demo_center_signspan の学習時と同じ)",
                    "手動で選ぶ(ROI・フレームの選び方)",
                ])
                .default(0)
                .interact()?;
            if idx == 0 {
                (PalmRoiMode::CenterSquare, SIGN_SPAN, SEQ_LEN)
            } else {
                (select_palm_roi(theme)?, select_frame_mode(theme)?, SEQ_LEN)
            }
        }
    };

    // 4. モデルと ONNX セッションを 1 回だけ読み込む
    eprintln!("モデルを読み込み中...");
    let mut engine = Engine::load(&model_dir, roi)?;
    // デバッグ書き出し(環境変数 LIVE_DEBUG_DUMP_DIR 指定時のみ。既定は何も保存しない)
    if let Some(dump) = DebugDump::from_env()? {
        eprintln!(
            "注意: {} が指定されているため、推論に渡したフレームを {} フレームに 1 枚(最大 {} 枚)\n\
             \x20     PNG として {} に書き出します。映像が残るので、確認後に削除してください",
            DEBUG_DUMP_ENV,
            DEBUG_DUMP_EVERY,
            DEBUG_DUMP_MAX,
            dump.dir.display()
        );
        engine.enable_debug_dump(dump);
    }

    // 5. 語彙一覧(語彙外の語を出すと必ず別の語になることを、見ている人に伝えるため)
    let labels = load_labels();
    print_vocabulary(engine.predictor.tags(), &labels);

    let settings = Settings {
        select,
        frames,
        topk: SHOW_TOPK,
    };
    let mut ui = TerminalUi { labels };

    match src_idx {
        0 => {
            let device: String = Input::with_theme(theme)
                .with_prompt("カメラのデバイス番号(avfoundation。FaceTime HD カメラは通常 0)")
                .default("0".into())
                .interact_text()?;
            println!(
                "\nカメラを起動します(初回はターミナルアプリにカメラ権限のダイアログが出ます。\n\
                 拒否した場合は システム設定 → プライバシーとセキュリティ → カメラ で許可)。"
            );
            let mut source = CameraSource::open(device.trim())?;
            println!(
                "操作: Enter = 記録開始 / もう一度 Enter = 停止して認識 / q + Enter = 終了\n\
                 (カメラ映像・特徴量は保存しません)\n"
            );
            let mut keys = StdinKeys::spawn();
            run_session(
                &mut engine,
                &mut source,
                &mut keys,
                &mut ManualSegmenter,
                &mut ui,
                &settings,
            )?;
            println!(
                "\n(推論が追いつかず捨てたカメラフレーム: {} 枚)",
                source.dropped_frames()
            );
        }
        _ => {
            let video = select_video(theme)?;
            let mut source = VideoFileSource::open(&video, 1)?;
            // 再生開始と同時に録画を始める(Toggle を 1 回押した扱い)。終端で認識される
            let mut keys = ScriptedKeys {
                queue: VecDeque::from([Key::Toggle]),
            };
            println!("\n動画を流します: {}\n", video.display());
            run_session(
                &mut engine,
                &mut source,
                &mut keys,
                &mut ManualSegmenter,
                &mut ui,
                &settings,
            )?;
        }
    }
    if let Some(summary) = engine.debug_dump_summary() {
        println!("\nデバッグ書き出し: {}", summary);
    }
    println!("\n終了します");
    Ok(())
}

/// 認識できる語の一覧を表示する(words.tsv の並び順。words.tsv に無いタグは末尾にタグ名で)
fn print_vocabulary(tags: &[String], labels: &Labels) {
    let mut shown: Vec<String> = Vec::new();
    for (romaji, label) in &labels.rows {
        if tags.iter().any(|t| t == romaji) {
            shown.push(label.clone());
        }
    }
    for t in tags {
        if !labels.rows.iter().any(|(r, _)| r == t) {
            shown.push(t.clone());
        }
    }
    println!(
        "\n認識できる語({}語): {}\n(この語彙の外の手話を出しても、必ずどれかの語になります)",
        tags.len(),
        shown.join(" / ")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// t = i / fps の時刻列
    fn times(n: usize, fps: f64) -> Vec<f64> {
        (0..n).map(|i| i as f64 / fps).collect()
    }

    fn hits_in(n: usize, ranges: &[(usize, usize)]) -> Vec<bool> {
        (0..n)
            .map(|i| ranges.iter().any(|(s, e)| i >= *s && i <= *e))
            .collect()
    }

    // [TEST向き] ライブの選択が既存の span_indices の規則(区間内の等間隔・端点を含む)に従うこと。
    // 10fps → gap=round(2.3)=2、margin=round(1.0)=1。手あり 10..=29 → 余白込み 9..=30 の 10 枚
    #[test]
    fn sign_span_selection_matches_existing_rules() {
        let t = times(40, 10.0);
        let h = hits_in(40, &[(10, 29)]);
        let sel = plan_selection(&t, &h, SIGN_SPAN, 10).unwrap();
        assert_eq!(sel.span, Some((9, 30)));
        assert_eq!(sel.indices, span_indices(9, 30, 10));
        assert_eq!(sel.indices, vec![9, 11, 13, 16, 18, 20, 23, 25, 27, 30]);
        assert!((sel.fps - 10.0).abs() < 1e-9);
    }

    // [TEST向き] 余白は秒換算: 30fps なら margin=3 フレーム、gap=7 フレーム
    #[test]
    fn margin_and_gap_scale_with_processed_fps() {
        let t = times(120, 30.0);
        let h = hits_in(120, &[(40, 79)]);
        let sel = plan_selection(&t, &h, SIGN_SPAN, 10).unwrap();
        assert_eq!(sel.span, Some((37, 82)));

        // 瞬断 6 フレーム(0.2 秒)は橋渡しされ、8 フレーム(0.27 秒)は橋渡しされない
        let bridged = hits_in(120, &[(30, 49), (56, 90)]);
        let sel = plan_selection(&t, &bridged, SIGN_SPAN, 10).unwrap();
        assert_eq!(sel.span.unwrap().0, 30 - 3);
        let split = hits_in(120, &[(30, 49), (58, 90)]);
        let sel = plan_selection(&t, &split, SIGN_SPAN, 10).unwrap();
        assert_eq!(sel.span.unwrap().0, 58 - 3, "長い方(58..=90)だけが選ばれるはず");
    }

    // [TEST向き] 手が 1 度も見えなければ選ばない(Uniform でも推論しない)
    #[test]
    fn no_hands_selects_nothing() {
        let t = times(30, 10.0);
        let h = vec![false; 30];
        assert_eq!(plan_selection(&t, &h, SIGN_SPAN, 10), None);
        assert_eq!(plan_selection(&t, &h, FrameSelectMode::Uniform, 10), None);
        assert_eq!(plan_selection(&[], &[], SIGN_SPAN, 10), None);
    }

    // [TEST向き] 等間隔モードはバッファ全体を downsample_indices と同じ規則で選ぶ
    #[test]
    fn uniform_selection_uses_downsample_indices() {
        let t = times(25, 10.0);
        let h = hits_in(25, &[(5, 12)]);
        let sel = plan_selection(&t, &h, FrameSelectMode::Uniform, 10).unwrap();
        assert_eq!(sel.indices, downsample_indices(25, 10));
        assert_eq!(sel.span, None);
    }

    // [TEST向き] 短い区間は target 枚ぶんまで広げられ、バッファ端でクランプされる
    #[test]
    fn short_span_is_widened_and_clamped() {
        let t = times(40, 10.0);
        let h = hits_in(40, &[(20, 22)]);
        let sel = plan_selection(&t, &h, SIGN_SPAN, 10).unwrap();
        let (s, e) = sel.span.unwrap();
        assert!(e - s + 1 >= 10, "区間が 10 フレーム未満: {:?}", sel.span);
        assert_eq!(sel.indices.len(), 10);
        assert!(sel.indices.iter().all(|&i| i >= s && i <= e));

        // 手あり区間がバッファの先頭に接している
        let h = hits_in(40, &[(0, 2)]);
        let (s, _) = plan_selection(&t, &h, SIGN_SPAN, 10).unwrap().span.unwrap();
        assert_eq!(s, 0);
    }

    // [TEST向き] バッファが target 未満でも落ちずに target 個の添字を返す(重複あり)
    #[test]
    fn buffer_shorter_than_target_still_returns_target_indices() {
        let t = times(6, 10.0);
        let h = vec![true; 6];
        let sel = plan_selection(&t, &h, SIGN_SPAN, 10).unwrap();
        assert_eq!(sel.indices.len(), 10);
        assert!(sel.indices.iter().all(|&i| i < 6));
    }

    #[test]
    fn frame_buffer_keeps_only_recent_seconds() {
        let mut b = FrameBuffer::default();
        for i in 0..20 {
            b.push(LiveFrame {
                t: i as f64 * 0.1,
                feature: [0.0; DICT_FEATURE_DIM],
                left: false,
                right: false,
            });
        }
        b.keep_last_secs(0.5);
        // 最新 1.9 から 0.5 秒以内 = 1.4..=1.9 の 6 枚
        assert_eq!(b.frames.len(), 6);
        assert!((b.frames[0].t - 1.4).abs() < 1e-9);
    }

    #[test]
    fn processed_fps_cases() {
        assert_eq!(processed_fps(&[]), 0.0);
        assert_eq!(processed_fps(&[1.0]), 0.0);
        assert!((processed_fps(&times(11, 8.0)) - 8.0).abs() < 1e-9);
    }

    #[test]
    fn manual_segmenter_toggles() {
        let mut s = ManualSegmenter;
        assert_eq!(
            s.on_toggle(false),
            Some(Boundary::Start { pre_roll_secs: 0.0 })
        );
        assert_eq!(s.on_toggle(true), Some(Boundary::Stop));
    }

    // カメラ起動の引数(実行はしない): avfoundation 入力・1920x1080 を 1620 幅に中央クロップ・rgb24
    #[test]
    fn camera_args_capture_then_crop_to_training_size() {
        let a = camera_ffmpeg_args("0");
        let joined = a.join(" ");
        assert!(joined.contains("-f avfoundation"));
        assert!(joined.contains("-framerate 30"));
        assert!(joined.contains("-video_size 1920x1080"));
        assert!(joined.contains("-i 0:none"));
        assert!(joined.contains("-vf crop=1620:1080"));
        // 無いと実機で 1 枚目が繰り返される(出力側のオプションなので -i より後)
        assert!(joined.contains("-fps_mode passthrough"));
        assert!(joined.ends_with("-pix_fmt rgb24 -f rawvideo -"));
        // 入力指定(-i)が出力側のオプション(-vf)より前にある
        let pos = |s: &str| a.iter().position(|x| x == s).unwrap();
        assert!(pos("-i") < pos("-vf"));
        assert!(pos("-i") < pos("-fps_mode"));
    }

    #[test]
    fn parse_words_tsv_skips_header_and_blank() {
        let text = "word_id\tromaji\tlabel_ja\tstage\ttarget_takes\n\
                    001\tkonnichiwa\tこんにちは\t1\t10\n\
                    \n\
                    002\tohayou\tおはよう\t1\t10\n\
                    bad line\n";
        let rows = parse_words_tsv(text);
        assert_eq!(
            rows,
            vec![
                ("konnichiwa".to_string(), "こんにちは".to_string()),
                ("ohayou".to_string(), "おはよう".to_string()),
            ]
        );
    }

    fn labels() -> Labels {
        Labels {
            rows: vec![
                ("arigatou".into(), "ありがとう".into()),
                ("onegaishimasu".into(), "おねがいします".into()),
                ("sumimasen".into(), "すみません".into()),
            ],
        }
    }

    #[test]
    fn format_result_confident_and_unsure() {
        let l = labels();
        let r = vec![
            ("arigatou".to_string(), 0.83),
            ("onegaishimasu".to_string(), 0.10),
            ("sumimasen".to_string(), 0.04),
        ];
        assert_eq!(
            format_result(&r, &l),
            "→ ありがとう (arigatou) 0.83   候補: おねがいします 0.10 / すみません 0.04"
        );
        let r = vec![
            ("arigatou".to_string(), 0.41),
            ("unknown_tag".to_string(), 0.33),
        ];
        assert_eq!(
            format_result(&r, &l),
            "→ (自信なし) ありがとう 0.41 / unknown_tag 0.33"
        );
        assert_eq!(format_result(&[], &l), "→ (結果なし)");
        // 閾値ちょうど(0.5)は自信あり扱い
        let r = vec![("arigatou".to_string(), 0.5)];
        assert_eq!(format_result(&r, &l), "→ ありがとう (arigatou) 0.50");
    }

    // ===== カメラ読み取りスレッド(byte 列を流して検証。カメラは開かない)=====

    // 1 フレーム分の rawvideo(rgb24, 行優先)が (高さ, 幅, 3) の Array3 になり、画素の位置が保たれる
    #[test]
    fn frame_reader_keeps_layout_and_stops_at_eof() {
        let (w, h) = (4usize, 2usize);
        // 画素 (y, x) の R=y*10+x, G=100+R, B=200+R
        let mut bytes = Vec::new();
        for _ in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let r = (y * 10 + x) as u8;
                    bytes.extend([r, 100 + r, 200 + r]);
                }
            }
        }
        let start = Instant::now();
        let shared = Arc::new((
            Mutex::new(LatestSlot {
                seq: 0,
                frame: None,
                stamp: start,
                closed: false,
            }),
            Condvar::new(),
        ));
        let handle = spawn_frame_reader(std::io::Cursor::new(bytes), w, h, Arc::clone(&shared));
        handle.join().unwrap();
        let s = lock(&shared.0);
        assert_eq!(s.seq, 3);
        assert!(s.closed);
        let f = s.frame.as_ref().expect("最新フレームが残っている");
        assert_eq!(f.shape(), &[h, w, 3]);
        assert_eq!(f[[1, 2, 0]], 12);
        assert_eq!(f[[1, 2, 1]], 112);
        assert_eq!(f[[1, 2, 2]], 212);
    }

    // 端数のフレーム(途中で切れた最後の 1 枚)は捨てて終了する
    #[test]
    fn frame_reader_drops_partial_tail() {
        let shared = Arc::new((
            Mutex::new(LatestSlot {
                seq: 0,
                frame: None,
                stamp: Instant::now(),
                closed: false,
            }),
            Condvar::new(),
        ));
        let bytes = vec![7u8; 2 * 2 * 3 + 5];
        spawn_frame_reader(std::io::Cursor::new(bytes), 2, 2, Arc::clone(&shared))
            .join()
            .unwrap();
        let s = lock(&shared.0);
        assert_eq!(s.seq, 1);
        assert!(s.closed);
    }

    // ===== デバッグ書き出し =====

    #[test]
    fn dump_dir_rejects_protected_locations() {
        for bad in [
            "transformer_burn/data/raw_jsl",
            "/x/transformer_burn/data/raw_jsl/001_konnichiwa",
            "/x/transformer_burn/models/rec_full",
            "/x/models",
            "/x/data/pose_dict_stage1.json",
            "/x/data/pose_dict_dump",
            // `..` で禁止領域へ抜ける指定も、畳んだあとで拒否される
            "/x/safe/../transformer_burn/models/y",
        ] {
            let abs = lexical_absolute(Path::new(bad));
            assert!(dump_dir_violation(&abs).is_some(), "拒否されるべき: {}", bad);
        }
        for ok in ["/tmp/live_dump", "/x/data/live_debug", "/x/model/dump"] {
            let abs = lexical_absolute(Path::new(ok));
            assert!(dump_dir_violation(&abs).is_none(), "許可されるべき: {}", ok);
        }
    }

    #[test]
    fn prepare_dump_dir_rejects_without_creating() {
        let base = std::env::temp_dir().join(format!("live_dump_reject_{}", std::process::id()));
        let bad = base.join("models").join("sub");
        let err = prepare_dump_dir(bad.to_str().unwrap()).unwrap_err().to_string();
        assert!(err.contains("使えません"), "{}", err);
        assert!(!bad.exists(), "拒否したディレクトリは作らない");
    }

    // N 枚に 1 枚・最大 20 枚。先頭 1 枚目を含む
    #[test]
    fn dump_sampling_every_30_up_to_20() {
        let dir = std::env::temp_dir().join(format!("live_dump_sampling_{}", std::process::id()));
        let mut d = DebugDump::new(dir.to_str().unwrap()).unwrap();
        let frame = Array3::<u8>::from_elem((2, 3, 3), 9);
        let mut taken = Vec::new();
        for i in 0..2000u64 {
            if d.should_dump() {
                taken.push(i);
                d.write(&frame, i % 2 == 0, false, Some(0.5));
            }
        }
        assert_eq!(taken.len(), DEBUG_DUMP_MAX);
        assert_eq!(&taken[..3], &[0, 30, 60]);
        assert_eq!(d.written, DEBUG_DUMP_MAX);
        let names: Vec<String> = std::fs::read_dir(&d.dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names.len(), DEBUG_DUMP_MAX);
        assert!(names.contains(&"live_00_frame00000_hand-L_palm0.50.png".to_string()));
        // 書き出した PNG は推論に渡した Array3 と同じ内容
        let back = load_png(&d.dir.join("live_01_frame00030_hand-L_palm0.50.png"));
        assert_eq!(back, frame);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // 環境変数が無ければ書き出しは作られない(既定オフ)
    #[test]
    fn dump_is_off_without_env() {
        // 他のテストと環境変数を共有するので、未設定のときだけ確認する
        if std::env::var(DEBUG_DUMP_ENV).is_err() {
            assert!(DebugDump::from_env().unwrap().is_none());
        }
    }

    // ===== ループ全体のテスト(ONNX なしの偽の認識器・偽の入力元)=====

    /// 左上画素が 0 以外のフレームを「手あり」とみなす
    struct FakeRecognizer;
    impl Recognizer for FakeRecognizer {
        fn observe(&mut self, frame: &Array3<u8>) -> Result<Observation> {
            let has = frame[[0, 0, 0]] > 0;
            let mut feature = [0.0f32; DICT_FEATURE_DIM];
            feature[0] = frame[[0, 0, 0]] as f32; // 識別用にフレーム値を載せる
            Ok(Observation {
                feature,
                left: false,
                right: has,
            })
        }
        fn predict(&self, flat: &[f32], frames: usize, k: usize) -> Vec<(String, f32)> {
            assert_eq!(flat.len(), frames * DICT_FEATURE_DIM);
            vec![("arigatou".to_string(), 0.9); k.min(1)]
        }
    }

    /// 値のリストを 1 フレームずつ返す(各フレームは 1x1 画素)。時刻は 10fps
    struct FakeSource {
        values: VecDeque<u8>,
        idx: usize,
    }
    impl FrameSource for FakeSource {
        fn next_frame(&mut self) -> Result<Option<SourceFrame>> {
            let Some(v) = self.values.pop_front() else {
                return Ok(None);
            };
            let t = self.idx as f64 / 10.0;
            self.idx += 1;
            Ok(Some(SourceFrame {
                frame: Array3::from_elem((1, 1, 3), v),
                t,
            }))
        }
    }

    #[derive(Default)]
    struct CollectUi {
        messages: Vec<String>,
        results: usize,
    }
    impl LiveUi for CollectUi {
        fn status(&mut self, _s: &StatusLine) {}
        fn message(&mut self, msg: &str) {
            self.messages.push(msg.to_string());
        }
        fn result(&mut self, _r: &LiveResult) {
            self.results += 1;
        }
    }

    fn settings() -> Settings {
        Settings {
            select: SIGN_SPAN,
            frames: 10,
            topk: 3,
        }
    }

    // Enter で開始 → 動画末尾で停止、の流れで 1 件認識され、選ばれたフレームが手あり区間内にあること
    #[test]
    fn loop_records_until_end_and_predicts_once() {
        // 0 が手なし、1 が手あり。手あり区間は 10..=29
        let values: Vec<u8> = (0..40).map(|i| if (10..=29).contains(&i) { 1 } else { 0 }).collect();
        let mut source = FakeSource {
            values: values.into(),
            idx: 0,
        };
        let mut keys = ScriptedKeys {
            queue: VecDeque::from([Key::Toggle]),
        };
        let mut ui = CollectUi::default();
        let results = run_session(
            &mut FakeRecognizer,
            &mut source,
            &mut keys,
            &mut ManualSegmenter,
            &mut ui,
            &settings(),
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(ui.results, 1);
        let r = &results[0];
        assert_eq!(r.buffered, 40);
        assert_eq!(r.selection.span, Some((9, 30)));
        // 余白 2 枚(両端)を除く 8 枚以上が手あり
        assert!(r.selected_with_hand >= 8, "{}", r.selected_with_hand);
    }

    // 手が一度も写らなければ認識せずメッセージだけ
    #[test]
    fn loop_without_hands_gives_message_not_result() {
        let mut source = FakeSource {
            values: vec![0u8; 20].into(),
            idx: 0,
        };
        let mut keys = ScriptedKeys {
            queue: VecDeque::from([Key::Toggle]),
        };
        let mut ui = CollectUi::default();
        let results = run_session(
            &mut FakeRecognizer,
            &mut source,
            &mut keys,
            &mut ManualSegmenter,
            &mut ui,
            &settings(),
        )
        .unwrap();
        assert!(results.is_empty());
        assert_eq!(ui.messages.len(), 1);
        assert!(ui.messages[0].contains("手が検出されませんでした"));
    }

    // キー無しなら録画は始まらない(待機のまま入力が尽きる)
    #[test]
    fn loop_without_toggle_never_records() {
        let mut source = FakeSource {
            values: vec![1u8; 20].into(),
            idx: 0,
        };
        let mut keys = ScriptedKeys {
            queue: VecDeque::new(),
        };
        let mut ui = CollectUi::default();
        let results = run_session(
            &mut FakeRecognizer,
            &mut source,
            &mut keys,
            &mut ManualSegmenter,
            &mut ui,
            &settings(),
        )
        .unwrap();
        assert!(results.is_empty());
        assert!(ui.messages.is_empty());
    }

    // q で即終了(録画中なら破棄)
    #[test]
    fn loop_quit_returns_immediately() {
        let mut source = FakeSource {
            values: vec![1u8; 20].into(),
            idx: 0,
        };
        let mut keys = ScriptedKeys {
            queue: VecDeque::from([Key::Toggle, Key::Quit]),
        };
        let mut ui = CollectUi::default();
        let results = run_session(
            &mut FakeRecognizer,
            &mut source,
            &mut keys,
            &mut ManualSegmenter,
            &mut ui,
            &settings(),
        )
        .unwrap();
        assert!(results.is_empty());
    }

    // ===== 実モデル + 実動画のスモーク(#[ignore])=====

    /// 動画ファイルを「カメラの代わり」にライブ経路へ流し、同じモデル・同じ動画を既存の
    /// 認識経路(extract_tag_features_impl → Predictor)に渡した結果と比べる。
    /// 処理フレーム数・区間の切り方が違うので完全一致は求めず、「大きく食い違わない」
    /// (一方の top-1 がもう一方の top-5 に入る)ことを確かめる。
    ///
    /// 環境変数:
    ///   LIVE_SMOKE_VIDEOS  対象動画(':' 区切り。既定 raw_jsl から 5 本)
    ///   LIVE_SMOKE_MODEL   モデルディレクトリ(既定 DEFAULT_MODEL_DIR)
    ///   LIVE_SMOKE_STRIDE  N フレームに 1 枚だけ使う(既定 1。4 で約 7.5fps 相当)
    ///
    /// ffmpeg + ONNX + 学習済みモデル + raw_jsl に依存するため #[ignore]。
    /// 手動実行: `cargo test live_file_smoke --release -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_file_smoke() {
        let model_dir = PathBuf::from(
            std::env::var("LIVE_SMOKE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL_DIR.into()),
        );
        for p in [model_dir.as_path(), Path::new(PALM_MODEL), Path::new(HAND_MODEL)] {
            if !p.exists() {
                eprintln!("skip: fixture not found: {}", p.display());
                return;
            }
        }
        let videos: Vec<PathBuf> = match std::env::var("LIVE_SMOKE_VIDEOS") {
            Ok(v) => v.split(':').map(PathBuf::from).collect(),
            Err(_) => crate::tests::sample_raw_jsl_takes(5),
        };
        if videos.is_empty() {
            eprintln!("skip: raw_jsl の動画が無い");
            return;
        }
        let stride: usize = std::env::var("LIVE_SMOKE_STRIDE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);

        // preprocess.json があればそれ、無ければライブ既定(center + sign-span)
        let (roi, select, frames) = match load_model_preprocess(&model_dir).unwrap() {
            Some(p) => (p.roi, p.select, p.frames.unwrap_or(SEQ_LEN)),
            None => (PalmRoiMode::CenterSquare, SIGN_SPAN, SEQ_LEN),
        };
        eprintln!(
            "model={} roi={} select={} stride={}",
            model_dir.display(),
            roi.label(),
            select.label(),
            stride
        );
        let mut engine = Engine::load(&model_dir, roi).unwrap();
        let labels = load_labels();
        let mut offline_palm = Session::builder().unwrap().commit_from_file(PALM_MODEL).unwrap();
        let mut offline_hand = Session::builder().unwrap().commit_from_file(HAND_MODEL).unwrap();
        let anchors = load_palm_anchors();

        let mut agree = 0usize;
        for video in &videos {
            let name = format!(
                "{}/{}",
                video.parent().and_then(|p| p.file_name()).unwrap_or_default().to_string_lossy(),
                video.file_name().unwrap_or_default().to_string_lossy()
            );
            // --- ライブ経路 ---
            let mut source = VideoFileSource::open(video, stride).unwrap();
            let mut keys = ScriptedKeys {
                queue: VecDeque::from([Key::Toggle]),
            };
            let mut ui = CollectUi::default();
            let t0 = Instant::now();
            let results = run_session(
                &mut engine,
                &mut source,
                &mut keys,
                &mut ManualSegmenter,
                &mut ui,
                &Settings {
                    select,
                    frames,
                    topk: 5,
                },
            )
            .unwrap();
            let live_secs = t0.elapsed().as_secs_f64();
            let live = results.first().unwrap_or_else(|| panic!("{}: 結果なし {:?}", name, ui.messages));

            // --- 既存の認識経路(extract_tag_features_impl → Predictor)---
            let mut person = if matches!(roi, PalmRoiMode::Person { .. }) {
                Some(PersonDetector::load(PERSON_MODEL).unwrap())
            } else {
                None
            };
            let entry = crate::extract_tag_features_impl(
                video,
                &mut offline_palm,
                &mut offline_hand,
                &anchors,
                frames,
                false,
                roi,
                person.as_mut(),
                select,
            )
            .unwrap();
            let flat: Vec<f32> = entry.sequence.iter().flatten().copied().collect();
            let offline = engine.predict(&flat, frames, 5);

            let fmt = |r: &[(String, f32)]| {
                r.iter()
                    .map(|(t, p)| format!("{} {:.2}", t, p))
                    .collect::<Vec<_>>()
                    .join(" / ")
            };
            eprintln!("\n=== {} ===", name);
            eprintln!(
                "live   : 処理{}枚({:.1}s分・推定{:.1}fps) 区間{:?} 選択{:?} 手あり{}/{}  ({:.1}s)",
                live.buffered,
                live.duration_secs,
                live.selection.fps,
                live.selection.span,
                live.selection.indices,
                live.selected_with_hand,
                live.selection.indices.len(),
                live_secs
            );
            eprintln!("live   : {}", fmt(&live.ranked));
            eprintln!(
                "offline: coverage L={:.0}% R={:.0}% / {}",
                entry.left_hand_coverage * 100.0,
                entry.right_hand_coverage * 100.0,
                fmt(&offline)
            );
            eprintln!("表示   : {}", format_result(&live.ranked, &labels));

            let live_top = &live.ranked[0].0;
            let off_top = &offline[0].0;
            let ok = offline.iter().any(|(t, _)| t == live_top)
                || live.ranked.iter().any(|(t, _)| t == off_top);
            if live_top == off_top {
                agree += 1;
            }
            assert!(
                ok,
                "{}: ライブと既存経路が大きく食い違う (live top1={}, offline top1={})",
                name, live_top, off_top
            );
        }
        eprintln!("\ntop-1 一致: {}/{}", agree, videos.len());
    }

    // ===== カメラ読み取りスレッドの再生テスト(#[ignore])=====

    fn load_png(path: &Path) -> Array3<u8> {
        let img = image::open(path).unwrap().to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        Array3::from_shape_vec((h, w, 3), img.into_raw()).unwrap()
    }

    /// 静止画を「ffmpeg rawvideo の byte 列」にしてカメラの読み取りスレッド経由で Engine に通す。
    /// (画像ファイルの ffmpeg デコードのみ。カメラは開かない)
    ///
    /// 外部の画像に依存する: 環境変数 LIVE_IMG_DIR が指すディレクトリに 1620x1080 の
    /// `cam.png`(実機カメラ由来のフレーム)が必要。リポジトリには含まれない。
    /// LIVE_IMG_DIR 未設定、または cam.png が無いときは何も検証せず skip する。
    /// 加えて ffmpeg・ONNX モデル・学習済みモデル(DEFAULT_MODEL_DIR)にも依存する。
    /// カメラ経路のバイト取り込み(サイズ・チャンネル順・縦横)が image::open と一致し、
    /// 手が検出されることを確かめる
    #[test]
    #[ignore]
    fn live_camera_reader_replay() {
        let Ok(dir) = std::env::var("LIVE_IMG_DIR") else {
            eprintln!("skip: LIVE_IMG_DIR 未設定");
            return;
        };
        let png = PathBuf::from(dir).join("cam.png");
        if !png.exists() {
            eprintln!("skip: {} が無い", png.display());
            return;
        }
        let out = Command::new("ffmpeg")
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(&png)
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .output()
            .unwrap();
        assert_eq!(out.stdout.len(), 1620 * 1080 * 3);
        let shared = Arc::new((
            Mutex::new(LatestSlot {
                seq: 0,
                frame: None,
                stamp: Instant::now(),
                closed: false,
            }),
            Condvar::new(),
        ));
        let mut bytes = out.stdout.clone();
        bytes.extend(out.stdout); // 2 枚流す
        spawn_frame_reader(std::io::Cursor::new(bytes), 1620, 1080, Arc::clone(&shared))
            .join()
            .unwrap();
        let frame = lock(&shared.0).frame.take().unwrap();
        let direct = load_png(&png);
        assert_eq!(frame, direct, "カメラ経路の取り込みが image::open と一致しない");
        let mut engine = Engine::load(
            &PathBuf::from(DEFAULT_MODEL_DIR),
            PalmRoiMode::CenterSquare,
        )
        .unwrap();
        let obs = engine.observe(&frame).unwrap();
        eprintln!("replay: left={} right={}", obs.left, obs.right);
        assert!(obs.left || obs.right);
    }
}
