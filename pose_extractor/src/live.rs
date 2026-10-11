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
//! - L1: `ManualSegmenter` = Enter で開始、もう一度 Enter で停止
//! - L2: `AutoSegmenter` = 手の出入りで自動的に開始/停止(Enter での手動操作も併用できる)。
//!   `Boundary::Start { pre_roll_secs }` で直前の何秒を録画に含めるかを返す
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

// アプリ専用の内部入口(標準入力/標準出力の JSON 行)。設計: docs/desktop-app-design.md §3.2。
// live.rs の部品(トレイト・Engine・Labels など)をそのまま使う別実装で、ここの子モジュールに閉じ込める
// (子モジュールは親の非公開アイテムを参照できるので、可視性は変えない)
mod json_engine;
pub use json_engine::run_engine_json;

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
/// 手が出始めた時点より前に録画へ含める秒数(自動区切りの pre-roll)。
/// 実動画で、サイン冒頭の短い手検出(0.2 秒)→ 0.23 秒の途切れ → 本体、という並びがあり、
/// 開始判定(START_HAND_SECS)の後ろから 0.5 秒だけ遡ると冒頭が落ちて別の語になった(kuru → iu)
const PRE_ROLL_BEFORE_ONSET_SECS: f64 = 0.5;
/// 待機中に保持しておく直近のフレーム長(L2 の「記録開始前の余白」用。手動開始は 0 秒で使わない)。
/// 開始判定に AUTO_START_HAND_SECS かかるので、その分を足して「手が出始めた 0.5 秒前」まで遡れるようにする
const PRE_ROLL_MAX_SECS: f64 = AUTO_START_HAND_SECS + PRE_ROLL_BEFORE_ONSET_SECS;
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
    /// `failure_reason` が ffmpeg を回収(wait)する**直前**に呼ぶ処理(ライブ用エンジンの PID 登録解除)。
    /// 回収した後の PID は別プロセスに使い回されうるので、外から kill される前に登録を外せるようにする
    before_reap: Option<fn()>,
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
            before_reap: None,
        })
    }

    /// ffmpeg を回収する直前に呼ぶ処理を登録する(json_engine が PID の登録解除に使う)
    fn set_before_reap(&mut self, f: fn()) {
        self.before_reap = Some(f);
    }

    /// ffmpeg が止まったときの理由(stderr)を取り出す
    fn failure_reason(&mut self) -> String {
        // kill した直後の ffmpeg は、wait するまでゾンビとして PID を保つ(= 使い回されない)。
        // その間に登録を外せば、外から見て「回収済みの PID に kill が飛ぶ」瞬間が無くなる
        let _ = self.child.kill();
        if let Some(f) = self.before_reap {
            f();
        }
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
    /// 録画を認識せずに捨てる(自動区切りで、手の出ている時間が短すぎた場合)
    Discard,
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

// ===== L2: 手の出入りによる自動区切り =====
//
// 待機 --(手あり が START_HAND_SECS 続く)--> 記録中
//      --(手なし が STOP_NO_HAND_SECS 続く)--> 確定(手ありの長さが MIN_SIGN_SECS 未満なら破棄)--> 待機
//      --(記録が MAX_RECORD_SECS を超える)--> 打ち切って確定 --> 待機(再アーム待ち)
// 数値はすべて仮置き(実機で調整する)。**秒**で持つ: 処理 fps は約 8〜17 で変わるため、
// フレーム数でなく `LiveFrame::t` の差で判定する。

/// 手あり がこの秒数続いたら記録開始
const AUTO_START_HAND_SECS: f64 = 0.3;
/// 手なし がこの秒数続いたら記録終了
const AUTO_STOP_NO_HAND_SECS: f64 = 0.5;
/// 記録がこの秒数を超えたら打ち切って確定
const AUTO_MAX_RECORD_SECS: f64 = 6.0;
/// 記録中に手が出ていた時間(最初の手あり〜最後の手あり)がこれ未満なら捨てる
const AUTO_MIN_SIGN_SECS: f64 = 0.4;
/// 打ち切り(または手動停止)の直後、手なしがこの秒数続くまで次の記録を始めない(再アーム)
const AUTO_REARM_NO_HAND_SECS: f64 = 0.5;
/// 記録開始時に含める直前の秒数(待機中のリングバッファ `PRE_ROLL_MAX_SECS` の範囲内)。
/// 開始判定の分(START_HAND_SECS)を含むので、手が出始めた瞬間より前に含まれるのは 0.5 秒
const AUTO_PRE_ROLL_SECS: f64 = PRE_ROLL_MAX_SECS;
/// 時刻差の比較の許容誤差(i/10 のような時刻の引き算で 0.39999.. になるのを吸収する)
const TIME_EPS: f64 = 1e-9;
/// 自動区切りが短すぎる記録を捨てたときに画面に出す文(結果行 `→ ...` とは区別する)
const DISCARD_MESSAGE: &str = "(短すぎる動きを無視しました)";

/// 自動区切りの閾値(テストで固定値を渡せるように struct にしてある)
#[derive(Debug, Clone, Copy)]
struct AutoParams {
    start_hand_secs: f64,
    stop_no_hand_secs: f64,
    max_record_secs: f64,
    min_sign_secs: f64,
    rearm_no_hand_secs: f64,
    pre_roll_secs: f64,
}

impl Default for AutoParams {
    fn default() -> Self {
        Self {
            start_hand_secs: AUTO_START_HAND_SECS,
            stop_no_hand_secs: AUTO_STOP_NO_HAND_SECS,
            max_record_secs: AUTO_MAX_RECORD_SECS,
            min_sign_secs: AUTO_MIN_SIGN_SECS,
            rearm_no_hand_secs: AUTO_REARM_NO_HAND_SECS,
            pre_roll_secs: AUTO_PRE_ROLL_SECS,
        }
    }
}

/// `elapsed` が `threshold` 秒に達したか(浮動小数の誤差を許容)
fn reached(elapsed: f64, threshold: f64) -> bool {
    elapsed + TIME_EPS >= threshold
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutoPhase {
    /// 待機中(記録していない)
    Idle,
    /// 手の出現で自動的に始めた記録。手なし・打ち切りで自動的に止まる
    Auto,
    /// Enter で手動に始めた記録。**自動では止めない**(止めるのも Enter)
    Manual,
}

/// L2: (時刻, 手あり)を 1 つずつ受け取り、境界(Start/Stop/Discard)を返す純粋な状態機械。
/// 実時間(Instant)は読まない。時刻はフレームの `t` だけを使うので、テストで時刻を与えて境界を固定できる。
///
/// 手あり判定は `LiveFrame::has_hand` 1 か所に集約している(§3.4 のキーボード上の手を
/// 手なし扱いにする条件を足すなら、そこを差し替える)。
#[derive(Debug)]
struct AutoSegmenter {
    p: AutoParams,
    phase: AutoPhase,
    /// 直前に見た時刻(巻き戻りの吸収用)
    last_t: Option<f64>,
    /// 待機中: 次の記録を始めてよいか。打ち切り・手動停止の直後は false で、
    /// 手なしが `rearm_no_hand_secs` 続くまで true にならない
    rearm_needed: bool,
    /// 待機中: 手あり が連続している区間の開始時刻
    hand_since: Option<f64>,
    /// 手なし が連続している区間の開始時刻(待機中は再アーム判定、自動記録中は停止判定に使う)
    absent_since: Option<f64>,
    /// 自動記録: 記録の開始時刻(= 開始フレームの時刻 - pre-roll)
    rec_start: f64,
    /// 自動記録: 最初/最後に手が見えた時刻(手ありの長さの判定用)
    first_hand: Option<f64>,
    last_hand: Option<f64>,
}

impl AutoSegmenter {
    fn new() -> Self {
        Self::with_params(AutoParams::default())
    }

    fn with_params(p: AutoParams) -> Self {
        Self {
            p,
            phase: AutoPhase::Idle,
            last_t: None,
            rearm_needed: false,
            hand_since: None,
            absent_since: None,
            rec_start: 0.0,
            first_hand: None,
            last_hand: None,
        }
    }

    /// 待機に戻して追跡状態をリセットする。`rearm` が true なら手なし待ち(再アーム)を要求する
    fn go_idle(&mut self, rearm: bool) {
        self.phase = AutoPhase::Idle;
        self.rearm_needed = rearm;
        self.hand_since = None;
        self.absent_since = None;
        self.first_hand = None;
        self.last_hand = None;
    }

    /// 1 フレーム進める。非有限の時刻は無視し、時刻の巻き戻りは直前の時刻に丸める
    /// (同時刻のフレームは経過 0 秒として扱うので、閾値を誤って超えない)
    fn step(&mut self, t: f64, hand: bool) -> Option<Boundary> {
        if !t.is_finite() {
            return None;
        }
        let t = self.last_t.map_or(t, |l| t.max(l));
        self.last_t = Some(t);
        match self.phase {
            AutoPhase::Idle => self.step_idle(t, hand),
            AutoPhase::Auto => self.step_auto(t, hand),
            // 手動で始めた記録は手なしでも打ち切りでも止めない(止めるのは Enter)
            AutoPhase::Manual => None,
        }
    }

    fn step_idle(&mut self, t: f64, hand: bool) -> Option<Boundary> {
        if hand {
            self.absent_since = None;
            if self.rearm_needed {
                // 打ち切り/手動停止の直後に手が写り続けている: 一度手なしになるまで始めない。
                // (再アームが無いと、6 秒打ち切りの直後に次の記録が即始まって同じ動きを二重に認識する)
                return None;
            }
            let since = *self.hand_since.get_or_insert(t);
            if reached(t - since, self.p.start_hand_secs) {
                self.phase = AutoPhase::Auto;
                self.rec_start = t - self.p.pre_roll_secs;
                self.first_hand = Some(since); // 手が出始めた時刻(判定待ちの間のフレームも手あり)
                self.last_hand = Some(t);
                self.absent_since = None;
                self.hand_since = None;
                return Some(Boundary::Start {
                    pre_roll_secs: self.p.pre_roll_secs,
                });
            }
        } else {
            self.hand_since = None; // 開始判定は手あり連続のみ。1 フレームの途切れでやり直し
            if self.rearm_needed {
                let since = *self.absent_since.get_or_insert(t);
                if reached(t - since, self.p.rearm_no_hand_secs) {
                    self.rearm_needed = false;
                }
            }
        }
        None
    }

    fn step_auto(&mut self, t: f64, hand: bool) -> Option<Boundary> {
        if hand {
            self.absent_since = None;
            self.first_hand.get_or_insert(t);
            self.last_hand = Some(t);
        } else {
            let since = *self.absent_since.get_or_insert(t);
            if reached(t - since, self.p.stop_no_hand_secs) {
                let sign_secs = match (self.first_hand, self.last_hand) {
                    (Some(a), Some(b)) => b - a,
                    _ => 0.0,
                };
                let short = !reached(sign_secs, self.p.min_sign_secs);
                self.go_idle(false); // すでに手なしが続いているので再アーム不要
                return Some(if short {
                    Boundary::Discard
                } else {
                    Boundary::Stop
                });
            }
        }
        if t - self.rec_start > self.p.max_record_secs + TIME_EPS {
            self.go_idle(true); // 手が写り続けていても、一度手なしになるまで再開しない
            return Some(Boundary::Stop);
        }
        None
    }
}

impl Segmenter for AutoSegmenter {
    /// 自動モードでも Enter で手動の開始/停止ができる。
    /// - 手動で始めた記録(Manual)は手の出入りでは止まらない(止めるのも Enter)。L1 と同じ操作感
    /// - 自動で始めた記録中に Enter を押すと、その場で停止して認識する(短さによる破棄はしない)
    /// - 停止後は、手が写り続けていても即再開しないよう再アームを要求する
    fn on_toggle(&mut self, recording: bool) -> Option<Boundary> {
        if recording {
            self.go_idle(true);
            Some(Boundary::Stop)
        } else {
            self.go_idle(false);
            self.phase = AutoPhase::Manual;
            Some(Boundary::Start { pre_roll_secs: 0.0 })
        }
    }

    fn on_frame(&mut self, frame: &LiveFrame, recording: bool) -> Option<Boundary> {
        // ループ側の録画状態とずれていたらループ側に合わせる(通常は起きない保険)
        if recording && self.phase == AutoPhase::Idle {
            self.phase = AutoPhase::Manual;
        } else if !recording && self.phase != AutoPhase::Idle {
            self.go_idle(false);
        }
        self.step(frame.t, frame.has_hand())
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
    /// 「このフレームに手があるか」の判定。自動区切り(`AutoSegmenter`)と最終的な区間選択
    /// (`finalize` の hits)の両方がここを通る。設計書 §3.4 のキーボード上の手を手なし扱いに
    /// する条件(手首の y が下端付近なら手なし、等)を足すなら、この 1 か所を差し替える
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
    /// 処理済みの 1 フレームを画面のプレビュー用に渡す(`run_session` がフレームごとに呼ぶ)。
    /// `raw` は認識に使った生フレームの**読み取り専用の参照**で、実装は書き換えない・保持しない
    /// (必要なら間引いたうえで自分用に縮小・複製する)。
    /// 既定は何もしない(`TerminalUi` はこれを使わないので挙動は変わらない)
    fn preview(&mut self, _raw: &Array3<u8>, _frame: &LiveFrame) {}
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
            (Boundary::Discard, true) => {
                self.recording = false;
                self.buf.clear();
                self.ui.message(DISCARD_MESSAGE);
            }
            // 録画中の Start / 待機中の Stop・Discard は無視
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
        lp.ui.preview(&sf.frame, &frame);
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

/// 区切り方
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentMode {
    /// 手の出入りで自動的に区切る(L2)。カメラでは Enter での手動操作も併用できる
    Auto,
    /// Enter で開始/停止(L1)。動画では「動画全体を 1 回の録画にする」
    Manual,
}

/// 区切り方を選ぶ(1 つ目 = 自動 = 既定)
fn choose_segment_mode(theme: &ColorfulTheme, items: &[&str; 2]) -> Result<SegmentMode> {
    let idx = Select::with_theme(theme)
        .with_prompt("区切り方")
        .items(items)
        .default(0)
        .interact()?;
    Ok(if idx == 0 {
        SegmentMode::Auto
    } else {
        SegmentMode::Manual
    })
}

fn make_segmenter(mode: SegmentMode) -> Box<dyn Segmenter> {
    match mode {
        SegmentMode::Auto => Box::new(AutoSegmenter::new()),
        SegmentMode::Manual => Box::new(ManualSegmenter),
    }
}

/// カメラ入力で最初に出す操作説明
fn camera_usage(mode: SegmentMode) -> &'static str {
    match mode {
        SegmentMode::Auto => {
            "操作: 手を出すと自動で記録開始 / 手を下げると自動で停止して認識\n\
             \x20     Enter = 手動で開始/停止も可(手動で始めた記録は手を下げても止まらず、Enter で止める)\n\
             \x20     q + Enter = 終了"
        }
        SegmentMode::Manual => {
            "操作: Enter = 記録開始 / もう一度 Enter = 停止して認識 / q + Enter = 終了"
        }
    }
}

/// メニュー「ライブ認識デモ」。TTY 必須(対話入力)。
pub fn live_wizard(theme: &ColorfulTheme) -> Result<()> {
    // 1. フレームの入力元
    let src_idx = Select::with_theme(theme)
        .with_prompt("フレームの入力元")
        .items(&[
            "カメラ(Mac 内蔵カメラ)",
            "動画ファイル(カメラの代わり。カメラ無しで区切りの経路を試せる)",
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
            let mode = choose_segment_mode(
                theme,
                &[
                    "自動(手の出入りで区切る。Enter で手動も可)",
                    "手動(Enter で開始/停止)",
                ],
            )?;
            let device: String = Input::with_theme(theme)
                .with_prompt("カメラのデバイス番号(avfoundation。FaceTime HD カメラは通常 0)")
                .default("0".into())
                .interact_text()?;
            println!(
                "\nカメラを起動します(初回はターミナルアプリにカメラ権限のダイアログが出ます。\n\
                 拒否した場合は システム設定 → プライバシーとセキュリティ → カメラ で許可)。"
            );
            let mut source = CameraSource::open(device.trim())?;
            println!("{}\n(カメラ映像・特徴量は保存しません)\n", camera_usage(mode));
            let mut keys = StdinKeys::spawn();
            let mut segmenter = make_segmenter(mode);
            run_session(
                &mut engine,
                &mut source,
                &mut keys,
                segmenter.as_mut(),
                &mut ui,
                &settings,
            )?;
            println!(
                "\n(推論が追いつかず捨てたカメラフレーム: {} 枚)",
                source.dropped_frames()
            );
        }
        _ => {
            let mode = choose_segment_mode(
                theme,
                &[
                    "自動(手の出入りで区切る。動画の中に複数のサインがあれば複数件出る)",
                    "動画全体を 1 回の録画にする(再生開始と同時に録画し、終端で認識)",
                ],
            )?;
            let video = select_video(theme)?;
            let mut source = VideoFileSource::open(&video, 1)?;
            // 自動: 手の出入りで区切るので Toggle は押さない。全体: 再生開始と同時に録画を始める
            // (Toggle を 1 回押した扱い)。どちらも終端で録画中なら認識される
            let mut keys = ScriptedKeys {
                queue: match mode {
                    SegmentMode::Auto => VecDeque::new(),
                    SegmentMode::Manual => VecDeque::from([Key::Toggle]),
                },
            };
            let mut segmenter = make_segmenter(mode);
            println!("\n動画を流します: {}\n", video.display());
            run_session(
                &mut engine,
                &mut source,
                &mut keys,
                segmenter.as_mut(),
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

    // ffmpeg が自分から止まって failure_reason が回収するとき、回収の前に before_reap が呼ばれる
    // (json_engine が PID の登録を外す。回収済みの PID に kill が飛ばないようにするため)。
    // カメラは開かず、`sleep` を ffmpeg の代わりの子プロセスにする
    #[cfg(unix)]
    #[test]
    fn camera_failure_reason_calls_before_reap_exactly_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        fn hook() {
            CALLS.fetch_add(1, Ordering::SeqCst);
        }
        let child = Command::new("sleep")
            .arg("30")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let start = Instant::now();
        let mut cam = CameraSource {
            child,
            // ffmpeg が止まったあとの状態(フレーム無し・closed)
            shared: Arc::new((
                Mutex::new(LatestSlot {
                    seq: 0,
                    frame: None,
                    stamp: start,
                    closed: true,
                }),
                Condvar::new(),
            )),
            reader: None,
            start,
            last_seq: 0,
            dropped: 0,
            before_reap: None,
        };
        cam.set_before_reap(hook);
        let err = match cam.next_frame() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("フレームが返った"),
        };
        assert!(err.contains("カメラ入力が止まりました"), "{err}");
        assert_eq!(CALLS.load(Ordering::SeqCst), 1);
        // 回収済み(kill されて wait 済み)なので、もう一度 try_wait しても終了済みと返る
        assert!(cam.child.try_wait().unwrap().is_some());
    }

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

    // ===== L2: AutoSegmenter(状態機械)=====

    /// テストで使う閾値。定数 AUTO_* を調整しても境界テストが壊れないよう、値を固定して渡す
    fn test_params() -> AutoParams {
        AutoParams {
            start_hand_secs: 0.3,
            stop_no_hand_secs: 0.5,
            max_record_secs: 6.0,
            min_sign_secs: 0.4,
            rearm_no_hand_secs: 0.5,
            pre_roll_secs: 0.5,
        }
    }

    /// 手あり(true)/なし(false)の列を 10fps(t = (offset + i) / 10)で流し、
    /// (フレーム番号 i, 境界) の一覧を返す
    fn drive(seg: &mut AutoSegmenter, offset: usize, pattern: &[bool]) -> Vec<(usize, Boundary)> {
        let mut out = Vec::new();
        for (i, &h) in pattern.iter().enumerate() {
            if let Some(b) = seg.step((offset + i) as f64 / 10.0, h) {
                out.push((i, b));
            }
        }
        out
    }

    /// `n` フレームぶんの true / false
    fn run(hand: bool, n: usize) -> Vec<bool> {
        vec![hand; n]
    }

    fn cat(parts: &[Vec<bool>]) -> Vec<bool> {
        parts.concat()
    }

    const START_PRE: Boundary = Boundary::Start { pre_roll_secs: 0.5 };

    // 開始: 手あり が 0.3 秒連続したフレームで Start(10fps なら 4 枚目)。それ未満では開始しない
    #[test]
    fn auto_starts_after_continuous_hand() {
        let mut s = AutoSegmenter::with_params(test_params());
        // 3 枚(経過 0.2 秒)では開始しない
        assert!(drive(&mut s, 0, &cat(&[run(true, 3), run(false, 3)])).is_empty());

        let mut s = AutoSegmenter::with_params(test_params());
        let b = drive(&mut s, 0, &run(true, 6));
        assert_eq!(b, vec![(3, START_PRE)]);
    }

    // 開始判定は連続のみ: 途中の手なし 1 枚でやり直しになる
    #[test]
    fn auto_start_run_resets_on_gap() {
        let mut s = AutoSegmenter::with_params(test_params());
        let b = drive(
            &mut s,
            0,
            &cat(&[run(true, 3), run(false, 1), run(true, 3)]), // 3 + 途切れ + 3(どちらも 0.2 秒)
        );
        assert!(b.is_empty(), "{:?}", b);
        // 途切れの後にあらためて 0.3 秒続けば開始
        let mut s = AutoSegmenter::with_params(test_params());
        let b = drive(&mut s, 0, &cat(&[run(true, 3), run(false, 1), run(true, 4)]));
        assert_eq!(b, vec![(7, START_PRE)]);
    }

    // 確定: 手なし が 0.5 秒続いたフレームで Stop(最初の手なしから 5 フレーム後)
    #[test]
    fn auto_stops_after_continuous_absence() {
        let mut s = AutoSegmenter::with_params(test_params());
        // 手あり 10 枚(i=0..=9)→ 手なし(i=10 が最初)。10 + 5 = 15 枚目で 0.5 秒
        let b = drive(&mut s, 0, &cat(&[run(true, 10), run(false, 8)]));
        assert_eq!(b, vec![(3, START_PRE), (15, Boundary::Stop)]);
    }

    // 瞬断: 手なしが 0.5 秒未満で手が戻れば確定しない。戻った後は手なしの数え直し
    #[test]
    fn auto_does_not_stop_on_brief_dropout() {
        let mut s = AutoSegmenter::with_params(test_params());
        let b = drive(
            &mut s,
            0,
            &cat(&[run(true, 8), run(false, 4), run(true, 3), run(false, 4), run(true, 3)]),
        );
        assert_eq!(b, vec![(3, START_PRE)], "0.4 秒の手なしが 2 回あっても止まらない");
    }

    // 6 秒打ち切り: 記録時間(開始フレームの時刻 - pre-roll から)が 6 秒を超えたフレームで Stop
    #[test]
    fn auto_cap_stops_after_max_record_secs() {
        let mut s = AutoSegmenter::with_params(test_params());
        let b = drive(&mut s, 0, &run(true, 100));
        // Start は i=3(t=0.3)。記録開始 = 0.3 - 0.5 = -0.2。t - (-0.2) > 6.0 は t=5.9(i=59)から
        assert_eq!(b, vec![(3, START_PRE), (59, Boundary::Stop)]);
    }

    // 再アーム: 打ち切り後に手が写り続けても次の記録は始まらない。手なし 0.5 秒で解除され、
    // その後 0.3 秒の手あり で再び始まる
    #[test]
    fn auto_rearms_only_after_hand_disappears() {
        let mut s = AutoSegmenter::with_params(test_params());
        let pattern = cat(&[
            run(true, 60),  // i=3 Start, i=59 打ち切り(Stop)。i=59 以降も手が写り続ける
            run(true, 20),  // 写り続けても再開しない
            run(false, 4),  // 手なし 0.3 秒(0.5 秒未満)→ まだ再アームしない
            run(true, 10),  // 手が戻っても再開しない
            run(false, 6),  // 手なし 0.5 秒以上 → 再アーム
            run(true, 5),   // 0.3 秒の手あり → Start
        ]);
        let b = drive(&mut s, 0, &pattern);
        let restart = 60 + 20 + 4 + 10 + 6 + 3;
        assert_eq!(
            b,
            vec![(3, START_PRE), (59, Boundary::Stop), (restart, START_PRE)]
        );
    }

    // 短すぎる区間の破棄: 手が出ていたのが 0.4 秒未満なら Discard、ちょうど 0.4 秒なら Stop
    #[test]
    fn auto_discards_too_short_sign() {
        // 手あり 4 枚(0.0〜0.3 = 0.3 秒)→ 開始はするが、確定時に 0.4 秒未満で破棄
        let mut s = AutoSegmenter::with_params(test_params());
        let b = drive(&mut s, 0, &cat(&[run(true, 4), run(false, 8)]));
        assert_eq!(b, vec![(3, START_PRE), (9, Boundary::Discard)]);

        // 手あり 5 枚(0.0〜0.4 = 0.4 秒)→ 確定
        let mut s = AutoSegmenter::with_params(test_params());
        let b = drive(&mut s, 0, &cat(&[run(true, 5), run(false, 8)]));
        assert_eq!(b, vec![(3, START_PRE), (10, Boundary::Stop)]);

        // 浮動小数の誤差: 0.7 - 0.3 = 0.39999999999999997 でも 0.4 秒として扱う
        let mut s = AutoSegmenter::with_params(test_params());
        let b = drive(&mut s, 3, &cat(&[run(true, 5), run(false, 8)]));
        assert_eq!(b[1].1, Boundary::Stop, "{:?}", b);
    }

    // 破棄・通常停止の後は、手なしが続いていた(0.5 秒)ので再アーム不要: すぐ次を始められる
    #[test]
    fn auto_can_restart_right_after_stop() {
        let mut s = AutoSegmenter::with_params(test_params());
        let pattern = cat(&[run(true, 10), run(false, 6), run(true, 4)]);
        let b = drive(&mut s, 0, &pattern);
        assert_eq!(
            b,
            vec![(3, START_PRE), (15, Boundary::Stop), (19, START_PRE)]
        );
    }

    // 手が一度も見えなければ何も起きない
    #[test]
    fn augo_idle_without_hands_does_nothing() {
        let mut s = AutoSegmenter::with_params(test_params());
        assert!(drive(&mut s, 0, &run(false, 100)).is_empty());
    }

    // pre-roll の秒数は Start に載る(既定値は PRE_ROLL_MAX_SECS = リングバッファの長さ以内)
    #[test]
    fn auto_start_carries_pre_roll_within_ring_buffer() {
        let mut s = AutoSegmenter::new();
        let b = drive(&mut s, 0, &run(true, 40));
        let Some((_, Boundary::Start { pre_roll_secs })) = b.first().copied() else {
            panic!("Start が出ない: {:?}", b);
        };
        assert!((pre_roll_secs - AUTO_PRE_ROLL_SECS).abs() < 1e-12);
        assert!(pre_roll_secs <= PRE_ROLL_MAX_SECS);
    }

    // 手動 Enter: 手動で始めた記録は、手なしが長く続いても・6 秒を超えても自動では止まらず、
    // Enter で止まる。停止後は手が写り続けていても即再開しない
    #[test]
    fn manual_enter_recording_is_not_auto_stopped() {
        let mut s = AutoSegmenter::with_params(test_params());
        assert_eq!(s.on_toggle(false), Some(Boundary::Start { pre_roll_secs: 0.0 }));
        let b = drive(&mut s, 0, &cat(&[run(true, 5), run(false, 100)]));
        assert!(b.is_empty(), "手動記録は自動停止しない: {:?}", b);
        let b = drive(&mut s, 105, &run(true, 100)); // 10 秒以上たっても打ち切らない
        assert!(b.is_empty(), "{:?}", b);
        assert_eq!(s.on_toggle(true), Some(Boundary::Stop));
        // 手が写り続けている → 再アーム待ち
        let b = drive(&mut s, 205, &run(true, 30));
        assert!(b.is_empty(), "{:?}", b);
        // 手なし 0.5 秒 → 手あり 0.3 秒で再び自動開始
        let b = drive(&mut s, 235, &cat(&[run(false, 6), run(true, 4)]));
        assert_eq!(b, vec![(9, START_PRE)]);
    }

    // 自動で始めた記録中に Enter を押すとその場で停止(破棄ではなく Stop)し、再アームを要求する
    #[test]
    fn enter_during_auto_recording_stops_and_requires_rearm() {
        let mut s = AutoSegmenter::with_params(test_params());
        let b = drive(&mut s, 0, &run(true, 6));
        assert_eq!(b, vec![(3, START_PRE)]);
        assert_eq!(s.on_toggle(true), Some(Boundary::Stop));
        let b = drive(&mut s, 6, &run(true, 20));
        assert!(b.is_empty(), "手が写り続けている間は再開しない: {:?}", b);
    }

    // 時刻の巻き戻り・同時刻・非有限値で壊れない
    #[test]
    fn auto_tolerates_odd_timestamps() {
        // 同時刻のフレームを何枚流しても経過 0 秒なので開始しない
        let mut s = AutoSegmenter::with_params(test_params());
        for _ in 0..50 {
            assert_eq!(s.step(1.0, true), None);
        }
        // 巻き戻った時刻は直前の時刻に丸められる(0.2 → 0.05 は 0.2 として扱う)
        let mut s = AutoSegmenter::with_params(test_params());
        assert_eq!(s.step(0.0, true), None);
        assert_eq!(s.step(0.1, true), None);
        assert_eq!(s.step(0.2, true), None);
        assert_eq!(s.step(0.05, true), None);
        assert_eq!(s.step(0.3, true), Some(START_PRE));
        // 非有限の時刻は無視する(状態も進めない)
        assert_eq!(s.step(f64::NAN, false), None);
        assert_eq!(s.step(f64::INFINITY, false), None);
        // 記録中の巻き戻りで手なし時間が負になって壊れない: 手なし開始 0.5 → 巻き戻し → 1.0 で確定
        assert_eq!(s.step(0.4, true), None); // 手ありの長さ 0.0〜0.4 = 0.4 秒(破棄されない長さ)
        assert_eq!(s.step(0.5, false), None);
        assert_eq!(s.step(0.1, false), None);
        assert_eq!(s.step(0.9, false), None);
        assert_eq!(s.step(1.0, false), Some(Boundary::Stop));
    }

    // 閾値は秒で決まる: 処理 fps が違っても(約 8 / 17 fps)同じ秒数で開始・確定する
    #[test]
    fn auto_thresholds_are_in_seconds_not_frames() {
        for fps in [8.0f64, 17.0] {
            let mut s = AutoSegmenter::with_params(test_params());
            let mut starts = Vec::new();
            let mut stops = Vec::new();
            // 手あり 0.0〜2.0 秒、その後 手なし
            for i in 0..((4.0 * fps) as usize) {
                let t = i as f64 / fps;
                match s.step(t, t <= 2.0) {
                    Some(Boundary::Start { .. }) => starts.push(t),
                    Some(Boundary::Stop) => stops.push(t),
                    _ => {}
                }
            }
            assert_eq!(starts.len(), 1, "fps={}", fps);
            assert_eq!(stops.len(), 1, "fps={}", fps);
            // 開始は手が出て 0.3 秒以上・1 フレーム以内、確定は最初の手なしから 0.5 秒以上・1 フレーム以内
            assert!(starts[0] >= 0.3 - 1e-9 && starts[0] < 0.3 + 1.0 / fps + 1e-9, "fps={} {:?}", fps, starts);
            let first_absent = (2.0 * fps).floor() / fps + 1.0 / fps;
            assert!(
                stops[0] >= first_absent + 0.5 - 1e-9 && stops[0] < first_absent + 0.5 + 1.0 / fps + 1e-9,
                "fps={} stop={:?} first_absent={}",
                fps,
                stops,
                first_absent
            );
        }
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

    // ===== L2: run_session に自動セグメンタを通す(偽の認識器・偽の入力元)=====

    /// 手あり(1)/なし(0)のフレーム列を FakeSource の値にする
    fn frames_from(parts: &[(u8, usize)]) -> Vec<u8> {
        parts.iter().flat_map(|&(v, n)| vec![v; n]).collect()
    }

    /// 流し終えたあとの結果と CollectUi を返す
    fn run_auto(
        values: Vec<u8>,
        keys: &mut dyn KeySource,
    ) -> (Vec<LiveResult>, CollectUi) {
        let mut source = FakeSource {
            values: values.into(),
            idx: 0,
        };
        let mut ui = CollectUi::default();
        let mut seg = AutoSegmenter::with_params(test_params());
        let results = run_session(
            &mut FakeRecognizer,
            &mut source,
            keys,
            &mut seg,
            &mut ui,
            &settings(),
        )
        .unwrap();
        (results, ui)
    }

    fn no_keys() -> ScriptedKeys {
        ScriptedKeys {
            queue: VecDeque::new(),
        }
    }

    /// i 回目のフレームを取る直前に Enter(Toggle)を押す。`poll` が None を返すたびに 1 フレーム進んだとみなす
    struct KeysAtFrames {
        schedule: VecDeque<(usize, Key)>,
        frame: usize,
    }
    impl KeySource for KeysAtFrames {
        fn poll(&mut self) -> Option<Key> {
            if self.schedule.front().is_some_and(|(f, _)| *f == self.frame) {
                return self.schedule.pop_front().map(|(_, k)| k);
            }
            self.frame += 1;
            None
        }
    }

    // 手あり/なしのフレーム列 → 結果が 1 件。pre-roll と余白込みの録画長・区間も確かめる
    #[test]
    fn auto_session_yields_one_result_per_sign() {
        // 手あり idx 20..=34 → Start は idx 23(t=2.3)、手なし idx 35 から 0.5 秒後の idx 40(t=4.0)で確定
        let values = frames_from(&[(0, 20), (1, 15), (0, 25)]);
        let (results, ui) = run_auto(values, &mut no_keys());
        assert_eq!(results.len(), 1);
        assert_eq!(ui.results, 1);
        assert!(ui.messages.is_empty(), "{:?}", ui.messages);
        let r = &results[0];
        // 録画 = pre-roll 0.5 秒前(idx 18)〜確定フレーム(idx 40)の 23 枚
        assert_eq!(r.buffered, 23);
        assert!((r.duration_secs - 2.2).abs() < 1e-6, "{}", r.duration_secs);
        // 手あり 15 枚(録画内の添字 2..=16)+ 余白 1 枚ずつ
        assert_eq!(r.selection.span, Some((1, 17)));
        assert!(r.selected_with_hand >= 8, "{}", r.selected_with_hand);
    }

    // 2 つのサイン(間に 1 秒の手なし)→ 2 件
    #[test]
    fn auto_session_yields_two_results_for_two_signs() {
        let values = frames_from(&[(0, 10), (1, 12), (0, 10), (1, 12), (0, 10)]);
        let (results, ui) = run_auto(values, &mut no_keys());
        assert_eq!(results.len(), 2);
        assert!(ui.messages.is_empty(), "{:?}", ui.messages);
    }

    // 手がかすめただけ(短すぎる)→ 結果なし、破棄メッセージだけ(結果行とは別)
    #[test]
    fn auto_session_discards_short_blip_with_message() {
        let values = frames_from(&[(0, 10), (1, 4), (0, 20)]);
        let (results, ui) = run_auto(values, &mut no_keys());
        assert!(results.is_empty());
        assert_eq!(ui.results, 0);
        assert_eq!(ui.messages, vec![DISCARD_MESSAGE.to_string()]);
        assert!(!DISCARD_MESSAGE.starts_with('→'));
    }

    // 破棄した後でも次のサインはちゃんと認識される(録画状態・バッファが戻っている)
    #[test]
    fn auto_session_recovers_after_discard() {
        let values = frames_from(&[(0, 10), (1, 4), (0, 10), (1, 12), (0, 10)]);
        let (results, ui) = run_auto(values, &mut no_keys());
        assert_eq!(results.len(), 1);
        assert_eq!(ui.messages, vec![DISCARD_MESSAGE.to_string()]);
    }

    // 手が一度も写らなければ推論しない・メッセージも出ない
    #[test]
    fn auto_session_without_hands_is_silent() {
        let (results, ui) = run_auto(vec![0u8; 80], &mut no_keys());
        assert!(results.is_empty());
        assert!(ui.messages.is_empty());
    }

    // 入力(動画)の終端で記録中なら、そこで確定して認識する(既存の挙動を維持)
    #[test]
    fn auto_session_finalizes_when_input_ends_while_recording() {
        let values = frames_from(&[(0, 10), (1, 15)]);
        let (results, _) = run_auto(values, &mut no_keys());
        assert_eq!(results.len(), 1);
    }

    // 6 秒打ち切りでも結果は出て、写り続けている手では二重に認識されない(再アーム)
    #[test]
    fn auto_session_cap_does_not_double_fire() {
        // 手あり 10 秒(100 枚)→ 手なし 1 秒。打ち切りで 1 件、再アームのため残りの手あり区間は始まらない
        let values = frames_from(&[(1, 100), (0, 10)]);
        let (results, ui) = run_auto(values, &mut no_keys());
        assert_eq!(results.len(), 1);
        assert!(ui.messages.is_empty(), "{:?}", ui.messages);
    }

    // 自動モード中の Enter(手動): 手動で始めた記録は、手が消えても止まらず Enter で止まる
    #[test]
    fn auto_session_manual_enter_records_until_enter() {
        // 手あり idx 8..=15、あとは手なし 40 枚まで。idx 5 の前で Enter(開始)、idx 35 の前で Enter(停止)
        let values = frames_from(&[(0, 8), (1, 8), (0, 24)]);
        let mut keys = KeysAtFrames {
            schedule: VecDeque::from([(5, Key::Toggle), (35, Key::Toggle)]),
            frame: 0,
        };
        let (results, ui) = run_auto(values, &mut keys);
        assert_eq!(results.len(), 1);
        assert!(ui.messages.is_empty(), "{:?}", ui.messages);
        // 録画 = idx 5..=34 の 30 枚(手が消えてから 1.9 秒たっても自動では止まらない)
        assert_eq!(results[0].buffered, 30);
    }

    // 自動で始めた記録中の Enter で停止し、手が写り続けていても即再開しない
    #[test]
    fn auto_session_enter_stops_auto_recording_without_restart() {
        let values = frames_from(&[(0, 5), (1, 35)]);
        // idx 5 から手あり。Start は idx 8。idx 12 の前で Enter(停止)
        let mut keys = KeysAtFrames {
            schedule: VecDeque::from([(12, Key::Toggle)]),
            frame: 0,
        };
        let (results, _) = run_auto(values, &mut keys);
        assert_eq!(results.len(), 1, "停止後は手が写り続けても再開しない");
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

    /// L2(自動区切り)の動画ファイルスモーク。raw_jsl の動画を「カメラの代わり」に自動区切りで流し、
    /// 1 本につき結果が概ね 1 件出るか、top-1 が既存の認識経路(extract_tag_features_impl → Predictor)と
    /// 大きく食い違わないかを見る。件数が 1 以外の動画は閾値との関係を数値付きで出力する
    /// (件数は assert せず報告のみ。top-1 の食い違いだけ assert)。
    ///
    /// 環境変数: LIVE_SMOKE_VIDEOS / LIVE_SMOKE_MODEL / LIVE_SMOKE_STRIDE(live_file_smoke と同じ)、
    ///   LIVE_SMOKE_AUTO_N  対象動画の本数(既定 8。LIVE_SMOKE_VIDEOS 未指定時)
    /// カメラは使わない。手動実行:
    ///   `cargo test live_file_auto_smoke --release -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_file_auto_smoke() {
        let model_dir = PathBuf::from(
            std::env::var("LIVE_SMOKE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL_DIR.into()),
        );
        for p in [model_dir.as_path(), Path::new(PALM_MODEL), Path::new(HAND_MODEL)] {
            if !p.exists() {
                eprintln!("skip: fixture not found: {}", p.display());
                return;
            }
        }
        let n: usize = std::env::var("LIVE_SMOKE_AUTO_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let videos: Vec<PathBuf> = match std::env::var("LIVE_SMOKE_VIDEOS") {
            Ok(v) => v.split(':').map(PathBuf::from).collect(),
            Err(_) => crate::tests::sample_raw_jsl_takes(n),
        };
        if videos.is_empty() {
            eprintln!("skip: raw_jsl の動画が無い");
            return;
        }
        let stride: usize = std::env::var("LIVE_SMOKE_STRIDE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let (roi, select, frames) = match load_model_preprocess(&model_dir).unwrap() {
            Some(p) => (p.roi, p.select, p.frames.unwrap_or(SEQ_LEN)),
            None => (PalmRoiMode::CenterSquare, SIGN_SPAN, SEQ_LEN),
        };
        eprintln!(
            "model={} roi={} select={} stride={} auto: start={}s stop={}s max={}s min_sign={}s",
            model_dir.display(),
            roi.label(),
            select.label(),
            stride,
            AUTO_START_HAND_SECS,
            AUTO_STOP_NO_HAND_SECS,
            AUTO_MAX_RECORD_SECS,
            AUTO_MIN_SIGN_SECS
        );
        let mut engine = Engine::load(&model_dir, roi).unwrap();
        let labels = load_labels();
        let mut offline_palm = Session::builder().unwrap().commit_from_file(PALM_MODEL).unwrap();
        let mut offline_hand = Session::builder().unwrap().commit_from_file(HAND_MODEL).unwrap();
        let anchors = load_palm_anchors();

        let (mut one, mut agree, mut cmp) = (0usize, 0usize, 0usize);
        for video in &videos {
            let name = format!(
                "{}/{}",
                video.parent().and_then(|p| p.file_name()).unwrap_or_default().to_string_lossy(),
                video.file_name().unwrap_or_default().to_string_lossy()
            );
            let mut source = VideoFileSource::open(video, stride).unwrap();
            let mut keys = no_keys();
            let mut ui = CollectUi::default();
            let mut seg = AutoSegmenter::new();
            let t0 = Instant::now();
            let results = run_session(
                &mut engine,
                &mut source,
                &mut keys,
                &mut seg,
                &mut ui,
                &Settings {
                    select,
                    frames,
                    topk: 5,
                },
            )
            .unwrap();
            let live_secs = t0.elapsed().as_secs_f64();

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
            let off_top = &offline[0].0;

            let fmt = |r: &[(String, f32)]| {
                r.iter()
                    .take(3)
                    .map(|(t, p)| format!("{} {:.2}", t, p))
                    .collect::<Vec<_>>()
                    .join(" / ")
            };
            eprintln!("\n=== {} ===  結果 {} 件 / 破棄 {} 件 ({:.1}s)", name, results.len(), ui.messages.len(), live_secs);
            for (i, r) in results.iter().enumerate() {
                eprintln!(
                    "  [{}] 録画{}枚 {:.2}s 推定{:.1}fps 区間{:?} 手あり{}/{}  {}",
                    i,
                    r.buffered,
                    r.duration_secs,
                    r.selection.fps,
                    r.selection.span,
                    r.selected_with_hand,
                    r.selection.indices.len(),
                    fmt(&r.ranked)
                );
                eprintln!("      表示: {}", format_result(&r.ranked, &labels));
            }
            for m in &ui.messages {
                eprintln!("  message: {}", m);
            }
            eprintln!("  offline: {}", fmt(&offline));
            match results.len() {
                1 => one += 1,
                0 => eprintln!("  !! 結果 0 件: 手の出ている時間・開始/破棄の閾値との関係を確認 (offline coverage L={:.0}% R={:.0}%)",
                    entry.left_hand_coverage * 100.0, entry.right_hand_coverage * 100.0),
                k => eprintln!("  !! 結果 {} 件に分割された: 手なし {}s 以上の途切れがサイン中にあるか、手の検出の途切れ", k, AUTO_STOP_NO_HAND_SECS),
            }
            if let Some(first) = results.first() {
                // 結果のうち、既存経路と最も合うものを見る(分割された場合に備え全件を対象)
                cmp += 1;
                let ok = results.iter().any(|r| {
                    r.ranked.iter().any(|(t, _)| t == off_top)
                        || offline.iter().any(|(t, _)| t == &r.ranked[0].0)
                });
                if first.ranked[0].0 == *off_top {
                    agree += 1;
                }
                assert!(
                    ok,
                    "{}: 自動区切りの結果が既存経路と大きく食い違う (offline top1={})",
                    name, off_top
                );
            }
        }
        eprintln!(
            "\n結果がちょうど 1 件: {}/{} 本 / 先頭結果の top-1 が既存経路と一致: {}/{} 本",
            one,
            videos.len(),
            agree,
            cmp
        );
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
