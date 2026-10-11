//! 撮影モード(R1): 録画 → 使えるか判定 → 採用すれば登録・保存。
//! 設計: docs/desktop-capture-design.md §3(境界)・§4.1(録画方式)・§4.2(プロトコル)・§4.5・§4.6。
//!
//! # 守っていること(ユーザー明示の境界)
//!
//! - **録画・保存のコードはこのファイルに閉じている**。ライブ認識(`run_session`・`JsonUi`・`serve_live`)から
//!   このモジュールを呼ぶ箇所は無い。ライブ認識と撮影モードが分かれるのは `json_engine::serve_dispatch` の 1 か所だけ
//!   (テスト `live_path_never_calls_capture` で、他から呼ばれていないことを固定している)
//! - 書き込みは次の 2 つだけ:
//!   (a) 一時領域への録画(リポジトリの外の `std::env::temp_dir()/attention-capture-<pid>/`)
//!   (b) 「採用」したときの、`raw_jsl/<word_id>_<romaji>/<NN>.mp4` への取り込みと `index.tsv` への記帳
//!   破棄・終了・異常終了のどれでも、一時ファイルは消す(強制終了で残ったものは、次の起動時に持ち主の PID が
//!   死んでいることを確かめてから掃除する)
//! - 語の推論はしない(認識モデルを読み込まない)。手検出(骨格・縁の警告に使う)とプレビューだけ動かす
//!
//! # 録画の方式(カメラは 1 本だけ開く)
//!
//! ```text
//! カメラ用 ffmpeg ──rawvideo──▶ 読み取りスレッド ─┬─▶ 最新フレーム置き場 ─▶ メインスレッド(手検出・プレビュー)
//!                                                 └─▶(録画中だけ)コピー ─▶ 録画用 ffmpeg(libx264)─▶ 一時領域の .mp4
//! ```
//!
//! - 読み取りスレッドが受けた**全フレーム**を録画に流す(手検出は追い付かなければ最新だけを見るが、録画は落とさない)
//! - 録画用のキューが詰まったら、そのフレームは**複製せず捨てて数える**(`dropped`)。画面に警告を出す
//! - 録画の fps は、カウントダウン中に実測したフレーム到着間隔から決める(`decide_record_fps`):
//!   30fps 前後なら 30、それ未満なら実測値をそのまま宣言する(フレームの複製はしない)
//!
//! # 判定と登録
//!
//! 判定は `progress.rs` の `HandChecker`(毎秒約 3 サンプル、連続して手が検出できた最長区間、瞬断 1 サンプルまで橋渡し)を
//! 録画した一時ファイルに対してそのまま使い、長さも `DURATION_MIN_MS` / `DURATION_MAX_MS` で評価する。
//! 登録は `progress.rs` の `read_words` / `scan_takes` / `load_index` / `merge_index` / `next_rep_idx` /
//! `new_take_row` / `stage_index` を使い、index.tsv は一時ファイル経由のアトミックな置き換えにする。

use super::json_engine::{
    clear_child_pid, pid_alive, register_child_pid, resolve_preview, round_to, CaptureWord,
    Command as EngineCommand, DeviceId, Emitter, Event, Input, JsonUi, PreviewCfg, SourceSpec, StartCmd, StopFlag, StopRequested,
    VerdictView,
};
use super::*;
use crate::progress::{self, WordSpec};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Sender, SyncSender, TrySendError};
use std::time::Duration;

// ===== 定数 =====

/// カウントダウンの既定(秒)。音に頼らないので、画面に大きく数字を出す
pub(super) const DEFAULT_COUNTDOWN_SECS: f64 = 3.0;
const MAX_COUNTDOWN_SECS: f64 = 10.0;
/// 自動停止する長さ(秒)。`progress.rs` の `DURATION_MAX_MS` と同じ
const MAX_RECORD_SECS: f64 = progress::DURATION_MAX_MS as f64 / 1000.0;
/// 既存データ(raw_jsl)の fps。カメラにもこれを要求している(`CAMERA_FPS`)
const NOMINAL_FPS: f64 = 30.0;
/// 実測 fps がこの範囲なら 30fps とみなす(カメラの到着間隔の揺れを吸収)
const FPS_SNAP_LOW: f64 = 28.5;
const FPS_SNAP_HIGH: f64 = 31.5;
/// fps を実測するのに必要な最小のフレーム数
const MIN_FPS_SAMPLES: usize = 8;
/// 実測に使う直近のフレーム到着時刻の個数(30fps で約 3 秒分)
const ARRIVAL_WINDOW: usize = 90;
/// 録画用エンコーダに渡すキューの長さ(フレーム数)。1 枚 約 5MB なので 24 枚で約 125MB
const RECORD_QUEUE_FRAMES: usize = 24;
/// 既存の raw_jsl の動画(約 7.7Mbps)に揃えたビットレート
const ENCODE_BITRATE: &str = "7700k";
const ENCODE_MAXRATE: &str = "9600k";
const ENCODE_BUFSIZE: &str = "15400k";
/// 状態の更新頻度(回/秒)。カウントダウン・録画中は速く、それ以外は 1 秒に 1 回
const STATE_ACTIVE_HZ: f64 = 10.0;
const STATE_IDLE_SECS: f64 = 1.0;
/// 入力を待つ最長時間。コマンドやカウントダウンをカメラのコマ待ちで遅らせないための周期
const POLL_TIMEOUT: Duration = Duration::from_millis(20);
/// エンコーダの終了を待つ最長時間(超えたら強制終了)
const ENCODER_FINISH_TIMEOUT: Duration = Duration::from_secs(15);
/// 一時領域のフォルダ名の接頭辞(`attention-capture-<pid>`)
const TEMP_PREFIX: &str = "attention-capture-";

// ===== 純粋な判断(fps・長さ・カウントダウン)=====

/// カウントダウン秒数の確定。省略や非有限は既定、範囲は 0〜10 秒
// [TEST向き] 純粋関数
pub(super) fn clamp_countdown(req: Option<f64>) -> f64 {
    match req {
        Some(v) if v.is_finite() => v.clamp(0.0, MAX_COUNTDOWN_SECS),
        _ => DEFAULT_COUNTDOWN_SECS,
    }
}

/// 自動停止までに録画するコマ数。`コマ数 ÷ fps` が**必ず 10 秒未満**になるようにする
/// (ちょうど 10.0 秒だと、長さの評価で `長すぎる` に誤って当たる境界を避けるため 1 コマ引く)
// [TEST向き] 純粋関数。30fps → 299 コマ(9.97 秒)、15fps → 149 コマ
pub(super) fn max_frames_for(fps: f64) -> u64 {
    if !(fps.is_finite() && fps > 0.0) {
        return 1;
    }
    ((MAX_RECORD_SECS * fps).ceil() as u64).saturating_sub(1).max(1)
}

/// 録画に使う fps と、画面に出す警告
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RecordFps {
    pub(super) fps: f64,
    pub(super) warning: Option<String>,
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// 実測のフレーム到着 fps から録画の fps を決める。
/// - 28.5〜31.5 → 30(カメラの揺れとみなす)
/// - それより遅い → 実測値をそのまま宣言し(フレームは複製しない)、警告を出す
/// - 測れなかった → 30 とみなし、警告を出す
// [TEST向き] 純粋関数。境界(28.5 / 31.5)・遅いカメラ・未計測
pub(super) fn decide_record_fps(measured: Option<f64>) -> RecordFps {
    match measured {
        Some(m) if m.is_finite() && m > 0.0 => {
            if (FPS_SNAP_LOW..=FPS_SNAP_HIGH).contains(&m) {
                RecordFps { fps: NOMINAL_FPS, warning: None }
            } else if m < FPS_SNAP_LOW {
                RecordFps {
                    fps: round3(m),
                    warning: Some(format!(
                        "カメラのコマが約 {:.1} fps しか届いていません(30 fps を要求)。コマは複製せず、実測の {:.1} fps で録画します。\
                         既存の撮影データ(30 fps)と動きの速さの扱いが違うので、このまま使うかは判断してください",
                        m, m
                    )),
                }
            } else {
                RecordFps { fps: round3(m), warning: None }
            }
        }
        _ => RecordFps {
            fps: NOMINAL_FPS,
            warning: Some(
                "カメラの fps を測れなかったため、30 fps として録画します(実際と違うと再生の速さがずれます)".into(),
            ),
        },
    }
}

/// 入力を開いてから、この秒数までのフレーム到着は fps の実測に使わない。
/// 起動直後はカメラ(や `-re` の動画)が溜まっていたコマを一気に出すことがあり、そのまま平均すると
/// 実際より速い fps に見えて、録画の再生が速くなってしまう(動画ファイル入力で 35 fps と測れた)
const FPS_WARMUP_SECS: f64 = 1.0;

/// 到着時刻の列から、起動直後の分を除いて fps を出す
// [TEST向き] 純粋関数。起動直後の一気読みを無視する
pub(super) fn measure_fps(arrivals: &[f64]) -> Option<f64> {
    let steady: Vec<f64> = arrivals.iter().copied().filter(|t| *t >= FPS_WARMUP_SECS).collect();
    fps_from_arrivals(&steady)
}

/// フレーム到着時刻(秒)の列から fps を出す。サンプルが少なければ None
// [TEST向き] 純粋関数
pub(super) fn fps_from_arrivals(times: &[f64]) -> Option<f64> {
    if times.len() < MIN_FPS_SAMPLES {
        return None;
    }
    let fps = processed_fps(times);
    (fps.is_finite() && fps > 0.0).then_some(fps)
}

fn format_fps(fps: f64) -> String {
    format!("{}", round3(fps))
}

/// 録画用 ffmpeg の引数(実行はしない純粋関数)。
/// 入力は rawvideo の rgb24(カメラ用 ffmpeg が出したものをそのまま)、出力は既存の raw_jsl の動画に揃えて
/// h264 High / yuv420p / 約 7.7Mbps / 一定フレームレート(`fps`)。音声なし
// [TEST向き] 純粋関数
pub(super) fn encoder_ffmpeg_args(width: usize, height: usize, fps: f64, out: &Path) -> Vec<String> {
    let mut a: Vec<String> = [
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    a.extend(["-video_size".into(), format!("{}x{}", width, height)]);
    a.extend(["-framerate".into(), format_fps(fps)]);
    a.extend(["-i".into(), "-".into(), "-an".into()]);
    a.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            ENCODE_BITRATE,
            "-maxrate",
            ENCODE_MAXRATE,
            "-bufsize",
            ENCODE_BUFSIZE,
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    a.push(out.to_string_lossy().to_string());
    a
}

/// 動画ファイルを「カメラの代わり」に流すときの ffmpeg 引数(開発・テスト用)。
/// `-re` で元の fps のまま実時間で流し、`-stream_loop -1` で終わらせない(カメラは終わらないため)
// [TEST向き] 純粋関数
pub(super) fn file_ffmpeg_args(path: &Path) -> Vec<String> {
    let mut a: Vec<String> = ["-nostdin", "-v", "error", "-re", "-stream_loop", "-1", "-i"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    a.push(path.to_string_lossy().to_string());
    a.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"].iter().map(|s| s.to_string()));
    a
}

// ===== 録画の分岐点(読み取りスレッドから録画用エンコーダへ)=====

/// 録画の統計。読み取りスレッドが更新し、メインスレッドが読む
#[derive(Debug, Default)]
pub(super) struct TeeStats {
    /// 録画キューに入れたコマ数
    sent: AtomicU64,
    /// キューが詰まって捨てたコマ数
    dropped: AtomicU64,
    /// 上限のコマ数に達して、録画への流し込みを止めた(= 自動停止)
    capped: AtomicBool,
    /// 録画用エンコーダが落ちて、流し込めなくなった
    closed: AtomicBool,
}

struct Tee {
    tx: SyncSender<Vec<u8>>,
    max_frames: u64,
    stats: Arc<TeeStats>,
}

/// 読み取りスレッドとメインスレッドで共有する「録画に流すか」の置き場。
/// 空のあいだ(録画していないあいだ)は何も流さない
#[derive(Clone, Default)]
pub(super) struct TeeSlot(Arc<Mutex<Option<Tee>>>);

impl TeeSlot {
    fn attach(&self, tee: Tee) {
        *lock(&self.0) = Some(tee);
    }

    /// 録画への流し込みを止める(送り口を手放す = 録画用エンコーダは入力の終わりを知る)
    fn detach(&self) {
        lock(&self.0).take();
    }

    /// 受け取った 1 フレーム(rgb24 の生バイト列)を、録画中なら録画キューへ流す。
    /// キューが詰まっていたら捨てて数える(複製はしない)。上限に達したら流し込みを止める
    pub(super) fn feed(&self, frame: &[u8]) {
        let mut g = lock(&self.0);
        let Some(t) = g.as_ref() else {
            return;
        };
        match t.tx.try_send(frame.to_vec()) {
            Ok(()) => {
                let n = t.stats.sent.fetch_add(1, Ordering::SeqCst) + 1;
                if n >= t.max_frames {
                    t.stats.capped.store(true, Ordering::SeqCst);
                    g.take();
                }
            }
            Err(TrySendError::Full(_)) => {
                t.stats.dropped.fetch_add(1, Ordering::SeqCst);
            }
            Err(TrySendError::Disconnected(_)) => {
                t.stats.closed.store(true, Ordering::SeqCst);
                g.take();
            }
        }
    }
}

// ===== 入力元(カメラ / 動画ファイル)=====

/// `poll_frame` の結果
pub(super) enum Polled {
    Frame(SourceFrame),
    /// 待っているあいだにフレームが来なかった(コマンドやカウントダウンを進めるための戻り)
    Idle,
    /// 入力が終わった(テストの偽の入力用。カメラと動画ファイルは終わらない)
    #[cfg_attr(not(test), allow(dead_code))]
    Ended,
}

/// 撮影モードの入力元。ライブ認識の `FrameSource` と違い、(1) 待ち時間に上限がある、
/// (2) 読み取りスレッドが全フレームを録画へ流せる、(3) フレーム到着の fps を実測できる
pub(super) trait CaptureInput {
    fn poll_frame(&mut self, timeout: Duration) -> Result<Polled>;
    /// 録画への分岐点
    fn tee(&self) -> TeeSlot;
    /// 直近のフレーム到着から実測した fps
    fn measured_fps(&self) -> Option<f64>;
    /// 手検出が追い付かず、最新だけを残して捨てたコマ数(録画には影響しない。`stopped` の報告用)
    fn dropped_frames(&self) -> u64;
}

/// 読み取りスレッド: パイプから全フレームを読み、(1) 到着時刻を記録、(2) 録画中なら録画へ流し、
/// (3) 最新フレーム置き場へ上書きする
fn spawn_capture_reader(
    mut stdout: impl Read + Send + 'static,
    width: usize,
    height: usize,
    shared: Arc<(Mutex<LatestSlot>, Condvar)>,
    tee: TeeSlot,
    arrivals: Arc<Mutex<VecDeque<f64>>>,
    start: Instant,
) -> std::thread::JoinHandle<()> {
    let frame_size = width * height * 3;
    std::thread::spawn(move || {
        let (slot, cv) = &*shared;
        while let Ok(Some(buf)) = read_one_frame(&mut stdout, frame_size) {
            {
                let mut a = lock(&arrivals);
                a.push_back(start.elapsed().as_secs_f64());
                while a.len() > ARRIVAL_WINDOW {
                    a.pop_front();
                }
            }
            // 録画への複製は、手検出に渡す(= 所有権を移す)より前に作る
            tee.feed(&buf);
            let Ok(frame) = Array3::from_shape_vec((height, width, 3), buf) else {
                break;
            };
            let mut s = lock(slot);
            s.seq += 1;
            s.frame = Some(frame);
            s.stamp = Instant::now();
            cv.notify_one();
        }
        lock(slot).closed = true;
        cv.notify_one();
    })
}

/// ffmpeg(カメラ / 動画ファイル)を入力元にする。`CameraSource` と同じ作りに、録画への分岐と到着時刻の記録を足したもの
struct FfmpegInput {
    child: Child,
    shared: Arc<(Mutex<LatestSlot>, Condvar)>,
    reader: Option<std::thread::JoinHandle<()>>,
    start: Instant,
    last_seq: u64,
    dropped: u64,
    tee: TeeSlot,
    arrivals: Arc<Mutex<VecDeque<f64>>>,
    what: &'static str,
}

impl FfmpegInput {
    fn spawn(args: Vec<String>, width: usize, height: usize, what: &'static str) -> Result<Self> {
        let mut child = Command::new("ffmpeg")
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("Failed to spawn ffmpeg ({what})"))?;
        let stdout = child.stdout.take().context("ffmpeg stdout not piped")?;
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
        let tee = TeeSlot::default();
        let arrivals = Arc::new(Mutex::new(VecDeque::new()));
        let reader = spawn_capture_reader(
            stdout,
            width,
            height,
            Arc::clone(&shared),
            tee.clone(),
            Arc::clone(&arrivals),
            start,
        );
        Ok(Self {
            child,
            shared,
            reader: Some(reader),
            start,
            last_seq: 0,
            dropped: 0,
            tee,
            arrivals,
            what,
        })
    }

    /// 注意: ここで**カメラが起動する**
    fn open_camera(device: &str) -> Result<Self> {
        Self::spawn(
            camera_ffmpeg_args(device),
            CAMERA_CROP_W as usize,
            CAMERA_CAPTURE_H as usize,
            "camera",
        )
    }

    fn open_file(path: &Path) -> Result<Self> {
        let info = probe_video(&path.to_path_buf())?;
        Self::spawn(
            file_ffmpeg_args(path),
            info.width as usize,
            info.height as usize,
            "file",
        )
    }

    /// ffmpeg が止まったときの理由(stderr)を取り出す。回収(wait)する直前に PID の登録を外す
    fn failure_reason(&mut self) -> String {
        let _ = self.child.kill();
        clear_child_pid();
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
}

impl CaptureInput for FfmpegInput {
    fn poll_frame(&mut self, timeout: Duration) -> Result<Polled> {
        let shared = Arc::clone(&self.shared);
        let (slot, cv) = &*shared;
        let mut s = lock(slot);
        if s.frame.is_none() && !s.closed {
            let (g, _) = cv.wait_timeout(s, timeout).unwrap_or_else(|e| e.into_inner());
            s = g;
        }
        if let Some(frame) = s.frame.take() {
            self.dropped += s.seq - self.last_seq - 1;
            self.last_seq = s.seq;
            let t = s.stamp.duration_since(self.start).as_secs_f64();
            return Ok(Polled::Frame(SourceFrame { frame, t }));
        }
        if !s.closed {
            return Ok(Polled::Idle);
        }
        drop(s);
        let what = self.what;
        let reason = self.failure_reason();
        anyhow::bail!(
            "入力が止まりました({}。権限・デバイス番号・解像度を確認してください):\n{}",
            what,
            reason
        )
    }

    fn tee(&self) -> TeeSlot {
        self.tee.clone()
    }

    fn measured_fps(&self) -> Option<f64> {
        let a = lock(&self.arrivals);
        let v: Vec<f64> = a.iter().copied().collect();
        measure_fps(&v)
    }

    fn dropped_frames(&self) -> u64 {
        self.dropped
    }
}

impl Drop for FfmpegInput {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(h) = self.reader.take() {
            let _ = h.join();
        }
    }
}

// ===== 手検出(語の推論はしない)=====

/// 1 フレームから手を検出する。実体は Palm + Hand の ONNX、テストでは差し替える
pub(super) trait HandObserver {
    fn observe(&mut self, frame: &Array3<u8>) -> Result<Observation>;
}

/// 実際の手検出。認識モデルは読み込まない(`Engine` と違い、`Predictor` を持たない)
struct HandOnly {
    palm: Session,
    hand: Session,
    anchors: Vec<[f32; 2]>,
    roi: PalmRoiMode,
}

impl HandOnly {
    fn load(roi: PalmRoiMode) -> Result<Self> {
        let palm = Session::builder()?
            .commit_from_file(PALM_MODEL)
            .with_context(|| format!("failed to load palm model: {}", PALM_MODEL))?;
        let hand = Session::builder()?
            .commit_from_file(HAND_MODEL)
            .with_context(|| format!("failed to load hand model: {}", HAND_MODEL))?;
        Ok(Self {
            palm,
            hand,
            anchors: load_palm_anchors(),
            roi,
        })
    }
}

impl HandObserver for HandOnly {
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
            None,
        )?;
        Ok(Observation { feature, left, right })
    }
}

// ===== 録画用エンコーダ =====

/// エンコーダの作成に必要な情報
#[derive(Debug, Clone)]
pub(super) struct EncoderSpec {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) fps: f64,
    pub(super) out: PathBuf,
}

/// 録画の書き込み先と、書き終えたあとの後始末(エンコーダの終了待ち)
pub(super) struct EncoderParts {
    pub(super) sink: Box<dyn Write + Send>,
    pub(super) finish: Box<dyn FnOnce() -> Result<()> + Send>,
}

pub(super) type EncoderFactory = Box<dyn Fn(&EncoderSpec) -> Result<EncoderParts>>;

/// 本番のエンコーダ: 別の ffmpeg(libx264)の標準入力に生フレームを書く
fn ffmpeg_encoder(spec: &EncoderSpec) -> Result<EncoderParts> {
    let mut child = Command::new("ffmpeg")
        .args(encoder_ffmpeg_args(spec.width, spec.height, spec.fps, &spec.out))
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn ffmpeg (encoder)")?;
    let stdin = child.stdin.take().context("ffmpeg stdin not piped")?;
    let finish = move || -> Result<()> {
        // 入力(stdin)は呼び出し側が閉じたあと。エンコーダが書き終えて終わるのを待つ
        let deadline = Instant::now() + ENCODER_FINISH_TIMEOUT;
        let status = loop {
            match child.try_wait()? {
                Some(st) => break st,
                None if Instant::now() >= deadline => {
                    let _ = child.kill();
                    let _ = child.wait();
                    anyhow::bail!("録画用の ffmpeg が終わらなかったため止めました");
                }
                None => std::thread::sleep(Duration::from_millis(10)),
            }
        };
        if !status.success() {
            let mut msg = String::new();
            if let Some(mut err) = child.stderr.take() {
                let _ = err.read_to_string(&mut msg);
            }
            anyhow::bail!("録画用の ffmpeg が異常終了しました({}): {}", status, msg.trim());
        }
        Ok(())
    };
    Ok(EncoderParts {
        sink: Box::new(stdin),
        finish: Box::new(finish),
    })
}

/// 録画中の状態
struct Recording {
    out: PathBuf,
    fps: f64,
    stats: Arc<TeeStats>,
    writer: Option<std::thread::JoinHandle<std::result::Result<u64, String>>>,
    started: Instant,
}

/// 録画を終えた結果
#[derive(Debug, PartialEq)]
struct RecordOutcome {
    frames: u64,
    dropped: u64,
}

/// 書き込みスレッド: キューのフレームをエンコーダへ書き、入力の終わり(送り口が全て閉じた)で後始末する。
/// 書けなくなったら受け取りをやめる(受け口を閉じる = 読み取りスレッド側が `closed` に気づく)
fn run_writer(rx: Receiver<Vec<u8>>, parts: EncoderParts) -> std::result::Result<u64, String> {
    let EncoderParts { mut sink, finish } = parts;
    let mut written = 0u64;
    let mut write_err: Option<String> = None;
    while let Ok(buf) = rx.recv() {
        if let Err(e) = sink.write_all(&buf) {
            write_err = Some(format!("エンコーダに書けません: {e}"));
            break;
        }
        written += 1;
    }
    drop(rx);
    drop(sink); // 入力を閉じる = エンコーダは書き終える
    let fin = finish();
    match (write_err, fin) {
        (Some(e), _) => Err(e),
        (None, Err(e)) => Err(format!("{e:#}")),
        (None, Ok(())) => Ok(written),
    }
}

fn start_recording(tee: &TeeSlot, factory: &EncoderFactory, spec: &EncoderSpec) -> Result<Recording> {
    let parts = factory(spec)?;
    let (tx, rx) = sync_channel::<Vec<u8>>(RECORD_QUEUE_FRAMES);
    let stats = Arc::new(TeeStats::default());
    let writer = std::thread::spawn(move || run_writer(rx, parts));
    tee.attach(Tee {
        tx,
        max_frames: max_frames_for(spec.fps),
        stats: Arc::clone(&stats),
    });
    Ok(Recording {
        out: spec.out.clone(),
        fps: spec.fps,
        stats,
        writer: Some(writer),
        started: Instant::now(),
    })
}

impl Recording {
    /// 録画に書いたコマ数から見た長さ(秒)
    fn secs(&self) -> f64 {
        self.stats.sent.load(Ordering::SeqCst) as f64 / self.fps
    }

    /// 自動停止の条件(コマ数の上限 / エンコーダが落ちた / コマが来なくて時間だけ過ぎた)
    fn should_auto_stop(&self) -> bool {
        self.stats.capped.load(Ordering::SeqCst)
            || self.stats.closed.load(Ordering::SeqCst)
            || self.started.elapsed().as_secs_f64() > MAX_RECORD_SECS + 3.0
    }

    /// 流し込みを止め、エンコーダが書き終えるのを待つ
    fn finish(mut self, tee: &TeeSlot) -> std::result::Result<RecordOutcome, String> {
        tee.detach();
        let dropped = self.stats.dropped.load(Ordering::SeqCst);
        let frames = match self.writer.take() {
            Some(h) => h.join().unwrap_or_else(|_| Err("録画スレッドが異常終了しました".into()))?,
            None => 0,
        };
        Ok(RecordOutcome { frames, dropped })
    }
}

// ===== 判定 =====

/// 録画した 1 本の判定の材料(手検出の最長連続区間と、動画の長さ)
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct JudgeOutcome {
    pub(super) left_secs: f32,
    pub(super) right_secs: f32,
    pub(super) duration_ms: u64,
}

pub(super) trait TakeJudge {
    fn judge(&mut self, video: &Path) -> Result<JudgeOutcome>;
}

/// 本番の判定: `progress.rs` の `HandChecker`(撮影セッションの取り込みと同じ規則・同じ関数)を使う
struct RealJudge {
    checker: progress::HandChecker,
}

impl TakeJudge for RealJudge {
    fn judge(&mut self, video: &Path) -> Result<JudgeOutcome> {
        let path = video.to_path_buf();
        let info = probe_video(&path)?;
        let duration_ms = (info.duration * 1000.0) as u64;
        let (left_secs, right_secs) = self.checker.coverage(&path)?;
        Ok(JudgeOutcome {
            left_secs,
            right_secs,
            duration_ms,
        })
    }
}

pub(super) type JudgeFactory = Box<dyn FnOnce() -> Result<Box<dyn TakeJudge>> + Send>;

/// 判定を別スレッドで行う(数秒かかる。メインスレッドの手検出・プレビューを止めない)。
/// モデルの読み込みもこのスレッドで、録画の前から始めておく
struct JudgeWorker {
    tx: Option<Sender<PathBuf>>,
    rx: Receiver<std::result::Result<JudgeOutcome, String>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl JudgeWorker {
    fn spawn(factory: JudgeFactory) -> Self {
        let (job_tx, job_rx) = mpsc::channel::<PathBuf>();
        let (out_tx, out_rx) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let mut judge = factory();
            while let Ok(path) = job_rx.recv() {
                let r = match &mut judge {
                    Ok(j) => j.judge(&path).map_err(|e| format!("{e:#}")),
                    Err(e) => Err(format!("{e:#}")),
                };
                if out_tx.send(r).is_err() {
                    break;
                }
            }
        });
        Self {
            tx: Some(job_tx),
            rx: out_rx,
            handle: Some(handle),
        }
    }

    fn submit(&self, path: PathBuf) -> bool {
        self.tx.as_ref().is_some_and(|t| t.send(path).is_ok())
    }

    /// 仕事を受け付けなくして、走っている判定が終わるのを待つ(ffmpeg を残さないため)
    fn shutdown(&mut self) {
        self.tx = None;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// 判定。画面に出す 1 文と、採用したときに index.tsv へ書く値を持つ
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Verdict {
    pub(super) usable: bool,
    pub(super) reasons: Vec<&'static str>,
    pub(super) summary: String,
    pub(super) left_secs: f32,
    pub(super) right_secs: f32,
    pub(super) duration_ms: u64,
    pub(super) quality_flag: String,
}

impl Verdict {
    pub(super) fn longest(&self) -> f32 {
        self.left_secs.max(self.right_secs)
    }

    fn view(&self) -> VerdictView {
        VerdictView {
            usable: self.usable,
            reasons: self.reasons.clone(),
            summary: self.summary.clone(),
            longest_run_secs: round_to(self.longest() as f64, 2),
            left_run_secs: round_to(self.left_secs as f64, 2),
            right_run_secs: round_to(self.right_secs as f64, 2),
            duration_secs: round_to(self.duration_ms as f64 / 1000.0, 2),
            quality_flag: self.quality_flag.clone(),
        }
    }
}

/// 判定の材料から結論を出す。手が連続して検出できた最長区間は `progress::quality_flag_for_run`
/// (撮影セッションの取り込みと同じ規則)、長さは `DURATION_MIN_MS` / `DURATION_MAX_MS`。
/// quality_flag は、手が足りなければ "ng_hands"、手は足りていて長さだけ範囲外なら "ng_duration"(どちらも目標数に数えない)
// [TEST向き] 純粋関数。手の境界(ちょうど 1.0 秒)・長さの境界(2 秒 / 10 秒)・理由の組み合わせ・コマ落ち
pub(super) fn evaluate(outcome: &JudgeOutcome, dropped: u64) -> Verdict {
    let longest = outcome.left_secs.max(outcome.right_secs);
    let hands_ok = progress::quality_flag_for_run(longest) == "ok";
    let mut reasons: Vec<&'static str> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    if !hands_ok {
        reasons.push("hands_short");
        texts.push(format!(
            "手が {:.1} 秒続いて検出されていません(最長 {:.2} 秒)",
            progress::MIN_HAND_RUN_SECONDS,
            longest
        ));
    }
    let secs = outcome.duration_ms as f64 / 1000.0;
    if outcome.duration_ms < progress::DURATION_MIN_MS {
        reasons.push("too_short");
        texts.push(format!(
            "短すぎます({:.1} 秒。{:.1} 秒以上が目安)",
            secs,
            progress::DURATION_MIN_MS as f64 / 1000.0
        ));
    } else if outcome.duration_ms > progress::DURATION_MAX_MS {
        reasons.push("too_long");
        texts.push(format!(
            "長すぎます({:.1} 秒。{:.1} 秒以下が目安)",
            secs,
            progress::DURATION_MAX_MS as f64 / 1000.0
        ));
    }
    let usable = reasons.is_empty();
    let quality_flag = if !hands_ok {
        "ng_hands"
    } else if !usable {
        "ng_duration"
    } else {
        "ok"
    };
    let mut summary = if usable {
        format!(
            "使えそう: 手が最長 {:.2} 秒つづけて検出できました(長さ {:.1} 秒)",
            longest, secs
        )
    } else {
        format!("使えない: {}", texts.join("、"))
    };
    if dropped > 0 {
        summary.push_str(&format!(
            "(コマ落ち {} 枚あり。動きが少し速く記録されています)",
            dropped
        ));
    }
    Verdict {
        usable,
        reasons,
        summary,
        left_secs: outcome.left_secs,
        right_secs: outcome.right_secs,
        duration_ms: outcome.duration_ms,
        quality_flag: quality_flag.to_string(),
    }
}

/// 判定そのものができなかったとき(モデルが読めない・ffprobe の失敗など)。
/// 撮影セッションの取り込みと同じく quality_flag は空にする(採用するなら、確認を求める)
pub(super) fn evaluate_failed(why: &str) -> Verdict {
    Verdict {
        usable: false,
        reasons: vec!["judge_failed"],
        summary: format!("判定できませんでした: {}", why.lines().next().unwrap_or("")),
        left_secs: 0.0,
        right_secs: 0.0,
        duration_ms: 0,
        quality_flag: String::new(),
    }
}

// ===== 状態機械(カウントダウン・録画・判定の遷移)=====

/// 撮影の段階
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase {
    Idle,
    Countdown,
    Recording,
    Judging,
    Judged,
}

impl Phase {
    pub(super) fn name(self) -> &'static str {
        match self {
            Phase::Idle => "idle",
            Phase::Countdown => "countdown",
            Phase::Recording => "recording",
            Phase::Judging => "judging",
            Phase::Judged => "judged",
        }
    }
}

/// 状態の遷移だけを持つ(時計は引数で受ける = 純粋にテストできる)
#[derive(Debug)]
pub(super) struct Machine {
    phase: Phase,
    countdown_secs: f64,
    countdown_ends: f64,
}

/// 受け付けない理由(画面に出す)
pub(super) type Rejected = String;

impl Machine {
    pub(super) fn new(countdown_secs: f64) -> Self {
        Self {
            phase: Phase::Idle,
            countdown_secs,
            countdown_ends: 0.0,
        }
    }

    pub(super) fn phase(&self) -> Phase {
        self.phase
    }

    fn reject(&self, what: &str) -> Rejected {
        let now = match self.phase {
            Phase::Idle => "待機中",
            Phase::Countdown => "カウントダウン中",
            Phase::Recording => "録画中",
            Phase::Judging => "判定中",
            Phase::Judged => "判定済み(採用か破棄を選んでください)",
        };
        format!("{what}できる状態ではありません(今は{now})")
    }

    /// 録画開始。待機中のみ
    pub(super) fn record(&mut self, now: f64) -> std::result::Result<(), Rejected> {
        if self.phase != Phase::Idle {
            return Err(self.reject("録画を始め"));
        }
        self.phase = Phase::Countdown;
        self.countdown_ends = now + self.countdown_secs;
        Ok(())
    }

    /// カウントダウンの残り秒数(カウントダウン中以外は 0)
    pub(super) fn countdown_left(&self, now: f64) -> f64 {
        if self.phase == Phase::Countdown {
            (self.countdown_ends - now).max(0.0)
        } else {
            0.0
        }
    }

    pub(super) fn countdown_done(&self, now: f64) -> bool {
        self.phase == Phase::Countdown && now >= self.countdown_ends
    }

    /// カウントダウンが終わって録画に入る
    pub(super) fn begin_recording(&mut self) {
        if self.phase == Phase::Countdown {
            self.phase = Phase::Recording;
        }
    }

    /// 録画を終えて判定に入る
    pub(super) fn begin_judging(&mut self) {
        if self.phase == Phase::Recording {
            self.phase = Phase::Judging;
        }
    }

    pub(super) fn finish_judging(&mut self) {
        if self.phase == Phase::Judging {
            self.phase = Phase::Judged;
        }
    }

    /// 停止の受け付け。録画中なら `Stop::Finish`(録画を終えて判定へ)、カウントダウン中なら `Stop::Cancel`(取り消し)
    pub(super) fn stop(&mut self) -> std::result::Result<Stop, Rejected> {
        match self.phase {
            Phase::Recording => Ok(Stop::Finish),
            Phase::Countdown => {
                self.phase = Phase::Idle;
                Ok(Stop::Cancel)
            }
            _ => Err(self.reject("停止")),
        }
    }

    /// 採用の受け付け(判定済みのみ)。成功したら呼び出し側が `reset` する
    pub(super) fn can_accept(&self) -> std::result::Result<(), Rejected> {
        if self.phase == Phase::Judged {
            Ok(())
        } else {
            Err(self.reject("採用"))
        }
    }

    /// 破棄の受け付け。カウントダウン(取り消し)・録画中(録画を捨てる)・判定済みで可。
    /// 判定中は判定が終わるまで待つ必要がある(判定の ffmpeg が一時ファイルを読んでいるため)
    pub(super) fn discard(&mut self) -> std::result::Result<Phase, Rejected> {
        match self.phase {
            Phase::Countdown | Phase::Recording | Phase::Judged => {
                let from = self.phase;
                self.phase = Phase::Idle;
                Ok(from)
            }
            _ => Err(self.reject("破棄")),
        }
    }

    pub(super) fn reset(&mut self) {
        self.phase = Phase::Idle;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Stop {
    /// 録画を終えて判定へ
    Finish,
    /// カウントダウンの取り消し
    Cancel,
}

// ===== 一時領域 =====

/// 録画の一時置き場(リポジトリの外)。`std::env::temp_dir()/attention-capture-<pid>/`。
/// 破棄・終了で消す。強制終了で残ったものは、次の起動時に持ち主の PID が死んでいれば掃除する
pub(super) struct TempArea {
    dir: PathBuf,
    counter: u32,
}

/// `attention-capture-<pid>` から pid を取り出す
// [TEST向き] 純粋関数
fn temp_dir_pid(name: &str) -> Option<u32> {
    let rest = name.strip_prefix(TEMP_PREFIX)?;
    if rest.is_empty() || !rest.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

impl TempArea {
    pub(super) fn create() -> Result<Self> {
        Self::create_in(&std::env::temp_dir())
    }

    /// `base` の下に作る(テストが専用の場所で試せるように分けてある)
    pub(super) fn create_in(base: &Path) -> Result<Self> {
        Self::sweep_stale(base);
        let dir = base.join(format!("{}{}", TEMP_PREFIX, std::process::id()));
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("一時領域を作れません: {}", dir.display()))?;
        Ok(Self { dir, counter: 0 })
    }

    pub(super) fn dir(&self) -> &Path {
        &self.dir
    }

    /// 次の録画の保存先(まだ作らない)
    pub(super) fn next_take_path(&mut self) -> PathBuf {
        self.counter += 1;
        self.dir.join(format!("take-{}.mp4", self.counter))
    }

    /// 強制終了で残った、持ち主の死んだ一時フォルダを掃除する。自分のものと、生きている持ち主のものには触れない。
    /// 中身は `take-*.mp4` だけを消し、そのほかのファイルがあればフォルダごと残す
    fn sweep_stale(base: &Path) {
        let Ok(entries) = std::fs::read_dir(base) else {
            return;
        };
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let Some(pid) = temp_dir_pid(&name) else {
                continue;
            };
            if pid == std::process::id() || pid_alive(pid) {
                continue;
            }
            let dir = e.path();
            if let Ok(files) = std::fs::read_dir(&dir) {
                for f in files.flatten() {
                    let fname = f.file_name().to_string_lossy().to_string();
                    if fname.starts_with("take-") && fname.ends_with(".mp4") {
                        let _ = std::fs::remove_file(f.path());
                    }
                }
            }
            let _ = std::fs::remove_dir(&dir); // 空のときだけ消える
        }
    }

    /// 一時領域をまるごと消す
    pub(super) fn cleanup(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl Drop for TempArea {
    fn drop(&mut self) {
        self.cleanup();
    }
}

// ===== 語の決定と登録(採用)=====

/// 録画する語を決める。`requested` は画面が出しているお題の語。省略(または空)なら、
/// `progress.rs` の `next_word` で「次の語」を選ぶ。**書き込まない**(words.tsv が無くてもテンプレートを作らない)
pub(super) fn resolve_word(data_dir: &Path, requested: Option<&str>) -> Result<CaptureWord> {
    let words = progress::read_words(data_dir)?;
    let (scanned, mut warnings) = progress::scan_takes(data_dir)?;
    let requested = requested.map(str::trim).filter(|s| !s.is_empty());
    let word: WordSpec = match requested {
        Some(id) => words
            .iter()
            .find(|w| w.word_id == id)
            .cloned()
            .with_context(|| format!("words.tsv に語 {} がありません", id))?,
        None => {
            let existing = progress::load_index(&data_dir.join(progress::INDEX_FILE))?;
            let rows = progress::merge_index(&scanned, &existing, &words, &mut warnings);
            progress::next_word(&words, &rows, &HashSet::new())
                .cloned()
                .context("すべての語が目標のテイク数に達しています")?
        }
    };
    Ok(CaptureWord {
        rep_idx: progress::scanned_next_rep(&scanned, &word.word_id),
        word_id: word.word_id,
        romaji: word.romaji,
        label: word.label_ja,
    })
}

/// 採用して登録した結果
#[derive(Debug, Clone)]
pub(super) struct Registered {
    /// `data_dir` からの相対パス("011_kare/04.mp4")
    pub(super) rel_path: String,
    pub(super) abs_path: PathBuf,
    pub(super) rep_idx: u32,
    pub(super) word: WordSpec,
    pub(super) quality_flag: String,
}

/// 一時領域の録画 `src` を `raw_jsl/<word_id>_<romaji>/<NN>.mp4` に移し、index.tsv に 1 行足す。
///
/// 壊さないことを最優先にした順序:
/// 1. 実ファイルと index を同期した行(`merge_index`)に新しい行を足し、**一時ファイルに**新しい index.tsv を書く
/// 2. 動画を移す(`rename`。別ボリュームならコピーして元を消す)。失敗したら一時 index を捨てて何も変えずに終わる
/// 3. 一時 index を `rename` で本物に置き換える。失敗したら動画を元の場所に戻す
/// `<NN>.mp4` が既にあれば番号を進める(上書きしない)
pub(super) fn register_take(
    data_dir: &Path,
    src: &Path,
    word_id: &str,
    verdict: &Verdict,
) -> Result<Registered> {
    let words = progress::read_words(data_dir)?;
    let word = words
        .iter()
        .find(|w| w.word_id == word_id)
        .cloned()
        .with_context(|| format!("words.tsv に語 {} がありません", word_id))?;
    let (scanned, mut warnings) = progress::scan_takes(data_dir)?;
    let index_path = data_dir.join(progress::INDEX_FILE);
    let existing = progress::load_index(&index_path)?;
    let mut rows = progress::merge_index(&scanned, &existing, &words, &mut warnings);
    for w in &warnings {
        eprintln!("警告: {}", w);
    }

    let mut rep = progress::next_rep_idx(&rows, &word.word_id);
    let dest_dir = data_dir.join(progress::take_dir_name(&word));
    let (dest_name, dest) = loop {
        let name = progress::take_file_name(rep, "mp4");
        let path = dest_dir.join(&name);
        if !path.exists() {
            break (name, path);
        }
        rep += 1;
    };
    rows.push(progress::new_take_row(
        &word,
        rep,
        &dest_name,
        progress::now_recorded_at(),
        verdict.duration_ms,
        verdict.quality_flag.clone(),
    ));

    let staged = progress::stage_index(&index_path, &rows)?;
    let created_dir = !dest_dir.exists();
    if let Err(e) = std::fs::create_dir_all(&dest_dir) {
        staged.abort();
        return Err(anyhow::Error::from(e)
            .context(format!("登録先のフォルダを作れません: {}", dest_dir.display())));
    }
    if let Err(e) = progress::move_file(src, &dest) {
        let _ = std::fs::remove_file(&dest); // コピー途中の欠片を残さない
        if created_dir {
            let _ = std::fs::remove_dir(&dest_dir);
        }
        staged.abort();
        return Err(e.context("録画を登録先へ移せませんでした(何も変更していません)"));
    }
    if let Err(e) = staged.commit() {
        // 動画だけが登録先にある状態を残さない: 元の場所へ戻す
        let _ = progress::move_file(&dest, src);
        if created_dir {
            let _ = std::fs::remove_dir(&dest_dir);
        }
        return Err(e.context("index.tsv を更新できませんでした(録画は一時領域に戻しました)"));
    }
    Ok(Registered {
        rel_path: format!("{}/{}", progress::take_dir_name(&word), dest_name),
        abs_path: dest,
        rep_idx: rep,
        quality_flag: verdict.quality_flag.clone(),
        word,
    })
}

// ===== セッション本体 =====

/// 撮影モードの部品一式(本番では ONNX・カメラ・ffmpeg、テストでは偽物)
pub(super) struct CaptureParts {
    pub(super) input: Box<dyn CaptureInput>,
    pub(super) detector: Box<dyn HandObserver>,
    pub(super) judge: JudgeFactory,
    pub(super) encoder: EncoderFactory,
    pub(super) tmp: TempArea,
    pub(super) roi: PalmRoiMode,
}

/// 撮影モードの設定(`start` から)
#[derive(Debug, Clone)]
pub(super) struct CaptureConfig {
    pub(super) data_dir: PathBuf,
    pub(super) countdown_secs: f64,
    pub(super) preview: Option<PreviewCfg>,
}

struct CaptureSession<'a> {
    emitter: &'a Emitter,
    stop: &'a StopFlag,
    rx: Receiver<Input>,
    cfg: CaptureConfig,
    data_dir_text: String,
    machine: Machine,
    input: Box<dyn CaptureInput>,
    detector: Box<dyn HandObserver>,
    judge: JudgeWorker,
    encoder: EncoderFactory,
    tmp: TempArea,
    ui: JsonUi,
    meter: RateMeter,
    t0: Instant,
    /// 最後に見たフレームの (幅, 高さ)。録画の寸法に使う
    dims: Option<(usize, usize)>,
    recording: Option<Recording>,
    /// 録画に使った fps(録画を始めてから)
    rec_fps: Option<f64>,
    rec_frames: u64,
    rec_dropped: u64,
    /// 録画した一時ファイル(判定中・判定済み)
    take_path: Option<PathBuf>,
    target: Option<CaptureWord>,
    verdict: Option<Verdict>,
    notes: Vec<String>,
    last_state_at: f64,
}

impl<'a> CaptureSession<'a> {
    fn now(&self) -> f64 {
        self.t0.elapsed().as_secs_f64()
    }

    // ----- 状態の通知 -----

    fn state_event(&self) -> Event {
        let phase = self.machine.phase();
        let now = self.now();
        let (rec_secs, dropped) = match (&self.recording, phase) {
            (Some(r), _) => (r.secs(), r.stats.dropped.load(Ordering::SeqCst)),
            (None, Phase::Judging | Phase::Judged) => (
                self.rec_fps.map_or(0.0, |f| self.rec_frames as f64 / f),
                self.rec_dropped,
            ),
            _ => (0.0, 0),
        };
        let fps = self
            .rec_fps
            .filter(|_| phase != Phase::Idle && phase != Phase::Countdown)
            .or_else(|| self.input.measured_fps());
        Event::CaptureState {
            phase: phase.name(),
            countdown_left: round_to(self.machine.countdown_left(now), 1),
            rec_secs: round_to(rec_secs, 2),
            max_secs: MAX_RECORD_SECS,
            fps: fps.map(|f| round_to(f, 1)),
            dropped,
            word: if phase == Phase::Idle { None } else { self.target.clone() },
            data_dir: self.data_dir_text.clone(),
            verdict: if phase == Phase::Judged {
                self.verdict.as_ref().map(Verdict::view)
            } else {
                None
            },
            notes: self.notes.clone(),
        }
    }

    /// 段階が変わったときなど、必ず届けたい状態の通知
    fn emit_state(&mut self) {
        self.last_state_at = self.now();
        self.emitter.send(self.state_event());
    }

    /// 定期の更新(カウントダウン・録画中は速く、それ以外は 1 秒に 1 回)。詰まっていたら捨てる
    fn emit_state_periodic(&mut self) {
        let interval = match self.machine.phase() {
            Phase::Countdown | Phase::Recording => 1.0 / STATE_ACTIVE_HZ,
            _ => STATE_IDLE_SECS,
        };
        if self.now() - self.last_state_at >= interval {
            self.last_state_at = self.now();
            self.emitter.try_send(self.state_event());
        }
    }

    // ----- コマンド -----

    /// 入力されたコマンドを処理する。終了(quit / EOF)なら true
    fn drain_inputs(&mut self) -> bool {
        loop {
            match self.rx.try_recv() {
                Ok(Input::Cmd(EngineCommand::Quit)) | Ok(Input::Eof) => return true,
                Ok(Input::Cmd(cmd)) => self.handle(cmd),
                Err(TryRecvError::Empty) => return false,
                Err(TryRecvError::Disconnected) => return true,
            }
        }
    }

    fn handle(&mut self, cmd: EngineCommand) {
        match cmd {
            EngineCommand::CaptureRecord { word_id } => self.cmd_record(word_id),
            EngineCommand::CaptureStop => self.cmd_stop(),
            EngineCommand::CaptureAccept { confirm_ng } => self.cmd_accept(confirm_ng),
            EngineCommand::CaptureDiscard => self.cmd_discard(),
            EngineCommand::Start(_) => self
                .emitter
                .message("warn", "start は受け付け済みです(2 回目以降は無視します)"),
            EngineCommand::Toggle | EngineCommand::SetSegmenter { .. } => self.emitter.message(
                "warn",
                "撮影モードでは使えないコマンドです(語の認識はしないので、区切り方はありません)",
            ),
            EngineCommand::Quit => {}
        }
    }

    fn warn(&self, text: impl Into<String>) {
        self.emitter.message("warn", text);
    }

    fn cmd_record(&mut self, word_id: Option<String>) {
        if self.machine.phase() != Phase::Idle {
            // 理由の文言は状態機械が作る
            if let Err(why) = self.machine.record(self.now()) {
                self.warn(why);
            }
            return;
        }
        let word = match resolve_word(&self.cfg.data_dir, word_id.as_deref()) {
            Ok(w) => w,
            Err(e) => {
                self.warn(format!("録画する語を決められません: {:#}", e));
                return;
            }
        };
        let now = self.now();
        if self.machine.record(now).is_err() {
            return;
        }
        self.target = Some(word);
        self.notes.clear();
        self.verdict = None;
        self.rec_fps = None;
        self.rec_frames = 0;
        self.rec_dropped = 0;
        self.emit_state();
    }

    fn cmd_stop(&mut self) {
        match self.machine.stop() {
            Ok(Stop::Cancel) => {
                self.target = None;
                self.emit_state();
            }
            Ok(Stop::Finish) => self.finish_recording(false),
            Err(why) => self.warn(why),
        }
    }

    fn cmd_discard(&mut self) {
        match self.machine.discard() {
            Ok(from) => {
                if from == Phase::Recording {
                    // 録画を止めて捨てる(判定には回さない)
                    if let Some(r) = self.recording.take() {
                        let out = r.out.clone();
                        let _ = r.finish(&self.input.tee());
                        let _ = std::fs::remove_file(out);
                    }
                }
                self.remove_take_file();
                self.clear_take();
                self.emit_state();
            }
            Err(why) => {
                if self.machine.phase() == Phase::Judging {
                    self.emitter
                        .message("info", "判定中です。終わってから破棄してください(判定は数秒で終わります)");
                } else {
                    self.warn(why);
                }
            }
        }
    }

    fn cmd_accept(&mut self, confirm_ng: bool) {
        if let Err(why) = self.machine.can_accept() {
            self.warn(why);
            return;
        }
        let (Some(verdict), Some(word), Some(src)) =
            (self.verdict.clone(), self.target.clone(), self.take_path.clone())
        else {
            self.warn("採用する録画がありません");
            return;
        };
        if !verdict.usable && !confirm_ng {
            self.warn("使えないと判定されたテイクです。NG フラグ付きで記録するには、確認のうえで採用してください");
            return;
        }
        match register_take(&self.cfg.data_dir, &src, &word.word_id, &verdict) {
            Ok(reg) => {
                self.emitter.send(Event::CaptureSaved {
                    file_path: reg.rel_path.clone(),
                    abs_path: reg.abs_path.to_string_lossy().to_string(),
                    word_id: reg.word.word_id.clone(),
                    label: reg.word.label_ja.clone(),
                    rep_idx: reg.rep_idx,
                    quality_flag: reg.quality_flag.clone(),
                    usable: verdict.usable,
                    longest_run_secs: round_to(verdict.longest() as f64, 2),
                    duration_secs: round_to(verdict.duration_ms as f64 / 1000.0, 2),
                    data_dir: self.data_dir_text.clone(),
                });
                self.machine.reset();
                self.take_path = None; // 移したので、消す対象は無い
                self.clear_take();
                self.emit_state();
            }
            Err(e) => {
                self.warn(format!("登録できませんでした: {:#}", e));
            }
        }
    }

    /// 一時領域の録画ファイルを消す
    fn remove_take_file(&mut self) {
        if let Some(p) = self.take_path.take() {
            let _ = std::fs::remove_file(p);
        }
    }

    fn clear_take(&mut self) {
        self.target = None;
        self.verdict = None;
        self.take_path = None;
        self.rec_fps = None;
        self.rec_frames = 0;
        self.rec_dropped = 0;
        self.notes.clear();
    }

    // ----- 録画 -----

    fn begin_recording(&mut self) {
        let Some((w, h)) = self.dims else {
            return; // まだ 1 枚も届いていない(寸法が決まるまでカウントダウン 0 のまま待つ)
        };
        let rf = decide_record_fps(self.input.measured_fps());
        let out = self.tmp.next_take_path();
        let spec = EncoderSpec {
            width: w,
            height: h,
            fps: rf.fps,
            out: out.clone(),
        };
        match start_recording(&self.input.tee(), &self.encoder, &spec) {
            Ok(rec) => {
                self.recording = Some(rec);
                self.take_path = Some(out);
                self.rec_fps = Some(rf.fps);
                if let Some(w) = rf.warning {
                    self.notes.push(w);
                }
                self.machine.begin_recording();
                self.emit_state();
            }
            Err(e) => {
                self.warn(format!("録画を始められません: {:#}", e));
                self.machine.reset();
                self.clear_take();
                self.emit_state();
            }
        }
    }

    /// 録画を止めて判定に回す。`auto` は自動停止(最長に達した等)
    fn finish_recording(&mut self, auto: bool) {
        let Some(rec) = self.recording.take() else {
            return;
        };
        let out = rec.out.clone();
        let result = rec.finish(&self.input.tee());
        match result {
            Ok(o) if o.frames > 0 => {
                if auto {
                    self.notes.push(format!(
                        "最長 {:.0} 秒に達したので、自動で止めました",
                        MAX_RECORD_SECS
                    ));
                }
                if o.dropped > 0 {
                    self.notes.push(format!(
                        "録画が間に合わず {} コマを捨てました(動きが少し速く記録されます)。負荷を下げて撮り直すことをお勧めします",
                        o.dropped
                    ));
                }
                self.rec_frames = o.frames;
                self.rec_dropped = o.dropped;
                self.machine.begin_judging();
                if !self.judge.submit(out) {
                    self.verdict = Some(evaluate_failed("判定の処理を始められませんでした"));
                    self.machine.finish_judging();
                }
            }
            Ok(_) => {
                self.warn("録画のコマが 1 枚も書かれませんでした。もう一度録画してください");
                let _ = std::fs::remove_file(&out);
                self.machine.reset();
                self.clear_take();
            }
            Err(e) => {
                self.warn(format!("録画に失敗しました: {e}"));
                let _ = std::fs::remove_file(&out);
                self.machine.reset();
                self.clear_take();
            }
        }
        self.emit_state();
    }

    fn poll_judge(&mut self) {
        if self.machine.phase() != Phase::Judging {
            return;
        }
        let Ok(res) = self.judge.rx.try_recv() else {
            return;
        };
        let verdict = match res {
            Ok(outcome) => evaluate(&outcome, self.rec_dropped),
            Err(why) => evaluate_failed(&why),
        };
        self.verdict = Some(verdict);
        self.machine.finish_judging();
        self.emit_state();
    }

    fn tick(&mut self) {
        let now = self.now();
        if self.machine.countdown_done(now) {
            self.begin_recording();
        }
        if self.recording.as_ref().is_some_and(Recording::should_auto_stop)
            && self.machine.phase() == Phase::Recording
        {
            self.finish_recording(true);
        }
        self.poll_judge();
        self.emit_state_periodic();
    }

    // ----- 1 フレーム -----

    fn process_frame(&mut self, sf: SourceFrame) -> Result<()> {
        self.dims = Some((sf.frame.shape()[1], sf.frame.shape()[0]));
        let obs = self.detector.observe(&sf.frame)?;
        let frame = LiveFrame {
            t: sf.t,
            feature: obs.feature,
            left: obs.left,
            right: obs.right,
        };
        let fps = self.meter.tick(sf.t);
        self.ui.preview(&sf.frame, &frame);
        let recording = self.machine.phase() == Phase::Recording;
        self.ui.status(&StatusLine {
            recording,
            rec_secs: self.recording.as_ref().map_or(0.0, Recording::secs),
            left: obs.left,
            right: obs.right,
            fps,
        });
        Ok(())
    }

    fn run(&mut self) -> Result<()> {
        loop {
            if self.stop.is_set() || self.emitter.is_broken() {
                return Ok(());
            }
            if self.drain_inputs() {
                return Ok(());
            }
            self.tick();
            match self.input.poll_frame(POLL_TIMEOUT)? {
                Polled::Frame(sf) => self.process_frame(sf)?,
                Polled::Idle => {}
                Polled::Ended => return Ok(()),
            }
        }
    }

    /// 後始末: 録画を止めて捨て、判定の終わりを待ち、一時領域を消す。ffmpeg(録画用・判定用)を残さない
    fn shutdown(&mut self) {
        if let Some(r) = self.recording.take() {
            let out = r.out.clone();
            let _ = r.finish(&self.input.tee());
            let _ = std::fs::remove_file(out);
        }
        self.judge.shutdown();
        self.tmp.cleanup();
    }
}

/// 撮影モードの本体。`start` を受けたあとに呼ぶ。終了(quit・EOF・SIGTERM)では後始末をして Ok で返る。
/// `ready` と最初の `capture_state` を出し、最後の `stopped` は呼び出し側ではなくここで出す
pub(super) fn serve_capture(
    rx: Receiver<Input>,
    emitter: &Emitter,
    stop: &StopFlag,
    cfg: CaptureConfig,
    parts: CaptureParts,
) -> Result<()> {
    let CaptureParts {
        input,
        detector,
        judge,
        encoder,
        tmp,
        roi,
    } = parts;
    let preview_cfg = cfg.preview;
    let t0 = Instant::now();
    let ui = JsonUi::new(
        emitter.clone(),
        Labels { rows: Vec::new() },
        roi,
        preview_cfg,
        Box::new(move || t0.elapsed().as_secs_f64()),
    );
    let data_dir_text = cfg.data_dir.to_string_lossy().to_string();
    let mut session = CaptureSession {
        emitter,
        stop,
        rx,
        machine: Machine::new(cfg.countdown_secs),
        cfg,
        data_dir_text,
        input,
        detector,
        judge: JudgeWorker::spawn(judge),
        encoder,
        tmp,
        ui,
        meter: RateMeter::new(),
        t0,
        dims: None,
        recording: None,
        rec_fps: None,
        rec_frames: 0,
        rec_dropped: 0,
        take_path: None,
        target: None,
        verdict: None,
        notes: Vec::new(),
        last_state_at: f64::NEG_INFINITY,
    };
    emitter.send(Event::Ready {
        vocab: Vec::new(),
        model_dir: String::new(),
        palm_roi: roi.label(),
        frame_select: String::new(),
        frames: 0,
        segmenter: "manual",
        preprocess: "default",
    });
    emitter.message(
        "info",
        format!(
            "撮影モード: 録画できます。登録先は {}。一時領域は {}(終了時に消します)",
            session.data_dir_text,
            session.tmp.dir().display()
        ),
    );
    session.emit_state();

    let outcome = session.run();
    session.shutdown();
    let dropped = session.input.dropped_frames();
    clear_child_pid();
    drop(session); // 入力(カメラ用 ffmpeg)を止める
    match outcome {
        Ok(()) => {}
        // 終了要求の最中に ffmpeg を止めたことによる入力断は、異常ではない
        Err(_) if stop.is_set() => {}
        Err(e) => return Err(e),
    }
    emitter.send(Event::Stopped {
        dropped_frames: dropped,
    });
    Ok(())
}

/// `start` から撮影モードの設定を作る。登録先は必須(フォルダで、words.tsv が読めること)。
/// どれかが欠けていれば、カメラを開く前に日本語の理由で失敗させる
pub(super) fn config_from_start(start: &StartCmd) -> Result<CaptureConfig> {
    let dir = start
        .data_dir
        .as_deref()
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .context("撮影モードには登録先(data_dir)の指定が必要です")?;
    let data_dir = PathBuf::from(dir);
    if !data_dir.is_dir() {
        anyhow::bail!("登録先がフォルダではありません(または見つかりません): {}", data_dir.display());
    }
    // 読めるか確かめるだけ(書かない)。読めない登録先には録画させない
    progress::read_words(&data_dir)
        .with_context(|| format!("登録先に words.tsv が読めません: {}", data_dir.display()))?;
    let (preview, warns) = resolve_preview(&start.preview);
    for w in warns {
        eprintln!("警告: {}", w);
    }
    Ok(CaptureConfig {
        data_dir,
        countdown_secs: clamp_countdown(start.countdown_secs),
        preview,
    })
}

/// 本番の撮影モード: 手検出の ONNX を読み、入力元(カメラ / 動画)を開いて `serve_capture` を回す
pub(super) fn serve_capture_real(
    rx: Receiver<Input>,
    emitter: &Emitter,
    stop: &StopFlag,
    start: &StartCmd,
) -> Result<()> {
    let cfg = config_from_start(start)?;
    let roi = PalmRoiMode::CenterSquare;
    let parts = match build_real(start, stop, roi) {
        Ok(p) => p,
        Err(e) if e.downcast_ref::<StopRequested>().is_some() => {
            emitter.send(Event::Stopped { dropped_frames: 0 });
            return Ok(());
        }
        Err(e) => return Err(e),
    };
    serve_capture(rx, emitter, stop, cfg, parts)
}

fn build_real(start: &StartCmd, stop: &StopFlag, roi: PalmRoiMode) -> Result<CaptureParts> {
    let detector = HandOnly::load(roi)?;
    let tmp = TempArea::create()?;
    // ここまでで時間が経っている。終了要求がもう来ていたら、カメラを起動せずに引き返す
    if stop.is_set() {
        return Err(StopRequested.into());
    }
    let input: Box<dyn CaptureInput> = match &start.source {
        SourceSpec::Camera { device } => {
            let device = device
                .as_ref()
                .map(DeviceId::as_string)
                .unwrap_or_else(|| "0".into());
            if device.is_empty() {
                anyhow::bail!("camera の device が空です");
            }
            // 注意: ここでカメラが起動する
            let cam = FfmpegInput::open_camera(&device)?;
            register_child_pid(cam.child.id());
            if stop.is_set() {
                super::json_engine::kill_registered_child();
            }
            Box::new(cam)
        }
        SourceSpec::File { path } => {
            let path = PathBuf::from(path);
            if !path.is_file() {
                anyhow::bail!("動画ファイルが見つかりません: {}", path.display());
            }
            let video = FfmpegInput::open_file(&path)?;
            register_child_pid(video.child.id());
            if stop.is_set() {
                super::json_engine::kill_registered_child();
            }
            Box::new(video)
        }
    };
    let judge: JudgeFactory = Box::new(|| {
        let checker = progress::HandChecker::new()?;
        Ok(Box::new(RealJudge { checker }) as Box<dyn TakeJudge>)
    });
    Ok(CaptureParts {
        input,
        detector: Box::new(detector),
        judge,
        encoder: Box::new(ffmpeg_encoder),
        tmp,
        roi,
    })
}

#[cfg(test)]
mod tests {
    use super::super::json_engine::{spawn_writer, StartMode};
    use super::*;
    use serde_json::Value;
    use std::sync::atomic::AtomicUsize;

    // ===== 道具 =====

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    /// テスト専用の作業フォルダ(リポジトリの外。本物の raw_jsl には触れない)
    fn scratch(tag: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let d = std::env::temp_dir().join(format!("attn_capture_test_{}_{}_{}", tag, std::process::id(), n));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    const WORDS: &str = "word_id\tromaji\tlabel_ja\tstage\ttarget_takes\n\
                         001\tkonnichiwa\tこんにちは\t1\t3\n\
                         011\tkare\tかれ\t2\t3\n";
    const INDEX_HEADER_LINE: &str =
        "file_path\tword_id\tword_romaji\tword_label_ja\trep_idx\trecorded_at\tduration_ms\tquality_flag\tnotes";
    const EXISTING_ROW: &str = "001_konnichiwa/01.mp4\t001\tkonnichiwa\tこんにちは\t1\t2026-10-01 10:00:00\t4000\tok\t手書きのメモ";

    /// words.tsv・既存の 1 テイク(001 の 01.mp4)・index.tsv が入ったデータフォルダを作る
    fn make_data_dir(root: &Path) -> PathBuf {
        let d = root.join("raw_jsl");
        std::fs::create_dir_all(d.join("001_konnichiwa")).unwrap();
        std::fs::write(d.join("words.tsv"), WORDS).unwrap();
        std::fs::write(d.join("001_konnichiwa/01.mp4"), b"existing take").unwrap();
        std::fs::write(d.join("index.tsv"), format!("{INDEX_HEADER_LINE}\n{EXISTING_ROW}\n")).unwrap();
        d
    }

    /// フォルダの中身(相対パス → 内容)。変わっていないことの確認用
    fn snapshot(root: &Path) -> std::collections::BTreeMap<String, Vec<u8>> {
        fn walk(root: &Path, dir: &Path, out: &mut std::collections::BTreeMap<String, Vec<u8>>) {
            let Ok(rd) = std::fs::read_dir(dir) else {
                return;
            };
            for e in rd.flatten() {
                let p = e.path();
                let rel = p.strip_prefix(root).unwrap().display().to_string();
                if p.is_dir() {
                    out.insert(format!("{rel}/"), Vec::new());
                    walk(root, &p, out);
                } else {
                    out.insert(rel, std::fs::read(&p).unwrap());
                }
            }
        }
        let mut out = Default::default();
        walk(root, root, &mut out);
        out
    }

    fn outcome(left: f32, right: f32, duration_ms: u64) -> JudgeOutcome {
        JudgeOutcome {
            left_secs: left,
            right_secs: right,
            duration_ms,
        }
    }

    // ===== 純粋な判断 =====

    #[test]
    fn countdown_is_clamped() {
        assert_eq!(clamp_countdown(None), 3.0);
        assert_eq!(clamp_countdown(Some(0.0)), 0.0);
        assert_eq!(clamp_countdown(Some(5.0)), 5.0);
        assert_eq!(clamp_countdown(Some(99.0)), 10.0);
        assert_eq!(clamp_countdown(Some(-1.0)), 0.0);
        assert_eq!(clamp_countdown(Some(f64::NAN)), 3.0);
    }

    #[test]
    fn max_frames_keeps_duration_strictly_under_ten_seconds() {
        assert_eq!(max_frames_for(30.0), 299);
        assert_eq!(max_frames_for(15.0), 149);
        assert_eq!(max_frames_for(29.97), 299);
        for fps in [10.0, 14.87, 15.0, 24.0, 25.0, 29.97, 30.0, 59.94, 60.0] {
            let n = max_frames_for(fps);
            assert!(n as f64 / fps < 10.0, "fps={fps} n={n}");
            assert!((n + 1) as f64 / fps >= 10.0 - 1e-9, "fps={fps} n={n}: 上限が小さすぎる");
        }
        assert_eq!(max_frames_for(0.0), 1);
        assert_eq!(max_frames_for(f64::NAN), 1);
    }

    #[test]
    fn record_fps_snaps_near_30_and_warns_when_slow() {
        // 28.5〜31.5 は 30(境界を含む)
        for m in [28.5, 29.4, 29.97, 30.0, 30.4, 31.5] {
            assert_eq!(decide_record_fps(Some(m)), RecordFps { fps: 30.0, warning: None }, "{m}");
        }
        // 遅いカメラ: 実測をそのまま宣言(複製しない)、警告
        let r = decide_record_fps(Some(14.87));
        assert_eq!(r.fps, 14.87);
        let w = r.warning.unwrap();
        assert!(w.contains("14.9 fps") && w.contains("複製せず"), "{w}");
        assert_eq!(decide_record_fps(Some(28.4)).fps, 28.4);
        assert!(decide_record_fps(Some(28.4)).warning.is_some());
        // 速すぎる(60fps)は実測のまま、警告なし
        assert_eq!(decide_record_fps(Some(60.0)), RecordFps { fps: 60.0, warning: None });
        // 測れなかった
        for m in [None, Some(0.0), Some(f64::NAN), Some(-3.0)] {
            let r = decide_record_fps(m);
            assert_eq!(r.fps, 30.0);
            assert!(r.warning.unwrap().contains("測れなかった"));
        }
    }

    #[test]
    fn fps_is_measured_from_arrival_times() {
        let t30: Vec<f64> = (0..60).map(|i| i as f64 / 30.0).collect();
        assert!((fps_from_arrivals(&t30).unwrap() - 30.0).abs() < 1e-9);
        let t15: Vec<f64> = (0..30).map(|i| 1.0 + i as f64 / 14.87).collect();
        assert!((fps_from_arrivals(&t15).unwrap() - 14.87).abs() < 1e-6);
        assert_eq!(fps_from_arrivals(&t30[..MIN_FPS_SAMPLES - 1]), None, "サンプルが少なすぎる");
        assert_eq!(fps_from_arrivals(&[]), None);
        assert_eq!(fps_from_arrivals(&[1.0; 20]), None, "時間が進んでいない");
    }

    #[test]
    fn warmup_burst_is_not_counted_in_the_measured_fps() {
        // 起動直後の 0.9 秒に 40 枚が届き(44 fps 相当)、そのあとは 30fps
        let mut t: Vec<f64> = (0..40).map(|i| i as f64 * 0.9 / 40.0).collect();
        t.extend((0..60).map(|i| 1.0 + i as f64 / 30.0));
        assert!(fps_from_arrivals(&t).unwrap() > 31.5, "起動直後を含めると速く測れてしまう");
        assert!((measure_fps(&t).unwrap() - 30.0).abs() < 1e-9);
        // 起動直後のものしか無ければ測れない
        assert_eq!(measure_fps(&t[..40]), None);
    }

    #[test]
    fn encoder_args_match_existing_raw_jsl_videos() {
        let a = encoder_ffmpeg_args(1620, 1080, 30.0, Path::new("/t/take-1.mp4"));
        let has = |pair: &[&str]| a.windows(pair.len()).any(|w| w.iter().zip(pair).all(|(x, y)| x == y));
        assert!(has(&["-video_size", "1620x1080"]), "{a:?}");
        assert!(has(&["-framerate", "30"]), "{a:?}");
        assert!(has(&["-pix_fmt", "rgb24"]));
        assert!(has(&["-i", "-"]));
        assert!(has(&["-c:v", "libx264"]));
        assert!(has(&["-profile:v", "high"]));
        assert!(has(&["-pix_fmt", "yuv420p"]));
        assert!(has(&["-b:v", "7700k"]));
        assert!(a.contains(&"-an".to_string()));
        assert_eq!(a.last().unwrap(), "/t/take-1.mp4");
        // 30 でない fps は実測値のまま渡す
        let a = encoder_ffmpeg_args(1620, 1080, 14.87, Path::new("/t/x.mp4"));
        assert!(a.windows(2).any(|w| w[0] == "-framerate" && w[1] == "14.87"), "{a:?}");
    }

    #[test]
    fn file_args_play_in_realtime_and_loop() {
        let a = file_ffmpeg_args(Path::new("/v/a b.mp4"));
        for k in ["-re", "-stream_loop", "-1", "rgb24", "rawvideo"] {
            assert!(a.contains(&k.to_string()), "{k}: {a:?}");
        }
        assert!(a.contains(&"/v/a b.mp4".to_string()));
    }

    // ===== 録画への分岐 =====

    fn tee_with(queue: usize, max_frames: u64) -> (TeeSlot, Receiver<Vec<u8>>, Arc<TeeStats>) {
        let slot = TeeSlot::default();
        let (tx, rx) = sync_channel(queue);
        let stats = Arc::new(TeeStats::default());
        slot.attach(Tee {
            tx,
            max_frames,
            stats: Arc::clone(&stats),
        });
        (slot, rx, stats)
    }

    #[test]
    fn tee_does_nothing_until_attached_and_after_detach() {
        let slot = TeeSlot::default();
        slot.feed(&[1, 2, 3]); // 録画していない: 何も起きない(パニックもしない)
        let (slot, rx, stats) = tee_with(8, 100);
        slot.feed(&[1]);
        slot.detach();
        slot.feed(&[2]);
        assert_eq!(rx.try_iter().count(), 1);
        assert_eq!(stats.sent.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn tee_stops_at_the_frame_cap() {
        let (slot, rx, stats) = tee_with(16, 3);
        for i in 0..5u8 {
            slot.feed(&[i]);
        }
        assert_eq!(stats.sent.load(Ordering::SeqCst), 3);
        assert!(stats.capped.load(Ordering::SeqCst));
        assert_eq!(rx.try_iter().collect::<Vec<_>>(), vec![vec![0u8], vec![1], vec![2]]);
        // 上限に達したら送り口は手放されている(書き込みスレッドは入力の終わりを知る)
        assert!(rx.recv().is_err());
    }

    #[test]
    fn tee_drops_without_duplicating_when_the_queue_is_full() {
        let (slot, rx, stats) = tee_with(2, 100);
        for i in 0..5u8 {
            slot.feed(&[i]);
        }
        assert_eq!(stats.sent.load(Ordering::SeqCst), 2);
        assert_eq!(stats.dropped.load(Ordering::SeqCst), 3);
        // 入ったのは最初の 2 枚だけ(捨てたコマの代わりに前のコマを複製したりしない)
        assert_eq!(rx.try_iter().collect::<Vec<_>>(), vec![vec![0u8], vec![1]]);
    }

    #[test]
    fn tee_notices_a_dead_encoder() {
        let (slot, rx, stats) = tee_with(2, 100);
        drop(rx);
        slot.feed(&[1]);
        assert!(stats.closed.load(Ordering::SeqCst));
        slot.feed(&[2]); // 二度目は何もしない
        assert_eq!(stats.sent.load(Ordering::SeqCst), 0);
    }

    // ===== 判定 =====

    #[test]
    fn verdict_boundaries_hands() {
        // ちょうど 1.0 秒は使える(progress.rs の quality_flag_for_run と同じ境界)
        let v = evaluate(&outcome(0.0, 1.0, 4000), 0);
        assert!(v.usable && v.reasons.is_empty());
        assert_eq!(v.quality_flag, "ok");
        assert!(v.summary.starts_with("使えそう"), "{}", v.summary);
        // 左右の良い方で判定
        assert!(evaluate(&outcome(1.33, 0.0, 4000), 0).usable);
        // 0.67 秒は足りない
        let v = evaluate(&outcome(0.67, 0.33, 4000), 0);
        assert!(!v.usable);
        assert_eq!(v.reasons, vec!["hands_short"]);
        assert_eq!(v.quality_flag, "ng_hands");
        assert!(v.summary.starts_with("使えない") && v.summary.contains("1.0 秒続いて検出されていません"), "{}", v.summary);
        assert!(v.summary.contains("0.67"), "{}", v.summary);
        assert_eq!(v.view().longest_run_secs, 0.67);
    }

    #[test]
    fn verdict_boundaries_duration() {
        // 2000ms ちょうどと 10000ms ちょうどは範囲内(DURATION_MIN_MS..=DURATION_MAX_MS)
        assert!(evaluate(&outcome(2.0, 0.0, 2000), 0).usable);
        assert!(evaluate(&outcome(2.0, 0.0, 10000), 0).usable);
        let short = evaluate(&outcome(2.0, 0.0, 1999), 0);
        assert_eq!(short.reasons, vec!["too_short"]);
        assert_eq!(short.quality_flag, "ng_duration", "手は足りていて長さだけ範囲外");
        assert!(short.summary.contains("短すぎます"), "{}", short.summary);
        let long = evaluate(&outcome(2.0, 0.0, 10001), 0);
        assert_eq!(long.reasons, vec!["too_long"]);
        assert!(long.summary.contains("長すぎます"), "{}", long.summary);
        // 手も長さもだめなら、手の理由が先で ng_hands
        let both = evaluate(&outcome(0.33, 0.0, 1000), 0);
        assert_eq!(both.reasons, vec!["hands_short", "too_short"]);
        assert_eq!(both.quality_flag, "ng_hands");
        assert!(both.summary.contains("、"));
    }

    #[test]
    fn verdict_mentions_dropped_frames_and_failure() {
        let v = evaluate(&outcome(2.0, 0.0, 4000), 7);
        assert!(v.usable, "コマ落ちだけでは使えないにしない");
        assert!(v.summary.contains("コマ落ち 7 枚"), "{}", v.summary);
        let f = evaluate_failed("モデルが読めません\n詳細");
        assert!(!f.usable);
        assert_eq!(f.reasons, vec!["judge_failed"]);
        assert_eq!(f.quality_flag, "", "判定できなかったときは取り込みと同じく空");
        assert!(f.summary.starts_with("判定できませんでした") && !f.summary.contains("詳細"), "{}", f.summary);
    }

    // ===== 状態機械 =====

    #[test]
    fn machine_happy_path() {
        let mut m = Machine::new(3.0);
        assert_eq!(m.phase(), Phase::Idle);
        assert_eq!(m.countdown_left(0.0), 0.0);
        m.record(10.0).unwrap();
        assert_eq!(m.phase(), Phase::Countdown);
        assert!((m.countdown_left(11.0) - 2.0).abs() < 1e-9);
        assert!(!m.countdown_done(12.99));
        assert!(m.countdown_done(13.0), "ちょうどで終わる");
        m.begin_recording();
        assert_eq!(m.phase(), Phase::Recording);
        assert_eq!(m.countdown_left(20.0), 0.0);
        assert_eq!(m.stop(), Ok(Stop::Finish));
        m.begin_judging();
        assert_eq!(m.phase(), Phase::Judging);
        assert!(m.can_accept().is_err(), "判定中は採用できない");
        m.finish_judging();
        assert_eq!(m.phase(), Phase::Judged);
        assert!(m.can_accept().is_ok());
        m.reset();
        assert_eq!(m.phase(), Phase::Idle);
    }

    #[test]
    fn machine_rejects_out_of_order_commands() {
        let mut m = Machine::new(0.0);
        // 待機中は停止・採用・破棄できない
        assert!(m.stop().is_err());
        assert!(m.can_accept().is_err());
        assert!(m.discard().is_err());
        m.record(0.0).unwrap();
        // 録画中の二重の開始は不可
        let e = m.record(1.0).unwrap_err();
        assert!(e.contains("録画を始められる状態ではありません") || e.contains("できる状態ではありません"), "{e}");
        m.begin_recording();
        assert!(m.record(2.0).is_err());
        assert!(m.can_accept().is_err(), "録画中は採用できない");
        m.begin_judging();
        // 判定中は破棄できない(判定が一時ファイルを読んでいる)
        assert!(m.discard().is_err());
        assert!(m.record(3.0).is_err());
        m.finish_judging();
        assert!(m.record(4.0).is_err(), "判定済みのまま次の録画は始められない(採用か破棄が先)");
        assert_eq!(m.discard(), Ok(Phase::Judged));
        assert_eq!(m.phase(), Phase::Idle);
    }

    #[test]
    fn machine_countdown_can_be_cancelled_and_recording_discarded() {
        let mut m = Machine::new(3.0);
        m.record(0.0).unwrap();
        assert_eq!(m.stop(), Ok(Stop::Cancel));
        assert_eq!(m.phase(), Phase::Idle);
        m.record(0.0).unwrap();
        assert_eq!(m.discard(), Ok(Phase::Countdown));
        m.record(0.0).unwrap();
        m.begin_recording();
        assert_eq!(m.discard(), Ok(Phase::Recording));
        assert_eq!(m.phase(), Phase::Idle);
    }

    #[test]
    fn machine_zero_countdown_starts_at_once() {
        let mut m = Machine::new(0.0);
        m.record(5.0).unwrap();
        assert!(m.countdown_done(5.0));
    }

    // ===== 一時領域 =====

    #[test]
    fn temp_dir_name_parsing() {
        assert_eq!(temp_dir_pid("attention-capture-123"), Some(123));
        assert_eq!(temp_dir_pid("attention-capture-"), None);
        assert_eq!(temp_dir_pid("attention-capture-12a"), None);
        assert_eq!(temp_dir_pid("other-123"), None);
    }

    #[test]
    fn temp_area_is_created_and_cleaned_up() {
        let base = scratch("tmparea");
        let dir;
        {
            let mut t = TempArea::create_in(&base).unwrap();
            dir = t.dir().to_path_buf();
            assert!(dir.starts_with(&base) && dir.is_dir());
            let p = t.next_take_path();
            std::fs::write(&p, b"x").unwrap();
            assert_ne!(t.next_take_path(), p, "番号は進む");
        }
        assert!(!dir.exists(), "Drop で消える");
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn stale_temp_dirs_of_dead_processes_are_swept_but_live_ones_are_kept() {
        let base = scratch("sweep");
        // 死んだ持ち主: すぐ終わる子プロセスの PID(終了を待ってから使う)
        let mut child = std::process::Command::new("true").spawn().unwrap();
        let dead = child.id();
        child.wait().unwrap();
        let stale = base.join(format!("{TEMP_PREFIX}{dead}"));
        std::fs::create_dir_all(&stale).unwrap();
        std::fs::write(stale.join("take-1.mp4"), b"leftover").unwrap();
        // 死んだ持ち主でも、知らないファイルが混ざっていればフォルダは残す(take-*.mp4 だけ消す)
        let odd = base.join(format!("{TEMP_PREFIX}{}", dead.wrapping_add(1_000_000)));
        std::fs::create_dir_all(&odd).unwrap();
        std::fs::write(odd.join("notes.txt"), b"keep me").unwrap();
        // 生きている持ち主(このテストのプロセスの親 = テストランナー)のものは触らない
        let alive_pid = std::os::unix::process::parent_id();
        assert!(pid_alive(alive_pid));
        let alive = base.join(format!("{TEMP_PREFIX}{alive_pid}"));
        std::fs::create_dir_all(&alive).unwrap();
        std::fs::write(alive.join("take-1.mp4"), b"in use").unwrap();
        // 無関係なフォルダも触らない
        let other = base.join("something-else");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join("take-1.mp4"), b"x").unwrap();

        let t = TempArea::create_in(&base).unwrap();
        assert!(!stale.exists(), "死んだ持ち主の一時フォルダは掃除される");
        assert!(odd.join("notes.txt").exists(), "知らないファイルがあるフォルダは残す");
        assert!(alive.join("take-1.mp4").exists(), "生きている持ち主のものは消さない");
        assert!(other.join("take-1.mp4").exists());
        assert!(t.dir().is_dir());
        drop(t);
        let _ = std::fs::remove_dir_all(base);
    }

    // ===== 語の決定と登録 =====

    #[test]
    fn resolve_word_uses_requested_or_next_and_never_writes() {
        let root = scratch("resolve");
        let data = make_data_dir(&root);
        let before = snapshot(&data);
        // 要求された語
        let w = resolve_word(&data, Some("011")).unwrap();
        assert_eq!((w.word_id.as_str(), w.romaji.as_str(), w.label.as_str(), w.rep_idx), ("011", "kare", "かれ", 1));
        // 001 は既に 01.mp4 があるので次は 2 本目
        assert_eq!(resolve_word(&data, Some("001")).unwrap().rep_idx, 2);
        // 省略 = 次の語(stage 1 の 001 が目標 3 に未達)
        let w = resolve_word(&data, None).unwrap();
        assert_eq!(w.word_id, "001");
        assert_eq!(resolve_word(&data, Some("  ")).unwrap().word_id, "001", "空文字は省略扱い");
        // 知らない語は理由つきのエラー
        let e = resolve_word(&data, Some("999")).unwrap_err();
        assert!(format!("{e:#}").contains("999"), "{e:#}");
        assert_eq!(snapshot(&data), before, "語を決めるだけではデータフォルダを変えない");
        // words.tsv が無ければテンプレートを書かずに失敗
        let empty = root.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(resolve_word(&empty, None).is_err());
        assert!(!empty.join("words.tsv").exists(), "words.tsv を作ってはいけない");
        let _ = std::fs::remove_dir_all(root);
    }

    fn verdict_ok() -> Verdict {
        evaluate(&outcome(0.0, 1.67, 4200), 0)
    }

    #[test]
    fn register_take_moves_the_file_and_appends_one_row_keeping_existing_rows() {
        let root = scratch("register");
        let data = make_data_dir(&root);
        let src = root.join("take-1.mp4");
        std::fs::write(&src, b"new take bytes").unwrap();

        let reg = register_take(&data, &src, "001", &verdict_ok()).unwrap();
        assert_eq!(reg.rel_path, "001_konnichiwa/02.mp4");
        assert_eq!(reg.rep_idx, 2);
        assert_eq!(reg.quality_flag, "ok");
        assert!(!src.exists(), "一時ファイルは移された");
        assert_eq!(std::fs::read(data.join("001_konnichiwa/02.mp4")).unwrap(), b"new take bytes");
        assert_eq!(std::fs::read(data.join("001_konnichiwa/01.mp4")).unwrap(), b"existing take", "既存の動画は変えない");

        let index = std::fs::read_to_string(data.join("index.tsv")).unwrap();
        let lines: Vec<&str> = index.lines().collect();
        assert_eq!(lines.len(), 3, "{index}");
        assert_eq!(lines[0], INDEX_HEADER_LINE);
        assert_eq!(lines[1], EXISTING_ROW, "既存の行(手書きのメモを含む)は 1 文字も変わらない");
        let cols: Vec<&str> = lines[2].split('\t').collect();
        assert_eq!(cols.len(), 9);
        assert_eq!(&cols[..5], &["001_konnichiwa/02.mp4", "001", "konnichiwa", "こんにちは", "2"]);
        assert_eq!(cols[6], "4200");
        assert_eq!(cols[7], "ok");
        assert_eq!(cols[8], "");
        // 日時は "YYYY-MM-DD HH:MM:SS"
        assert_eq!(cols[5].len(), 19, "{}", cols[5]);

        // 一時ファイルが残っていない
        let names: Vec<String> = std::fs::read_dir(&data)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert!(names.iter().all(|n| !n.contains(".tmp")), "{names:?}");

        // 続けて登録すると番号が進む。別の語は別フォルダ
        let src2 = root.join("take-2.mp4");
        std::fs::write(&src2, b"third").unwrap();
        assert_eq!(register_take(&data, &src2, "001", &verdict_ok()).unwrap().rel_path, "001_konnichiwa/03.mp4");
        let src3 = root.join("take-3.mp4");
        std::fs::write(&src3, b"kare").unwrap();
        let r = register_take(&data, &src3, "011", &verdict_ok()).unwrap();
        assert_eq!(r.rel_path, "011_kare/01.mp4");
        assert_eq!(std::fs::read_to_string(data.join("index.tsv")).unwrap().lines().count(), 5);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn register_take_records_ng_flag_and_advances_the_number() {
        let root = scratch("registerng");
        let data = make_data_dir(&root);
        let src = root.join("take-1.mp4");
        std::fs::write(&src, b"bad").unwrap();
        let bad = evaluate(&outcome(0.33, 0.0, 4000), 0);
        let reg = register_take(&data, &src, "001", &bad).unwrap();
        assert_eq!(reg.quality_flag, "ng_hands");
        let index = std::fs::read_to_string(data.join("index.tsv")).unwrap();
        assert!(index.lines().last().unwrap().contains("\tng_hands\t"), "{index}");
        // 番号は進む(撮影セッションの ng の扱いと同じ)。ng は目標数に数えない
        let src2 = root.join("take-2.mp4");
        std::fs::write(&src2, b"good").unwrap();
        assert_eq!(register_take(&data, &src2, "001", &verdict_ok()).unwrap().rep_idx, 3);
        let rows = progress::load_index(&data.join("index.tsv")).unwrap();
        let counts = progress::take_counts(&rows);
        assert_eq!(counts["001"], (2, 1), "有効 2(01 と 03)、NG 1(02)");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn register_take_failure_changes_nothing() {
        let root = scratch("registerfail");
        let data = make_data_dir(&root);
        let before = snapshot(&data);
        // 移す元が無い → 失敗。動画も index も変わらず、一時 index も残らず、新しいフォルダも作られない
        let missing = root.join("no-such-take.mp4");
        let e = register_take(&data, &missing, "011", &verdict_ok()).unwrap_err();
        assert!(format!("{e:#}").contains("移せませんでした"), "{e:#}");
        assert_eq!(snapshot(&data), before);
        // 知らない語も同様
        let src = root.join("take-1.mp4");
        std::fs::write(&src, b"x").unwrap();
        assert!(register_take(&data, &src, "999", &verdict_ok()).is_err());
        assert_eq!(snapshot(&data), before);
        assert!(src.exists(), "失敗したら録画は元の場所に残る");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn register_take_does_not_overwrite_an_existing_numbered_file() {
        let root = scratch("registerclash");
        let data = make_data_dir(&root);
        // index に無い 02.mp4 が既にある(別のツールが置いた直後など)
        std::fs::write(data.join("001_konnichiwa/02.mp4"), b"other tool").unwrap();
        let src = root.join("take-1.mp4");
        std::fs::write(&src, b"mine").unwrap();
        let reg = register_take(&data, &src, "001", &verdict_ok()).unwrap();
        assert_eq!(reg.rep_idx, 3);
        assert_eq!(std::fs::read(data.join("001_konnichiwa/02.mp4")).unwrap(), b"other tool");
        let _ = std::fs::remove_dir_all(root);
    }

    // ===== セッション全体(偽のカメラ・手検出・エンコーダ・判定)=====

    #[derive(Clone, Default)]
    struct SharedBuf(Arc<Mutex<Vec<u8>>>);
    impl Write for SharedBuf {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            lock(&self.0).extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl SharedBuf {
        /// 書き終えた行だけを JSON として読む
        fn events(&self) -> Vec<Value> {
            let bytes = lock(&self.0).clone();
            let text = String::from_utf8_lossy(&bytes).to_string();
            let mut lines: Vec<&str> = text.split('\n').collect();
            lines.pop(); // 最後の(書きかけかもしれない)要素
            lines.iter().filter(|l| !l.is_empty()).map(|l| serde_json::from_str(l).unwrap()).collect()
        }
    }

    const FW: usize = 8;
    const FH: usize = 6;

    /// 偽のカメラ: 8x6 のフレームを(約 1000 枚/秒で)出し続け、録画中なら全フレームを録画へ流す
    struct FakeInput {
        tee: TeeSlot,
        idx: u64,
        fps: Option<f64>,
    }
    impl CaptureInput for FakeInput {
        fn poll_frame(&mut self, _timeout: Duration) -> Result<Polled> {
            std::thread::sleep(Duration::from_millis(1));
            let bytes = vec![(self.idx % 200) as u8 + 1; FW * FH * 3];
            self.tee.feed(&bytes);
            let t = self.idx as f64 / 30.0;
            self.idx += 1;
            Ok(Polled::Frame(SourceFrame {
                frame: Array3::from_shape_vec((FH, FW, 3), bytes).unwrap(),
                t,
            }))
        }
        fn tee(&self) -> TeeSlot {
            self.tee.clone()
        }
        fn measured_fps(&self) -> Option<f64> {
            self.fps
        }
        fn dropped_frames(&self) -> u64 {
            0
        }
    }

    struct FakeDetector;
    impl HandObserver for FakeDetector {
        fn observe(&mut self, _frame: &Array3<u8>) -> Result<Observation> {
            Ok(Observation {
                feature: [0.0; DICT_FEATURE_DIM],
                left: false,
                right: true,
            })
        }
    }

    struct FakeJudge(std::result::Result<JudgeOutcome, &'static str>);
    impl TakeJudge for FakeJudge {
        fn judge(&mut self, video: &Path) -> Result<JudgeOutcome> {
            assert!(video.is_file(), "判定は一時ファイルに対して行う: {}", video.display());
            self.0.map_err(|e| anyhow::anyhow!(e))
        }
    }

    struct Harness {
        buf: SharedBuf,
        tx: mpsc::Sender<Input>,
        handle: Option<std::thread::JoinHandle<Result<()>>>,
        root: PathBuf,
        data: PathBuf,
        tmp_base: PathBuf,
        specs: Arc<Mutex<Vec<EncoderSpec>>>,
    }

    struct Opts {
        countdown: f64,
        judge: std::result::Result<JudgeOutcome, &'static str>,
        fps: Option<f64>,
    }

    impl Default for Opts {
        fn default() -> Self {
            Self {
                countdown: 0.0,
                judge: Ok(outcome(0.0, 1.67, 4200)),
                fps: Some(30.0),
            }
        }
    }

    impl Harness {
        fn start(o: Opts) -> Self {
            let root = scratch("session");
            let data = make_data_dir(&root);
            let tmp_base = root.join("tmp");
            std::fs::create_dir_all(&tmp_base).unwrap();
            let buf = SharedBuf::default();
            let (tx, rx) = mpsc::channel();
            let specs: Arc<Mutex<Vec<EncoderSpec>>> = Arc::default();
            let (b2, d2, t2, s2) = (buf.clone(), data.clone(), tmp_base.clone(), Arc::clone(&specs));
            let handle = std::thread::spawn(move || {
                let (emitter, writer) = spawn_writer(b2);
                let stop = StopFlag::default();
                let input = FakeInput {
                    tee: TeeSlot::default(),
                    idx: 0,
                    fps: o.fps,
                };
                let judge_result = o.judge;
                let encoder: EncoderFactory = Box::new(move |spec: &EncoderSpec| {
                    lock(&s2).push(spec.clone());
                    let f = std::fs::File::create(&spec.out)?;
                    Ok(EncoderParts {
                        sink: Box::new(f),
                        finish: Box::new(|| Ok(())),
                    })
                });
                let parts = CaptureParts {
                    input: Box::new(input),
                    detector: Box::new(FakeDetector),
                    judge: Box::new(move || Ok(Box::new(FakeJudge(judge_result)) as Box<dyn TakeJudge>)),
                    encoder,
                    tmp: TempArea::create_in(&t2).unwrap(),
                    roi: PalmRoiMode::CenterSquare,
                };
                let cfg = CaptureConfig {
                    data_dir: d2,
                    countdown_secs: o.countdown,
                    preview: None,
                };
                let r = serve_capture(rx, &emitter, &stop, cfg, parts);
                drop(emitter);
                writer.finish();
                r
            });
            Self {
                buf,
                tx,
                handle: Some(handle),
                root,
                data,
                tmp_base,
                specs,
            }
        }

        fn send(&self, cmd: EngineCommand) {
            self.tx.send(Input::Cmd(cmd)).unwrap();
        }

        fn record(&self, word: &str) {
            self.send(EngineCommand::CaptureRecord { word_id: Some(word.into()) });
        }

        /// 条件に合うイベントが来るまで待つ(これまでに届いた分も見る)
        fn wait(&self, what: &str, pred: impl Fn(&Value) -> bool) -> Value {
            let deadline = Instant::now() + Duration::from_secs(20);
            loop {
                if let Some(e) = self.buf.events().into_iter().find(|e| pred(e)) {
                    return e;
                }
                assert!(Instant::now() < deadline, "{what} が来ない。受信: {:#?}", self.types());
                std::thread::sleep(Duration::from_millis(5));
            }
        }

        fn wait_phase(&self, phase: &'static str) -> Value {
            // 直前の状態より後に出た phase を拾うため、最新の capture_state が phase になるまで待つ
            let deadline = Instant::now() + Duration::from_secs(20);
            loop {
                let ev = self.buf.events();
                if let Some(last) = ev.iter().rev().find(|e| e["type"] == "capture_state") {
                    if last["phase"] == phase {
                        return last.clone();
                    }
                }
                assert!(Instant::now() < deadline, "phase {phase} にならない。最新: {:?}", ev.iter().rev().find(|e| e["type"] == "capture_state"));
                std::thread::sleep(Duration::from_millis(5));
            }
        }

        fn types(&self) -> Vec<String> {
            self.buf.events().iter().map(|e| e["type"].as_str().unwrap().to_string()).collect()
        }

        fn tmp_files(&self) -> Vec<PathBuf> {
            let mut out = Vec::new();
            if let Ok(rd) = std::fs::read_dir(&self.tmp_base) {
                for d in rd.flatten() {
                    if let Ok(files) = std::fs::read_dir(d.path()) {
                        out.extend(files.flatten().map(|f| f.path()));
                    }
                }
            }
            out
        }

        /// quit を送って終わるのを待ち、全イベントと結果を返す
        fn finish(mut self) -> (Vec<Value>, Result<()>, PathBuf, PathBuf) {
            self.send(EngineCommand::Quit);
            let r = self.handle.take().unwrap().join().unwrap();
            (self.buf.events(), r, self.root.clone(), self.tmp_base.clone())
        }
    }

    fn words_dir_state_after(h: &Harness) -> std::collections::BTreeMap<String, Vec<u8>> {
        snapshot(&h.data)
    }

    // 録画 → 停止 → 判定 → 採用: 一時領域に録画され、採用で正しい名前で登録・記帳される
    #[test]
    fn record_stop_judge_accept_registers_the_take() {
        let h = Harness::start(Opts::default());
        let idle = h.wait_phase("idle");
        assert_eq!(idle["data_dir"], h.data.to_string_lossy().as_ref(), "登録先を常に伝える");
        assert!(idle["word"].is_null());
        let before = words_dir_state_after(&h);

        h.record("011");
        let rec = h.wait_phase("recording");
        assert_eq!(rec["word"]["word_id"], "011");
        assert_eq!(rec["word"]["rep_idx"], 1);
        // 録画の途中: 録画の長さが進み、一時領域にファイルがあり、登録先はまだ変わっていない
        h.wait("rec_secs が進む", |e| e["type"] == "capture_state" && e["phase"] == "recording" && e["rec_secs"].as_f64().unwrap() > 0.15);
        assert_eq!(h.tmp_files().len(), 1, "{:?}", h.tmp_files());
        assert!(h.tmp_files()[0].starts_with(&h.tmp_base), "一時領域はリポジトリの外(テストでは専用フォルダ)");
        assert_eq!(words_dir_state_after(&h), before, "採用するまで登録先に何も書かない");

        h.send(EngineCommand::CaptureStop);
        let judged = h.wait_phase("judged");
        assert_eq!(judged["verdict"]["usable"], true);
        assert!(judged["verdict"]["summary"].as_str().unwrap().starts_with("使えそう"));
        assert_eq!(judged["verdict"]["quality_flag"], "ok");
        assert_eq!(judged["word"]["label"], "かれ");

        h.send(EngineCommand::CaptureAccept { confirm_ng: false });
        let saved = h.wait("capture_saved", |e| e["type"] == "capture_saved");
        assert_eq!(saved["file_path"], "011_kare/01.mp4");
        assert_eq!(saved["rep_idx"], 1);
        assert_eq!(saved["quality_flag"], "ok");
        assert_eq!(saved["usable"], true);
        assert_eq!(saved["data_dir"], h.data.to_string_lossy().as_ref());
        h.wait_phase("idle");
        assert!(h.data.join("011_kare/01.mp4").is_file());
        assert!(h.tmp_files().is_empty(), "移したので一時領域は空: {:?}", h.tmp_files());
        let index = std::fs::read_to_string(h.data.join("index.tsv")).unwrap();
        assert!(index.contains(EXISTING_ROW), "既存の行は変わらない");
        assert!(index.lines().last().unwrap().starts_with("011_kare/01.mp4\t011\tkare\tかれ\t1\t"), "{index}");
        // エンコーダには、フレームの寸法と fps(実測 30)が渡された
        let specs = lock(&h.specs).clone();
        assert_eq!(specs.len(), 1);
        assert_eq!((specs[0].width, specs[0].height, specs[0].fps), (FW, FH, 30.0));

        let (events, res, root, tmp_base) = h.finish();
        res.unwrap();
        assert_eq!(events.last().unwrap()["type"], "stopped");
        assert_eq!(events.iter().filter(|e| e["type"] == "capture_saved").count(), 1);
        assert!(!tmp_base.join(format!("{TEMP_PREFIX}{}", std::process::id())).exists(), "終了で一時領域が消える");
        let _ = std::fs::remove_dir_all(root);
    }

    // 録画中は「録画中」が status にも出る(画面が常に明示できる材料)
    #[test]
    fn status_says_recording_only_while_recording() {
        let h = Harness::start(Opts::default());
        h.wait_phase("idle");
        h.record("001");
        h.wait_phase("recording");
        h.wait("status recording", |e| e["type"] == "status" && e["state"] == "recording");
        h.send(EngineCommand::CaptureStop);
        h.wait_phase("judged");
        let (events, res, root, _) = h.finish();
        res.unwrap();
        // 録画が終わったあとに出た status は idle に戻っている
        let last_status = events.iter().rev().find(|e| e["type"] == "status").unwrap();
        assert_eq!(last_status["state"], "idle");
        let _ = std::fs::remove_dir_all(root);
    }

    // 破棄: 一時ファイルが消え、登録先は何も変わらない
    #[test]
    fn discard_removes_the_temp_file_and_leaves_the_data_alone() {
        let h = Harness::start(Opts::default());
        h.wait_phase("idle");
        let before = words_dir_state_after(&h);
        h.record("011");
        h.wait("録画のコマが貯まる", |e| e["type"] == "capture_state" && e["phase"] == "recording" && e["rec_secs"].as_f64().unwrap() > 0.1);
        h.send(EngineCommand::CaptureStop);
        h.wait_phase("judged");
        assert_eq!(h.tmp_files().len(), 1);
        h.send(EngineCommand::CaptureDiscard);
        let idle = h.wait_phase("idle");
        assert!(idle["word"].is_null() && idle["verdict"].is_null());
        assert!(h.tmp_files().is_empty(), "破棄したら一時ファイルは残らない: {:?}", h.tmp_files());
        assert_eq!(words_dir_state_after(&h), before);
        assert!(!h.types().contains(&"capture_saved".to_string()));
        let (_, res, root, _) = h.finish();
        res.unwrap();
        let _ = std::fs::remove_dir_all(root);
    }

    // 撮り直し = 破棄 → 次の録画(同じ語の同じ番号)。録画中の破棄も一時ファイルを残さない
    #[test]
    fn retake_and_discard_during_recording_leave_no_files() {
        let h = Harness::start(Opts::default());
        h.wait_phase("idle");
        h.record("011");
        h.wait("録画のコマが貯まる", |e| e["type"] == "capture_state" && e["phase"] == "recording" && e["rec_secs"].as_f64().unwrap() > 0.1);
        h.send(EngineCommand::CaptureDiscard); // 録画中に捨てる
        h.wait_phase("idle");
        assert!(h.tmp_files().is_empty(), "{:?}", h.tmp_files());
        assert!(!h.types().contains(&"capture_saved".to_string()));
        // もう一度録画できる(2 回目のエンコーダ)
        h.record("011");
        h.wait_phase("recording");
        assert_eq!(lock(&h.specs).len(), 2);
        let (_, res, root, tmp_base) = h.finish();
        res.unwrap();
        assert!(!tmp_base.join(format!("{TEMP_PREFIX}{}", std::process::id())).exists());
        let _ = std::fs::remove_dir_all(root);
    }

    // 使えないテイク: 確認なしでは記録しない。確認つきなら ng 付きで記録し、番号は進む
    #[test]
    fn unusable_take_needs_confirmation_and_is_recorded_as_ng() {
        let h = Harness::start(Opts {
            judge: Ok(outcome(0.0, 0.33, 4000)),
            ..Opts::default()
        });
        h.wait_phase("idle");
        h.record("001");
        h.wait_phase("recording");
        h.send(EngineCommand::CaptureStop);
        let judged = h.wait_phase("judged");
        assert_eq!(judged["verdict"]["usable"], false);
        assert_eq!(judged["verdict"]["reasons"][0], "hands_short");
        assert_eq!(judged["verdict"]["quality_flag"], "ng_hands");
        assert!(judged["verdict"]["summary"].as_str().unwrap().starts_with("使えない"));
        let before = words_dir_state_after(&h);
        // 確認なしの採用は断られ、何も書かれない(判定済みのまま)
        h.send(EngineCommand::CaptureAccept { confirm_ng: false });
        h.wait("断りの警告", |e| e["type"] == "message" && e["level"] == "warn" && e["text"].as_str().unwrap().contains("使えないと判定"));
        assert_eq!(words_dir_state_after(&h), before);
        assert!(!h.types().contains(&"capture_saved".to_string()));
        assert_eq!(h.tmp_files().len(), 1);
        // 確認つき → ng_hands で記録、番号 02
        h.send(EngineCommand::CaptureAccept { confirm_ng: true });
        let saved = h.wait("capture_saved", |e| e["type"] == "capture_saved");
        assert_eq!(saved["file_path"], "001_konnichiwa/02.mp4");
        assert_eq!(saved["quality_flag"], "ng_hands");
        assert_eq!(saved["usable"], false);
        let index = std::fs::read_to_string(h.data.join("index.tsv")).unwrap();
        assert!(index.lines().last().unwrap().contains("\tng_hands\t"), "{index}");
        let (_, res, root, _) = h.finish();
        res.unwrap();
        let _ = std::fs::remove_dir_all(root);
    }

    // 判定そのものの失敗: 判定できなかったと出し、確認つきなら flag 空で記録
    #[test]
    fn judge_failure_is_reported_and_needs_confirmation() {
        let h = Harness::start(Opts {
            judge: Err("モデルが読めません"),
            ..Opts::default()
        });
        h.wait_phase("idle");
        h.record("011");
        h.wait_phase("recording");
        h.send(EngineCommand::CaptureStop);
        let judged = h.wait_phase("judged");
        assert_eq!(judged["verdict"]["reasons"][0], "judge_failed");
        assert!(judged["verdict"]["summary"].as_str().unwrap().contains("判定できませんでした"));
        h.send(EngineCommand::CaptureAccept { confirm_ng: false });
        h.wait("断りの警告", |e| e["type"] == "message" && e["level"] == "warn");
        assert!(!h.data.join("011_kare").exists());
        h.send(EngineCommand::CaptureAccept { confirm_ng: true });
        let saved = h.wait("capture_saved", |e| e["type"] == "capture_saved");
        assert_eq!(saved["quality_flag"], "");
        let (_, res, root, _) = h.finish();
        res.unwrap();
        let _ = std::fs::remove_dir_all(root);
    }

    // 最長 10 秒で自動停止(手動で止めなくても、コマ数の上限で止まり、警告が付き、10 秒未満の動画になる)
    #[test]
    fn auto_stop_at_the_frame_cap_with_a_note() {
        let h = Harness::start(Opts::default());
        h.wait_phase("idle");
        h.record("011");
        let judged = h.wait_phase("judged");
        let notes: Vec<String> = judged["notes"].as_array().unwrap().iter().map(|n| n.as_str().unwrap().to_string()).collect();
        assert!(notes.iter().any(|n| n.contains("自動で止めました")), "{notes:?}");
        let secs = judged["rec_secs"].as_f64().unwrap();
        assert!(secs < 10.0 && secs > 9.9, "299 コマ / 30fps = 9.97 秒: {secs}");
        // 一時ファイルには 299 コマぶんのバイト列
        let files = h.tmp_files();
        assert_eq!(files.len(), 1);
        assert_eq!(std::fs::metadata(&files[0]).unwrap().len(), (299 * FW * FH * 3) as u64);
        let (_, res, root, _) = h.finish();
        res.unwrap();
        let _ = std::fs::remove_dir_all(root);
    }

    // カメラが 30fps 出ていないとき: 実測の fps で録画し(複製しない)、警告を出す
    #[test]
    fn slow_camera_records_at_measured_fps_with_a_warning() {
        let h = Harness::start(Opts {
            fps: Some(14.87),
            ..Opts::default()
        });
        h.wait_phase("idle");
        h.record("011");
        let rec = h.wait_phase("recording");
        let notes = rec["notes"].as_array().unwrap();
        assert!(notes.iter().any(|n| n.as_str().unwrap().contains("14.9 fps")), "{notes:?}");
        assert_eq!(rec["fps"], 14.9);
        assert_eq!(lock(&h.specs)[0].fps, 14.87);
        let judged = h.wait_phase("judged"); // 自動停止 = 148 コマ(148 / 14.87 = 9.95 秒)
        let files = h.tmp_files();
        assert_eq!(std::fs::metadata(&files[0]).unwrap().len(), (148 * FW * FH * 3) as u64);
        assert!(judged["rec_secs"].as_f64().unwrap() < 10.0);
        let (_, res, root, _) = h.finish();
        res.unwrap();
        let _ = std::fs::remove_dir_all(root);
    }

    // カウントダウン: 録画は始まらず(エンコーダも作られず)、取り消せる
    #[test]
    fn countdown_does_not_record_and_can_be_cancelled() {
        let h = Harness::start(Opts {
            countdown: 30.0,
            ..Opts::default()
        });
        h.wait_phase("idle");
        h.record("011");
        let cd = h.wait_phase("countdown");
        let left = cd["countdown_left"].as_f64().unwrap();
        assert!(left > 25.0 && left <= 30.0, "{left}");
        std::thread::sleep(Duration::from_millis(100));
        assert!(lock(&h.specs).is_empty(), "カウントダウン中はエンコーダを作らない");
        assert!(h.tmp_files().is_empty());
        h.send(EngineCommand::CaptureStop); // 取り消し
        let idle = h.wait_phase("idle");
        assert!(idle["word"].is_null());
        assert!(lock(&h.specs).is_empty());
        let (_, res, root, _) = h.finish();
        res.unwrap();
        let _ = std::fs::remove_dir_all(root);
    }

    // 録画中に終了(quit)しても、一時ファイルは消え、登録先は変わらず、stopped が最後
    #[test]
    fn quit_during_recording_cleans_up_everything() {
        let h = Harness::start(Opts::default());
        h.wait_phase("idle");
        let before = words_dir_state_after(&h);
        h.record("011");
        h.wait("録画のコマが貯まる", |e| e["type"] == "capture_state" && e["phase"] == "recording" && e["rec_secs"].as_f64().unwrap() > 0.1);
        assert_eq!(h.tmp_files().len(), 1);
        let data = h.data.clone();
        let (events, res, root, tmp_base) = h.finish();
        res.unwrap();
        assert_eq!(events.last().unwrap()["type"], "stopped");
        assert!(!tmp_base.join(format!("{TEMP_PREFIX}{}", std::process::id())).exists());
        assert_eq!(snapshot(&data), before);
        let _ = std::fs::remove_dir_all(root);
    }

    // 順序違い・知らない語・撮影モードで使えないコマンドは、警告して何も変えない
    #[test]
    fn invalid_commands_are_warned_and_ignored() {
        let h = Harness::start(Opts::default());
        h.wait_phase("idle");
        let before = words_dir_state_after(&h);
        h.send(EngineCommand::CaptureStop);
        h.send(EngineCommand::CaptureAccept { confirm_ng: true });
        h.send(EngineCommand::CaptureDiscard);
        h.send(EngineCommand::Toggle);
        h.record("999");
        h.wait("知らない語の警告", |e| e["type"] == "message" && e["text"].as_str().unwrap().contains("録画する語を決められません"));
        let warns = h.buf.events().iter().filter(|e| e["type"] == "message" && e["level"] == "warn").count();
        assert!(warns >= 5, "{warns}");
        assert_eq!(h.wait_phase("idle")["phase"], "idle");
        assert!(lock(&h.specs).is_empty());
        assert_eq!(words_dir_state_after(&h), before);
        let (_, res, root, _) = h.finish();
        res.unwrap();
        let _ = std::fs::remove_dir_all(root);
    }

    // お題の語が画面とエンジンで食い違っても、録画する語は「録画開始で画面が指定した語」(省略時はエンジンの次の語)
    #[test]
    fn record_without_word_id_uses_the_engines_next_word() {
        let h = Harness::start(Opts::default());
        h.wait_phase("idle");
        h.send(EngineCommand::CaptureRecord { word_id: None });
        let rec = h.wait_phase("recording");
        assert_eq!(rec["word"]["word_id"], "001");
        assert_eq!(rec["word"]["rep_idx"], 2, "001 は 01.mp4 が既にある");
        let (_, res, root, _) = h.finish();
        res.unwrap();
        let _ = std::fs::remove_dir_all(root);
    }

    // ===== 設定の検証 =====

    fn start_with(data_dir: Option<&str>) -> StartCmd {
        StartCmd {
            source: SourceSpec::File { path: "/dummy.mp4".into() },
            model_dir: None,
            segmenter: Default::default(),
            preview: None,
            mode: StartMode::Capture,
            data_dir: data_dir.map(String::from),
            countdown_secs: None,
        }
    }

    #[test]
    fn config_requires_a_readable_data_dir_and_never_creates_one() {
        let e = config_from_start(&start_with(None)).unwrap_err();
        assert!(format!("{e:#}").contains("data_dir"), "{e:#}");
        assert!(config_from_start(&start_with(Some("  "))).is_err());
        let root = scratch("config");
        let missing = root.join("nothing-here");
        let e = config_from_start(&start_with(Some(missing.to_str().unwrap()))).unwrap_err();
        assert!(format!("{e:#}").contains("見つかりません"), "{e:#}");
        assert!(!missing.exists(), "登録先を作ってはいけない");
        // フォルダはあるが words.tsv が無い → 失敗し、words.tsv を作らない
        let bare = root.join("bare");
        std::fs::create_dir_all(&bare).unwrap();
        assert!(config_from_start(&start_with(Some(bare.to_str().unwrap()))).is_err());
        assert!(!bare.join("words.tsv").exists());
        // 正しい登録先
        let data = make_data_dir(&root);
        let cfg = config_from_start(&start_with(Some(data.to_str().unwrap()))).unwrap();
        assert_eq!(cfg.data_dir, data);
        assert_eq!(cfg.countdown_secs, 3.0);
        assert!(cfg.preview.is_some(), "プレビューは省略 = 既定");
        let _ = std::fs::remove_dir_all(root);
    }

    // ===== 境界: ライブ認識の経路から撮影モードを呼ばない =====

    // ライブ認識側のソース(live.rs と json_engine.rs のテスト以外の部分)に、撮影モードを呼ぶ箇所が
    // `serve_dispatch` の 1 か所だけであること。ファイルを書く API を直接呼ぶ箇所が json_engine.rs に増えていないことも確認する
    #[test]
    fn live_path_never_calls_capture() {
        fn non_test_source(full: &str) -> String {
            let cut = full.find("\n#[cfg(test)]\nmod tests {").unwrap_or(full.len());
            full[..cut]
                .lines()
                .filter(|l| !l.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n")
        }
        let live = non_test_source(include_str!("../live.rs"));
        let engine = non_test_source(include_str!("json_engine.rs"));
        // live.rs: モジュール宣言のほかに capture を使わない
        assert!(!live.contains("capture::"), "live.rs が capture を呼んでいる");
        // json_engine.rs: 呼び出しは serve_dispatch の 1 か所だけ
        let calls: Vec<&str> = engine.lines().filter(|l| l.contains("capture::")).collect();
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert!(calls[0].contains("capture::serve_capture_real"), "{calls:?}");
        let dispatch_start = engine.find("fn serve_dispatch(").unwrap();
        let call_pos = engine.find("capture::serve_capture_real").unwrap();
        let next_fn = engine[dispatch_start + 1..].find("\nfn ").map(|p| p + dispatch_start + 1).unwrap();
        assert!(dispatch_start < call_pos && call_pos < next_fn, "呼び出しが serve_dispatch の外にある");
        // ライブ認識の本体(serve_live 以降、run_session)は、ファイルを書く API を使わない
        let live_body_start = engine.find("fn serve_live(").unwrap();
        let live_body_end = engine[live_body_start..].find("\npub(super) struct StopRequested").map(|p| p + live_body_start).unwrap();
        let body = &engine[live_body_start..live_body_end];
        for bad in ["File::create", "fs::write", "fs::rename", "fs::copy", "create_dir", "remove_file", "OpenOptions"] {
            assert!(!body.contains(bad), "serve_live が `{bad}` を使っている");
        }
    }
}
