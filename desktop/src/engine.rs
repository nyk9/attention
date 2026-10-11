//! 認識エンジン(`pose-extract --engine-json`)を子プロセスとして起動し、JSON の会話をする。
//!
//! # スレッドの構成(Rust 初心者向けのメモ)
//!
//! ```text
//! [エンジン] --標準出力--> [読み取りスレッド] --Inbox--> [UI スレッド(定期的に取り出す)]
//!    ^                    JSON を解析、JPEG は                     |
//!    |                    デコード+鏡像+重ね描きまで               |
//!    +------標準入力----------------------(コマンドを書く)---------+
//! [エラー出力の読み取りスレッド] (パイプが詰まらないよう読み捨てる。末尾だけ覚えて異常終了時の説明に使う)
//! ```
//!
//! - 重い処理(JPEG デコードなど)は読み取りスレッドで済ませ、UI スレッドを止めない。
//! - `Inbox` は「UI が次に見るもの」の置き場。プレビュー画像と status は**最新の 1 件だけ**を持ち、
//!   上書きされた古いものは捨てる(UI が遅れても溜まらない)。`result` や `message` は順序が大事で
//!   件数も少ないので、キューに積む。
//! - `Arc` = 参照カウント付きの共有所有、`Mutex` = 同時に 1 つのスレッドだけが触れる錠。
//!
//! # 後始末(アプリを閉じたとき・クラッシュしたとき)
//!
//! エンジンは標準入力が EOF になると、カメラ用 ffmpeg を含めて後始末して終了する。
//! そのため (1) 通常終了では `quit` を送って標準入力を閉じる、(2) アプリが異常終了しても
//! OS がパイプを閉じるのでエンジンが自分で後始末する、(3) それでも終わらなければ SIGTERM、最後に kill。
//!
//! 映像・特徴量・プレビュー画像は、メモリ上でのみ扱いディスクには一切書かない。

use crate::capture_guide::{self, edge::EdgeStatus, HandSample};
use crate::overlay;
use crate::protocol::{parse_event, Command, EngineEvent};
use gpui::RenderImage;
use image::Frame;
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command as Proc, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Inbox のキューの上限(暴走したエンジンがメモリを食い尽くさないように)
const QUEUE_MAX: usize = 1000;
/// エラー出力のうち覚えておく行数
const STDERR_KEEP: usize = 12;
/// 撮影ガイド用の手サンプルのキューの上限(status は最大 10 回/秒なので、UI が 50 秒止まっても溢れない)
const HAND_SAMPLES_MAX: usize = 512;

/// プレビューに何を重ねるか(セッションの開始時に決め、途中では変えない)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayKind {
    /// ライブ認識: 検出範囲の枠と手の骨格(従来どおり)
    Live,
    /// 撮影ガイド: 上に加えて立ち位置ガイド枠、縁に近づいたら検出範囲の枠を赤く。
    /// 受信時刻つきの手サンプルと縁の判定も Inbox に入れる
    CaptureGuide,
}

// ===== エンジンの場所 =====

#[derive(Debug, Clone, PartialEq)]
pub struct EngineLocation {
    pub root: PathBuf,
    /// エンジンの作業ディレクトリ(`pose_extractor/`。モデルのパスが相対のため必須)
    pub cwd: PathBuf,
    pub bin: PathBuf,
}

/// `git rev-parse --git-common-dir` の出力から、本体チェックアウトのルートを求める(撮影ガイドのデータ探し用)。
/// 出力は絶対パスのことも、実行場所からの相対パス(`.git`)のこともある。共通 git ディレクトリの親が本体。
/// 撮影ガイドのソース(`capture_guide/`)は書き込み・プロセス起動 API を禁止しているので、git を呼ぶ処理はここに置く
pub fn main_checkout_from_common_dir(output: &str, cwd: &Path) -> Option<PathBuf> {
    let line = output.lines().next()?.trim();
    if line.is_empty() {
        return None;
    }
    let p = Path::new(line);
    let abs = if p.is_absolute() { p.to_path_buf() } else { cwd.join(p) };
    // 相対形でも `..` を含まないよう正規化する(失敗したら素のまま)
    let abs = std::fs::canonicalize(&abs).unwrap_or(abs);
    abs.parent().map(Path::to_path_buf)
}

/// 実際に git を呼んで、git worktree の本体チェックアウトのルートを探す(読み取り系の `rev-parse` のみ。失敗は None)
pub fn find_main_checkout(repo_root: &Path) -> Option<PathBuf> {
    let out = Proc::new("git").args(["rev-parse", "--git-common-dir"]).current_dir(repo_root).output().ok()?;
    if !out.status.success() {
        return None;
    }
    main_checkout_from_common_dir(&String::from_utf8_lossy(&out.stdout), repo_root)
}

/// リポジトリのルート: 環境変数 `ATTENTION_ROOT`、無ければコンパイル時の `CARGO_MANIFEST_DIR`(= desktop/)の親
pub fn resolve_root(env_root: Option<&str>, manifest_dir: &Path) -> PathBuf {
    match env_root {
        Some(r) if !r.trim().is_empty() => PathBuf::from(r.trim()),
        _ => manifest_dir.parent().unwrap_or(manifest_dir).to_path_buf(),
    }
}

/// ルートからエンジンの場所を決める。バイナリが無ければ、次に何をすればよいかを日本語で返す
pub fn locate_in(root: &Path) -> Result<EngineLocation, String> {
    let cwd = root.join("pose_extractor");
    let bin = cwd.join("target/release/pose-extract");
    if !bin.is_file() {
        return Err(format!(
            "認識エンジンが見つかりません: {}\n\
             先に次のコマンドでビルドしてください。\n\
             cd {} && cargo build --release",
            bin.display(),
            cwd.display()
        ));
    }
    Ok(EngineLocation { root: root.to_path_buf(), cwd, bin })
}

pub fn locate() -> Result<EngineLocation, String> {
    let env_root = std::env::var("ATTENTION_ROOT").ok();
    let root = resolve_root(env_root.as_deref(), Path::new(env!("CARGO_MANIFEST_DIR")));
    locate_in(&root)
}

// ===== Inbox =====

/// 読み取りスレッドから UI スレッドへ渡すもの(エンジンのイベント以外も含む)
#[derive(Debug, Clone, PartialEq)]
pub enum Incoming {
    Event(EngineEvent),
    /// 解釈できなかった行(ログに出す)
    BadLine(String),
    /// エンジンのプロセスが終わった。`detail` は異常終了時の説明(終了コード・エラー出力の末尾)
    Exited { code: Option<i32>, detail: String },
}

pub struct Drained<F> {
    pub frame: Option<F>,
    pub status: Option<EngineEvent>,
    pub events: Vec<Incoming>,
    /// 撮影ガイドのときだけ入る: status の手の有無と受信時刻(全件、時刻順)
    pub hand_samples: Vec<HandSample>,
    /// 撮影ガイドのときだけ入る: 最新のプレビューでの縁の判定
    pub edge: Option<EdgeStatus>,
}

struct InboxInner<F> {
    frame: Option<F>,
    status: Option<EngineEvent>,
    queue: VecDeque<Incoming>,
    hand_samples: VecDeque<HandSample>,
    edge: Option<EdgeStatus>,
}

pub struct Inbox<F> {
    inner: Mutex<InboxInner<F>>,
}

impl<F> Default for Inbox<F> {
    fn default() -> Self {
        Self {
            inner: Mutex::new(InboxInner {
                frame: None,
                status: None,
                queue: VecDeque::new(),
                hand_samples: VecDeque::new(),
                edge: None,
            }),
        }
    }
}

impl<F> Inbox<F> {
    /// 最新のプレビュー画像で置き換える(古い未取得の画像は捨てる)
    pub fn put_frame(&self, f: F) {
        if let Ok(mut g) = self.inner.lock() {
            g.frame = Some(f);
        }
    }
    /// 最新の status で置き換える
    pub fn put_status(&self, s: EngineEvent) {
        if let Ok(mut g) = self.inner.lock() {
            g.status = Some(s);
        }
    }
    pub fn push(&self, i: Incoming) {
        if let Ok(mut g) = self.inner.lock() {
            if g.queue.len() >= QUEUE_MAX {
                g.queue.pop_front();
            }
            g.queue.push_back(i);
        }
    }
    /// 撮影ガイド用: 手サンプルを積む(上限を超えたら古いものから捨てる)
    pub fn push_hand_sample(&self, s: HandSample) {
        if let Ok(mut g) = self.inner.lock() {
            if g.hand_samples.len() >= HAND_SAMPLES_MAX {
                g.hand_samples.pop_front();
            }
            g.hand_samples.push_back(s);
        }
    }
    /// 撮影ガイド用: 最新の縁の判定で置き換える
    pub fn put_edge(&self, e: EdgeStatus) {
        if let Ok(mut g) = self.inner.lock() {
            g.edge = Some(e);
        }
    }
    /// たまっているものをすべて取り出す(UI スレッドが定期的に呼ぶ)
    pub fn drain(&self) -> Drained<F> {
        match self.inner.lock() {
            Ok(mut g) => Drained {
                frame: g.frame.take(),
                status: g.status.take(),
                events: g.queue.drain(..).collect(),
                hand_samples: g.hand_samples.drain(..).collect(),
                edge: g.edge.take(),
            },
            Err(_) => Drained { frame: None, status: None, events: Vec::new(), hand_samples: Vec::new(), edge: None },
        }
    }
}

pub type AppInbox = Inbox<Arc<RenderImage>>;

// ===== 子プロセスの待ち・止め =====

/// 終了を待つ。`term_after` を過ぎても動いていれば SIGTERM、さらに `kill_after` 過ぎたら kill
fn wait_or_kill(child: &Mutex<Child>, term_after: Duration, kill_after: Duration) -> Option<ExitStatus> {
    let start = Instant::now();
    let mut termed = false;
    loop {
        {
            let Ok(mut c) = child.lock() else { return None };
            match c.try_wait() {
                Ok(Some(st)) => return Some(st),
                Ok(None) => {}
                Err(_) => return None,
            }
            let t = start.elapsed();
            if !termed && t >= term_after {
                // SIGTERM: エンジンはこれを受けても後始末して終わる(ffmpeg を残さない)
                unsafe {
                    libc::kill(c.id() as libc::pid_t, libc::SIGTERM);
                }
                termed = true;
            } else if termed && t >= term_after + kill_after {
                let _ = c.kill();
                return c.wait().ok();
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// 開発用のデバッグ書き出し(推論に渡した映像のフレームを PNG で保存する)を有効にする環境変数。
/// pose_extractor 側の `DEBUG_DUMP_ENV` と同じ名前
const ENGINE_DEBUG_DUMP_ENV: &str = "LIVE_DEBUG_DUMP_DIR";

/// エンジンを起動するコマンドを組み立てる(起動はしない。テストで中身を確かめられるよう分けてある)。
///
/// `LIVE_DEBUG_DUMP_DIR` は子プロセスに**引き継がない**。これが設定されたシェルからアプリを起動すると、
/// エンジンが映像をディスクに書き出してしまうのに、画面からは気づけない。映像をディスクに書かない
/// アプリの約束を守るため、環境変数を外して起動する(書き出したいときは、エンジンを端末から直接起動する)
fn engine_command(loc: &EngineLocation) -> Proc {
    let mut cmd = Proc::new(&loc.bin);
    cmd.arg("--engine-json")
        .current_dir(&loc.cwd)
        .env_remove(ENGINE_DEBUG_DUMP_ENV)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd
}

pub struct EngineHandle {
    child: Arc<Mutex<Child>>,
    stdin: Option<ChildStdin>,
    inbox: Arc<AppInbox>,
}

impl EngineHandle {
    /// エンジンを起動する(まだ start は送らない)。プレビューはライブ認識の重ね描き
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn spawn(loc: &EngineLocation) -> Result<Self, String> {
        Self::spawn_with(loc, OverlayKind::Live)
    }

    /// `spawn` の、プレビューの重ね描きの種類を指定できる版
    pub fn spawn_with(loc: &EngineLocation, kind: OverlayKind) -> Result<Self, String> {
        let mut child = engine_command(loc)
            .spawn()
            .map_err(|e| format!("認識エンジンを起動できません({}): {e}", loc.bin.display()))?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().ok_or("エンジンの標準出力を取得できません")?;
        let stderr = child.stderr.take().ok_or("エンジンのエラー出力を取得できません")?;
        let child = Arc::new(Mutex::new(child));
        let inbox: Arc<AppInbox> = Arc::new(Inbox::default());

        // エラー出力: 読み捨てつつ末尾だけ覚える(ディスクには書かない)
        let tail: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(VecDeque::new()));
        {
            let tail = Arc::clone(&tail);
            std::thread::Builder::new()
                .name("engine-stderr".into())
                .spawn(move || {
                    for line in BufReader::new(stderr).split(b'\n').map_while(Result::ok) {
                        let s = String::from_utf8_lossy(&line).into_owned();
                        if let Ok(mut t) = tail.lock() {
                            t.push_back(s);
                            while t.len() > STDERR_KEEP {
                                t.pop_front();
                            }
                        }
                    }
                })
                .map_err(|e| format!("スレッドを作れません: {e}"))?;
        }

        // 標準出力: JSON 行を読み、Inbox に入れる
        {
            let inbox = Arc::clone(&inbox);
            let child = Arc::clone(&child);
            std::thread::Builder::new()
                .name("engine-stdout".into())
                .spawn(move || reader_loop(stdout, &inbox, &child, &tail, kind))
                .map_err(|e| format!("スレッドを作れません: {e}"))?;
        }

        Ok(Self { child, stdin, inbox })
    }

    pub fn inbox(&self) -> Arc<AppInbox> {
        Arc::clone(&self.inbox)
    }

    /// コマンドを 1 行書く。パイプが壊れていたら Err(エンジンが既に終わっている)
    pub fn send(&mut self, cmd: &Command) -> Result<(), String> {
        let stdin = self.stdin.as_mut().ok_or("エンジンの入力は閉じています")?;
        let mut line = cmd.to_line();
        line.push('\n');
        stdin
            .write_all(line.as_bytes())
            .and_then(|_| stdin.flush())
            .map_err(|e| format!("エンジンに送れません: {e}"))
    }

    /// 止める要求(待たない)。quit を送り、標準入力を閉じる。
    /// 3 秒で終わらなければ見張りスレッドが SIGTERM、さらに 2 秒で kill する
    pub fn request_stop(&mut self) {
        if self.stdin.is_some() {
            let _ = self.send(&Command::Quit);
            self.stdin = None; // 閉じる = EOF(エンジンはこれでも後始末して終わる)
        }
        let child = Arc::clone(&self.child);
        let _ = std::thread::Builder::new()
            .name("engine-watchdog".into())
            .spawn(move || {
                wait_or_kill(&child, Duration::from_secs(3), Duration::from_secs(2));
            });
    }
}

impl Drop for EngineHandle {
    /// アプリ終了時など: 止めて、終わるまで待つ(最大 約 5 秒)。既に終わっていれば即戻る
    fn drop(&mut self) {
        if let Some(mut s) = self.stdin.take() {
            let _ = writeln!(s, "{}", Command::Quit.to_line());
            let _ = s.flush();
        }
        wait_or_kill(&self.child, Duration::from_secs(3), Duration::from_secs(2));
    }
}

fn reader_loop(
    stdout: std::process::ChildStdout,
    inbox: &AppInbox,
    child: &Mutex<Child>,
    tail: &Mutex<VecDeque<String>>,
    kind: OverlayKind,
) {
    // split(b'\n') + lossy: 万一 UTF-8 でない行が来ても読み続ける
    for raw in BufReader::with_capacity(1 << 16, stdout).split(b'\n').map_while(Result::ok) {
        let line = String::from_utf8_lossy(&raw);
        match parse_event(&line) {
            Ok(None) => {}
            Ok(Some(EngineEvent::Preview { jpeg_b64, hands, crop, .. })) => {
                // JPEG デコード+鏡像+重ね描きはここ(UI スレッドの外)で済ませる
                let composed = match kind {
                    OverlayKind::Live => overlay::compose_preview(&jpeg_b64, &hands, crop),
                    OverlayKind::CaptureGuide => {
                        capture_guide::compose_guide_preview(&jpeg_b64, &hands, crop).map(|(img, edge)| {
                            inbox.put_edge(edge);
                            img
                        })
                    }
                };
                match composed {
                    Ok(mut rgba) => {
                        overlay::rgba_to_bgra(&mut rgba);
                        let img = RenderImage::new(SmallVec::<[Frame; 1]>::from_elem(Frame::new(rgba), 1));
                        inbox.put_frame(Arc::new(img));
                    }
                    Err(e) => inbox.push(Incoming::BadLine(format!("プレビューを表示できません: {e}"))),
                }
            }
            Ok(Some(ev @ EngineEvent::Status { .. })) => {
                if kind == OverlayKind::CaptureGuide {
                    // 受信時刻はここ(行を読んだ直後)で記録する。理由は capture_guide/mod.rs の冒頭
                    if let EngineEvent::Status { hands, .. } = &ev {
                        inbox.push_hand_sample(HandSample { at: Instant::now(), hands: hands.clone() });
                    }
                }
                inbox.put_status(ev)
            }
            // 知らない種類のイベント(エンジンの方が新しい)は、画面には出さず読み捨てる。
            // 診断用に、行の頭だけ標準エラーへ出す(映像などは含まれない種別の先頭 80 文字まで)
            Ok(Some(EngineEvent::Unknown)) => {
                let head: String = line.chars().take(80).collect();
                eprintln!("[engine] 知らない種類のイベントを読み捨てました: {head}");
            }
            Ok(Some(ev)) => inbox.push(Incoming::Event(ev)),
            Err(e) => {
                // 長い行を丸ごとログに出さない
                let head: String = line.chars().take(80).collect();
                inbox.push(Incoming::BadLine(format!("解釈できない出力: {head} ({e})")));
            }
        }
    }
    // 標準出力が閉じた = エンジンが終わった(または終わりつつある)。終了コードを待つ
    let st = wait_or_kill(child, Duration::from_secs(3), Duration::from_secs(2));
    let code = st.and_then(|s| s.code());
    let detail = if code == Some(0) {
        String::new()
    } else {
        let t = tail.lock().map(|t| t.iter().cloned().collect::<Vec<_>>().join("\n")).unwrap_or_default();
        format!("終了コード {}\n{}", code.map(|c| c.to_string()).unwrap_or_else(|| "なし(シグナル終了)".into()), t)
    };
    inbox.push(Incoming::Exited { code, detail });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn main_checkout_from_git_output() {
        let cwd = Path::new("/nonexistent/wt");
        assert_eq!(main_checkout_from_common_dir("/main/.git\n", cwd), Some(PathBuf::from("/main")));
        assert_eq!(main_checkout_from_common_dir("", cwd), None);
        assert_eq!(
            main_checkout_from_common_dir("../.git\n", Path::new("/nonexistent/a")),
            Some(PathBuf::from("/nonexistent/a/.."))
        );
    }

    /// この worktree(または本体)から実 git で解決できるか(手動実行・読むだけ)。
    /// 本体の raw_jsl が実在する環境で ATTENTION_DATA_DIR 未設定なら、本体側が選ばれ「次の語」が出る
    ///   cargo test --release real_resolution_finds_main_data -- --ignored --nocapture
    #[test]
    #[ignore]
    fn real_resolution_finds_main_data_and_next_word() {
        use crate::capture_guide::progress as gp;
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
        let c = gp::choose_data_dir(None, &root, |p| p.is_dir(), || find_main_checkout(&root));
        eprintln!("{c:?}");
        let v = gp::load_progress_for(&c).expect("本体の raw_jsl が読めるはず");
        eprintln!("{}", gp::next_word_text(&v));
        assert!(v.next.is_some(), "次の語が出る");
    }

    #[test]
    fn root_resolution_prefers_env_then_manifest_parent() {
        let m = Path::new("/repo/desktop");
        assert_eq!(resolve_root(Some("/x/y"), m), PathBuf::from("/x/y"));
        assert_eq!(resolve_root(Some("  /x/y \n"), m), PathBuf::from("/x/y"));
        assert_eq!(resolve_root(None, m), PathBuf::from("/repo"));
        assert_eq!(resolve_root(Some(""), m), PathBuf::from("/repo"));
        assert_eq!(resolve_root(Some("   "), m), PathBuf::from("/repo"));
    }

    #[test]
    fn locate_reports_missing_binary_in_japanese() {
        let dir = std::env::temp_dir().join(format!("attn_locate_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("pose_extractor")).unwrap();
        let err = locate_in(&dir).unwrap_err();
        assert!(err.contains("認識エンジンが見つかりません"), "{err}");
        assert!(err.contains("cargo build --release"), "{err}");
        // バイナリがあれば見つかる。cwd は pose_extractor/
        std::fs::create_dir_all(dir.join("pose_extractor/target/release")).unwrap();
        std::fs::write(dir.join("pose_extractor/target/release/pose-extract"), b"").unwrap();
        let loc = locate_in(&dir).unwrap();
        assert_eq!(loc.cwd, dir.join("pose_extractor"));
        assert!(loc.bin.ends_with("target/release/pose-extract"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn inbox_keeps_only_latest_frame_and_status() {
        let ib: Inbox<u32> = Inbox::default();
        ib.put_frame(1);
        ib.put_frame(2);
        ib.put_frame(3);
        ib.put_status(EngineEvent::Status { state: "idle".into(), hands: "none".into(), fps: 1.0, rec_secs: 0.0 });
        ib.put_status(EngineEvent::Status { state: "recording".into(), hands: "R".into(), fps: 2.0, rec_secs: 1.0 });
        ib.push(Incoming::BadLine("a".into()));
        ib.push(Incoming::BadLine("b".into()));
        let d = ib.drain();
        assert_eq!(d.frame, Some(3), "古いフレームは捨てる");
        match d.status {
            Some(EngineEvent::Status { state, .. }) => assert_eq!(state, "recording"),
            other => panic!("{other:?}"),
        }
        // キューは順序どおり全部残る
        assert_eq!(d.events.len(), 2);
        assert_eq!(d.events[0], Incoming::BadLine("a".into()));
        // 取り出したあとは空
        let d = ib.drain();
        assert!(d.frame.is_none() && d.status.is_none() && d.events.is_empty());
    }

    #[test]
    fn inbox_queue_is_bounded() {
        let ib: Inbox<u32> = Inbox::default();
        for i in 0..(QUEUE_MAX + 50) {
            ib.push(Incoming::BadLine(i.to_string()));
        }
        let d = ib.drain();
        assert_eq!(d.events.len(), QUEUE_MAX);
        // 古いものから捨てる
        assert_eq!(d.events[0], Incoming::BadLine("50".into()));
    }

    // GUI が起動するエンジンには LIVE_DEBUG_DUMP_DIR を引き継がない(映像がディスクに書かれるのに
    // 画面から気づけないため)。`Command::get_envs` は「環境変数に加える/外す」指定を返し、
    // 外す指定は値が None で表れる。また、他の設定(引数・作業ディレクトリ)は変わっていない
    #[test]
    fn engine_command_removes_debug_dump_env() {
        let loc = EngineLocation {
            root: PathBuf::from("/r"),
            cwd: PathBuf::from("/r/pose_extractor"),
            bin: PathBuf::from("/r/pose_extractor/target/release/pose-extract"),
        };
        let cmd = engine_command(&loc);
        let removed: Vec<_> = cmd
            .get_envs()
            .filter(|(k, v)| *k == std::ffi::OsStr::new("LIVE_DEBUG_DUMP_DIR") && v.is_none())
            .collect();
        assert_eq!(removed.len(), 1, "env_remove(LIVE_DEBUG_DUMP_DIR) が指定されていない");
        assert_eq!(cmd.get_program(), loc.bin.as_os_str());
        assert_eq!(cmd.get_args().collect::<Vec<_>>(), vec![std::ffi::OsStr::new("--engine-json")]);
        assert_eq!(cmd.get_current_dir(), Some(loc.cwd.as_path()));
    }

    /// 偽のエンジン(シェルスクリプト)を起動して、JSON 行の受信・コマンド送信・EOF での終了・Exited を確かめる。
    /// 本物のエンジン・モデル・カメラは使わない
    #[test]
    fn spawn_fake_engine_roundtrip_and_exit() {
        let dir = std::env::temp_dir().join(format!("attn_fake_engine_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("pose_extractor/target/release")).unwrap();
        let bin = dir.join("pose_extractor/target/release/pose-extract");
        // 偽のエンジンが受け取った行を残すファイル(送ったコマンドが本当にエンジンへ届いたかを見る)
        let received = dir.join("received.log");
        // 起動したら ready を出し、標準入力の行を読んで(届いた行はファイルに追記して)、
        // "quit" を含む行で stopped を出して終わる
        std::fs::write(
            &bin,
            format!(
                "#!/bin/sh\n\
                 echo '{{\"type\":\"ready\",\"vocab\":[]}}'\n\
                 echo '{{\"type\":\"status\",\"state\":\"idle\",\"hands\":\"none\",\"fps\":0.0,\"rec_secs\":0.0}}'\n\
                 echo 'これはJSONではない'\n\
                 echo '{{\"type\":\"future_event\",\"x\":1}}'\n\
                 while read line; do\n\
                   echo \"$line\" >> '{log}'\n\
                   case \"$line\" in *quit*) echo '{{\"type\":\"stopped\",\"dropped_frames\":0}}'; exit 0;; esac\n\
                 done\n\
                 echo '{{\"type\":\"stopped\",\"dropped_frames\":0}}'\n",
                log = received.display()
            ),
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        let loc = locate_in(&dir).unwrap();
        let mut h = EngineHandle::spawn(&loc).unwrap();
        let inbox = h.inbox();
        h.send(&Command::Toggle).unwrap();
        h.request_stop();

        let mut events = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            let d = inbox.drain();
            events.extend(d.events);
            if events.iter().any(|e| matches!(e, Incoming::Exited { .. })) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(matches!(events.first(), Some(Incoming::Event(EngineEvent::Ready { .. }))), "{events:?}");
        assert!(events.iter().any(|e| matches!(e, Incoming::BadLine(_))), "JSON でない行は BadLine");
        assert!(events.iter().any(|e| matches!(e, Incoming::Event(EngineEvent::Stopped { .. }))));
        // 知らない type の行は読み捨てられ、画面向けのキューには何も出ない(BadLine は JSON でない 1 行だけ)
        assert_eq!(
            events.iter().filter(|e| matches!(e, Incoming::BadLine(_))).count(),
            1,
            "{events:?}"
        );
        assert!(!events.iter().any(|e| matches!(e, Incoming::Event(EngineEvent::Unknown))), "{events:?}");
        // 送った toggle と quit が、この順でエンジンの標準入力に届いている
        let got = std::fs::read_to_string(&received).unwrap();
        let lines: Vec<&str> = got.lines().collect();
        assert_eq!(lines, vec![r#"{"cmd":"toggle"}"#, r#"{"cmd":"quit"}"#], "{got:?}");
        match events.last() {
            Some(Incoming::Exited { code, detail }) => {
                assert_eq!(*code, Some(0));
                assert!(detail.is_empty());
            }
            other => panic!("{other:?}"),
        }
        drop(h);
        let _ = std::fs::remove_dir_all(dir);
    }

    // ===== 実エンジンを使う手動テスト(#[ignore])=====
    // 実行: cargo test --release -- --ignored --nocapture --test-threads=1
    // 先に pose_extractor で cargo build --release が必要。カメラは開かない(偽の ffmpeg を PATH に置く)

    fn process_exists(pattern: &str) -> bool {
        Proc::new("pgrep").args(["-f", pattern]).output().map(|o| o.status.success()).unwrap_or(false)
    }

    fn real_engine() -> Option<EngineLocation> {
        let loc = locate().ok()?;
        let ok = ["models/palm_detection_mediapipe.onnx", "models/handpose_estimation_mediapipe.onnx"]
            .iter()
            .all(|p| loc.cwd.join(p).exists());
        ok.then_some(loc)
    }

    fn wait_for(inbox: &AppInbox, secs: u64, mut f: impl FnMut(&Incoming) -> bool) -> Vec<Incoming> {
        let mut all = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(secs);
        while Instant::now() < deadline {
            let d = inbox.drain();
            let hit = d.events.iter().any(&mut f);
            all.extend(d.events);
            if hit {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        all
    }

    /// 偽の ffmpeg(引数を無視して寝るだけ = 映像が来ない最悪のケース)でカメラ経路を再現し、
    /// ハンドルの Drop(アプリ終了時の処理)でエンジンと ffmpeg が残らないことを確かめる
    #[test]
    #[ignore]
    fn drop_cleans_up_camera_engine_and_ffmpeg() {
        let Some(loc) = real_engine() else { return eprintln!("skip: エンジンかモデルが無い") };
        let dir = std::env::temp_dir().join(format!("attn_fake_ffmpeg_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("ffmpeg");
        std::fs::write(&script, "#!/bin/sh\nexec sleep 987661\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::env::set_var("PATH", format!("{}:{}", dir.display(), std::env::var("PATH").unwrap_or_default()));

        let mut h = EngineHandle::spawn(&loc).unwrap();
        let inbox = h.inbox();
        h.send(&Command::Start {
            source: crate::protocol::SourceSpec::Camera { device: "0".into() },
            segmenter: crate::protocol::SegmenterMode::Auto,
        })
        .unwrap();
        let ev = wait_for(&inbox, 120, |e| matches!(e, Incoming::Event(EngineEvent::Ready { .. })));
        assert!(ev.iter().any(|e| matches!(e, Incoming::Event(EngineEvent::Ready { .. }))), "ready が来ない: {ev:?}");
        let up = Instant::now();
        while !process_exists("sleep 987661") && up.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(process_exists("sleep 987661"), "偽の ffmpeg が起動していない");
        let t0 = Instant::now();
        drop(h);
        eprintln!("Drop に {:.2} 秒", t0.elapsed().as_secs_f64());
        std::thread::sleep(Duration::from_millis(300));
        assert!(!process_exists("sleep 987661"), "ffmpeg が残っている");
        assert!(!process_exists("pose-extract --engine-json"), "エンジンが残っている");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// 動画入力の途中で request_stop → Exited(終了コード 0)、ffmpeg も残らない
    #[test]
    #[ignore]
    fn request_stop_during_file_playback() {
        let Some(loc) = real_engine() else { return eprintln!("skip: エンジンかモデルが無い") };
        let video = loc.root.join("transformer_burn/data/raw_jsl/001_konnichiwa/05.mp4");
        if !video.exists() {
            return eprintln!("skip: 動画が無い");
        }
        let mut h = EngineHandle::spawn(&loc).unwrap();
        let inbox = h.inbox();
        h.send(&Command::Start {
            source: crate::protocol::SourceSpec::File { path: video.display().to_string() },
            segmenter: crate::protocol::SegmenterMode::Auto,
        })
        .unwrap();
        wait_for(&inbox, 120, |e| matches!(e, Incoming::Event(EngineEvent::Ready { .. })));
        let t0 = Instant::now();
        h.request_stop();
        let ev = wait_for(&inbox, 20, |e| matches!(e, Incoming::Exited { .. }));
        eprintln!("request_stop から Exited まで {:.2} 秒", t0.elapsed().as_secs_f64());
        match ev.last() {
            Some(Incoming::Exited { code, .. }) => assert_eq!(*code, Some(0)),
            other => panic!("{other:?}"),
        }
        std::thread::sleep(Duration::from_millis(300));
        assert!(!process_exists(&video.display().to_string()), "ffmpeg が残っている");
        drop(h);
    }
}
