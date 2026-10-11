//! `pose-extract --engine-json`(アプリ専用の内部入口)を**実バイナリとして起動**して確かめる統合テスト。
//! 標準入力に JSON コマンドを流し、標準出力が JSON 行だけであること・イベントの順序・終了コード・
//! 子プロセス(ffmpeg)の後始末を見る。
//!
//! - カメラは開かない。カメラ経路の後始末は、PATH の先頭に置いた偽の `ffmpeg`(寝るだけのシェルスクリプト)で確かめる
//! - 実モデル(ONNX + 学習済みモデル)と raw_jsl の動画が要るものは `#[ignore]`。手動実行:
//!   `cargo test --release --test engine_json_binary -- --ignored --nocapture`
//!   (先に `cargo build --release` は不要。テストが必要なバイナリをビルドする)

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

/// 実行中のエンジン。標準出力の各行を別スレッドで受け、チャンネルで取り出せるようにしてある
struct Engine {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
    stderr_tail: std::sync::Arc<std::sync::Mutex<String>>,
    /// これまでに受け取った全イベント
    events: Vec<Value>,
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

impl Engine {
    /// `path_prefix` を PATH の先頭に足して起動する(偽の ffmpeg を差し込むため)
    fn spawn(path_prefix: Option<&Path>) -> Self {
        Self::spawn_with_env(path_prefix, &[])
    }

    /// `envs` の環境変数を足して起動する
    fn spawn_with_env(path_prefix: Option<&Path>, envs: &[(&str, &str)]) -> Self {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_pose-extract"));
        cmd.envs(envs.iter().copied());
        cmd.arg("--engine-json")
            .current_dir(manifest_dir())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(dir) = path_prefix {
            let old = std::env::var("PATH").unwrap_or_default();
            cmd.env("PATH", format!("{}:{}", dir.display(), old));
        }
        let mut child = cmd.spawn().expect("バイナリを起動できない");
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let (tx, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for l in BufReader::new(stdout).lines().map_while(Result::ok) {
                if tx.send(l).is_err() {
                    break;
                }
            }
        });
        let stderr_tail = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let tail = std::sync::Arc::clone(&stderr_tail);
        std::thread::spawn(move || {
            for l in BufReader::new(stderr).lines().map_while(Result::ok) {
                let mut t = tail.lock().unwrap();
                t.push_str(&l);
                t.push('\n');
                if t.len() > 4000 {
                    let cut = t.len() - 4000;
                    *t = t.chars().skip(cut).collect();
                }
            }
        });
        Self {
            child,
            stdin,
            lines,
            stderr_tail,
            events: Vec::new(),
        }
    }

    fn send(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().expect("stdin は閉じ済み");
        writeln!(stdin, "{line}").unwrap();
        stdin.flush().unwrap();
    }

    fn close_stdin(&mut self) {
        self.stdin = None;
    }

    /// 次の 1 行を JSON として読む。JSON でない行があれば即失敗(標準出力は JSON 行だけ、の検証)
    fn next_event(&mut self, timeout: Duration) -> Option<Value> {
        match self.lines.recv_timeout(timeout) {
            Ok(l) => {
                let v: Value = serde_json::from_str(&l)
                    .unwrap_or_else(|e| panic!("標準出力に JSON でない行: {l:?} ({e})"));
                assert!(v["type"].is_string(), "type が無い: {l}");
                self.events.push(v.clone());
                Some(v)
            }
            Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => None,
        }
    }

    /// 種類 `kind` のイベントが来るまで読む(来なければ失敗)
    fn wait_for(&mut self, kind: &str, timeout: Duration) -> Value {
        let deadline = Instant::now() + timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.next_event(left.max(Duration::from_millis(1))) {
                Some(v) if v["type"] == kind => return v,
                Some(_) => {}
                None => panic!(
                    "{kind} が来ない(timeout)。受信済み: {:?}\nstderr:\n{}",
                    self.types(),
                    self.stderr_tail.lock().unwrap()
                ),
            }
        }
    }

    /// 出力が尽きる(プロセスが終わる)まで読み、終了コードを返す
    fn finish(&mut self, timeout: Duration) -> Option<i32> {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            while let Some(_v) = self.next_event(Duration::from_millis(20)) {}
            if let Some(status) = self.child.try_wait().unwrap() {
                // 残りの行も読み切る
                while let Some(_v) = self.next_event(Duration::from_millis(100)) {}
                return status.code();
            }
        }
        None
    }

    fn types(&self) -> Vec<String> {
        self.events
            .iter()
            .map(|e| e["type"].as_str().unwrap().to_string())
            .collect()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // テストが途中で失敗しても、エンジンを残さない
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// 同じ動画を使うテストを 1 本ずつ動かすための錠。どちらも「動画のパスを持つ ffmpeg が残っていないか」を
/// `pgrep -f` で見るので、並列に走ると相手の ffmpeg を自分の孤児と見間違える
static VIDEO_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// `pgrep -f <pattern>` が何か見つけたか(ffmpeg の孤児チェック用)
fn process_exists(pattern: &str) -> bool {
    Command::new("pgrep")
        .args(["-f", pattern])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// raw_jsl の動画を 1 本選ぶ(無ければ None)
fn sample_video() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ENGINE_JSON_VIDEO") {
        return Some(PathBuf::from(p));
    }
    let root = manifest_dir().join("../transformer_burn/data/raw_jsl");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    for d in dirs {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&d)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "mp4"))
            .collect();
        files.sort();
        if let Some(f) = files.into_iter().next() {
            return Some(f.canonicalize().unwrap_or(f));
        }
    }
    None
}

fn models_present() -> bool {
    ["models/palm_detection_mediapipe.onnx", "models/handpose_estimation_mediapipe.onnx"]
        .iter()
        .all(|p| manifest_dir().join(p).exists())
        && manifest_dir()
            .join("../transformer_burn/models/rec_demo_center_signspan/model.bin")
            .exists()
}

// ===== モデル不要のテスト(常時実行)=====

// 不正な行・未知のコマンドは warn で無視され、回復不能な失敗(存在しないモデル)は
// error イベント + 非 0 の終了コードになる。標準出力は終始 JSON 行だけ
#[test]
fn bad_input_is_warned_and_unrecoverable_failure_exits_nonzero() {
    let mut e = Engine::spawn(None);
    e.send("this is not json");
    e.send(r#"{"cmd":"explode"}"#);
    e.send(r#"{"cmd":"toggle"}"#);
    e.send(
        &json!({
            "cmd": "start",
            "source": {"kind": "file", "path": "/nonexistent/video.mp4"},
            "model_dir": "/nonexistent/model_dir",
        })
        .to_string(),
    );
    let err = e.wait_for("error", Duration::from_secs(60));
    assert!(err["text"].as_str().unwrap().len() > 3);
    e.close_stdin();
    let code = e.finish(Duration::from_secs(20)).expect("終了しない");
    assert_ne!(code, 0, "回復不能な失敗は非 0 で終わる");
    let t = e.types();
    // 先頭の 3 件(不正 JSON・未知のコマンド・start 前の toggle)はどれも warn
    assert_eq!(&t[..3], &["message", "message", "message"], "{t:?}");
    assert!(e.events[..3].iter().all(|v| v["level"] == "warn"));
    assert!(!t.contains(&"ready".to_string()));
}

// start の前に標準入力が閉じられたら、何も始めずに stopped を出して正常終了(終了コード 0)
#[test]
fn eof_before_start_exits_zero_with_stopped() {
    let mut e = Engine::spawn(None);
    e.close_stdin();
    let code = e.finish(Duration::from_secs(20)).expect("終了しない");
    assert_eq!(code, 0);
    assert_eq!(e.types(), vec!["stopped"]);
}

// ===== 実モデル + 実動画 =====

// 動画ファイル入力: ready → … → result → … → stopped、終了コード 0。ffmpeg が残らない
#[test]
#[ignore]
fn file_input_end_to_end() {
    let _serial = VIDEO_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    if !models_present() {
        eprintln!("skip: モデルが無い");
        return;
    }
    let Some(video) = sample_video() else {
        eprintln!("skip: raw_jsl の動画が無い");
        return;
    };
    let mut e = Engine::spawn(None);
    let t0 = Instant::now();
    e.send(
        &json!({
            "cmd": "start",
            "source": {"kind": "file", "path": video.display().to_string()},
            "segmenter": "auto",
            "preview": {"max_fps": 10, "width": 640},
        })
        .to_string(),
    );
    let ready = e.wait_for("ready", Duration::from_secs(120));
    eprintln!(
        "ready: palm_roi={} frame_select={} frames={} segmenter={} preprocess={} vocab={}語 ({:.1}s)",
        ready["palm_roi"], ready["frame_select"], ready["frames"], ready["segmenter"],
        ready["preprocess"], ready["vocab"].as_array().unwrap().len(), t0.elapsed().as_secs_f64()
    );
    // stdin は開けたまま stopped を待つ(動画が尽きれば自分で終わる)
    let _ = e.wait_for("stopped", Duration::from_secs(120));
    e.close_stdin();
    let code = e.finish(Duration::from_secs(20)).expect("終了しない");
    assert_eq!(code, 0);

    let t = e.types();
    assert_eq!(t[0], "ready");
    assert_eq!(t.last().unwrap(), "stopped");
    let mut counts: std::collections::BTreeMap<&str, usize> = Default::default();
    for x in &t {
        *counts.entry(x.as_str()).or_default() += 1;
    }
    eprintln!("動画: {}\n行の種類と件数: {counts:?}", video.display());
    for r in e.events.iter().filter(|v| v["type"] == "result") {
        eprintln!(
            "result: {} ({}) p={} low_confidence={} span={} buffered={} 手あり {}/{}",
            r["tag"], r["label"], r["p"], r["low_confidence"], r["span"], r["buffered"],
            r["selected_with_hand"], r["selected_total"]
        );
    }
    assert!(counts.get("result").copied().unwrap_or(0) >= 1, "result が出ない");
    assert!(counts.get("preview").copied().unwrap_or(0) >= 1, "preview が出ない");
    assert!(counts.get("status").copied().unwrap_or(0) >= 1);
    let pv = e.events.iter().find(|v| v["type"] == "preview").unwrap();
    assert_eq!(pv["width"], 640);
    assert!(pv["jpeg_b64"].as_str().unwrap().len() > 1000);
    assert!(!process_exists(&video.display().to_string()), "動画用の ffmpeg が残っている");
}

// 動画入力の途中で標準入力を閉じる(= GUI が落ちた)と、すぐに後始末して終了する
#[test]
#[ignore]
fn stdin_eof_during_file_playback_cleans_up() {
    let _serial = VIDEO_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    if !models_present() {
        eprintln!("skip: モデルが無い");
        return;
    }
    let Some(video) = sample_video() else {
        eprintln!("skip: raw_jsl の動画が無い");
        return;
    };
    let mut e = Engine::spawn(None);
    e.send(
        &json!({"cmd":"start","source":{"kind":"file","path":video.display().to_string()},"preview":null})
            .to_string(),
    );
    e.wait_for("ready", Duration::from_secs(120));
    e.close_stdin();
    let t0 = Instant::now();
    let code = e.finish(Duration::from_secs(20)).expect("EOF で終了しない");
    eprintln!("EOF から終了まで {:.2}s", t0.elapsed().as_secs_f64());
    assert_eq!(code, 0);
    assert_eq!(e.types().last().unwrap(), "stopped");
    assert!(!process_exists(&video.display().to_string()), "ffmpeg が残っている");
}

/// 偽の ffmpeg(引数を無視してただ寝る)を置いたディレクトリ。カメラを開かずにカメラ経路の後始末を確かめる。
/// 寝る秒数を固有の値にして、`pgrep -f` で孤児を探せるようにする
fn fake_ffmpeg_dir(tag: &str, sleep_secs: u32) -> PathBuf {
    fake_ffmpeg_dir_with_marker(tag, sleep_secs, None)
}

/// `marker` を渡すと、偽の ffmpeg は起動した印としてそのファイルを作ってから寝る
/// (「ffmpeg が一度も起動されなかった」ことを、プロセスの有無ではなくファイルで確かめるため)
fn fake_ffmpeg_dir_with_marker(tag: &str, sleep_secs: u32, marker: Option<&Path>) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fake_ffmpeg_{}_{}", tag, std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let script = dir.join("ffmpeg");
    let touch = marker
        .map(|m| format!("echo started > '{}'\n", m.display()))
        .unwrap_or_default();
    std::fs::write(&script, format!("#!/bin/sh\n{touch}exec sleep {sleep_secs}\n")).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    dir
}

/// カメラ経路: 偽の ffmpeg(映像を一切出さない = 1 枚目を待ち続ける最悪のケース)で、
/// 終了のきっかけごとに「エンジンが終わり、ffmpeg が残らない」ことを確かめる
fn camera_cleanup(tag: &str, sleep_secs: u32, trigger: &str) {
    if !models_present() {
        eprintln!("skip: モデルが無い");
        return;
    }
    let dir = fake_ffmpeg_dir(tag, sleep_secs);
    let mut e = Engine::spawn(Some(&dir));
    e.send(&json!({"cmd":"start","source":{"kind":"camera","device":"0"},"preview":null}).to_string());
    e.wait_for("ready", Duration::from_secs(120));
    let marker = format!("sleep {sleep_secs}");
    // ready は ffmpeg を spawn した直後に出るので、sh が sleep に exec するまで少し待つ
    let up = Instant::now();
    while !process_exists(&marker) && up.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(process_exists(&marker), "偽の ffmpeg が起動していない");
    let t0 = Instant::now();
    match trigger {
        "eof" => e.close_stdin(),
        "quit" => e.send(r#"{"cmd":"quit"}"#),
        "sigterm" | "sigint" => {
            // sigint = Ctrl+C 相当(端末からの割り込み)
            let sig = if trigger == "sigint" { "-INT" } else { "-TERM" };
            let pid = e.child.id().to_string();
            Command::new("kill").args([sig, &pid]).status().unwrap();
        }
        other => panic!("{other}"),
    }
    let code = e.finish(Duration::from_secs(20)).unwrap_or_else(|| {
        panic!("{trigger}: 終了しない。stderr:\n{}", e.stderr_tail.lock().unwrap())
    });
    eprintln!("{trigger}: 終了まで {:.2}s 終了コード {code}", t0.elapsed().as_secs_f64());
    assert_eq!(code, 0, "{trigger}");
    assert_eq!(e.types().last().unwrap(), "stopped", "{trigger}");
    // 後始末のあと少し待って確認(kill 直後の回収待ち)
    std::thread::sleep(Duration::from_millis(300));
    assert!(!process_exists(&marker), "{trigger}: ffmpeg が残っている");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[ignore]
fn camera_path_cleans_up_on_stdin_eof() {
    camera_cleanup("eof", 987_651, "eof");
}

#[test]
#[ignore]
fn camera_path_cleans_up_on_quit_command() {
    camera_cleanup("quit", 987_652, "quit");
}

#[test]
#[ignore]
fn camera_path_cleans_up_on_sigterm() {
    camera_cleanup("sigterm", 987_653, "sigterm");
}

// Ctrl+C(SIGINT)でも、SIGTERM と同じ後始末をして正常終了し、ffmpeg が残らない
#[test]
#[ignore]
fn camera_path_cleans_up_on_sigint() {
    camera_cleanup("sigint", 987_654, "sigint");
}

// 終了要求(quit)が start のすぐあとに来ていたら、モデル読み込みのあとでもカメラ(ffmpeg)を起動しない。
// 偽の ffmpeg は起動したら印のファイルを作るので、それが作られないことを見る
#[test]
fn quit_right_after_start_never_launches_ffmpeg() {
    if !models_present() {
        eprintln!("skip: モデルが無い");
        return;
    }
    let marker = std::env::temp_dir().join(format!("fake_ffmpeg_started_{}", std::process::id()));
    let _ = std::fs::remove_file(&marker);
    let dir = fake_ffmpeg_dir_with_marker("l1", 987_655, Some(&marker));
    let mut e = Engine::spawn(Some(&dir));
    // 1 回の書き込みで続けて送る(エンジンがモデルを読んでいる間に quit が届く)
    e.send(&json!({"cmd":"start","source":{"kind":"camera","device":"0"},"preview":null}).to_string());
    e.send(r#"{"cmd":"quit"}"#);
    let code = e.finish(Duration::from_secs(60)).unwrap_or_else(|| {
        panic!("終了しない。stderr:\n{}", e.stderr_tail.lock().unwrap())
    });
    assert_eq!(code, 0);
    let t = e.types();
    assert_eq!(t, vec!["stopped"], "ready を出さずに stopped だけで終わるはず: {t:?}");
    std::thread::sleep(Duration::from_millis(300));
    assert!(!marker.exists(), "終了要求のあとなのに偽の ffmpeg が起動された");
    assert!(!process_exists("sleep 987655"), "ffmpeg が残っている");
    let _ = std::fs::remove_dir_all(dir);
}

// エンジンが後始末なしで死んでも(kill -9)、カメラ用 ffmpeg が生きたままでも、画面側(読み手)に EOF が届く。
// JSON 用の fd に FD_CLOEXEC が付いていないと、ffmpeg が書き込み口を引き継いで EOF が永遠に来ない
#[test]
fn json_pipe_reaches_eof_after_engine_is_killed_even_if_ffmpeg_lives() {
    if !models_present() {
        eprintln!("skip: モデルが無い");
        return;
    }
    let secs = 987_656;
    let marker = format!("sleep {secs}");
    let dir = fake_ffmpeg_dir("kill9", secs);
    let mut e = Engine::spawn(Some(&dir));
    e.send(&json!({"cmd":"start","source":{"kind":"camera","device":"0"},"preview":null}).to_string());
    e.wait_for("ready", Duration::from_secs(120));
    let up = Instant::now();
    while !process_exists(&marker) && up.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(process_exists(&marker), "偽の ffmpeg が起動していない");

    // エンジンを SIGKILL(後始末なし)。ffmpeg は孤児として生き残る
    e.child.kill().unwrap();
    let _ = e.child.wait();
    assert!(process_exists(&marker), "前提: エンジンが死んでも偽の ffmpeg は生きている");

    // 読み手(テスト側)に EOF が届く = 読み取りスレッドが終わってチャンネルが閉じる
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut eof = false;
    while Instant::now() < deadline {
        match e.lines.recv_timeout(Duration::from_millis(100)) {
            Ok(_) => {}
            Err(RecvTimeoutError::Disconnected) => {
                eof = true;
                break;
            }
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
    // 後始末(孤児の偽 ffmpeg を止める)は、結果にかかわらず先にやる
    let _ = Command::new("pkill").args(["-f", &marker]).status();
    let _ = std::fs::remove_dir_all(dir);
    assert!(eof, "エンジンが死んだのに、JSON のパイプに EOF が届かない(ffmpeg が書き込み口を握っている)");
}

// GUI から起動したエンジンで LIVE_DEBUG_DUMP_DIR が有効になったときは、画面にも警告(message の warn)を出す
// (標準エラーだけでは、映像が書き出されることに画面から気づけない)。
// 存在しない動画を指定して、フレームを一枚も処理しないうちに error で終わらせる(= 何も書き出されない)
#[test]
fn debug_dump_env_is_announced_as_a_warn_message() {
    if !models_present() {
        eprintln!("skip: モデルが無い");
        return;
    }
    let dump = std::env::temp_dir().join(format!("engine_json_dump_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dump);
    let mut e = Engine::spawn_with_env(None, &[("LIVE_DEBUG_DUMP_DIR", dump.to_str().unwrap())]);
    e.send(
        &json!({"cmd":"start","source":{"kind":"file","path":"/nonexistent/video.mp4"},"preview":null})
            .to_string(),
    );
    e.wait_for("error", Duration::from_secs(60));
    e.close_stdin();
    let _ = e.finish(Duration::from_secs(20));
    let warned = e.events.iter().any(|v| {
        v["type"] == "message" && v["level"] == "warn" && v["text"].as_str().is_some_and(|t| t.contains("LIVE_DEBUG_DUMP_DIR"))
    });
    let written = std::fs::read_dir(&dump).map(|d| d.count()).unwrap_or(0);
    let _ = std::fs::remove_dir_all(&dump);
    assert!(warned, "書き出しが有効になったのに画面向けの警告が無い: {:?}", e.events);
    assert_eq!(written, 0, "フレームを処理していないので何も書かれないはず");
}

// ===== 撮影モード(R1): 録画 → 判定 → 採用 =====
//
// 書き込みを伴うテストは、すべて専用の一時フォルダを登録先(data_dir)と TMPDIR にして行う。
// 本物の transformer_burn/data/raw_jsl には書かない(読むのは入力動画としてだけ)。

use std::collections::BTreeMap;

/// テスト専用の作業フォルダ(登録先と TMPDIR を入れる)
fn capture_scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("engine_capture_{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

const CAPTURE_WORDS: &str = "word_id\tromaji\tlabel_ja\tstage\ttarget_takes\n\
                             001\tkonnichiwa\tこんにちは\t1\t3\n\
                             011\tkare\tかれ\t2\t3\n";

/// 登録先(words.tsv だけ)と TMPDIR 用のフォルダを作る。(登録先, TMPDIR)
fn capture_dirs(root: &Path) -> (PathBuf, PathBuf) {
    let data = root.join("raw_jsl");
    let tmp = root.join("tmp");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(data.join("words.tsv"), CAPTURE_WORDS).unwrap();
    (data, tmp)
}

/// フォルダの中身(相対パス → (大きさ, 内容))。増減・変更の検出用
fn snapshot_dir(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
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
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

fn hand_models_present() -> bool {
    ["models/palm_detection_mediapipe.onnx", "models/handpose_estimation_mediapipe.onnx"]
        .iter()
        .all(|p| manifest_dir().join(p).exists())
}

fn capture_start_line(source: Value, data: &Path, countdown: f64) -> String {
    json!({
        "cmd": "start", "mode": "capture", "source": source,
        "data_dir": data.display().to_string(), "countdown_secs": countdown, "preview": null,
    })
    .to_string()
}

// モデル不要: 登録先が無い / words.tsv が無い撮影モードは、カメラを開く前に error で終わり、何も作らない
#[test]
fn capture_start_with_a_bad_data_dir_fails_without_creating_anything() {
    let root = capture_scratch("baddir");
    let (data, tmp) = capture_dirs(&root);
    let missing = root.join("not-there");
    let bare = root.join("bare");
    std::fs::create_dir_all(&bare).unwrap();
    let before = snapshot_dir(&root);
    for (label, line) in [
        ("存在しない", capture_start_line(json!({"kind":"camera","device":"0"}), &missing, 0.0)),
        ("words.tsv が無い", capture_start_line(json!({"kind":"camera","device":"0"}), &bare, 0.0)),
        (
            "data_dir 省略",
            json!({"cmd":"start","mode":"capture","source":{"kind":"camera"}}).to_string(),
        ),
    ] {
        let marker = std::env::temp_dir().join(format!("fake_ffmpeg_started_cap_{}", std::process::id()));
        let _ = std::fs::remove_file(&marker);
        let dir = fake_ffmpeg_dir_with_marker("capbad", 987_660, Some(&marker));
        let mut e = Engine::spawn_with_env(Some(&dir), &[("TMPDIR", tmp.to_str().unwrap())]);
        e.send(&line);
        let err = e.wait_for("error", Duration::from_secs(60));
        assert!(err["text"].as_str().unwrap().len() > 5, "{label}");
        e.close_stdin();
        let code = e.finish(Duration::from_secs(20)).expect("終了しない");
        assert_ne!(code, 0, "{label}");
        assert!(!marker.exists(), "{label}: カメラ(ffmpeg)を起動してはいけない");
        assert!(!e.types().contains(&"ready".to_string()), "{label}");
        let _ = std::fs::remove_dir_all(dir);
    }
    assert_eq!(snapshot_dir(&root), before, "登録先・TMPDIR・その周りに何も作らない");
    assert!(data.join("words.tsv").is_file());
    let _ = std::fs::remove_dir_all(root);
}

/// 撮影モードの終了のきっかけごとに、(映像が来ない最悪のカメラでも)ffmpeg が残らず、一時領域が消えることを確かめる
fn capture_camera_cleanup(tag: &str, sleep_secs: u32, trigger: &str) {
    if !hand_models_present() {
        eprintln!("skip: モデルが無い");
        return;
    }
    let root = capture_scratch(tag);
    let (data, tmp) = capture_dirs(&root);
    let dir = fake_ffmpeg_dir(tag, sleep_secs);
    let mut e = Engine::spawn_with_env(Some(&dir), &[("TMPDIR", tmp.to_str().unwrap())]);
    e.send(&capture_start_line(json!({"kind":"camera","device":"0"}), &data, 3.0));
    e.wait_for("ready", Duration::from_secs(120));
    let state = e.wait_for("capture_state", Duration::from_secs(10));
    assert_eq!(state["phase"], "idle");
    assert_eq!(state["data_dir"], data.display().to_string(), "登録先を画面へ伝える");
    let marker = format!("sleep {sleep_secs}");
    let up = Instant::now();
    while !process_exists(&marker) && up.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(process_exists(&marker), "偽の ffmpeg が起動していない");
    // 一時領域はこの間だけ存在する
    let tmp_dir = tmp.join(format!("attention-capture-{}", e.child.id()));
    assert!(tmp_dir.is_dir(), "一時領域がある: {}", tmp_dir.display());
    match trigger {
        "eof" => e.close_stdin(),
        "quit" => e.send(r#"{"cmd":"quit"}"#),
        "sigterm" => {
            let pid = e.child.id().to_string();
            Command::new("kill").args(["-TERM", &pid]).status().unwrap();
        }
        other => panic!("{other}"),
    }
    let code = e.finish(Duration::from_secs(20)).unwrap_or_else(|| {
        panic!("{trigger}: 終了しない。stderr:\n{}", e.stderr_tail.lock().unwrap())
    });
    assert_eq!(code, 0, "{trigger}");
    assert_eq!(e.types().last().unwrap(), "stopped", "{trigger}");
    std::thread::sleep(Duration::from_millis(300));
    assert!(!process_exists(&marker), "{trigger}: ffmpeg が残っている");
    assert!(!tmp_dir.exists(), "{trigger}: 一時領域が残っている");
    assert_eq!(snapshot_dir(&data).keys().collect::<Vec<_>>(), vec!["words.tsv"], "{trigger}: 登録先に何も増えていない");
    let _ = std::fs::remove_dir_all(dir);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn capture_camera_path_cleans_up_on_stdin_eof() {
    capture_camera_cleanup("capeof", 987_661, "eof");
}

#[test]
fn capture_camera_path_cleans_up_on_quit_command() {
    capture_camera_cleanup("capquit", 987_662, "quit");
}

#[test]
fn capture_camera_path_cleans_up_on_sigterm() {
    capture_camera_cleanup("capterm", 987_663, "sigterm");
}

// 境界: ライブ認識モードで start → toggle → quit を流しても、一時領域にも登録先にもファイルが増えない
// (撮影用のコマンドを送っても、録画も保存も始まらない)
#[test]
fn live_mode_never_writes_to_temp_or_data_dir() {
    if !models_present() {
        eprintln!("skip: モデルが無い");
        return;
    }
    let root = capture_scratch("livenowrite");
    let (data, tmp) = capture_dirs(&root);
    let before = snapshot_dir(&root);
    let sleep = 987_664;
    let dir = fake_ffmpeg_dir("livenowrite", sleep);
    let mut e = Engine::spawn_with_env(
        Some(&dir),
        &[("TMPDIR", tmp.to_str().unwrap()), ("ATTENTION_DATA_DIR", data.to_str().unwrap())],
    );
    // data_dir を付けても、ライブ認識(mode 省略)では使われない
    e.send(
        &json!({"cmd":"start","source":{"kind":"camera","device":"0"},"data_dir":data.display().to_string(),"preview":null})
            .to_string(),
    );
    e.wait_for("ready", Duration::from_secs(120));
    e.send(r#"{"cmd":"toggle"}"#);
    e.send(r#"{"cmd":"capture_record","word_id":"001"}"#);
    e.send(r#"{"cmd":"capture_accept","confirm_ng":true}"#);
    e.send(r#"{"cmd":"toggle"}"#);
    // (映像が来ないカメラでは、ループが 1 枚目を待っているあいだコマンドは読まれない。
    //  撮影用のコマンドが警告されて無視されることは、ユニットテスト live_mode_ignores_capture_commands で確認している)
    e.send(r#"{"cmd":"quit"}"#);
    let code = e.finish(Duration::from_secs(20)).expect("終了しない");
    assert_eq!(code, 0);
    assert!(!e.types().iter().any(|t| t.starts_with("capture_")), "{:?}", e.types());
    std::thread::sleep(Duration::from_millis(300));
    assert!(!process_exists(&format!("sleep {sleep}")), "ffmpeg が残っている");
    assert_eq!(snapshot_dir(&root), before, "ライブ認識で一時領域・登録先が変わった");
    let _ = std::fs::remove_dir_all(dir);
    let _ = std::fs::remove_dir_all(root);
}

/// ffprobe で動画の主要な性質を読む
fn probe(path: &Path) -> Value {
    let out = Command::new("ffprobe")
        .args([
            "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,pix_fmt,bit_rate,nb_read_frames,profile,duration",
            "-of", "json",
        ])
        .arg(path)
        .output()
        .expect("ffprobe を起動できない");
    assert!(out.status.success(), "ffprobe 失敗: {}", String::from_utf8_lossy(&out.stderr));
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    v["streams"][0].clone()
}

// 実モデル + 実 ffmpeg + 動画ファイル入力(カメラ無し): 撮影モードの一連の流れ。
//  start → record → stop → 判定 → accept で、登録先に正しい名前の mp4 と index.tsv の行ができ、
//  その mp4 が 1620x1080 / 30fps / h264 / yuv420p になること。discard で一時ファイルが消えること。
//  終了後に ffmpeg が残らないこと。入力動画は読むだけ
#[test]
#[ignore]
fn capture_file_input_end_to_end() {
    let _serial = VIDEO_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    if !hand_models_present() {
        eprintln!("skip: モデルが無い");
        return;
    }
    let Some(video) = sample_video() else {
        eprintln!("skip: raw_jsl の動画が無い(ENGINE_JSON_VIDEO で指定できる)");
        return;
    };
    let root = capture_scratch("e2e");
    let (data, tmp) = capture_dirs(&root);
    let mut e = Engine::spawn_with_env(None, &[("TMPDIR", tmp.to_str().unwrap())]);
    // 入力動画: カウントダウンを 2.5 秒にして、カウントダウンも通す(fps の実測は起動直後の 1 秒を除くので、それより長く)
    e.send(&capture_start_line(json!({"kind":"file","path":video.display().to_string()}), &data, 2.5));
    e.wait_for("ready", Duration::from_secs(120));
    let state = e.wait_for("capture_state", Duration::from_secs(10));
    assert_eq!(state["phase"], "idle");
    let tmp_dir = tmp.join(format!("attention-capture-{}", e.child.id()));
    assert!(tmp_dir.is_dir());

    // --- 1 本目: 録画 → 停止 → 判定 → 採用 ---
    e.send(r#"{"cmd":"capture_record","word_id":"011"}"#);
    let cd = e.wait_for("capture_state", Duration::from_secs(10));
    assert_eq!(cd["phase"], "countdown", "{cd}");
    assert!(cd["countdown_left"].as_f64().unwrap() <= 2.5);
    loop {
        let s = e.wait_for("capture_state", Duration::from_secs(20));
        if s["phase"] == "recording" {
            assert_eq!(s["word"]["word_id"], "011");
            assert_eq!(s["fps"], 30.0, "動画は 30fps: {s}");
            break;
        }
    }
    // 録画中は一時領域にファイルがあり、登録先にはまだ何も無い
    std::thread::sleep(Duration::from_millis(3200));
    let in_tmp: Vec<_> = std::fs::read_dir(&tmp_dir).unwrap().flatten().collect();
    assert_eq!(in_tmp.len(), 1, "録画中の一時ファイル");
    assert!(!data.join("011_kare").exists(), "採用するまで登録先に書かない");
    e.send(r#"{"cmd":"capture_stop"}"#);
    let judged = loop {
        let s = e.wait_for("capture_state", Duration::from_secs(120));
        if s["phase"] == "judged" {
            break s;
        }
    };
    let v = &judged["verdict"];
    eprintln!("判定: {}", v);
    assert!(v["summary"].as_str().unwrap().len() > 5);
    assert!(v["duration_secs"].as_f64().unwrap() > 2.5, "約 3.2 秒以上録画したはず: {v}");
    let usable = v["usable"].as_bool().unwrap();
    e.send(&json!({"cmd":"capture_accept","confirm_ng":!usable}).to_string());
    let saved = e.wait_for("capture_saved", Duration::from_secs(30));
    assert_eq!(saved["file_path"], "011_kare/01.mp4");
    assert_eq!(saved["usable"], usable);
    assert_eq!(saved["quality_flag"], if usable { "ok" } else { v["quality_flag"].as_str().unwrap() });
    let dest = data.join("011_kare/01.mp4");
    assert!(dest.is_file());
    assert_eq!(std::fs::read_dir(&tmp_dir).unwrap().count(), 0, "移したので一時領域は空");
    let index = std::fs::read_to_string(data.join("index.tsv")).unwrap();
    assert_eq!(index.lines().count(), 2, "{index}");
    assert!(index.lines().nth(1).unwrap().starts_with("011_kare/01.mp4\t011\tkare\tかれ\t1\t"), "{index}");

    // 確認用: ENGINE_CAPTURE_KEEP にフォルダを指定すると、録画した mp4 をコピーして残す
    // (build-dict の経路に通すテスト captured_take_goes_through_build_dict_path で使う)
    if let Ok(keep) = std::env::var("ENGINE_CAPTURE_KEEP") {
        std::fs::create_dir_all(&keep).unwrap();
        std::fs::copy(&dest, Path::new(&keep).join("captured-01.mp4")).unwrap();
    }

    // 録画した mp4 の性質(既存の raw_jsl の動画と同じ: 1620x1080 / 30fps / h264 / yuv420p / 約 7.7Mbps)
    let p = probe(&dest);
    eprintln!("録画の性質: {p}");
    assert_eq!(p["codec_name"], "h264");
    assert_eq!((p["width"].as_u64(), p["height"].as_u64()), (Some(1620), Some(1080)));
    assert_eq!(p["r_frame_rate"], "30/1");
    assert_eq!(p["pix_fmt"], "yuv420p");
    assert_eq!(p["profile"], "High");
    let bitrate: f64 = p["bit_rate"].as_str().unwrap().parse().unwrap();
    assert!((3.0e6..12.0e6).contains(&bitrate), "ビットレートが約 7.7Mbps から外れている: {bitrate}");
    let frames: f64 = p["nb_read_frames"].as_str().unwrap().parse().unwrap();
    let dur = v["duration_secs"].as_f64().unwrap();
    assert!((frames / 30.0 - dur).abs() < 0.2, "コマ数 {frames} と長さ {dur} が合わない");

    // --- 2 本目: 録画 → 停止 → 判定 → 破棄(一時ファイルが残らない)---
    e.send(r#"{"cmd":"capture_record","word_id":"011"}"#);
    loop {
        let s = e.wait_for("capture_state", Duration::from_secs(20));
        if s["phase"] == "recording" {
            assert_eq!(s["word"]["rep_idx"], 2, "次は 02.mp4");
            break;
        }
    }
    std::thread::sleep(Duration::from_millis(2500));
    e.send(r#"{"cmd":"capture_stop"}"#);
    loop {
        if e.wait_for("capture_state", Duration::from_secs(120))["phase"] == "judged" {
            break;
        }
    }
    assert_eq!(std::fs::read_dir(&tmp_dir).unwrap().count(), 1);
    let data_before = snapshot_dir(&data);
    e.send(r#"{"cmd":"capture_discard"}"#);
    loop {
        if e.wait_for("capture_state", Duration::from_secs(20))["phase"] == "idle" {
            break;
        }
    }
    assert_eq!(std::fs::read_dir(&tmp_dir).unwrap().count(), 0, "破棄したので一時ファイルは残らない");
    assert_eq!(snapshot_dir(&data), data_before, "破棄で登録先は変わらない");

    // --- 終了: ffmpeg が残らず、一時領域が消える ---
    e.send(r#"{"cmd":"quit"}"#);
    let code = e.finish(Duration::from_secs(30)).expect("終了しない");
    assert_eq!(code, 0);
    assert_eq!(e.types().last().unwrap(), "stopped");
    std::thread::sleep(Duration::from_millis(300));
    // pgrep -f は親シェルのコマンド行にもマッチしうるので、このテストの ffmpeg にしか現れない引数と組み合わせて探す
    let input_pat = format!("-re -stream_loop -1 -i {}", video.display());
    let encoder_pat = format!("rawvideo.*{}", tmp_dir.display());
    assert!(!process_exists(&input_pat), "入力動画用の ffmpeg が残っている");
    assert!(!process_exists(&encoder_pat), "録画用の ffmpeg が残っている");
    assert!(!tmp_dir.exists(), "一時領域が残っている");
    let _ = std::fs::remove_dir_all(root);
}

// 録画を始めてから最長(10 秒)で自動停止し、10 秒未満の動画になる(実 ffmpeg・動画ファイル入力)
#[test]
#[ignore]
fn capture_auto_stops_before_ten_seconds() {
    let _serial = VIDEO_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    if !hand_models_present() {
        eprintln!("skip: モデルが無い");
        return;
    }
    let Some(video) = sample_video() else {
        eprintln!("skip: raw_jsl の動画が無い");
        return;
    };
    let root = capture_scratch("autostop");
    let (data, tmp) = capture_dirs(&root);
    let mut e = Engine::spawn_with_env(None, &[("TMPDIR", tmp.to_str().unwrap())]);
    e.send(&capture_start_line(json!({"kind":"file","path":video.display().to_string()}), &data, 0.0));
    e.wait_for("ready", Duration::from_secs(120));
    e.send(r#"{"cmd":"capture_record","word_id":"001"}"#);
    let judged = loop {
        let s = e.wait_for("capture_state", Duration::from_secs(120));
        if s["phase"] == "judged" {
            break s;
        }
    };
    let secs = judged["rec_secs"].as_f64().unwrap();
    assert!((9.5..10.0).contains(&secs), "最長に近いが 10 秒未満: {secs}");
    assert!(judged["notes"].as_array().unwrap().iter().any(|n| n.as_str().unwrap().contains("自動で止めました")));
    assert!(judged["verdict"]["reasons"].as_array().unwrap().iter().all(|r| r != "too_long"), "自動停止で『長すぎる』にしない");
    e.send(r#"{"cmd":"capture_discard"}"#);
    e.send(r#"{"cmd":"quit"}"#);
    assert_eq!(e.finish(Duration::from_secs(30)), Some(0));
    let _ = std::fs::remove_dir_all(root);
}
