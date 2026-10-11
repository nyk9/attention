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
