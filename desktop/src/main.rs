//! 手話認識のライブ画面(GPUI)。T1: docs/desktop-app-design.md §3.3。
//!
//! 認識そのものは別プロセスのエンジン(`pose-extract --engine-json`)が行い、このアプリは
//! 標準入出力の JSON 行で会話する(`protocol.rs` / `engine.rs`)。カメラを開くのもエンジン側で、
//! このアプリはカメラに触れない。
//!
//! GPUI の状態管理の超要約(Rust 初心者向け):
//! - 画面の部品は `Render` を実装した構造体(ここでは `LiveApp`)。状態はその構造体のフィールドに持つ。
//! - 構造体は `Entity<LiveApp>` という「ハンドル」を通してのみ更新できる(`cx.new(...)` で作る)。
//!   更新したら `cx.notify()` で「再描画して」と知らせる。
//! - `Context<LiveApp>`(cx)は、この構造体の更新中に使える道具箱(タイマー・非同期・通知など)。
//! - 非同期の処理(`cx.spawn_in`)は UI スレッド上で動く。重い処理や待ち受けは別スレッド(engine.rs)で行い、
//!   ここでは一定間隔で結果を取り出すだけにして、UI を止めない。

mod capture_guide;
#[cfg(test)]
mod capture_guide_tests;
mod capture_rec;
mod engine;
mod overlay;
mod protocol;
mod state;

use capture_guide::{hand_run, progress as guide_progress, GuideState};
use engine::{AppInbox, EngineHandle, EngineLocation, Incoming, OverlayKind};
use gpui::{
    actions, div, img, prelude::*, px, rgb, rgba, size, App, Application, Bounds, Context, FocusHandle,
    Focusable, FontWeight, IntoElement, KeyBinding, ObjectFit, Render, RenderImage, SharedString,
    StyledImage, TitlebarOptions, Window, WindowBounds, WindowOptions,
};
use gpui_component::{
    button::{Button, ButtonVariants},
    h_flex,
    progress::Progress,
    v_flex, ActiveTheme, Disableable, Root, Sizable, Theme, ThemeMode,
};
use protocol::{Command, SegmenterMode, SourceSpec};
use capture_rec::Tone;
use state::{format_secs, hands_text, percent, AppMode, AppModel, CapturePhase, LogLevel, Phase, Recognized};
use std::sync::Arc;
use std::time::{Duration, Instant};

// キー操作・メニューの動作(アクション)。スペース = 手動の開始/停止、cmd-q = 終了
actions!(attention_app, [ToggleRecording, QuitApp]);

/// 日本語を確実に出すフォント。gpui-component の既定は ".SystemUIFont" だが、
/// 日本語の字形を Hiragino に固定して見た目を安定させる
const JP_FONT: &str = "Hiragino Sans";

/// 開発用の診断出力(標準エラー)を出すか。環境変数 `ATTENTION_APP_DEBUG` が設定されていれば有効。
/// 映像・特徴量は出さない(イベントの種類と件数、レイアウトの寸法だけ)
fn debug_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATTENTION_APP_DEBUG").is_some())
}

/// レイアウト診断: 子要素の位置と大きさを、変化したときだけ標準エラーへ出す
fn log_bounds(tag: &'static str, children: &[gpui::Bounds<gpui::Pixels>]) {
    static LAST: std::sync::Mutex<Vec<(&'static str, String)>> = std::sync::Mutex::new(Vec::new());
    let text = children
        .iter()
        .map(|b| {
            format!(
                "[x={:.0} y={:.0} w={:.0} h={:.0}]",
                f32::from(b.origin.x),
                f32::from(b.origin.y),
                f32::from(b.size.width),
                f32::from(b.size.height)
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    if let Ok(mut last) = LAST.lock() {
        match last.iter_mut().find(|(t, _)| *t == tag) {
            Some((_, old)) if *old == text => {}
            Some((_, old)) => {
                *old = text.clone();
                eprintln!("[layout] {tag}: {text}");
            }
            None => {
                last.push((tag, text.clone()));
                eprintln!("[layout] {tag}: {text}");
            }
        }
    }
}

/// UI がエンジンの出力を取り出す間隔(約 30 回/秒。プレビューは最大 10fps なので十分)
const POLL_INTERVAL: Duration = Duration::from_millis(33);

struct LiveApp {
    model: AppModel,
    engine: Option<EngineHandle>,
    inbox: Option<Arc<AppInbox>>,
    /// いま表示しているプレビュー画像(鏡像・重ね描き済み)
    image: Option<Arc<RenderImage>>,
    focus_handle: FocusHandle,
    /// 診断用: 表示したプレビューの枚数
    frames_shown: u64,
    /// 撮影ガイドの状態(手の連続検出・縁の判定・お題と進捗)。ライブ認識では使わない
    guide: GuideState,
    /// 診断用: 前回ログに出した撮影ガイドの表示内容(変化したときだけ出す)
    last_guide_log: String,
}

impl LiveApp {
    fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();
        focus_handle.focus(window);

        // 定期的に Inbox を見に行くタスク。`this` は弱い参照で、画面が閉じられれば更新に失敗して終わる
        cx.spawn_in(window, async move |this, cx| loop {
            cx.background_executor().timer(POLL_INTERVAL).await;
            let alive = this.update_in(cx, |this, window, cx| this.poll(window, cx));
            if alive.is_err() {
                break;
            }
        })
        .detach();

        // 開発・検証用: ATTENTION_APP_QUIT_AFTER=<秒> で、その秒数後にアプリを終了する
        // (ウィンドウを閉じる操作なしに、終了処理 = エンジンの後始末を確かめるため)
        if let Some(secs) = std::env::var("ATTENTION_APP_QUIT_AFTER").ok().and_then(|v| v.parse::<f64>().ok()) {
            cx.spawn(async move |this, cx| {
                cx.background_executor().timer(Duration::from_secs_f64(secs)).await;
                let _ = this.update(cx, |_, cx| cx.quit());
            })
            .detach();
        }

        // 開発・検証用: ATTENTION_APP_CAPTURE_SCRIPT="record:2,stop:6,accept:20" のように、起動からの秒数で
        // 撮影(録画)のボタン操作を自動で行う(画面を手で触れない環境で、ボタン → コマンド → エンジン → 表示の
        // 一連を確かめるため)。操作は実際のボタンと同じメソッドを通る(送ってよいかは AppModel が決める)
        if let Ok(script) = std::env::var("ATTENTION_APP_CAPTURE_SCRIPT") {
            for step in script.split(',') {
                let Some((action, secs)) = step.trim().split_once(':') else { continue };
                let (Ok(secs), action) = (secs.trim().parse::<f64>(), action.trim().to_string()) else { continue };
                cx.spawn(async move |this, cx| {
                    cx.background_executor().timer(Duration::from_secs_f64(secs)).await;
                    let _ = this.update(cx, |this, cx| this.run_script_action(&action, cx));
                })
                .detach();
            }
        }

        // 撮影ガイドのデータディレクトリ: ATTENTION_DATA_DIR、無ければ <リポジトリ>/transformer_burn/data/raw_jsl、
        // それも無ければ git worktree の本体チェックアウトの raw_jsl
        let root = engine::resolve_root(
            std::env::var("ATTENTION_ROOT").ok().as_deref(),
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
        );
        let data_choice = guide_progress::choose_data_dir(
            std::env::var(guide_progress::DATA_DIR_ENV).ok().as_deref(),
            &root,
            |p| p.is_dir(),
            || engine::find_main_checkout(&root),
        );

        let mut app = Self {
            model: AppModel::new(),
            engine: None,
            inbox: None,
            image: None,
            focus_handle,
            frames_shown: 0,
            guide: GuideState::from_choice(data_choice),
            last_guide_log: String::new(),
        };

        // 開発・検証用: ATTENTION_APP_MODE=capture-guide で撮影ガイドモードから始める
        if std::env::var("ATTENTION_APP_MODE").is_ok_and(|v| v.trim() == "capture-guide") {
            app.model.mode = AppMode::CaptureGuide;
        }
        // ATTENTION_APP_MODE=capture で撮影(録画あり)から始める
        if std::env::var("ATTENTION_APP_MODE").is_ok_and(|v| v.trim() == "capture") {
            app.model.mode = AppMode::Capture;
        }

        // 開発・検証用: 動画ファイルがあれば、カメラを使わずに自動で開始する
        if let Ok(path) = std::env::var("ATTENTION_APP_VIDEO") {
            if !path.trim().is_empty() {
                app.start_session(SourceSpec::File { path: path.trim().to_string() }, cx);
            }
        }
        app
    }

    /// エンジンを起動して start を送る
    fn start_session(&mut self, source: SourceSpec, cx: &mut Context<Self>) {
        if self.engine.is_some() {
            return;
        }
        self.model.begin_session();
        self.guide.reset_session();
        let kind = match self.model.mode {
            AppMode::Live => OverlayKind::Live,
            AppMode::CaptureGuide | AppMode::Capture => OverlayKind::CaptureGuide,
        };
        let result: Result<EngineHandle, String> = engine::locate().and_then(|loc: EngineLocation| {
            let mut h = EngineHandle::spawn_with(&loc, kind)?;
            // 区切り方は、開始前に選んでおいた値(model.segmenter)をそのまま送る。
            // 撮影(録画)では、登録先(R0 で解決したデータの場所)を付けた撮影モードの start を送る
            // 登録先は絶対パスにして送る(エンジンは別の作業フォルダで動くので、相対パスだと別の場所を指す)
            let data_dir = std::path::absolute(&self.guide.data_dir).unwrap_or_else(|_| self.guide.data_dir.clone());
            h.send(&self.model.start_command(source, &data_dir))?;
            Ok(h)
        });
        match result {
            Ok(h) => {
                self.inbox = Some(h.inbox());
                self.engine = Some(h);
            }
            Err(e) => {
                self.model.error = Some(e);
                self.model.phase = Phase::Idle;
            }
        }
        cx.notify();
    }

    fn start_camera(&mut self, cx: &mut Context<Self>) {
        // 「カメラで開始」: デバイス "0"。区切り方は開始前に選んだ値のまま(ここで上書きしない)
        self.start_session(SourceSpec::Camera { device: "0".into() }, cx);
    }

    /// 停止を要求する(待たない。終了は poll が Exited で気づく)
    fn stop(&mut self, cx: &mut Context<Self>) {
        if let Some(e) = self.engine.as_mut() {
            if matches!(self.model.phase, Phase::Starting | Phase::Running) {
                self.model.phase = Phase::Stopping;
                e.request_stop();
            }
        }
        cx.notify();
    }

    /// 画面が出しているお題の語の ID(撮影で録画する語。エンジンはこの語として録画する)
    fn displayed_word_id(&self) -> Option<String> {
        match &self.guide.progress {
            Some(Ok(v)) => v.next.as_ref().map(|n| n.word_id.clone()),
            _ => None,
        }
    }

    /// 撮影のコマンドを送る(送ってよいかは AppModel が決める。ここは送るだけ)
    fn send_capture(&mut self, cmds: Vec<Command>, cx: &mut Context<Self>) {
        if let Some(e) = self.engine.as_mut() {
            for c in cmds {
                if let Err(err) = e.send(&c) {
                    self.model.push_log(LogLevel::Warn, err);
                    break;
                }
            }
        }
        cx.notify();
    }

    fn capture_record(&mut self, cx: &mut Context<Self>) {
        let w = self.displayed_word_id();
        let cmds: Vec<Command> = self.model.record_command(w.as_deref()).into_iter().collect();
        self.send_capture(cmds, cx);
    }

    fn capture_stop(&mut self, cx: &mut Context<Self>) {
        let cmds: Vec<Command> = self.model.stop_command().into_iter().collect();
        self.send_capture(cmds, cx);
    }

    fn capture_accept(&mut self, cx: &mut Context<Self>) {
        let cmds: Vec<Command> = self.model.accept_command().into_iter().collect();
        self.send_capture(cmds, cx);
    }

    fn capture_discard(&mut self, cx: &mut Context<Self>) {
        let cmds: Vec<Command> = self.model.discard_command().into_iter().collect();
        self.send_capture(cmds, cx);
    }

    fn capture_retake(&mut self, cx: &mut Context<Self>) {
        let w = self.displayed_word_id();
        let cmds = self.model.retake_commands(w.as_deref());
        self.send_capture(cmds, cx);
    }

    /// 開発・検証用の自動操作(ATTENTION_APP_CAPTURE_SCRIPT)。実際のボタンと同じメソッドを呼ぶ
    fn run_script_action(&mut self, action: &str, cx: &mut Context<Self>) {
        if debug_enabled() {
            eprintln!("[script] {action}");
        }
        match action {
            "record" => self.capture_record(cx),
            "stop" => self.capture_stop(cx),
            "accept" => self.capture_accept(cx),
            "discard" => self.capture_discard(cx),
            "retake" => self.capture_retake(cx),
            "space" => self.toggle(cx),
            _ => {}
        }
    }

    fn toggle(&mut self, cx: &mut Context<Self>) {
        // 撮影(録画)では、スペースキーは録画の開始/停止
        if self.model.mode == AppMode::Capture {
            let w = self.displayed_word_id();
            let cmds: Vec<Command> = self.model.space_command(w.as_deref()).into_iter().collect();
            self.send_capture(cmds, cx);
            return;
        }
        // 撮影ガイドでは送らない(スペースキーで届いても何もしない)
        if self.model.toggle_allowed() {
            if let Some(e) = self.engine.as_mut() {
                if let Err(err) = e.send(&Command::Toggle) {
                    self.model.push_log(LogLevel::Warn, err);
                }
            }
        }
        cx.notify();
    }

    /// 画面のモードを切り替える(待機中だけ)
    fn set_mode(&mut self, mode: AppMode, cx: &mut Context<Self>) {
        if !self.model.mode_selectable() || self.model.mode == mode {
            return;
        }
        self.model.mode = mode;
        self.guide.reset_session();
        cx.notify();
    }

    fn set_segmenter(&mut self, mode: SegmenterMode, cx: &mut Context<Self>) {
        // 起動中・停止中は何もしない(ボタンも無効にしてあるが、キー操作などで届いても表示だけ変えない)
        if !self.model.segmenter_selectable() {
            return;
        }
        if self.model.phase == Phase::Running {
            // エンジンが反映したら `segmenter` イベントで表示が変わる
            if let Some(e) = self.engine.as_mut() {
                if let Err(err) = e.send(&Command::SetSegmenter(mode)) {
                    self.model.push_log(LogLevel::Warn, err);
                }
            }
        } else {
            // 開始前は選んだ値を覚えておくだけ(次の start で使う)
            self.model.segmenter = mode;
        }
        cx.notify();
    }

    /// アプリ終了時の後始末: エンジン(とカメラ用 ffmpeg)を止めて、終わるまで待つ
    fn shutdown(&mut self) {
        // EngineHandle の Drop が quit 送信 → 待機 → SIGTERM → kill の順で止める
        self.engine = None;
        self.inbox = None;
    }

    /// Inbox から最新の状態を取り出して反映する(UI スレッドで定期的に呼ばれる)
    fn poll(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // 撮影ガイド: お題と進捗を定期的に読み直す(読むだけ。エンジンが動いていなくても表示する)
        if self.model.mode.shows_guide() && self.guide.reload_due(Instant::now()) {
            self.guide.reload_progress(Instant::now());
            cx.notify();
        }
        let Some(inbox) = self.inbox.clone() else {
            self.log_guide();
            return;
        };
        let d = inbox.drain();
        let mut changed = false;

        if let Some(ev) = d.status {
            self.model.apply(&ev);
            changed = true;
        }
        // 撮影ガイド用(エンジンを撮影ガイドで起動したときだけ入っている)
        for s in &d.hand_samples {
            self.guide.push_sample(s);
            changed = true;
        }
        if let Some(e) = d.edge {
            self.guide.edge = e;
            changed = true;
        }
        for item in d.events {
            changed = true;
            match item {
                Incoming::Event(ev) => {
                    if debug_enabled() {
                        match &ev {
                            protocol::EngineEvent::Result { label, p, low_confidence, .. } => {
                                eprintln!("[event] result {label} p={p:.3} low_confidence={low_confidence}")
                            }
                            protocol::EngineEvent::Ready { vocab, .. } => {
                                eprintln!("[event] ready vocab={}", vocab.len())
                            }
                            other => eprintln!("[event] {}", format!("{other:?}").chars().take(100).collect::<String>()),
                        }
                    }
                    // 登録できたら、次の語・本数をすぐ読み直す(3 秒ごとの読み直しを待たない)
                    if matches!(ev, protocol::EngineEvent::CaptureSaved { .. }) {
                        self.guide.force_reload();
                    }
                    self.model.apply(&ev)
                }
                Incoming::BadLine(text) => self.model.push_log(LogLevel::Warn, text),
                Incoming::Exited { code, detail } => {
                    if debug_enabled() {
                        eprintln!("[event] exited code={code:?} detail={detail:?}");
                    }
                    // エンジンの終了。ハンドルを手放す(既に終わっているので Drop は即戻る)
                    self.engine = None;
                    self.inbox = None;
                    if !detail.is_empty() && self.model.error.is_none() {
                        self.model.error = Some(format!("認識エンジンが異常終了しました。\n{detail}"));
                    }
                    self.model.engine_exited();
                    self.guide.reset_session();
                    self.drop_image(window, cx);
                }
            }
        }
        if let Some(new_img) = d.frame {
            if debug_enabled() {
                self.frames_shown += 1;
                if self.frames_shown == 1 || self.frames_shown.is_multiple_of(20) {
                    eprintln!("[frame] 表示したプレビュー {} 枚目", self.frames_shown);
                }
            }
            if self.engine.is_some() {
                // 古い画像は GPU のテクスチャ置き場(アトラス)から外す。外さないと溜まり続ける
                if let Some(old) = self.image.replace(new_img) {
                    cx.drop_image(old, Some(window));
                }
                changed = true;
            } else {
                // 終了後に届いた最後の 1 枚は表示しない(古い映像を残さない)
                cx.drop_image(new_img, Some(window));
            }
        }
        if changed {
            cx.notify();
        }
        self.log_guide();
    }

    /// 診断用(ATTENTION_APP_DEBUG): 撮影ガイドの画面に渡す文字・色の判定を、変わったときだけ標準エラーへ出す。
    /// 画面のスクリーンショットが撮れない環境で、描画に渡る状態を確かめるため。映像・座標の生データは出さない
    fn log_guide(&mut self) {
        if !debug_enabled() || !self.model.mode.shows_guide() {
            return;
        }
        let g = &self.guide;
        let cur = g.runs.current_best();
        let last = g.runs.last_ended_best();
        let progress = match &g.progress {
            Some(Ok(v)) => guide_progress::next_word_text(v),
            Some(Err(e)) => format!("読めない: {}", e.lines().next().unwrap_or("")),
            None => "未読み込み".into(),
        };
        let cap = if self.model.mode == AppMode::Capture {
            let b = capture_rec::banner(&self.model.capture);
            format!(
                " capture=[{} | dest={} | verdict={} | notes={}]",
                b.text,
                capture_rec::dest_line(&self.model.capture.data_dir, &self.guide.data_dir.to_string_lossy()),
                self.model.capture.verdict.as_ref().map(|v| v.summary.as_str()).unwrap_or("-"),
                self.model.capture.notes.len()
            )
        } else {
            String::new()
        };
        let text = format!(
            "hands={} run={} pass={} last={} edge={:?} crop_red={} progress={}{cap}",
            capture_guide::hands_big_text(g.hands.as_deref()),
            capture_guide::run_text(cur),
            hand_run::passes(cur),
            last.map(capture_guide::run_text).unwrap_or_else(|| "-".into()),
            g.edge.message(),
            g.edge.any(),
            progress
        );
        if text != self.last_guide_log {
            eprintln!("[guide] {text}");
            self.last_guide_log = text;
        }
    }

    fn drop_image(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(old) = self.image.take() {
            cx.drop_image(old, Some(window));
        }
    }
}

impl Focusable for LiveApp {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

// ===== 描画 =====

fn bar_color(rank: usize) -> gpui::Hsla {
    // 1 位は強調、以降は控えめ
    if rank == 0 {
        rgb(0x4fc3f7).into()
    } else {
        rgb(0x78909c).into()
    }
}

impl Render for LiveApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let fg = cx.theme().foreground;
        let muted = cx.theme().muted_foreground;
        let border = cx.theme().border;
        let running = self.model.phase == Phase::Running;
        let busy = !matches!(self.model.phase, Phase::Idle);

        let dbg = debug_enabled();
        div()
            .id("root")
            .key_context("LiveApp")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|this, _: &ToggleRecording, _w, cx| this.toggle(cx)))
            .size_full()
            .bg(cx.theme().background)
            .text_color(fg)
            .font_family(JP_FONT)
            .child(
                // gpui-component の h_flex は items_center なので、列を上端に揃えて高さは各列に h_full で持たせる
                h_flex()
                    .size_full()
                    .p_4()
                    .gap_4()
                    .items_start()
                    .when(dbg, |d| d.on_children_prepainted(|b, _, _| log_bounds("columns(left,right)", &b)))
                    // ===== 左: プレビューと操作 =====
                    .child(
                        v_flex()
                            .flex_1()
                            .min_w_0()
                            .h_full()
                            .gap_3()
                            .when(dbg, |d| d.on_children_prepainted(|b, _, _| log_bounds("left(preview,status,controls)", &b)))
                            .child(self.render_preview(cx))
                            .child(self.render_status(cx))
                            .child(self.render_controls(running, busy, cx)),
                    )
                    // ===== 右: 認識結果・履歴・語彙(撮影ガイドでは、お題・手の検出・検出範囲)=====
                    .child(match self.model.mode {
                        AppMode::Live => v_flex()
                            .w(px(420.))
                            .h_full()
                            .min_h_0()
                            .flex_shrink_0()
                            .gap_3()
                            .when(dbg, |d| d.on_children_prepainted(|b, _, _| log_bounds("right(result,history,log,vocab)", &b)))
                            .child(self.render_result(cx))
                            .child(self.render_history(cx))
                            .child(self.render_log(cx))
                            .child(self.render_vocab(border, muted, cx))
                            .into_any_element(),
                        AppMode::CaptureGuide | AppMode::Capture => v_flex()
                            // on_children_prepainted は id を付ける前(Div)にしか呼べない
                            .when(dbg, |d| d.on_children_prepainted(|b, _, _| log_bounds("right(guide)", &b)))
                            .id("guide-column")
                            .w(px(420.))
                            .h_full()
                            .min_h_0()
                            .flex_shrink_0()
                            .gap_3()
                            .overflow_y_scroll()
                            .when(self.model.mode == AppMode::Capture, |d| d.child(self.render_capture_panel(cx)))
                            .child(self.render_guide_topic(cx))
                            .child(self.render_guide_hands(cx))
                            .child(self.render_guide_edge(cx))
                            .child(self.render_guide_legend(cx))
                            .child(self.render_log(cx))
                            .into_any_element(),
                    }),
            )
    }
}

impl LiveApp {
    fn render_preview(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let placeholder: SharedString = match self.model.phase {
            Phase::Idle => "「カメラで開始」を押すと、ここにカメラ映像が出ます".into(),
            Phase::Starting => "認識エンジンを起動しています(モデルの読み込みに数秒かかります)".into(),
            Phase::Running => "映像を待っています".into(),
            Phase::Stopping => "停止しています".into(),
        };
        let mut frame = div()
            .when(debug_enabled(), |d| d.on_children_prepainted(|b, _, _| log_bounds("preview-content", &b)))
            .id("preview")
            .relative()
            .flex_1()
            .min_h(px(240.))
            .w_full()
            .rounded_lg()
            .overflow_hidden()
            .bg(rgb(0x000000))
            .border_1()
            .border_color(cx.theme().border)
            .flex()
            .items_center()
            .justify_center();
        frame = match &self.image {
            // 鏡像・骨格・検出範囲の枠は、画像側(overlay.rs)で描き込み済み
            Some(image) => frame.child(
                img(image.clone())
                    .size_full()
                    .object_fit(ObjectFit::Contain),
            ),
            None => frame.child(
                div()
                    .px_6()
                    .text_size(px(18.))
                    .text_color(rgb(0xb0bec5))
                    .child(placeholder),
            ),
        };
        // 撮影(録画): 録画中は常に「録画中」、カウントダウン中は大きな数字を、映像の上に重ねる(音に頼らない)
        if self.model.mode == AppMode::Capture && self.model.phase == Phase::Running {
            let c = &self.model.capture;
            match c.phase {
                CapturePhase::Recording => {
                    frame = frame.child(
                        div()
                            .absolute()
                            .top_3()
                            .left_3()
                            .px_3()
                            .py_1()
                            .rounded_md()
                            .bg(rgb(0xb71c1c))
                            .text_color(rgb(0xffffff))
                            .text_size(px(26.))
                            .font_weight(FontWeight::BOLD)
                            .child(SharedString::from(format!("● 録画中 {}", format_secs(c.rec_secs)))),
                    );
                }
                CapturePhase::Countdown => {
                    frame = frame.child(
                        div()
                            .absolute()
                            .top_0()
                            .left_0()
                            .size_full()
                            .flex()
                            .flex_col()
                            .items_center()
                            .justify_center()
                            .child(
                                div()
                                    .px_8()
                                    .py_2()
                                    .rounded_lg()
                                    .bg(rgba(0x000000b3))
                                    .text_color(rgb(0xffffff))
                                    .text_size(px(180.))
                                    .line_height(px(200.))
                                    .font_weight(FontWeight::BOLD)
                                    .child(SharedString::from(capture_rec::countdown_digit(c.countdown_left))),
                            )
                            .child(
                                div()
                                    .mt_2()
                                    .px_4()
                                    .py_1()
                                    .rounded_md()
                                    .bg(rgba(0x000000b3))
                                    .text_color(rgb(0xffb300))
                                    .text_size(px(24.))
                                    .font_weight(FontWeight::BOLD)
                                    .child("録画開始まで(席に戻ってください。取り消しはスペース)"),
                            ),
                    );
                }
                _ => {}
            }
        }
        frame
    }

    fn render_status(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let (state_text, state_color): (SharedString, gpui::Hsla) = match (self.model.phase, &self.model.status) {
            (Phase::Starting, _) => ("起動中".into(), rgb(0x90a4ae).into()),
            (Phase::Stopping, _) => ("停止中".into(), rgb(0x90a4ae).into()),
            (Phase::Idle, _) => ("停止".into(), rgb(0x90a4ae).into()),
            // 撮影ガイドは記録しないので「待機中/記録中」ではなく、録画しないことを明示する
            (Phase::Running, _) if self.model.mode == AppMode::CaptureGuide => {
                ("撮影ガイド中(録画しません)".into(), rgb(0x80deea).into())
            }
            // 撮影(録画): 録画中は常に「● 録画中」と文字で示す
            (Phase::Running, _) if self.model.mode == AppMode::Capture => {
                let c = &self.model.capture;
                match c.phase {
                    CapturePhase::Recording => {
                        (format!("● 録画中 {}", format_secs(c.rec_secs)).into(), rgb(0xff5252).into())
                    }
                    CapturePhase::Countdown => (
                        format!("録画開始まで {} 秒", capture_rec::countdown_digit(c.countdown_left)).into(),
                        rgb(0xffb300).into(),
                    ),
                    CapturePhase::Judging => ("判定中".into(), rgb(0x90a4ae).into()),
                    CapturePhase::Judged => ("判定済み".into(), rgb(0x80deea).into()),
                    CapturePhase::Idle => ("撮影(待機中。録画していません)".into(), rgb(0x80deea).into()),
                }
            }
            (Phase::Running, Some(s)) if s.recording => {
                (format!("記録中 {}", format_secs(s.rec_secs)).into(), rgb(0xff5252).into())
            }
            (Phase::Running, _) => ("待機中".into(), cx.theme().success),
        };
        let hands = match (&self.model.status, self.model.phase) {
            (Some(s), Phase::Running) => hands_text(&s.hands),
            _ => "-",
        };
        let fps = match (&self.model.status, self.model.phase) {
            (Some(s), Phase::Running) => format!("処理 {:.1} fps", s.fps),
            _ => "処理 - fps".to_string(),
        };
        h_flex()
            .gap_4()
            .items_center()
            .child(
                div()
                    .px_4()
                    .py_1()
                    .rounded_full()
                    .bg(state_color)
                    .text_color(rgb(0x101010))
                    .text_size(px(20.))
                    .font_weight(FontWeight::BOLD)
                    .child(state_text),
            )
            .child(div().text_size(px(18.)).child(hands))
            .child(
                div()
                    .text_size(px(16.))
                    .text_color(cx.theme().muted_foreground)
                    .child(fps),
            )
    }

    fn render_controls(&self, running: bool, busy: bool, cx: &mut Context<Self>) -> impl IntoElement {
        let seg = self.model.segmenter;
        // 起動中・停止中は区切り方を変えられない(押しても表示だけ変わって、ready で戻ってしまうため)
        let seg_ok = self.model.segmenter_selectable();
        let recording = self.model.status.as_ref().is_some_and(|s| s.recording);
        let toggle_label = if recording { "手動で停止(スペース)" } else { "手動で開始(スペース)" };

        let guide_mode = self.model.mode.shows_guide();
        let capture_mode = self.model.mode == AppMode::Capture;
        let mut start = Button::new("start")
            .primary()
            .large()
            .label("カメラで開始")
            .disabled(busy)
            .on_click(cx.listener(|this, _, _w, cx| this.start_camera(cx)));
        if busy {
            start = start.outline();
        }
        let mode_ok = self.model.mode_selectable();
        v_flex()
            .gap_2()
            .child(
                // モード切替(待機中だけ押せる)
                h_flex()
                    .gap_3()
                    .items_center()
                    .child(div().text_size(px(16.)).child("モード"))
                    .child(mode_button("mode-live", "ライブ認識", !guide_mode, AppMode::Live, mode_ok, cx))
                    .child(mode_button(
                        "mode-guide",
                        "撮影ガイド(録画なし)",
                        self.model.mode == AppMode::CaptureGuide,
                        AppMode::CaptureGuide,
                        mode_ok,
                        cx,
                    ))
                    .child(mode_button("mode-capture", "撮影(録画あり)", capture_mode, AppMode::Capture, mode_ok, cx))
                    .child(
                        div()
                            .text_size(px(14.))
                            .text_color(cx.theme().muted_foreground)
                            .child(if mode_ok { "停止中に切り替えられます" } else { "切り替えるには先に停止してください" }),
                    ),
            )
            .child(
                h_flex()
                    .gap_3()
                    .items_center()
                    .child(start)
                    .child(
                        Button::new("stop")
                            .danger()
                            .large()
                            .label("停止")
                            .disabled(!matches!(self.model.phase, Phase::Starting | Phase::Running))
                            .on_click(cx.listener(|this, _, _w, cx| this.stop(cx))),
                    )
                    // 撮影ガイドでは手動の開始/停止を出さない(語の認識をしない)
                    .when(!guide_mode, |d| {
                        d.child(
                            Button::new("toggle")
                                .large()
                                .outline()
                                .label(toggle_label)
                                .disabled(!running)
                                .on_click(cx.listener(|this, _, _w, cx| this.toggle(cx))),
                        )
                    }),
            )
            .child(if capture_mode {
                div()
                    .text_size(px(16.))
                    .text_color(cx.theme().muted_foreground)
                    .child("撮影(録画あり): 右の「録画開始」を押すと、カウントダウンのあとに録画します。録画中は画面に「● 録画中」と出ます。採用するまで登録先には何も書きません。語の認識はしません")
                    .into_any_element()
            } else if guide_mode {
                div()
                    .text_size(px(16.))
                    .text_color(cx.theme().muted_foreground)
                    .child("撮影ガイド: 録画・保存はしません。語の認識もしません。撮影は Photo Booth などで続けてください")
                    .into_any_element()
            } else {
                h_flex()
                    .gap_3()
                    .items_center()
                    .child(div().text_size(px(16.)).child("区切り方"))
                    .child(seg_button("seg-auto", "自動", seg == SegmenterMode::Auto, SegmenterMode::Auto, seg_ok, cx))
                    .child(seg_button("seg-manual", "手動", seg == SegmenterMode::Manual, SegmenterMode::Manual, seg_ok, cx))
                    .child(
                        div()
                            .text_size(px(14.))
                            .text_color(cx.theme().muted_foreground)
                            .child("自動: 手の出入りで区切る / 手動: スペースキーで開始・停止"),
                    )
                    .into_any_element()
            })
            .children(self.model.error.as_ref().map(|e| {
                div()
                    .p_3()
                    .rounded_md()
                    .bg(rgb(0x5d1a1a))
                    .border_1()
                    .border_color(rgb(0xff5252))
                    .text_size(px(16.))
                    .text_color(rgb(0xffffff))
                    .child(SharedString::from(e.clone()))
            }))
    }

    fn render_result(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let card = v_flex()
            .p_4()
            .gap_2()
            .rounded_lg()
            .border_1()
            .border_color(cx.theme().border)
            .bg(cx.theme().secondary);
        match &self.model.current {
            None => card
                .child(div().text_size(px(16.)).text_color(cx.theme().muted_foreground).child("認識結果"))
                .child(
                    div()
                        .h(px(84.))
                        .flex()
                        .items_center()
                        .text_size(px(22.))
                        .text_color(cx.theme().muted_foreground)
                        .child(if self.model.phase == Phase::Running {
                            "サインをどうぞ"
                        } else {
                            "まだ結果はありません"
                        }),
                ),
            Some(r) => card
                .child(
                    h_flex()
                        .items_center()
                        .gap_3()
                        .child(div().text_size(px(16.)).text_color(cx.theme().muted_foreground).child("認識結果"))
                        .children(r.low_confidence.then(|| {
                            div()
                                .px_3()
                                .py_0p5()
                                .rounded_full()
                                .bg(cx.theme().warning)
                                .text_color(rgb(0x101010))
                                .text_size(px(16.))
                                .font_weight(FontWeight::BOLD)
                                .child("自信なし")
                        })),
                )
                .child(
                    div()
                        // 最長の語(8 文字)でも 2 行までに収まる大きさ。語が長いと折り返す
                        .text_size(px(52.))
                        .font_weight(FontWeight::BOLD)
                        .line_height(px(62.))
                        .min_h(px(62.))
                        .text_color(if r.low_confidence { cx.theme().warning } else { cx.theme().foreground })
                        .child(SharedString::from(r.label.clone())),
                )
                .children(r.top.iter().enumerate().map(|(i, item)| {
                    h_flex()
                        .gap_2()
                        .items_center()
                        .child(
                            div()
                                .w(px(190.))
                                .flex_shrink_0()
                                .text_size(px(18.))
                                .truncate()
                                .child(SharedString::from(format!("{}. {}", i + 1, item.label))),
                        )
                        .child(
                            div()
                                .flex_1()
                                .child(Progress::new().value(percent(item.p)).bg(bar_color(i)).h(px(12.))),
                        )
                        .child(
                            div()
                                .w(px(56.))
                                .flex_shrink_0()
                                .text_size(px(16.))
                                .child(SharedString::from(format!("{:.0}%", percent(item.p)))),
                        )
                })),
        }
    }

    fn render_history(&self, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .gap_1()
            .child(div().text_size(px(16.)).text_color(cx.theme().muted_foreground).child("履歴(新しい順)"))
            .child(if self.model.history.is_empty() {
                div()
                    .text_size(px(16.))
                    .text_color(cx.theme().muted_foreground)
                    .child("-")
                    .into_any_element()
            } else {
                v_flex()
                    .gap_0p5()
                    .children(self.model.history.iter().map(history_row))
                    .into_any_element()
            })
    }

    fn render_log(&self, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .gap_0p5()
            .child(div().text_size(px(16.)).text_color(cx.theme().muted_foreground).child("メッセージ"))
            .children(self.model.log.iter().take(4).map(|l| {
                let (mark, color): (&str, gpui::Hsla) = match l.level {
                    LogLevel::Info => ("・", cx.theme().muted_foreground),
                    LogLevel::Warn => ("注意: ", cx.theme().warning),
                };
                div()
                    .text_size(px(14.))
                    .text_color(color)
                    .line_clamp(2)
                    .child(SharedString::from(format!("{mark}{}", l.text)))
            }))
            .when(self.model.log.is_empty(), |d| {
                d.child(div().text_size(px(14.)).text_color(cx.theme().muted_foreground).child("-"))
            })
    }

    fn render_vocab(&self, border: gpui::Hsla, muted: gpui::Hsla, _cx: &mut Context<Self>) -> impl IntoElement {
        let current_tag = self.model.current.as_ref().map(|r| r.tag.clone());
        v_flex()
            .flex_1()
            .min_h(px(60.))
            .gap_1()
            .child(
                div()
                    .text_size(px(16.))
                    .text_color(muted)
                    .child(SharedString::from(format!("認識できる語({} 語)", self.model.vocab.len()))),
            )
            .child(
                div()
                    .id("vocab")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .p_2()
                    .rounded_md()
                    .border_1()
                    .border_color(border)
                    .child(
                        h_flex().flex_wrap().gap_1p5().children(self.model.vocab.iter().map(|v| {
                            let hit = current_tag.as_deref() == Some(v.tag.as_str());
                            div()
                                .px_2()
                                .py_0p5()
                                .rounded_md()
                                .text_size(px(16.))
                                .when(hit, |d| d.bg(rgb(0x4fc3f7)).text_color(rgb(0x101010)))
                                .when(!hit, |d| d.bg(rgb(0x2a3138)))
                                .child(SharedString::from(v.label.clone()))
                        })),
                    ),
            )
    }
}

// ===== 撮影ガイドの右列 =====

/// 合格ライン以上 = 緑、未満 = 琥珀色。色だけに頼らないよう、文字でも「合格ライン以上/未満」を出す
fn run_colors(pass: bool) -> (gpui::Hsla, gpui::Hsla) {
    if pass {
        (rgb(0x00c853).into(), rgb(0x0b0b0b).into())
    } else {
        (rgb(0xffb300).into(), rgb(0x0b0b0b).into())
    }
}

fn guide_card(cx: &mut Context<LiveApp>) -> gpui::Div {
    v_flex()
        .p_4()
        .gap_2()
        .rounded_lg()
        .border_1()
        .border_color(cx.theme().border)
        .bg(cx.theme().secondary)
}

/// 見出しの色(背景, 文字)。コントラスト比は背景と文字の組で 4.5 以上
fn tone_colors(t: Tone) -> (gpui::Hsla, gpui::Hsla) {
    match t {
        Tone::Neutral => (rgb(0x37474f).into(), rgb(0xffffff).into()),
        Tone::Recording => (rgb(0xb71c1c).into(), rgb(0xffffff).into()),
        Tone::Warning => (rgb(0xffb300).into(), rgb(0x0b0b0b).into()),
        Tone::Good => (rgb(0x00c853).into(), rgb(0x0b0b0b).into()),
        Tone::Bad => (rgb(0x7f0000).into(), rgb(0xffffff).into()),
    }
}

/// パスを折り返せるようにする(`/` の後ろに幅 0 の空白を入れる。表示だけで、中身は変えない)
fn wrappable_path(p: &str) -> SharedString {
    SharedString::from(p.replace('/', "/\u{200B}"))
}

impl LiveApp {
    /// 撮影(録画あり)の操作パネル。録画の状態・カウントダウン・判定・採用/破棄/撮り直し・登録先
    fn render_capture_panel(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let muted = cx.theme().muted_foreground;
        let c = &self.model.capture;
        let running = self.model.phase == Phase::Running;
        let banner = capture_rec::banner(c);
        let (bg, fg) = tone_colors(banner.tone);
        let word_id = self.displayed_word_id();

        let mut card = guide_card(cx)
            .child(div().text_size(px(16.)).text_color(muted).child("撮影(録画あり)"))
            .child(
                div()
                    .px_4()
                    .py_2()
                    .rounded_lg()
                    .bg(bg)
                    .text_color(fg)
                    .text_size(px(32.))
                    .line_height(px(40.))
                    .font_weight(FontWeight::BOLD)
                    .child(SharedString::from(if running { banner.text } else { "エンジンが動いていません".to_string() })),
            );

        // 判定(判定済みのとき): エンジンの 1 文をそのまま + 数字 + 補足
        if running && c.phase == CapturePhase::Judged {
            if let Some(v) = &c.verdict {
                card = card
                    .child(
                        div()
                            .text_size(px(20.))
                            .font_weight(FontWeight::BOLD)
                            .child(SharedString::from(v.summary.clone())),
                    )
                    .child(div().text_size(px(15.)).child(SharedString::from(capture_rec::verdict_detail(v))));
                for r in &v.reasons {
                    let hint = capture_rec::reason_hint(r);
                    if !hint.is_empty() {
                        card = card.child(div().text_size(px(16.)).text_color(rgb(0xffb300)).child(hint));
                    }
                }
            }
        }

        // ボタン(段階に応じて、押せるものだけを出す)
        let mut buttons = h_flex().gap_2().flex_wrap();
        if running {
            match c.phase {
                CapturePhase::Idle => {
                    buttons = buttons.child(
                        Button::new("cap-record")
                            .primary()
                            .large()
                            .label("録画開始(スペース)")
                            .disabled(word_id.is_none())
                            .on_click(cx.listener(|this, _, _w, cx| this.capture_record(cx))),
                    );
                }
                CapturePhase::Countdown => {
                    buttons = buttons.child(
                        Button::new("cap-cancel")
                            .large()
                            .outline()
                            .label("取り消し(スペース)")
                            .on_click(cx.listener(|this, _, _w, cx| this.capture_stop(cx))),
                    );
                }
                CapturePhase::Recording => {
                    buttons = buttons
                        .child(
                            Button::new("cap-stop")
                                .danger()
                                .large()
                                .label("停止(スペース)")
                                .on_click(cx.listener(|this, _, _w, cx| this.capture_stop(cx))),
                        )
                        .child(
                            Button::new("cap-abort")
                                .large()
                                .outline()
                                .label("録画を捨てる")
                                .on_click(cx.listener(|this, _, _w, cx| this.capture_discard(cx))),
                        );
                }
                CapturePhase::Judging => {
                    buttons = buttons.child(Button::new("cap-judging").large().outline().label("判定中…").disabled(true));
                }
                CapturePhase::Judged => {
                    let usable = c.verdict.as_ref().is_some_and(|v| v.usable);
                    let accept = Button::new("cap-accept")
                        .large()
                        .label(SharedString::from(capture_rec::accept_label(c)))
                        .on_click(cx.listener(|this, _, _w, cx| this.capture_accept(cx)));
                    buttons = buttons
                        .child(if usable { accept.primary() } else { accept.danger() })
                        .child(
                            Button::new("cap-retake")
                                .large()
                                .outline()
                                .label("撮り直し")
                                .disabled(word_id.is_none())
                                .on_click(cx.listener(|this, _, _w, cx| this.capture_retake(cx))),
                        )
                        .child(
                            Button::new("cap-discard")
                                .large()
                                .outline()
                                .label("破棄")
                                .on_click(cx.listener(|this, _, _w, cx| this.capture_discard(cx))),
                        );
                }
            }
        }
        card = card.child(buttons);
        if running && c.phase == CapturePhase::Idle && word_id.is_none() {
            card = card.child(
                div()
                    .text_size(px(16.))
                    .text_color(rgb(0xffb300))
                    .child("お題の語が読めていないので録画できません(下の「次の語」を確認してください)"),
            );
        }
        if running && c.phase == CapturePhase::Judged && !c.verdict.as_ref().is_some_and(|v| v.usable) {
            card = card.child(div().text_size(px(15.)).text_color(muted).child(capture_rec::NG_ACCEPT_NOTE));
        }

        // 録画の fps・コマ落ち・注意
        if running && c.phase != CapturePhase::Idle {
            card = card.child(
                div()
                    .text_size(px(15.))
                    .text_color(muted)
                    .child(SharedString::from(format!("録画の fps: {}", capture_rec::fps_text(c.fps)))),
            );
        }
        if let Some(d) = capture_rec::dropped_text(c.dropped) {
            card = card.child(div().text_size(px(16.)).text_color(rgb(0xffb300)).child(SharedString::from(format!("注意: {d}"))));
        }
        for n in &c.notes {
            card = card.child(div().text_size(px(15.)).text_color(rgb(0xffb300)).child(SharedString::from(format!("注意: {n}"))));
        }

        // 直近の登録
        if let Some(s) = &c.last_saved {
            card = card.child(
                div()
                    .text_size(px(16.))
                    .font_weight(FontWeight::BOLD)
                    .text_color(if s.usable { rgb(0x69f0ae) } else { rgb(0xffb300) })
                    .child(SharedString::from(capture_rec::saved_text(s))),
            );
        }

        // 登録先(常に表示)と、その決まり方。本物のデータに書き込むときは、そのことを文字で明示する
        let origin = self.guide.data_dir_choice.origin;
        let dest = capture_rec::dest_line(&c.data_dir, &self.guide.data_dir.to_string_lossy());
        card.child(
            v_flex()
                .gap_0p5()
                .child(
                    div()
                        .text_size(px(15.))
                        .font_weight(FontWeight::BOLD)
                        .child(wrappable_path(&dest)),
                )
                .child(
                    div()
                        .text_size(px(14.))
                        .text_color(if origin == capture_guide::progress::DataDirOrigin::Env { muted } else { rgb(0xffb300).into() })
                        .child(SharedString::from(capture_rec::origin_note(origin))),
                )
                .child(
                    div()
                        .text_size(px(13.))
                        .text_color(muted)
                        .child("書き込みは、録画の一時ファイル(リポジトリの外)と、採用したときの登録先への取り込みだけです"),
                ),
        )
    }

    /// お題と進捗(読み取りのみ)
    fn render_guide_topic(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let muted = cx.theme().muted_foreground;
        let card = guide_card(cx).child(div().text_size(px(16.)).text_color(muted).child("次の語(進捗は読み取りのみ)"));
        match &self.guide.progress {
            None => card.child(div().text_size(px(20.)).child("読み込み中")),
            Some(Err(e)) => card.child(
                div()
                    .p_3()
                    .rounded_md()
                    .bg(rgb(0x5d1a1a))
                    .border_1()
                    .border_color(rgb(0xff5252))
                    .text_size(px(16.))
                    .text_color(rgb(0xffffff))
                    .child(SharedString::from(format!("お題と進捗を読めません。他の機能は使えます。\n{e}"))),
            ),
            Some(Ok(v)) => {
                let card = match guide_progress::next_word_parts(v) {
                    Some((word, take)) => card
                        .child(
                            div()
                                .text_size(px(48.))
                                .line_height(px(58.))
                                .font_weight(FontWeight::BOLD)
                                .child(SharedString::from(word)),
                        )
                        .child(
                            div()
                                .text_size(px(32.))
                                .font_weight(FontWeight::BOLD)
                                .text_color(rgb(0x80deea))
                                .child(SharedString::from(take)),
                        ),
                    None => card.child(
                        div()
                            .text_size(px(26.))
                            .font_weight(FontWeight::BOLD)
                            .child(SharedString::from(guide_progress::next_word_text(v))),
                    ),
                };
                card.child(div().text_size(px(16.)).text_color(muted).child(SharedString::from(guide_progress::summary_text(v))))
                    .child(
                        div()
                            .text_size(px(13.))
                            .text_color(muted)
                            .child(SharedString::from(format!("データ: {}", self.guide.data_dir.display()))),
                    )
            }
        }
    }

    /// 左右どちらの手が見えているか、連続して検出できている秒数
    fn render_guide_hands(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let muted = cx.theme().muted_foreground;
        let running = self.model.phase == Phase::Running;
        let hands = if running { self.guide.hands.as_deref() } else { None };
        let hands_color: gpui::Hsla = match hands {
            Some("L") | Some("R") | Some("LR") => rgb(0xffffff).into(),
            _ => rgb(0x90a4ae).into(),
        };
        let cur = if running { self.guide.runs.current_best() } else { 0.0 };
        let pass = hand_run::passes(cur);
        let (bg, fg) = run_colors(pass);
        let last = if running { self.guide.runs.last_ended_best() } else { None };
        guide_card(cx)
            .child(div().text_size(px(16.)).text_color(muted).child("手の検出"))
            .child(
                div()
                    .text_size(px(56.))
                    .line_height(px(64.))
                    .font_weight(FontWeight::BOLD)
                    .text_color(hands_color)
                    .child(capture_guide::hands_big_text(hands)),
            )
            .child(
                h_flex()
                    .gap_3()
                    .items_center()
                    .child(div().text_size(px(18.)).child("連続"))
                    .child(
                        div()
                            .px_4()
                            .py_1()
                            .rounded_lg()
                            .bg(bg)
                            .text_color(fg)
                            .text_size(px(40.))
                            .font_weight(FontWeight::BOLD)
                            .child(SharedString::from(capture_guide::run_text(cur))),
                    )
                    .child(
                        div()
                            .text_size(px(18.))
                            .font_weight(FontWeight::BOLD)
                            .child(if pass { "合格ライン以上" } else { "1.0 秒未満" }),
                    ),
            )
            .child(div().text_size(px(16.)).child(SharedString::from(match last {
                Some(s) => format!(
                    "直前の連続区間: {}({})",
                    capture_guide::run_text(s),
                    if hand_run::passes(s) { "合格ライン以上" } else { "1.0 秒未満" }
                ),
                None => "直前の連続区間: -".to_string(),
            })))
            .child(
                div()
                    .text_size(px(14.))
                    .text_color(muted)
                    .child("合格ラインは 1.0 秒(撮影後の手検出チェックと同じ基準。瞬断 1 回までは続いているとみなす)"),
            )
    }

    /// 検出範囲(黄色の枠)の縁に手が近づいたか
    fn render_guide_edge(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let muted = cx.theme().muted_foreground;
        let running = self.model.phase == Phase::Running;
        let msg = if running { self.guide.edge.message() } else { None };
        let card = guide_card(cx).child(div().text_size(px(16.)).text_color(muted).child("検出範囲"));
        match msg {
            Some(m) => card.bg(rgb(0x7f0000)).border_color(rgb(0xff5252)).child(
                div()
                    .text_size(px(24.))
                    .font_weight(FontWeight::BOLD)
                    .text_color(rgb(0xffffff))
                    .child(SharedString::from(format!("{m}。枠の内側へ"))),
            ),
            None => card.child(div().text_size(px(20.)).child(if !running {
                "-"
            } else if self.guide.hands.as_deref().is_some_and(|h| h != "none") {
                "手は検出範囲の内側です"
            } else {
                "手が見えたら判定します"
            })),
        }
    }

    fn render_guide_legend(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let muted = cx.theme().muted_foreground;
        v_flex()
            .gap_1()
            .child(div().text_size(px(14.)).text_color(muted).child(capture_guide::silhouette::GUIDE_LEGEND))
            .child(
                div()
                    .text_size(px(14.))
                    .text_color(muted)
                    .child("黄色の枠: 手を検出する範囲。手が縁に近づくと赤く太くなります"),
            )
            .child(div().text_size(px(14.)).text_color(muted).child(capture_guide::HANDS_NOTE))
    }
}

fn mode_button(
    id: &'static str,
    label: &'static str,
    selected: bool,
    mode: AppMode,
    enabled: bool,
    cx: &mut Context<LiveApp>,
) -> Button {
    let b = Button::new(id)
        .label(label)
        .disabled(!enabled)
        .on_click(cx.listener(move |this, _, _w, cx| this.set_mode(mode, cx)));
    if selected {
        b.primary()
    } else {
        b.outline()
    }
}

fn seg_button(
    id: &'static str,
    label: &'static str,
    selected: bool,
    mode: SegmenterMode,
    enabled: bool,
    cx: &mut Context<LiveApp>,
) -> Button {
    let b = Button::new(id)
        .label(label)
        .disabled(!enabled)
        .on_click(cx.listener(move |this, _, _w, cx| this.set_segmenter(mode, cx)));
    if selected {
        b.primary()
    } else {
        b.outline()
    }
}

fn history_row(r: &Recognized) -> impl IntoElement {
    h_flex()
        .gap_3()
        .items_baseline()
        .child(
            div()
                .w(px(190.))
                .flex_shrink_0()
                .text_size(px(20.))
                .truncate()
                .child(SharedString::from(r.label.clone())),
        )
        .child(
            div()
                .text_size(px(16.))
                .child(SharedString::from(format!("{:.0}%", percent(r.p)))),
        )
        .children(r.low_confidence.then(|| {
            div().text_size(px(14.)).text_color(rgb(0xffb300)).child("自信なし")
        }))
}

fn main() {
    Application::new().run(|cx: &mut App| {
        // gpui-component の部品を使う前に必ず呼ぶ
        gpui_component::init(cx);
        Theme::change(ThemeMode::Dark, None, cx);
        Theme::global_mut(cx).font_family = JP_FONT.into();

        if debug_enabled() {
            // 日本語フォントが解決でき、かな・漢字の字形を持つかの診断
            let ts = cx.text_system();
            let id = ts.resolve_font(&gpui::font(JP_FONT));
            // 存在しない名前を解決させると代替フォントになる。それと違う id なら、指定のフォントが使われている
            let fallback = ts.resolve_font(&gpui::font("存在しないフォント名 xyz"));
            eprintln!(
                "[font] {JP_FONT}: 代替フォントとは別物={} / 'こ' の字形={} / '認' の字形={}",
                id != fallback,
                ts.typographic_bounds(id, px(16.), 'こ').is_ok(),
                ts.typographic_bounds(id, px(16.), '認').is_ok()
            );
        }

        cx.bind_keys([
            KeyBinding::new("space", ToggleRecording, Some("LiveApp")),
            KeyBinding::new("cmd-q", QuitApp, None),
        ]);
        cx.on_action(|_: &QuitApp, cx| cx.quit());

        // ウィンドウを閉じたらアプリも終了(終了時に on_app_quit でエンジンを止める)
        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();

        let bounds = Bounds::centered(None, size(px(1240.), px(800.)), cx);
        let window = cx
            .open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    titlebar: Some(TitlebarOptions {
                        title: Some("手話認識 ライブ".into()),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                |window, cx| {
                    let view = cx.new(|cx| LiveApp::new(window, cx));

                    // アプリ終了時(cmd-q・ウィンドウを閉じる等)にエンジンとカメラ用 ffmpeg を確実に止める。
                    // `on_app_quit` の登録は Subscription を持ち続ける必要があるので、view に結びつけて離さない
                    let weak = view.downgrade();
                    let sub = cx.on_app_quit(move |cx| {
                        let weak = weak.clone();
                        let _ = weak.update(cx, |app, _| app.shutdown());
                        async {}
                    });
                    std::mem::forget(sub); // アプリの生存期間と同じ。終了まで登録を保つ

                    cx.new(|cx| Root::new(view, window, cx))
                },
            )
            .expect("ウィンドウを開けません");
        let _ = window;
        cx.activate(true);
    });
}
