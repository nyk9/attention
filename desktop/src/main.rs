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

mod engine;
mod overlay;
mod protocol;
mod state;

use engine::{AppInbox, EngineHandle, EngineLocation, Incoming};
use gpui::{
    actions, div, img, prelude::*, px, rgb, size, App, Application, Bounds, Context, FocusHandle,
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
use state::{format_secs, hands_text, percent, AppModel, LogLevel, Phase, Recognized};
use std::sync::Arc;
use std::time::Duration;

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

        let mut app = Self {
            model: AppModel::new(),
            engine: None,
            inbox: None,
            image: None,
            focus_handle,
            frames_shown: 0,
        };

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
        let result: Result<EngineHandle, String> = engine::locate().and_then(|loc: EngineLocation| {
            let mut h = EngineHandle::spawn(&loc)?;
            // 区切り方は、開始前に選んでおいた値(model.segmenter)をそのまま送る
            h.send(&self.model.start_command(source))?;
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

    fn toggle(&mut self, cx: &mut Context<Self>) {
        if self.model.phase == Phase::Running {
            if let Some(e) = self.engine.as_mut() {
                if let Err(err) = e.send(&Command::Toggle) {
                    self.model.push_log(LogLevel::Warn, err);
                }
            }
        }
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
        let Some(inbox) = self.inbox.clone() else { return };
        let d = inbox.drain();
        let mut changed = false;

        if let Some(ev) = d.status {
            self.model.apply(&ev);
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
                    // ===== 右: 認識結果・履歴・語彙 =====
                    .child(
                        v_flex()
                            .w(px(420.))
                            .h_full()
                            .min_h_0()
                            .flex_shrink_0()
                            .gap_3()
                            .when(dbg, |d| d.on_children_prepainted(|b, _, _| log_bounds("right(result,history,log,vocab)", &b)))
                            .child(self.render_result(cx))
                            .child(self.render_history(cx))
                            .child(self.render_log(cx))
                            .child(self.render_vocab(border, muted, cx)),
                    ),
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
        frame
    }

    fn render_status(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let (state_text, state_color): (SharedString, gpui::Hsla) = match (self.model.phase, &self.model.status) {
            (Phase::Starting, _) => ("起動中".into(), rgb(0x90a4ae).into()),
            (Phase::Stopping, _) => ("停止中".into(), rgb(0x90a4ae).into()),
            (Phase::Idle, _) => ("停止".into(), rgb(0x90a4ae).into()),
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

        let mut start = Button::new("start")
            .primary()
            .large()
            .label("カメラで開始")
            .disabled(busy)
            .on_click(cx.listener(|this, _, _w, cx| this.start_camera(cx)));
        if busy {
            start = start.outline();
        }
        v_flex()
            .gap_2()
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
                    .child(
                        Button::new("toggle")
                            .large()
                            .outline()
                            .label(toggle_label)
                            .disabled(!running)
                            .on_click(cx.listener(|this, _, _w, cx| this.toggle(cx))),
                    ),
            )
            .child(
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
                    ),
            )
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
