//! クイズモードの操作と画面(右カラム)。進行・判定は `quiz.rs`(UI 非依存)、ここは GPUI の描画と操作の橋渡しだけ。
//! 判定は文字と形(○ △ ×)でも示し、色だけ・音だけに頼らない。

use crate::quiz::{
    Judgement, QuestionRecord, Stage, ANSWER_TIMEOUT_SECS, AUTO_ADVANCE_SECS, POINTS_CLOSE, POINTS_CORRECT,
};
use crate::state::LogLevel;
use crate::{top_bar_rows, LiveApp};
use gpui::{div, px, rgb, AnyElement, Context, FontWeight, IntoElement, ParentElement, SharedString, Styled};
use gpui_component::{
    button::{Button, ButtonVariants},
    h_flex,
    progress::Progress,
    v_flex, ActiveTheme, Disableable, Sizable,
};

/// 判定の色(背景, 文字)。色は補助で、文字と形が主
fn judgement_colors(j: Judgement) -> (gpui::Hsla, gpui::Hsla) {
    match j {
        Judgement::Correct => (rgb(0x2e7d32).into(), rgb(0xffffff).into()),
        Judgement::Close => (rgb(0xffb300).into(), rgb(0x101010).into()),
        Judgement::Wrong => (rgb(0xc62828).into(), rgb(0xffffff).into()),
        Judgement::TimedOut | Judgement::Passed => (rgb(0x546e7a).into(), rgb(0xffffff).into()),
    }
}

/// 総括の表・バナーに出す判定の文字(人間判定は印を付ける)
fn record_judgement_text(r: &QuestionRecord) -> String {
    let j = r.effective();
    if r.human_override {
        format!("{} {}※人", j.symbol(), j.text())
    } else {
        format!("{} {}", j.symbol(), j.text())
    }
}

impl LiveApp {
    /// アプリ起動からの秒数(クイズの時計)
    pub(crate) fn quiz_now(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// ゲーム開始 / もう一度。クイズが動いていて、開始前か総括のときだけ
    pub(crate) fn quiz_start(&mut self, cx: &mut Context<Self>) {
        if !self.model.quiz_running() || !matches!(self.model.quiz.stage, Stage::Ready | Stage::Summary) {
            return;
        }
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(1);
        let now = self.quiz_now();
        if self.model.quiz.start(&self.model.vocab, seed, now).is_err() {
            self.model.push_log(LogLevel::Warn, "出題できる語がありません(語彙が空、または除外語だけです)");
        }
        cx.notify();
    }

    pub(crate) fn quiz_pass(&mut self, cx: &mut Context<Self>) {
        if self.model.quiz_running() {
            self.model.quiz.pass();
        }
        cx.notify();
    }

    pub(crate) fn quiz_correct(&mut self, cx: &mut Context<Self>) {
        if self.model.quiz_running() {
            self.model.quiz.mark_correct();
        }
        cx.notify();
    }

    pub(crate) fn quiz_next(&mut self, cx: &mut Context<Self>) {
        if self.model.quiz_running() {
            self.model.quiz.next();
        }
        cx.notify();
    }

    /// スペースキー: 開始前・総括では開始、判定表示中は次へ。出題中は何もしない(うっかりパスにならないように)
    pub(crate) fn quiz_space(&mut self, cx: &mut Context<Self>) {
        match self.model.quiz.stage {
            Stage::Ready | Stage::Summary => self.quiz_start(cx),
            Stage::Judged { .. } => self.quiz_next(cx),
            Stage::Asking { .. } => {}
        }
    }

    /// 診断用(ATTENTION_APP_DEBUG): クイズの状態を、変わったときだけ標準エラーへ出す。映像・特徴量は出さない
    pub(crate) fn log_quiz(&mut self) {
        if !crate::debug_enabled() || self.model.mode != crate::state::AppMode::Quiz {
            return;
        }
        let q = &self.model.quiz;
        let last = q
            .records()
            .last()
            .map(|r| format!("{} -> {}({}点)", r.target.tag, record_judgement_text(r), r.points()))
            .unwrap_or_else(|| "-".into());
        let stage = match q.stage {
            Stage::Ready => "ready",
            Stage::Asking { .. } => "asking",
            Stage::Judged { .. } => "judged",
            Stage::Summary => "summary",
        };
        let text = format!(
            "stage={stage} n={}/{} target={} score={}/{} last=[{last}]",
            q.number(),
            q.total(),
            q.current_target().map(|t| t.tag.as_str()).unwrap_or("-"),
            q.score(),
            q.max_score(),
        );
        if text != self.last_quiz_log {
            eprintln!("[quiz] {text}");
            self.last_quiz_log = text;
        }
    }

    pub(crate) fn render_quiz_panel(&self, cx: &mut Context<Self>) -> AnyElement {
        let q = &self.model.quiz;
        let muted = cx.theme().muted_foreground;
        let card = || {
            v_flex()
                .p_4()
                .gap_2()
                .rounded_lg()
                .border_1()
                .border_color(cx.theme().border)
                .bg(cx.theme().secondary)
        };
        let caption = |t: &'static str| div().text_size(px(16.)).text_color(muted).child(t);

        // 進捗と合計点(開始前は満点の案内)
        let header = h_flex()
            .justify_between()
            .items_baseline()
            .child(
                div()
                    .text_size(px(26.))
                    .font_weight(FontWeight::BOLD)
                    .child(SharedString::from(match q.stage {
                        Stage::Ready => "クイズ".to_string(),
                        Stage::Summary => "結果".to_string(),
                        _ => format!("{} / {} 問目", q.number(), q.total()),
                    })),
            )
            .child(div().text_size(px(22.)).font_weight(FontWeight::BOLD).child(SharedString::from(match q.stage {
                Stage::Ready => format!("{}問・満点 {} 点", q.config.questions_per_game, q.config.questions_per_game as u32 * POINTS_CORRECT),
                _ => format!("合計 {} / {} 点", q.score(), q.max_score()),
            })));

        let body: AnyElement = match q.stage {
            Stage::Ready => card()
                .child(div().text_size(px(20.)).child(if self.model.quiz_running() {
                    "出題された語のサインをしてください。"
                } else {
                    "「カメラで開始」を押してから、スタートします。"
                }))
                .child(div().text_size(px(16.)).text_color(muted).child(SharedString::from(format!(
                    "正解 {POINTS_CORRECT} 点 / おしい(上位3位以内) {POINTS_CLOSE} 点 / ちがう・パス・時間切れ 0 点。1問 {ANSWER_TIMEOUT_SECS:.0} 秒"
                ))))
                .child(
                    Button::new("quiz-start")
                        .primary()
                        .large()
                        .label("スタート(スペース)")
                        .disabled(!self.model.quiz_running())
                        .on_click(cx.listener(|this, _, _w, cx| this.quiz_start(cx))),
                )
                .into_any_element(),
            Stage::Asking { .. } => {
                let left = q.time_left().unwrap_or(0.0);
                let bar_color: gpui::Hsla = if left <= 10.0 { rgb(0xffb300).into() } else { rgb(0x4fc3f7).into() };
                card()
                    .child(caption("お題(このサインをしてください)"))
                    .child(
                        div()
                            .text_size(px(60.))
                            .font_weight(FontWeight::BOLD)
                            .line_height(px(70.))
                            .min_h(px(70.))
                            .child(SharedString::from(q.current_target().map(|t| t.label.clone()).unwrap_or_default())),
                    )
                    .child(
                        h_flex()
                            .gap_2()
                            .items_center()
                            .child(div().flex_1().child(
                                Progress::new().value(left as f32 / ANSWER_TIMEOUT_SECS as f32 * 100.0).bg(bar_color).h(px(14.)),
                            ))
                            .child(
                                div()
                                    .w(px(88.))
                                    .flex_shrink_0()
                                    .text_size(px(18.))
                                    .child(SharedString::from(format!("残り {} 秒", left.ceil() as u32))),
                            ),
                    )
                    .child(div().text_size(px(16.)).text_color(muted).child("手を出して、サインが終わったら手を下げてください"))
                    .child(
                        Button::new("quiz-pass")
                            .outline()
                            .label("パス")
                            .on_click(cx.listener(|this, _, _w, cx| this.quiz_pass(cx))),
                    )
                    .into_any_element()
            }
            Stage::Judged { .. } => {
                let Some(r) = q.current_record() else { return div().into_any_element() };
                let (bg, fg) = judgement_colors(r.effective());
                let can_override = matches!(r.judgement, Judgement::Close | Judgement::Wrong) && !r.human_override;
                let last = q.number() >= q.total();
                let banner = h_flex()
                    .p_4()
                    .gap_4()
                    .items_center()
                    .rounded_lg()
                    .bg(bg)
                    .text_color(fg)
                    .child(div().text_size(px(72.)).font_weight(FontWeight::BOLD).line_height(px(80.)).child(r.effective().symbol()))
                    .child(
                        v_flex()
                            .child(div().text_size(px(44.)).font_weight(FontWeight::BOLD).child(r.effective().text()))
                            .child(div().text_size(px(18.)).child(SharedString::from(if r.human_override {
                                format!("人の判定で正解にしました(+{POINTS_CORRECT} 点)")
                            } else {
                                format!("+{} 点", r.points())
                            }))),
                    );
                card()
                    .child(banner)
                    .child(div().text_size(px(20.)).child(SharedString::from(format!("お題: {}", r.target.label))))
                    .child(div().text_size(px(20.)).child(SharedString::from(if r.top.is_empty() {
                        "AIの答え: なし".to_string()
                    } else {
                        format!("AIの答え: {}", r.ai_label())
                    })))
                    .children(top_bar_rows(&r.top))
                    .child(
                        h_flex()
                            .gap_3()
                            .items_center()
                            .child(
                                Button::new("quiz-next")
                                    .primary()
                                    .large()
                                    .label(if last { "結果を見る(スペース)" } else { "次へ(スペース)" })
                                    .on_click(cx.listener(|this, _, _w, cx| this.quiz_next(cx))),
                            )
                            .children(can_override.then(|| {
                                Button::new("quiz-correct")
                                    .outline()
                                    .large()
                                    .label("正解にする")
                                    .on_click(cx.listener(|this, _, _w, cx| this.quiz_correct(cx)))
                            })),
                    )
                    .child(div().text_size(px(14.)).text_color(muted).child(SharedString::from(format!(
                        "あと {} 秒で自動で次へ進みます",
                        q.advance_left().unwrap_or(AUTO_ADVANCE_SECS).ceil() as u32
                    ))))
                    .into_any_element()
            }
            Stage::Summary => {
                let cell = |w: Option<f32>, t: String, bold: bool| {
                    let d = div().text_size(px(15.)).truncate().child(SharedString::from(t));
                    let d = if bold { d.font_weight(FontWeight::BOLD) } else { d };
                    match w {
                        Some(w) => d.w(px(w)).flex_shrink_0(),
                        None => d.flex_1().min_w_0(),
                    }
                };
                let any_human = q.records().iter().any(|r| r.human_override);
                card()
                    .child(
                        div()
                            .text_size(px(56.))
                            .font_weight(FontWeight::BOLD)
                            .child(SharedString::from(format!("{} / {} 点", q.score(), q.max_score()))),
                    )
                    .child(
                        h_flex()
                            .gap_2()
                            .child(cell(Some(20.), "#".into(), true))
                            .child(cell(None, "お題".into(), true))
                            .child(cell(None, "AIの答え".into(), true))
                            .child(cell(Some(104.), "判定".into(), true))
                            .child(cell(Some(28.), "点".into(), true)),
                    )
                    .children(q.records().iter().enumerate().map(|(i, r)| {
                        let (bg, _) = judgement_colors(r.effective());
                        h_flex()
                            .gap_2()
                            .items_center()
                            .child(cell(Some(20.), format!("{}", i + 1), false))
                            .child(cell(None, r.target.label.clone(), false))
                            .child(cell(None, r.ai_label().to_string(), false))
                            .child(
                                div()
                                    .w(px(104.))
                                    .flex_shrink_0()
                                    .px_1()
                                    .rounded_sm()
                                    .border_l_4()
                                    .border_color(bg)
                                    .child(cell(None, record_judgement_text(r), true)),
                            )
                            .child(cell(Some(28.), format!("{}", r.points()), false))
                    }))
                    .children(any_human.then(|| div().text_size(px(14.)).text_color(muted).child("※人: 人間が「正解にする」で判定した問")))
                    .child(
                        Button::new("quiz-again")
                            .primary()
                            .large()
                            .label("もう一度(スペース)")
                            .on_click(cx.listener(|this, _, _w, cx| this.quiz_start(cx))),
                    )
                    .into_any_element()
            }
        };
        v_flex().gap_3().child(header).child(body).into_any_element()
    }
}
