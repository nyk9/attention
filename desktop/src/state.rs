//! 画面に出す状態(UI 非依存)。エンジンからのイベントを `apply` で畳み込む。
//! GPUI に触れないので、ユニットテストで挙動を確かめられる。

use crate::protocol::{Command, EngineEvent, RankedItem, SegmenterMode, SourceSpec, VocabEntry};
use std::collections::VecDeque;

/// 履歴に残す件数(直近 5 件)
pub const HISTORY_MAX: usize = 5;
/// ログ欄(message)に残す件数
pub const LOG_MAX: usize = 6;

/// エンジンとの接続の段階
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// エンジンは動いていない
    Idle,
    /// 起動して start を送った。ready(モデル読み込み完了)を待っている
    Starting,
    /// ready を受け取り、認識中
    Running,
    /// quit を送った。終了を待っている
    Stopping,
}

/// 画面のモード。待機中(エンジンが動いていない間)だけ切り替えられる
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    /// ライブ認識(従来の画面)
    Live,
    /// 撮影ガイド(R0)。区切り方「手動」で起動し、開始操作は送らない = 語の認識はしない。録画・保存もしない
    CaptureGuide,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Recognized {
    pub tag: String,
    pub label: String,
    pub p: f64,
    pub top: Vec<RankedItem>,
    pub low_confidence: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Warn,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogLine {
    pub level: LogLevel,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StatusView {
    pub recording: bool,
    /// "L" | "R" | "LR" | "none"
    pub hands: String,
    pub fps: f64,
    pub rec_secs: f64,
}

pub struct AppModel {
    pub phase: Phase,
    pub vocab: Vec<VocabEntry>,
    pub status: Option<StatusView>,
    /// 最新の認識結果(大きく表示するもの)
    pub current: Option<Recognized>,
    /// 直近の履歴(新しいものが先頭)
    pub history: VecDeque<Recognized>,
    pub log: VecDeque<LogLine>,
    /// 画面に出すエラー(エンジンの error や、起動失敗)
    pub error: Option<String>,
    /// 現在の区切り方(エンジンが反映したもの。start 前は選択中の値)。
    /// 撮影ガイドのセッション中は、エンジンからの値で上書きしない(ライブ認識に戻ったときに選択を保つ)
    pub segmenter: SegmenterMode,
    pub mode: AppMode,
}

impl AppModel {
    pub fn new() -> Self {
        Self {
            phase: Phase::Idle,
            vocab: Vec::new(),
            status: None,
            current: None,
            history: VecDeque::new(),
            log: VecDeque::new(),
            error: None,
            segmenter: SegmenterMode::Auto,
            mode: AppMode::Live,
        }
    }

    /// 新しいセッションを始めるとき(認識結果・状態を空にする。語彙とログは次の ready/操作で更新)
    pub fn begin_session(&mut self) {
        self.phase = Phase::Starting;
        self.status = None;
        self.current = None;
        self.history.clear();
        self.error = None;
    }

    pub fn push_log(&mut self, level: LogLevel, text: impl Into<String>) {
        self.log.push_front(LogLine { level, text: text.into() });
        while self.log.len() > LOG_MAX {
            self.log.pop_back();
        }
    }

    /// エンジンのプロセスが終わったとき(stopped を受け取った後、または異常終了)
    pub fn engine_exited(&mut self) {
        self.phase = Phase::Idle;
        self.status = None;
    }

    /// プレビューと status 以外のイベントの反映(status は `set_status`、preview は画面側で扱う)
    pub fn apply(&mut self, ev: &EngineEvent) {
        match ev {
            EngineEvent::Ready { vocab, segmenter, .. } => {
                self.vocab = vocab.clone();
                if self.mode == AppMode::Live {
                    if let Some(m) = SegmenterMode::parse(segmenter) {
                        self.segmenter = m;
                    }
                }
                // 停止要求の後に遅れて ready が来ても、Stopping を上書きしない
                if self.phase == Phase::Starting {
                    self.phase = Phase::Running;
                }
            }
            EngineEvent::Status { state, hands, fps, rec_secs } => {
                self.status = Some(StatusView {
                    recording: state == "recording",
                    hands: hands.clone(),
                    fps: *fps,
                    rec_secs: *rec_secs,
                });
            }
            EngineEvent::Preview { .. } => {}
            // 撮影ガイドでは認識結果を出さない(開始操作を送らないので、カメラでは来ない。
            // 動画入力 + 手動では、エンジンが再生開始と同時に記録し終端で認識するので来ることがある)
            EngineEvent::Result { .. } if self.mode == AppMode::CaptureGuide => {}
            EngineEvent::Result { tag, label, p, ranked, low_confidence, .. } => {
                let r = Recognized {
                    tag: tag.clone(),
                    label: label.clone(),
                    p: *p,
                    top: ranked.iter().take(3).cloned().collect(),
                    low_confidence: *low_confidence,
                };
                self.current = Some(r.clone());
                self.history.push_front(r);
                while self.history.len() > HISTORY_MAX {
                    self.history.pop_back();
                }
            }
            EngineEvent::Message { level, text } => {
                let lv = if level == "warn" { LogLevel::Warn } else { LogLevel::Info };
                self.push_log(lv, text.clone());
            }
            EngineEvent::Error { text } => {
                self.error = Some(text.clone());
                self.push_log(LogLevel::Warn, format!("エラー: {text}"));
            }
            EngineEvent::Stopped { .. } => {
                // プロセスの終了(EOF)を受けて engine_exited を呼ぶ。ここでは何も変えない
            }
            EngineEvent::Segmenter { mode } => {
                if self.mode == AppMode::Live {
                    if let Some(m) = SegmenterMode::parse(mode) {
                        self.segmenter = m;
                    }
                }
            }
            // 知らない種類のイベントは何もしない(読み捨て)
            EngineEvent::Unknown => {}
        }
    }

    /// 区切り方のボタンを押せる段階か。待機中(次の start で使う値を選ぶ)と認識中(エンジンへ切り替えを頼む)は
    /// 押せる。起動中・停止中は押せない: start はもう送った後(または終わる途中)なので、押しても表示だけが
    /// 変わって、エンジンの `ready` / `segmenter` で元に戻ってしまう
    pub fn segmenter_selectable(&self) -> bool {
        self.mode == AppMode::Live && matches!(self.phase, Phase::Idle | Phase::Running)
    }

    /// モードを切り替えられるか(待機中だけ。セッション中に変えるとエンジンの起動設定と食い違うため)
    pub fn mode_selectable(&self) -> bool {
        self.phase == Phase::Idle
    }

    /// 手動の開始/停止(toggle)を送ってよいか。撮影ガイドでは送らない(語の認識をしない)
    pub fn toggle_allowed(&self) -> bool {
        self.mode == AppMode::Live && self.phase == Phase::Running
    }

    /// `start` コマンドを作る。ライブ認識では、区切り方は開始前に選んでおいた値(`self.segmenter`)をそのまま送る
    /// (「カメラで開始」が選択を自動に戻してしまわないように、組み立てをここに集めてある)。
    /// 撮影ガイドでは常に「手動」(開始操作を送らないので記録も認識も始まらない)。`self.segmenter` は変えない
    pub fn start_command(&self, source: SourceSpec) -> Command {
        let segmenter = match self.mode {
            AppMode::Live => self.segmenter,
            AppMode::CaptureGuide => SegmenterMode::Manual,
        };
        Command::Start { source, segmenter }
    }
}

impl Default for AppModel {
    fn default() -> Self {
        Self::new()
    }
}

/// 確率 → 表示用の百分率(0〜100)
pub fn percent(p: f64) -> f32 {
    (p * 100.0).clamp(0.0, 100.0) as f32
}

/// 経過秒数の表示("3.2 秒")
pub fn format_secs(s: f64) -> String {
    format!("{:.1} 秒", if s.is_finite() { s.max(0.0) } else { 0.0 })
}

/// 手の見え方 → 日本語
pub fn hands_text(h: &str) -> &'static str {
    match h {
        "L" => "左手が見えています",
        "R" => "右手が見えています",
        "LR" => "両手が見えています",
        _ => "手が見えていません",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{parse_event, RankedItem};

    fn result_ev(label: &str, p: f64, low: bool) -> EngineEvent {
        EngineEvent::Result {
            tag: label.into(),
            label: label.into(),
            p,
            ranked: vec![
                RankedItem { tag: "a".into(), label: "あ".into(), p },
                RankedItem { tag: "b".into(), label: "い".into(), p: 0.2 },
                RankedItem { tag: "c".into(), label: "う".into(), p: 0.1 },
                RankedItem { tag: "d".into(), label: "え".into(), p: 0.05 },
            ],
            low_confidence: low,
            span: None,
            buffered: 0,
            duration_secs: 1.0,
            selected_with_hand: 5,
            selected_total: 10,
        }
    }

    #[test]
    fn history_is_capped_newest_first() {
        let mut m = AppModel::new();
        for i in 0..8 {
            m.apply(&result_ev(&format!("w{i}"), 0.9, false));
        }
        assert_eq!(m.history.len(), HISTORY_MAX);
        assert_eq!(m.history[0].label, "w7");
        assert_eq!(m.history[HISTORY_MAX - 1].label, "w3");
        assert_eq!(m.current.as_ref().unwrap().label, "w7");
    }

    #[test]
    fn result_keeps_top3_and_low_confidence() {
        let mut m = AppModel::new();
        m.apply(&result_ev("x", 0.4, true));
        let c = m.current.unwrap();
        assert_eq!(c.top.len(), 3);
        assert!(c.low_confidence);
    }

    #[test]
    fn log_is_capped() {
        let mut m = AppModel::new();
        for i in 0..20 {
            m.apply(&EngineEvent::Message { level: "info".into(), text: format!("m{i}") });
        }
        assert_eq!(m.log.len(), LOG_MAX);
        assert_eq!(m.log[0].text, "m19");
    }

    #[test]
    fn message_levels_and_error() {
        let mut m = AppModel::new();
        m.apply(&EngineEvent::Message { level: "warn".into(), text: "w".into() });
        m.apply(&EngineEvent::Message { level: "info".into(), text: "i".into() });
        assert_eq!(m.log[0].level, LogLevel::Info);
        assert_eq!(m.log[1].level, LogLevel::Warn);
        m.apply(&EngineEvent::Error { text: "壊れた".into() });
        assert_eq!(m.error.as_deref(), Some("壊れた"));
        // 次のセッション開始でエラーは消える
        m.begin_session();
        assert!(m.error.is_none());
    }

    #[test]
    fn phase_transitions() {
        let mut m = AppModel::new();
        assert_eq!(m.phase, Phase::Idle);
        m.begin_session();
        assert_eq!(m.phase, Phase::Starting);
        let ready = parse_event(
            r#"{"type":"ready","vocab":[{"tag":"a","label":"あ"}],"segmenter":"manual"}"#,
        )
        .unwrap()
        .unwrap();
        m.apply(&ready);
        assert_eq!(m.phase, Phase::Running);
        assert_eq!(m.vocab.len(), 1);
        assert_eq!(m.segmenter, SegmenterMode::Manual);
        m.engine_exited();
        assert_eq!(m.phase, Phase::Idle);
    }

    #[test]
    fn late_ready_does_not_override_stopping() {
        let mut m = AppModel::new();
        m.begin_session();
        m.phase = Phase::Stopping;
        m.apply(&parse_event(r#"{"type":"ready"}"#).unwrap().unwrap());
        assert_eq!(m.phase, Phase::Stopping);
    }

    #[test]
    fn status_and_segmenter_event() {
        let mut m = AppModel::new();
        m.apply(&parse_event(
            r#"{"type":"status","state":"recording","hands":"R","fps":14.9,"rec_secs":2.34}"#,
        ).unwrap().unwrap());
        let s = m.status.clone().unwrap();
        assert!(s.recording);
        assert_eq!(s.hands, "R");
        m.apply(&EngineEvent::Segmenter { mode: "manual".into() });
        assert_eq!(m.segmenter, SegmenterMode::Manual);
        m.engine_exited();
        assert!(m.status.is_none());
    }

    // 開始前に「手動」を選んでから start を作ると、segmenter が "manual" で送られる
    // (以前は「カメラで開始」が選択を自動に上書きしていた)
    #[test]
    fn start_command_uses_the_segmenter_chosen_before_start() {
        let camera = || SourceSpec::Camera { device: "0".into() };
        let mut m = AppModel::new();
        // 既定は自動
        assert_eq!(
            m.start_command(camera()),
            Command::Start { source: camera(), segmenter: SegmenterMode::Auto }
        );
        // 待機中に手動を選ぶ → begin_session を挟んでも保たれ、start の行に "manual" が出る
        m.segmenter = SegmenterMode::Manual;
        m.begin_session();
        let line = m.start_command(camera()).to_line();
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["segmenter"], "manual", "{line}");
        assert_eq!(v["source"]["kind"], "camera");
        // 動画ファイルでも同じ
        let v: serde_json::Value = serde_json::from_str(
            &m.start_command(SourceSpec::File { path: "/a.mp4".into() }).to_line(),
        )
        .unwrap();
        assert_eq!(v["segmenter"], "manual");
    }

    // 区切り方のボタンは、待機中と認識中だけ押せる(起動中・停止中は無効)
    #[test]
    fn segmenter_buttons_are_disabled_while_starting_or_stopping() {
        let mut m = AppModel::new();
        let want = [
            (Phase::Idle, true),
            (Phase::Starting, false),
            (Phase::Running, true),
            (Phase::Stopping, false),
        ];
        for (phase, ok) in want {
            m.phase = phase;
            assert_eq!(m.segmenter_selectable(), ok, "{phase:?}");
        }
    }

    // 撮影ガイド: start は常に手動、ライブ認識の選択(自動)は保たれる。toggle は送らない。認識結果は出さない
    #[test]
    fn capture_guide_mode_starts_manual_and_ignores_results() {
        let camera = || SourceSpec::Camera { device: "0".into() };
        let mut m = AppModel::new();
        assert_eq!(m.mode, AppMode::Live);
        assert!(m.mode_selectable());
        m.mode = AppMode::CaptureGuide;
        assert_eq!(
            m.start_command(camera()),
            Command::Start { source: camera(), segmenter: SegmenterMode::Manual }
        );
        assert_eq!(m.segmenter, SegmenterMode::Auto, "ライブ認識の選択は変えない");
        m.begin_session();
        assert!(!m.mode_selectable(), "セッション中はモードを変えられない");
        m.apply(&parse_event(r#"{"type":"ready","vocab":[],"segmenter":"manual"}"#).unwrap().unwrap());
        assert_eq!(m.phase, Phase::Running);
        assert_eq!(m.segmenter, SegmenterMode::Auto, "エンジンの manual で上書きしない");
        assert!(!m.toggle_allowed(), "撮影ガイドでは toggle を送らない");
        assert!(!m.segmenter_selectable());
        m.apply(&result_ev("こんにちは", 0.99, false));
        assert!(m.current.is_none() && m.history.is_empty(), "撮影ガイドでは認識結果を出さない");
        // ライブ認識に戻すと従来どおり
        m.engine_exited();
        m.mode = AppMode::Live;
        m.begin_session();
        m.apply(&parse_event(r#"{"type":"ready","vocab":[],"segmenter":"auto"}"#).unwrap().unwrap());
        assert!(m.toggle_allowed());
        m.apply(&result_ev("こんにちは", 0.99, false));
        assert_eq!(m.current.as_ref().unwrap().label, "こんにちは");
    }

    // 知らない種類のイベントは何も変えない
    #[test]
    fn unknown_event_changes_nothing() {
        let mut m = AppModel::new();
        m.begin_session();
        m.segmenter = SegmenterMode::Manual;
        let ev = parse_event(r#"{"type":"future_event","x":1}"#).unwrap().unwrap();
        m.apply(&ev);
        assert_eq!(m.phase, Phase::Starting);
        assert_eq!(m.segmenter, SegmenterMode::Manual);
        assert!(m.log.is_empty() && m.error.is_none() && m.current.is_none());
    }

    #[test]
    fn format_helpers() {
        assert_eq!(percent(0.987), 98.7);
        assert_eq!(percent(1.5), 100.0);
        assert_eq!(percent(-1.0), 0.0);
        assert_eq!(format_secs(3.24), "3.2 秒");
        assert_eq!(format_secs(f64::NAN), "0.0 秒");
        assert_eq!(hands_text("LR"), "両手が見えています");
        assert_eq!(hands_text("none"), "手が見えていません");
    }
}
