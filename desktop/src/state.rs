//! 画面に出す状態(UI 非依存)。エンジンからのイベントを `apply` で畳み込む。
//! GPUI に触れないので、ユニットテストで挙動を確かめられる。

use crate::protocol::{Command, EngineEvent, RankedItem, SegmenterMode, SourceSpec, Verdict, VocabEntry};
use crate::protocol::CaptureWord;
use crate::quiz::{Quiz, QuizConfig};
use std::path::Path;
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
    /// 撮影(R1)。撮影ガイドの表示に加えて、エンジンが録画・判定・登録をする。
    /// エンジンを撮影モード(`mode: "capture"`)で起動する = **このモードだけが、書き込みうるエンジンを起動する**
    Capture,
    /// クイズ。エンジンはライブ認識と同じ設定(自動区切り)で起動し、`result` を画面側のクイズの判定に流す
    Quiz,
}

impl AppMode {
    /// 撮影ガイドの表示(立ち位置の枠・手の連続秒数・縁の警告・お題)を使うモードか
    pub fn shows_guide(self) -> bool {
        matches!(self, AppMode::CaptureGuide | AppMode::Capture)
    }
}

/// 撮影の段階(エンジンの `capture_state.phase`)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CapturePhase {
    #[default]
    Idle,
    Countdown,
    Recording,
    Judging,
    Judged,
}

impl CapturePhase {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "idle" => Some(Self::Idle),
            "countdown" => Some(Self::Countdown),
            "recording" => Some(Self::Recording),
            "judging" => Some(Self::Judging),
            "judged" => Some(Self::Judged),
            _ => None,
        }
    }
}

/// 直近に登録したテイク(画面の「登録しました」表示)
#[derive(Debug, Clone, PartialEq)]
pub struct SavedView {
    pub file_path: String,
    pub label: String,
    pub rep_idx: u32,
    pub quality_flag: String,
}

/// 撮影モードの画面の状態(エンジンの `capture_state` / `capture_saved` を畳み込んだもの)
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CaptureView {
    pub phase: CapturePhase,
    pub countdown_left: f64,
    pub rec_secs: f64,
    pub max_secs: f64,
    pub fps: Option<f64>,
    pub dropped: u64,
    pub word: Option<CaptureWord>,
    /// エンジンが実際に書き込む先(まだ `capture_state` が来ていなければ空)
    pub data_dir: String,
    pub verdict: Option<Verdict>,
    pub notes: Vec<String>,
    pub last_saved: Option<SavedView>,
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
    /// 撮影モード(`AppMode::Capture`)の状態。ほかのモードでは使わない
    pub capture: CaptureView,
    /// クイズモード(`AppMode::Quiz`)の進行。ほかのモードでは使わない。メモリのみ(保存しない)
    pub quiz: Quiz,
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
            capture: CaptureView::default(),
            quiz: Quiz::new(QuizConfig::default()),
        }
    }

    /// 新しいセッションを始めるとき(認識結果・状態を空にする。語彙とログは次の ready/操作で更新)
    pub fn begin_session(&mut self) {
        self.phase = Phase::Starting;
        self.status = None;
        self.current = None;
        self.history.clear();
        self.error = None;
        self.capture = CaptureView::default();
        self.quiz.reset();
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
        // 録画の途中状態は引きずらない(直近の登録結果だけ残す)
        let last_saved = self.capture.last_saved.take();
        self.capture = CaptureView { last_saved, ..CaptureView::default() };
        // ゲームの途中でエンジンが止まったら中断する(結果が来ないまま時間切れを積まないように)
        self.quiz.reset();
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
            // クイズでは結果をクイズの判定に流す(お題を待っているときだけ受け付ける。判定は Quiz が決める)
            EngineEvent::Result { .. } if self.mode == AppMode::Quiz => {
                self.quiz.on_event(ev);
            }
            // 撮影ガイドでは認識結果を出さない(開始操作を送らないので、カメラでは来ない。
            // 動画入力 + 手動では、エンジンが再生開始と同時に記録し終端で認識するので来ることがある)
            EngineEvent::Result { .. } if self.mode != AppMode::Live => {}
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
            EngineEvent::CaptureState {
                phase,
                countdown_left,
                rec_secs,
                max_secs,
                fps,
                dropped,
                word,
                data_dir,
                verdict,
                notes,
            } => {
                // 知らない段階名は無視する(エンジンの方が新しいとき)
                if let Some(p) = CapturePhase::parse(phase) {
                    let last_saved = self.capture.last_saved.take();
                    self.capture = CaptureView {
                        phase: p,
                        countdown_left: *countdown_left,
                        rec_secs: *rec_secs,
                        max_secs: *max_secs,
                        fps: *fps,
                        dropped: *dropped,
                        word: word.clone(),
                        data_dir: data_dir.clone(),
                        verdict: verdict.clone(),
                        notes: notes.clone(),
                        last_saved,
                    };
                }
            }
            EngineEvent::CaptureSaved { file_path, label, rep_idx, quality_flag, .. } => {
                self.capture.last_saved = Some(SavedView {
                    file_path: file_path.clone(),
                    label: label.clone(),
                    rep_idx: *rep_idx,
                    quality_flag: quality_flag.clone(),
                });
                let flag_text = quality_flag_text(quality_flag);
                self.push_log(LogLevel::Info, format!("登録しました: {file_path}({flag_text})"));
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

    /// クイズが動いている(クイズモードで、エンジンが認識中)か。クイズの操作を受け付けてよい段階
    pub fn quiz_running(&self) -> bool {
        self.mode == AppMode::Quiz && self.phase == Phase::Running
    }

    /// 手動の開始/停止(toggle)を送ってよいか。撮影ガイドでは送らない(語の認識をしない)
    pub fn toggle_allowed(&self) -> bool {
        self.mode == AppMode::Live && self.phase == Phase::Running
    }

    /// `start` コマンドを作る。ライブ認識では、区切り方は開始前に選んでおいた値(`self.segmenter`)をそのまま送る
    /// (「カメラで開始」が選択を自動に戻してしまわないように、組み立てをここに集めてある)。
    /// 撮影ガイドでは常に「手動」(開始操作を送らないので記録も認識も始まらない)。`self.segmenter` は変えない。
    /// 撮影(録画)では、撮影モードの start に登録先 `data_dir` を付ける(エンジンが書き込むのはここだけ)
    pub fn start_command(&self, source: SourceSpec, data_dir: &Path) -> Command {
        let segmenter = match self.mode {
            AppMode::Live => self.segmenter,
            AppMode::CaptureGuide => SegmenterMode::Manual,
            // クイズはライブ認識と同じ(自動区切り)。画面の区切り方の選択には左右されない
            AppMode::Quiz => SegmenterMode::Auto,
            AppMode::Capture => {
                return Command::StartCapture {
                    source,
                    data_dir: data_dir.to_string_lossy().to_string(),
                    countdown_secs: None,
                }
            }
        };
        Command::Start { source, segmenter }
    }

    // ----- 撮影(録画)の操作。送ってよいときだけ Some(コマンド)を返す(送る側は判断しない)-----

    fn capture_running(&self) -> bool {
        self.mode == AppMode::Capture && self.phase == Phase::Running
    }

    /// 撮影が動いていて、判定済み(採用か破棄を選ぶ段階)か。採用の確認を閉じるかの判断に使う
    pub fn capture_running_judged(&self) -> bool {
        self.capture_running() && self.capture.phase == CapturePhase::Judged
    }

    /// 録画開始。待機中で、画面にお題の語が出ているときだけ。語は画面が出しているもの(エンジンはそれを録画する)
    pub fn record_command(&self, displayed_word_id: Option<&str>) -> Option<Command> {
        (self.capture_running() && self.capture.phase == CapturePhase::Idle)
            .then(|| displayed_word_id.map(|w| Command::CaptureRecord { word_id: Some(w.to_string()) }))
            .flatten()
    }

    /// 停止(録画中)/ 取り消し(カウントダウン中)
    pub fn stop_command(&self) -> Option<Command> {
        (self.capture_running() && matches!(self.capture.phase, CapturePhase::Countdown | CapturePhase::Recording))
            .then_some(Command::CaptureStop)
    }

    /// 採用。判定済みで、**使えると判定されたテイクだけ**(使えない判定・判定できなかったテイクには None。
    /// 破棄か撮り直しだけ。ユーザー決定 2026-10-12)。エンジンも同じ理由で拒否する
    pub fn accept_command(&self) -> Option<Command> {
        if !(self.capture_running() && self.capture.phase == CapturePhase::Judged) {
            return None;
        }
        self.capture.verdict.as_ref().is_some_and(|v| v.usable).then_some(Command::CaptureAccept)
    }

    /// 破棄。カウントダウン・録画中・判定済みのとき(判定中は、判定が終わるまで破棄できない)
    pub fn discard_command(&self) -> Option<Command> {
        (self.capture_running()
            && matches!(
                self.capture.phase,
                CapturePhase::Countdown | CapturePhase::Recording | CapturePhase::Judged
            ))
        .then_some(Command::CaptureDiscard)
    }

    /// 撮り直し = 破棄して、同じお題でもう一度録画。判定済みのときだけ
    pub fn retake_commands(&self, displayed_word_id: Option<&str>) -> Vec<Command> {
        let (Some(word), true) = (displayed_word_id, self.capture_running() && self.capture.phase == CapturePhase::Judged)
        else {
            return Vec::new();
        };
        vec![Command::CaptureDiscard, Command::CaptureRecord { word_id: Some(word.to_string()) }]
    }

    /// スペースキー: 待機中は録画開始、カウントダウン/録画中は停止(取り消し)。判定中・判定済みでは何もしない
    /// (判定済みで押して、うっかり採用や撮り直しにならないように)
    pub fn space_command(&self, displayed_word_id: Option<&str>) -> Option<Command> {
        match self.capture.phase {
            CapturePhase::Idle => self.record_command(displayed_word_id),
            CapturePhase::Countdown | CapturePhase::Recording => self.stop_command(),
            CapturePhase::Judging | CapturePhase::Judged => None,
        }
    }
}

/// index.tsv の quality_flag の説明(画面用)。撮影モードが書くのは "ok" だけ(使えないテイクは登録しない)。
/// ng_* の説明は、撮影セッション(ingest)が書く値や、判定結果の種別を表示するための汎用の対応表
pub fn quality_flag_text(flag: &str) -> &'static str {
    match flag {
        "ok" => "有効なテイク",
        "ng_hands" => "NG: 手の検出が足りない",
        "ng_duration" => "NG: 長さが範囲外",
        "" => "判定なし",
        _ => "NG",
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
            m.start_command(camera(), Path::new("/d")),
            Command::Start { source: camera(), segmenter: SegmenterMode::Auto }
        );
        // 待機中に手動を選ぶ → begin_session を挟んでも保たれ、start の行に "manual" が出る
        m.segmenter = SegmenterMode::Manual;
        m.begin_session();
        let line = m.start_command(camera(), Path::new("/d")).to_line();
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["segmenter"], "manual", "{line}");
        assert_eq!(v["source"]["kind"], "camera");
        // 動画ファイルでも同じ
        let v: serde_json::Value = serde_json::from_str(
            &m.start_command(SourceSpec::File { path: "/a.mp4".into() }, Path::new("/d")).to_line(),
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
            m.start_command(camera(), Path::new("/d")),
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

    // ===== 撮影(録画あり)=====

    fn cap_state(phase: &str, usable: Option<bool>) -> EngineEvent {
        parse_event(
            &serde_json::json!({
                "type": "capture_state", "phase": phase, "countdown_left": 2.0, "rec_secs": 1.5, "max_secs": 10.0,
                "fps": 30.0, "dropped": 0,
                "word": {"word_id": "011", "romaji": "kare", "label": "かれ", "rep_idx": 4},
                "data_dir": "/data/raw_jsl",
                "verdict": usable.map(|u| serde_json::json!({
                    "usable": u, "reasons": [], "summary": "判定", "longest_run_secs": 1.0, "left_run_secs": 0.0,
                    "right_run_secs": 1.0, "duration_secs": 4.0, "quality_flag": if u { "ok" } else { "ng_hands" }
                })),
                "notes": ["注意"],
            })
            .to_string(),
        )
        .unwrap()
        .unwrap()
    }

    fn running_capture_model() -> AppModel {
        let mut m = AppModel::new();
        m.mode = AppMode::Capture;
        m.begin_session();
        m.apply(&parse_event(r#"{"type":"ready","vocab":[],"segmenter":"manual"}"#).unwrap().unwrap());
        assert_eq!(m.phase, Phase::Running);
        m
    }

    // 撮影(録画): start は登録先つきの撮影モード。ライブ認識の選択は変えない。認識結果は出さない
    #[test]
    fn capture_mode_starts_in_capture_mode_with_the_data_dir() {
        let camera = || SourceSpec::Camera { device: "0".into() };
        let mut m = AppModel::new();
        m.mode = AppMode::Capture;
        assert!(m.mode.shows_guide());
        assert!(AppMode::CaptureGuide.shows_guide() && !AppMode::Live.shows_guide());
        assert_eq!(
            m.start_command(camera(), Path::new("/a b/raw_jsl")),
            Command::StartCapture { source: camera(), data_dir: "/a b/raw_jsl".into(), countdown_secs: None }
        );
        assert_eq!(m.segmenter, SegmenterMode::Auto, "ライブ認識の選択は変えない");
        // ライブ認識・撮影ガイドの start には data_dir が付かない(書き込みうるエンジンを起動するのは撮影だけ)
        for mode in [AppMode::Live, AppMode::CaptureGuide] {
            m.mode = mode;
            assert!(matches!(m.start_command(camera(), Path::new("/d")), Command::Start { .. }), "{mode:?}");
        }
        m.mode = AppMode::Capture;
        m.begin_session();
        m.apply(&parse_event(r#"{"type":"ready","vocab":[],"segmenter":"manual"}"#).unwrap().unwrap());
        assert!(!m.toggle_allowed(), "撮影では toggle を送らない");
        assert!(!m.segmenter_selectable());
        m.apply(&result_ev("こんにちは", 0.99, false));
        assert!(m.current.is_none() && m.history.is_empty(), "撮影では認識結果を出さない");
    }

    #[test]
    fn capture_state_is_folded_and_phase_gates_the_commands() {
        let mut m = running_capture_model();
        let word = Some("011");
        // 待機: 録画開始だけ。お題が出ていなければ開始できない
        m.apply(&cap_state("idle", None));
        assert_eq!(m.capture.phase, CapturePhase::Idle);
        assert_eq!(m.record_command(word), Some(Command::CaptureRecord { word_id: Some("011".into()) }));
        assert_eq!(m.record_command(None), None, "お題の語が読めていなければ録画しない");
        assert_eq!(m.stop_command(), None);
        assert_eq!(m.accept_command(), None);
        assert_eq!(m.discard_command(), None);
        assert!(m.retake_commands(word).is_empty());
        assert_eq!(m.space_command(word), m.record_command(word));
        // カウントダウン: 取り消し(停止・破棄)だけ
        m.apply(&cap_state("countdown", None));
        assert_eq!(m.capture.countdown_left, 2.0);
        assert_eq!(m.record_command(word), None, "二重の開始はしない");
        assert_eq!(m.stop_command(), Some(Command::CaptureStop));
        assert_eq!(m.discard_command(), Some(Command::CaptureDiscard));
        assert_eq!(m.space_command(word), Some(Command::CaptureStop));
        // 録画中
        m.apply(&cap_state("recording", None));
        assert_eq!(m.capture.rec_secs, 1.5);
        assert_eq!(m.capture.word.as_ref().unwrap().label, "かれ");
        assert_eq!(m.capture.notes, vec!["注意"]);
        assert_eq!(m.stop_command(), Some(Command::CaptureStop));
        assert_eq!(m.accept_command(), None, "録画中は採用できない");
        assert_eq!(m.space_command(word), Some(Command::CaptureStop));
        // 判定中: 何もできない(スペースも効かない)
        m.apply(&cap_state("judging", None));
        assert_eq!(m.stop_command(), None);
        assert_eq!(m.discard_command(), None);
        assert_eq!(m.accept_command(), None);
        assert_eq!(m.space_command(word), None);
        // 判定済み(使える): 採用 / 撮り直し / 破棄。スペースは何もしない
        m.apply(&cap_state("judged", Some(true)));
        assert_eq!(m.accept_command(), Some(Command::CaptureAccept));
        assert_eq!(m.discard_command(), Some(Command::CaptureDiscard));
        assert_eq!(
            m.retake_commands(word),
            vec![Command::CaptureDiscard, Command::CaptureRecord { word_id: Some("011".into()) }]
        );
        assert!(m.retake_commands(None).is_empty());
        assert_eq!(m.space_command(word), None, "判定済みでスペースを押しても、採用も撮り直しもしない");
        // 判定済み(使えない): 採用は作らない(NG として記録する経路は無い)。破棄・撮り直しは作れる
        m.apply(&cap_state("judged", Some(false)));
        assert_eq!(m.accept_command(), None);
        assert_eq!(m.discard_command(), Some(Command::CaptureDiscard));
        assert_eq!(m.retake_commands(word).len(), 2);
        // 判定済みなのに判定の内容が無い(来るはずのない形)ときも、採用は作らない
        m.apply(&cap_state("judged", None));
        assert_eq!(m.capture.phase, CapturePhase::Judged);
        assert_eq!(m.accept_command(), None);
        // 知らない段階名は無視(直前の状態のまま)
        m.apply(&cap_state("future_phase", None));
        assert_eq!(m.capture.phase, CapturePhase::Judged);
    }

    #[test]
    fn capture_commands_are_unavailable_outside_a_running_capture_session() {
        let mut m = AppModel::new();
        m.capture.phase = CapturePhase::Judged;
        // ライブ認識・撮影ガイドでは撮影のコマンドは作らない
        for mode in [AppMode::Live, AppMode::CaptureGuide] {
            m.mode = mode;
            m.phase = Phase::Running;
            assert_eq!(m.accept_command(), None, "{mode:?}");
            assert_eq!(m.record_command(Some("011")), None);
            assert_eq!(m.space_command(Some("011")), None);
        }
        // 撮影でも、エンジンが動いていない・起動中・停止中は作らない
        m.mode = AppMode::Capture;
        for phase in [Phase::Idle, Phase::Starting, Phase::Stopping] {
            m.phase = phase;
            assert_eq!(m.accept_command(), None, "{phase:?}");
            assert_eq!(m.discard_command(), None);
        }
    }

    #[test]
    fn capture_saved_is_remembered_and_logged_and_exit_resets_the_rest() {
        let mut m = running_capture_model();
        m.apply(&cap_state("recording", None));
        m.apply(
            &parse_event(
                r#"{"type":"capture_saved","file_path":"011_kare/04.mp4","label":"かれ","rep_idx":4,"quality_flag":"ok","data_dir":"/d"}"#,
            )
            .unwrap()
            .unwrap(),
        );
        let saved = m.capture.last_saved.clone().unwrap();
        assert_eq!((saved.file_path.as_str(), saved.rep_idx), ("011_kare/04.mp4", 4));
        assert!(m.log[0].text.contains("011_kare/04.mp4") && m.log[0].text.contains("有効"), "{:?}", m.log[0]);
        // 次の capture_state が来ても、直近の登録結果は残る
        m.apply(&cap_state("idle", None));
        assert!(m.capture.last_saved.is_some());
        // エンジンが終わったら録画の途中状態は消え、登録結果だけ残る
        m.apply(&cap_state("recording", None));
        m.engine_exited();
        assert_eq!(m.capture.phase, CapturePhase::Idle);
        assert!(m.capture.word.is_none() && m.capture.notes.is_empty());
        assert!(m.capture.last_saved.is_some());
        // 次のセッション開始で全部消える
        m.begin_session();
        assert_eq!(m.capture, CaptureView::default());
    }

    #[test]
    fn quality_flag_texts() {
        assert_eq!(quality_flag_text("ok"), "有効なテイク");
        assert!(quality_flag_text("ng_hands").starts_with("NG"));
        assert!(quality_flag_text("ng_duration").starts_with("NG"));
        assert_eq!(quality_flag_text(""), "判定なし");
        assert_eq!(CapturePhase::parse("judging"), Some(CapturePhase::Judging));
        assert_eq!(CapturePhase::parse("x"), None);
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
    #[test]
    fn quiz_mode_routes_results_to_the_quiz_only() {
        use crate::quiz::Stage;
        let vocab: Vec<VocabEntry> =
            ["a", "b", "c"].iter().map(|t| VocabEntry { tag: t.to_string(), label: t.to_string() }).collect();
        let mut m = AppModel::new();
        m.mode = AppMode::Quiz;
        m.quiz.config.excluded_tags.clear();
        m.begin_session();
        m.apply(&EngineEvent::Ready { vocab: vocab.clone(), model_dir: String::new(), palm_roi: String::new(),
            frame_select: String::new(), frames: 10, segmenter: "manual".into(), preprocess: String::new() });
        assert!(m.quiz_running());
        // ready の segmenter(manual)は、ライブ以外では選択値を書き換えない
        assert_eq!(m.segmenter, SegmenterMode::Auto);
        m.quiz.start(&m.vocab, 1, 0.0).unwrap();
        let target = m.quiz.current_target().unwrap().tag.clone();
        let ev = EngineEvent::Result {
            tag: target.clone(), label: target.clone(), p: 0.9,
            ranked: vec![RankedItem { tag: target, label: "x".into(), p: 0.9 }],
            low_confidence: false, span: None, buffered: 0, duration_secs: 1.0,
            selected_with_hand: 5, selected_total: 10,
        };
        m.apply(&ev);
        assert_eq!(m.quiz.score(), crate::quiz::POINTS_CORRECT);
        // ライブの履歴・現在の結果には流れない
        assert!(m.current.is_none() && m.history.is_empty());
        // エンジン終了でゲームは中断される
        m.engine_exited();
        assert_eq!(m.quiz.stage, Stage::Ready);
        assert!(!m.quiz_running());
    }

    #[test]
    fn quiz_starts_the_engine_like_live_auto_and_results_are_ignored_in_capture_modes() {
        let mut m = AppModel::new();
        m.mode = AppMode::Quiz;
        m.segmenter = SegmenterMode::Manual; // ライブ側の選択が手動でも、クイズは自動で起動する
        let cmd = m.start_command(SourceSpec::Camera { device: "0".into() }, Path::new("/d"));
        assert_eq!(cmd, Command::Start { source: SourceSpec::Camera { device: "0".into() }, segmenter: SegmenterMode::Auto });
        // 撮影モードではクイズに流さない(従来どおり無視)
        for mode in [AppMode::CaptureGuide, AppMode::Capture] {
            let mut m = AppModel::new();
            m.mode = mode;
            m.quiz.config.excluded_tags.clear();
            m.quiz.start(&[VocabEntry { tag: "a".into(), label: "a".into() }], 1, 0.0).unwrap();
            m.apply(&result_ev("a", 0.9, false));
            assert!(m.quiz.records().is_empty(), "{mode:?}");
        }
        // モード切替は待機中だけ(既存の規則のまま)
        let mut m = AppModel::new();
        assert!(m.mode_selectable());
        m.phase = Phase::Running;
        assert!(!m.mode_selectable());
        assert!(!m.toggle_allowed() || m.mode == AppMode::Live);
    }
}
