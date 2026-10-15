//! クイズモードの進行・判定・点数(UI 非依存)。GPUI に触れないので、ユニットテストで挙動を確かめられる。
//! 設計: docs/quiz-mode-design.md。
//!
//! 流れ: お題表示(Asking)→ 結果受信 → 判定表示(Judged)→ 次へ(スペース or 一定秒数で自動)→ … → 総括(Summary)。
//! 時刻は呼び出し側が `tick(now)` で渡す秒数(単調増加)を使う。`Instant` を直接読まないので、テストで時間を進められる。
//! 映像・特徴量は扱わない。点数と記録はメモリ上だけで、ファイルには書かない。

use crate::protocol::{EngineEvent, RankedItem, VocabEntry};

/// 1 ゲームの問題数
pub const QUESTIONS_PER_GAME: usize = 5;
/// 点数(ユーザー要望「点数があったほうがわかりやすい」への提案値。変えるときはここだけ)
pub const POINTS_CORRECT: u32 = 3;
pub const POINTS_CLOSE: u32 = 1;
/// 1 問あたりの待ち時間(秒)。超えると時間切れ
pub const ANSWER_TIMEOUT_SECS: f64 = 30.0;
/// 判定表示から次の問へ自動で進むまでの秒数
pub const AUTO_ADVANCE_SECS: f64 = 3.0;
/// 「おしい」とみなす順位(top-3 に入る)
pub const CLOSE_RANK: usize = 3;
/// 出題から除く語(2026-10-12 の rotate 評価で外れやすかった語。仮の初期値)
pub const DEFAULT_EXCLUDED_TAGS: [&str; 6] = ["atsui", "are-sore", "kore", "juu", "ichi", "anata"];

/// 満点(問題数 × 正解の点)
pub fn max_score(questions: usize) -> u32 {
    questions as u32 * POINTS_CORRECT
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuizConfig {
    /// 出題から除く語のタグ(設定として持つ。環境変数 ATTENTION_QUIZ_EXCLUDE で上書き: main.rs)
    pub excluded_tags: Vec<String>,
    pub questions_per_game: usize,
}

impl Default for QuizConfig {
    fn default() -> Self {
        Self {
            excluded_tags: DEFAULT_EXCLUDED_TAGS.iter().map(|s| s.to_string()).collect(),
            questions_per_game: QUESTIONS_PER_GAME,
        }
    }
}

/// カンマ区切りの語タグ → 除外リスト("a, b,,c" → [a, b, c])。空文字は「除外なし」
pub fn parse_excluded(s: &str) -> Vec<String> {
    s.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Judgement {
    Correct,
    Close,
    Wrong,
    TimedOut,
    Passed,
}

impl Judgement {
    pub fn points(self) -> u32 {
        match self {
            Judgement::Correct => POINTS_CORRECT,
            Judgement::Close => POINTS_CLOSE,
            _ => 0,
        }
    }
    /// 文字(色だけに頼らないため、形と文字を併記する)
    pub fn text(self) -> &'static str {
        match self {
            Judgement::Correct => "正解",
            Judgement::Close => "おしい",
            Judgement::Wrong => "ちがう",
            Judgement::TimedOut => "時間切れ",
            Judgement::Passed => "パス",
        }
    }
    pub fn symbol(self) -> &'static str {
        match self {
            Judgement::Correct => "○",
            Judgement::Close => "△",
            Judgement::Wrong | Judgement::TimedOut => "×",
            Judgement::Passed => "－",
        }
    }
}

/// 判定: top-1 がお題と一致 = 正解、top-3 に入る = おしい、それ以外 = ちがう。
/// `top1_tag` は結果イベントの tag。`ranked` が空でも top-1 の一致だけで判定できる
pub fn judge(target_tag: &str, top1_tag: &str, ranked: &[RankedItem]) -> Judgement {
    if top1_tag == target_tag {
        Judgement::Correct
    } else if ranked.iter().take(CLOSE_RANK).any(|r| r.tag == target_tag) {
        Judgement::Close
    } else {
        Judgement::Wrong
    }
}

/// 1 問の記録(総括の表になる)
#[derive(Debug, Clone, PartialEq)]
pub struct QuestionRecord {
    pub target: VocabEntry,
    /// AI の結果から決めた判定。人間判定で書き換えない(総括で元の判定も見えるように)
    pub judgement: Judgement,
    /// AI の上位 3 件(時間切れ・パスでは空)
    pub top: Vec<RankedItem>,
    /// 人間が「正解にする」を押した問
    pub human_override: bool,
}

impl QuestionRecord {
    /// 点数に使う判定(人間判定なら正解)
    pub fn effective(&self) -> Judgement {
        if self.human_override {
            Judgement::Correct
        } else {
            self.judgement
        }
    }
    pub fn points(&self) -> u32 {
        self.effective().points()
    }
    pub fn ai_label(&self) -> &str {
        self.top.first().map(|r| r.label.as_str()).unwrap_or("-")
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Stage {
    /// ゲーム開始前(エンジン起動直後・終了後)
    Ready,
    /// お題を出して結果を待っている。`since` はこの問を出した時刻
    Asking { since: f64 },
    /// 判定を表示している。`since` は判定した時刻(自動で進むまでの計測)
    Judged { since: f64 },
    Summary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartError {
    /// 語彙が空(エンジンの ready 前)、または除外後に出題できる語が無い
    NoCandidates,
}

#[derive(Debug, Clone)]
pub struct Quiz {
    pub config: QuizConfig,
    pub stage: Stage,
    questions: Vec<VocabEntry>,
    records: Vec<QuestionRecord>,
    /// `tick` で最後に受け取った時刻(結果受信時の「判定した時刻」に使う)
    clock: f64,
}

impl Quiz {
    pub fn new(config: QuizConfig) -> Self {
        Self { config, stage: Stage::Ready, questions: Vec::new(), records: Vec::new(), clock: 0.0 }
    }

    /// ゲームを中断して開始前に戻す(エンジン終了・セッション開始時)
    pub fn reset(&mut self) {
        self.stage = Stage::Ready;
        self.questions.clear();
        self.records.clear();
    }

    /// 出題候補: 語彙から除外語を引いたもの
    pub fn candidates(&self, vocab: &[VocabEntry]) -> Vec<VocabEntry> {
        vocab.iter().filter(|v| !self.config.excluded_tags.contains(&v.tag)).cloned().collect()
    }

    /// ゲームを始める。出題は候補からランダム(seed 固定で再現可能)で重複なし。
    /// 候補が `questions_per_game` に満たなければ、候補の数だけ出題する
    pub fn start(&mut self, vocab: &[VocabEntry], seed: u64, now: f64) -> Result<(), StartError> {
        let mut c = self.candidates(vocab);
        if c.is_empty() {
            return Err(StartError::NoCandidates);
        }
        shuffle(&mut c, seed);
        c.truncate(self.config.questions_per_game.max(1));
        self.questions = c;
        self.records.clear();
        self.clock = now;
        self.stage = Stage::Asking { since: now };
        Ok(())
    }

    /// 結果イベントを判定に流す。**お題を待っているときだけ**受け付ける(判定済み・総括・開始前に届いたものは無視)。
    /// 受け付けたら true
    pub fn on_event(&mut self, ev: &EngineEvent) -> bool {
        let EngineEvent::Result { tag, ranked, .. } = ev else { return false };
        let Stage::Asking { .. } = self.stage else { return false };
        let Some(target) = self.current_target().cloned() else { return false };
        let judgement = judge(&target.tag, tag, ranked);
        let top = ranked.iter().take(3).cloned().collect();
        self.finish_question(target, judgement, top);
        true
    }

    /// パス(お題を待っているときだけ)
    pub fn pass(&mut self) -> bool {
        self.finish_without_result(Judgement::Passed)
    }

    fn finish_without_result(&mut self, j: Judgement) -> bool {
        let Stage::Asking { .. } = self.stage else { return false };
        let Some(target) = self.current_target().cloned() else { return false };
        self.finish_question(target, j, Vec::new());
        true
    }

    fn finish_question(&mut self, target: VocabEntry, judgement: Judgement, top: Vec<RankedItem>) {
        self.records.push(QuestionRecord { target, judgement, top, human_override: false });
        self.stage = Stage::Judged { since: self.clock };
    }

    /// 人間判定「正解にする」。判定表示中で、AI の読みが外れた問(おしい・ちがう)だけ。
    /// 時間切れ・パスは AI の読みが無いので対象外
    pub fn mark_correct(&mut self) -> bool {
        let Stage::Judged { .. } = self.stage else { return false };
        match self.records.last_mut() {
            Some(r) if matches!(r.judgement, Judgement::Close | Judgement::Wrong) && !r.human_override => {
                r.human_override = true;
                true
            }
            _ => false,
        }
    }

    /// 次へ(判定表示中のみ)。最後の問なら総括へ
    pub fn next(&mut self) -> bool {
        let Stage::Judged { .. } = self.stage else { return false };
        self.stage = if self.records.len() >= self.questions.len() {
            Stage::Summary
        } else {
            Stage::Asking { since: self.clock }
        };
        true
    }

    /// 時刻を進める。時間切れと自動で次へ進む処理をここで行う。状態が変わったら true
    pub fn tick(&mut self, now: f64) -> bool {
        self.clock = now;
        match self.stage {
            Stage::Asking { since } if now - since >= ANSWER_TIMEOUT_SECS => {
                self.finish_without_result(Judgement::TimedOut)
            }
            Stage::Judged { since } if now - since >= AUTO_ADVANCE_SECS => self.next(),
            _ => false,
        }
    }

    // ----- 表示用の問い合わせ -----

    pub fn total(&self) -> usize {
        self.questions.len()
    }
    /// 「N / 総数」の N(出題中・判定中の問の番号。1 始まり)
    pub fn number(&self) -> usize {
        match self.stage {
            Stage::Asking { .. } => self.records.len() + 1,
            Stage::Judged { .. } => self.records.len(),
            Stage::Summary => self.total(),
            Stage::Ready => 0,
        }
    }
    /// いま出している(判定中なら出していた)お題
    pub fn current_target(&self) -> Option<&VocabEntry> {
        match self.stage {
            Stage::Asking { .. } => self.questions.get(self.records.len()),
            Stage::Judged { .. } => self.records.last().map(|r| &r.target),
            _ => None,
        }
    }
    /// 判定表示中の記録
    pub fn current_record(&self) -> Option<&QuestionRecord> {
        matches!(self.stage, Stage::Judged { .. }).then(|| self.records.last()).flatten()
    }
    pub fn records(&self) -> &[QuestionRecord] {
        &self.records
    }
    pub fn score(&self) -> u32 {
        self.records.iter().map(QuestionRecord::points).sum()
    }
    pub fn max_score(&self) -> u32 {
        max_score(self.total())
    }
    /// 残り時間(秒)。出題中のみ
    pub fn time_left(&self) -> Option<f64> {
        match self.stage {
            Stage::Asking { since } => Some((ANSWER_TIMEOUT_SECS - (self.clock - since)).clamp(0.0, ANSWER_TIMEOUT_SECS)),
            _ => None,
        }
    }
    /// 自動で次へ進むまでの残り秒数。判定表示中のみ
    pub fn advance_left(&self) -> Option<f64> {
        match self.stage {
            Stage::Judged { since } => Some((AUTO_ADVANCE_SECS - (self.clock - since)).clamp(0.0, AUTO_ADVANCE_SECS)),
            _ => None,
        }
    }
}

/// 乱数クレートを増やさないための小さな擬似乱数(xorshift64*)で Fisher-Yates シャッフルする。
/// 暗号用途ではない。seed が 0 でも動くよう定数を混ぜる
fn shuffle<T>(v: &mut [T], seed: u64) {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    for i in (1..v.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(tags: &[&str]) -> Vec<VocabEntry> {
        tags.iter().map(|t| VocabEntry { tag: t.to_string(), label: format!("L-{t}") }).collect()
    }

    /// 人工の結果イベント。top1 と 2・3 位を指定する
    fn result(top: [&str; 3]) -> EngineEvent {
        EngineEvent::Result {
            tag: top[0].into(),
            label: format!("L-{}", top[0]),
            p: 0.9,
            ranked: top
                .iter()
                .enumerate()
                .map(|(i, t)| RankedItem { tag: t.to_string(), label: format!("L-{t}"), p: 0.9 - 0.3 * i as f64 })
                .collect(),
            low_confidence: false,
            span: None,
            buffered: 0,
            duration_secs: 1.0,
            selected_with_hand: 5,
            selected_total: 10,
        }
    }

    /// 除外語なし・語彙 8 語・5 問のゲームを始め、お題のタグを返す
    fn started() -> (Quiz, Vec<String>) {
        let mut q = Quiz::new(QuizConfig { excluded_tags: vec![], ..QuizConfig::default() });
        q.start(&vocab(&["a", "b", "c", "d", "e", "f", "g", "h"]), 7, 0.0).unwrap();
        let targets = q.questions.iter().map(|v| v.tag.clone()).collect();
        (q, targets)
    }

    fn other_than(t: &str) -> [&'static str; 3] {
        // お題を含まない 3 件
        let pool = ["x1", "x2", "x3"];
        assert!(!pool.contains(&t));
        pool
    }

    #[test]
    fn correct_when_top1_matches() {
        let (mut q, t) = started();
        assert!(q.on_event(&result([&t[0], "x1", "x2"])));
        assert_eq!(q.current_record().unwrap().judgement, Judgement::Correct);
        assert_eq!(q.score(), POINTS_CORRECT);
    }

    #[test]
    fn close_when_in_top3_but_not_top1() {
        for pos in [1, 2] {
            let (mut q, t) = started();
            let mut top = ["x1", "x2", "x3"];
            top[pos] = &t[0];
            q.on_event(&result(top));
            assert_eq!(q.current_record().unwrap().judgement, Judgement::Close, "pos={pos}");
            assert_eq!(q.score(), POINTS_CLOSE);
        }
    }

    #[test]
    fn wrong_when_not_in_top3() {
        let (mut q, t) = started();
        q.on_event(&result(other_than(&t[0])));
        assert_eq!(q.current_record().unwrap().judgement, Judgement::Wrong);
        assert_eq!(q.score(), 0);
    }

    #[test]
    fn judge_uses_only_top3_of_ranked() {
        let ranked: Vec<RankedItem> =
            ["a", "b", "c", "t"].iter().map(|t| RankedItem { tag: t.to_string(), label: String::new(), p: 0.1 }).collect();
        assert_eq!(judge("t", "a", &ranked), Judgement::Wrong); // 4 位は対象外
        assert_eq!(judge("a", "a", &[]), Judgement::Correct); // ranked が空でも top-1 で判定
    }

    #[test]
    fn timeout_after_limit_scores_zero_and_has_no_ai_answer() {
        let (mut q, _) = started();
        assert!(!q.tick(ANSWER_TIMEOUT_SECS - 0.1));
        assert_eq!(q.time_left().map(|t| t > 0.0), Some(true));
        assert!(q.tick(ANSWER_TIMEOUT_SECS));
        let r = q.current_record().unwrap();
        assert_eq!(r.judgement, Judgement::TimedOut);
        assert_eq!(r.ai_label(), "-");
        assert_eq!(q.score(), 0);
    }

    #[test]
    fn pass_scores_zero_and_only_while_asking() {
        let (mut q, _) = started();
        assert!(q.pass());
        assert_eq!(q.current_record().unwrap().judgement, Judgement::Passed);
        assert!(!q.pass(), "判定表示中はパスできない");
        assert_eq!(q.records().len(), 1);
        assert_eq!(q.score(), 0);
    }

    #[test]
    fn human_override_counts_as_correct_but_keeps_the_ai_judgement() {
        let (mut q, t) = started();
        q.on_event(&result(other_than(&t[0])));
        assert!(q.mark_correct());
        let r = q.current_record().unwrap();
        assert!(r.human_override);
        assert_eq!(r.judgement, Judgement::Wrong, "AI の判定は残す");
        assert_eq!(r.effective(), Judgement::Correct);
        assert_eq!(q.score(), POINTS_CORRECT);
        assert!(!q.mark_correct(), "二重に押しても点は増えない");
        assert_eq!(q.score(), POINTS_CORRECT);
    }

    #[test]
    fn human_override_is_limited_to_wrong_or_close() {
        // AI が正解 → 不要
        let (mut q, t) = started();
        q.on_event(&result([&t[0], "x1", "x2"]));
        assert!(!q.mark_correct());
        // パス・時間切れ → AI の読みが無いので対象外
        let (mut q, _) = started();
        q.pass();
        assert!(!q.mark_correct());
        let (mut q, _) = started();
        q.tick(ANSWER_TIMEOUT_SECS);
        assert!(!q.mark_correct());
        // 出題中・開始前
        let (mut q, _) = started();
        assert!(!q.mark_correct());
        assert!(!Quiz::new(QuizConfig::default()).mark_correct());
    }

    #[test]
    fn stale_results_are_ignored() {
        let mut q = Quiz::new(QuizConfig { excluded_tags: vec![], ..QuizConfig::default() });
        // 開始前
        assert!(!q.on_event(&result(["a", "b", "c"])));
        q.start(&vocab(&["a", "b", "c", "d", "e", "f"]), 1, 0.0).unwrap();
        let t0 = q.current_target().unwrap().tag.clone();
        assert!(q.on_event(&result([&t0, "x1", "x2"])));
        // 判定表示中に届いた結果(次の問の判定に使われない・記録も増えない)
        assert!(!q.on_event(&result(["x1", "x2", "x3"])));
        assert_eq!(q.records().len(), 1);
        assert_eq!(q.score(), POINTS_CORRECT);
        // 次の問に移った後は新しい問を判定する
        assert!(q.next());
        assert!(q.on_event(&result(["x1", "x2", "x3"])));
        assert_eq!(q.records().len(), 2);
        // 結果以外のイベントは無視
        assert!(!q.on_event(&EngineEvent::Message { level: "info".into(), text: "m".into() }));
    }

    #[test]
    fn excluded_words_are_never_asked() {
        let mut v: Vec<&str> = DEFAULT_EXCLUDED_TAGS.to_vec();
        v.extend(["a", "b", "c", "d", "e", "f"]);
        for seed in 0..50 {
            let mut q = Quiz::new(QuizConfig::default());
            q.start(&vocab(&v), seed, 0.0).unwrap();
            assert_eq!(q.total(), QUESTIONS_PER_GAME);
            assert!(q.questions.iter().all(|x| !DEFAULT_EXCLUDED_TAGS.contains(&x.tag.as_str())));
            let mut tags: Vec<_> = q.questions.iter().map(|x| x.tag.clone()).collect();
            tags.sort();
            tags.dedup();
            assert_eq!(tags.len(), QUESTIONS_PER_GAME, "重複なし");
        }
    }

    #[test]
    fn exclusion_is_configurable_and_small_pools_shorten_the_game() {
        assert_eq!(parse_excluded(" a, b ,,c"), vec!["a", "b", "c"]);
        assert!(parse_excluded("").is_empty());
        let mut q = Quiz::new(QuizConfig { excluded_tags: parse_excluded("a,b"), ..QuizConfig::default() });
        q.start(&vocab(&["a", "b", "c", "d"]), 3, 0.0).unwrap();
        assert_eq!(q.total(), 2, "候補 2 語 = 2 問");
        assert_eq!(q.max_score(), 2 * POINTS_CORRECT);
        // 候補が無ければ始められない
        let mut q = Quiz::new(QuizConfig { excluded_tags: parse_excluded("a"), ..QuizConfig::default() });
        assert_eq!(q.start(&vocab(&["a"]), 3, 0.0), Err(StartError::NoCandidates));
        assert_eq!(q.start(&[], 3, 0.0), Err(StartError::NoCandidates));
        assert_eq!(q.stage, Stage::Ready);
    }

    #[test]
    fn five_questions_end_in_summary_with_total_score() {
        let (mut q, t) = started();
        assert_eq!((q.total(), q.max_score()), (5, 15));
        // 1:正解 2:おしい 3:ちがう(→人間判定で正解) 4:パス 5:時間切れ
        assert_eq!(q.number(), 1);
        q.on_event(&result([&t[0], "x1", "x2"]));
        q.next();
        assert_eq!(q.number(), 2);
        q.on_event(&result(["x1", &t[1], "x2"]));
        q.next();
        q.on_event(&result(["x1", "x2", "x3"]));
        q.mark_correct();
        q.next();
        q.pass();
        q.next();
        assert_eq!(q.number(), 5);
        q.tick(1000.0);
        assert!(matches!(q.stage, Stage::Judged { .. }));
        q.next();
        assert_eq!(q.stage, Stage::Summary);
        let pts: Vec<u32> = q.records().iter().map(|r| r.points()).collect();
        assert_eq!(pts, vec![3, 1, 3, 0, 0]);
        assert_eq!(q.score(), 7);
        let kinds: Vec<_> = q.records().iter().map(|r| (r.judgement, r.human_override)).collect();
        assert_eq!(
            kinds,
            vec![
                (Judgement::Correct, false),
                (Judgement::Close, false),
                (Judgement::Wrong, true),
                (Judgement::Passed, false),
                (Judgement::TimedOut, false)
            ]
        );
        // 総括後は何も受け付けない
        assert!(!q.on_event(&result([&t[0], "x1", "x2"])) && !q.next() && !q.pass());
    }

    #[test]
    fn judged_advances_automatically_after_the_wait() {
        let (mut q, t) = started();
        q.tick(1.0);
        q.on_event(&result([&t[0], "x1", "x2"])); // 判定時刻 = 1.0
        assert!(!q.tick(1.0 + AUTO_ADVANCE_SECS - 0.1));
        assert!(q.advance_left().unwrap() > 0.0);
        assert!(q.tick(1.0 + AUTO_ADVANCE_SECS));
        assert!(matches!(q.stage, Stage::Asking { .. }));
        assert_eq!(q.number(), 2);
        // 新しい問の制限時間は切り替えた時刻から数え直す
        assert!(!q.tick(1.0 + AUTO_ADVANCE_SECS + ANSWER_TIMEOUT_SECS - 0.1));
    }

    #[test]
    fn reset_returns_to_ready_and_can_restart() {
        let (mut q, _) = started();
        q.pass();
        q.reset();
        assert_eq!(q.stage, Stage::Ready);
        assert!(q.records().is_empty() && q.current_target().is_none());
        q.start(&vocab(&["a", "b"]), 1, 5.0).unwrap();
        assert_eq!((q.number(), q.score()), (1, 0));
    }

    #[test]
    fn same_seed_gives_the_same_questions() {
        let v = vocab(&["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]);
        let ask = |seed| {
            let mut q = Quiz::new(QuizConfig { excluded_tags: vec![], ..QuizConfig::default() });
            q.start(&v, seed, 0.0).unwrap();
            q.questions.iter().map(|x| x.tag.clone()).collect::<Vec<_>>()
        };
        assert_eq!(ask(9), ask(9));
        assert!((0..20).any(|s| ask(s) != ask(9)), "seed を変えれば出題が変わる");
    }

    #[test]
    fn symbols_and_texts_do_not_rely_on_color() {
        let all = [Judgement::Correct, Judgement::Close, Judgement::Wrong, Judgement::TimedOut, Judgement::Passed];
        assert_eq!(all.map(Judgement::symbol), ["○", "△", "×", "×", "－"]);
        assert_eq!(all.map(Judgement::text), ["正解", "おしい", "ちがう", "時間切れ", "パス"]);
    }
}
