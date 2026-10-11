//! 「連続して手が検出できている秒数」の算出(純関数・UI 非依存)。
//!
//! 規則は撮影後の合否判定(pose_extractor/src/progress.rs)に合わせる:
//! - 合格ライン `MIN_HAND_RUN_SECONDS` = 1.0 秒(progress.rs 64 行目と同じ値)
//! - 瞬断は `MAX_RUN_GAP_SAMPLES` = 1 サンプルまで同じ区間として橋渡しする(progress.rs 67 行目と同じ)
//! - 左手・右手は別々に数え、良い方(長い方)で判定する(progress.rs の `ingest` の `left.max(right)`)
//!
//! progress.rs との違い(どちらも「だいたい同じ基準を撮影中に見せる」ための近似で、厳密な一致ではない):
//! - 時間の数え方: progress.rs は「検出できたサンプル数 / 1 秒あたりのサンプル数」(約 3 サンプル/秒、
//!   瞬断のサンプルは長さに含めない)。ここでは「区間の最初の検出時刻から最後の検出時刻まで」の経過時間
//!   (橋渡しした瞬断の時間は含み、最後の 1 間隔ぶんは含まない)。差はおおむね 1 サンプル間隔ぶん
//! - 1 サンプルの間隔: エンジンの `status` は最大 10 回/秒なので、ここでの「1 サンプルの瞬断」は約 0.1 秒。
//!   progress.rs(約 3 サンプル/秒)では約 0.33 秒。サンプル数の規則は同じだが、時間にすると短い
//!
//! 時刻は「エンジンの status 行を受け取った時刻(読み取りスレッドで記録)」を使う。理由は mod.rs の冒頭。

/// 合格ライン(秒)。progress.rs の `MIN_HAND_RUN_SECONDS` と同じ値
pub const MIN_HAND_RUN_SECONDS: f64 = 1.0;
/// 同じ区間として橋渡しする瞬断のサンプル数。progress.rs の `MAX_RUN_GAP_SAMPLES` と同じ値
pub const MAX_RUN_GAP_SAMPLES: usize = 1;

/// 終わった区間(直前の連続区間の表示に使う)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EndedRun {
    /// 区間の最後の検出時刻(秒)
    pub ended_at: f64,
    /// 区間の長さ(秒)
    pub secs: f64,
}

/// 片手ぶんの連続区間の追跡。サンプル(時刻, 検出できたか)を時刻順に `push` する
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunTracker {
    /// いまの区間の最初の検出時刻(区間が無ければ None)
    start: Option<f64>,
    /// いまの区間の最後の検出時刻
    last_hit: f64,
    /// いまの区間の後ろに続いている「検出できなかった」サンプル数
    gap: usize,
    /// 最後に終わった区間
    last_ended: Option<EndedRun>,
}

impl RunTracker {
    pub fn push(&mut self, t: f64, hit: bool) {
        if !t.is_finite() {
            return;
        }
        if hit {
            match self.start {
                // 瞬断(gap <= MAX_RUN_GAP_SAMPLES)のあとの検出は、同じ区間の続き
                Some(_) => {}
                None => self.start = Some(t),
            }
            self.last_hit = t;
            self.gap = 0;
        } else if let Some(start) = self.start {
            self.gap += 1;
            if self.gap > MAX_RUN_GAP_SAMPLES {
                // 途切れた。区間を閉じる
                self.last_ended = Some(EndedRun { ended_at: self.last_hit, secs: self.last_hit - start });
                self.start = None;
                self.gap = 0;
            }
        }
    }

    /// いま続いている区間の長さ(秒)。区間が無ければ 0
    pub fn current_secs(&self) -> f64 {
        match self.start {
            Some(s) => (self.last_hit - s).max(0.0),
            None => 0.0,
        }
    }

    pub fn last_ended(&self) -> Option<EndedRun> {
        self.last_ended
    }
}

/// 左右の手をまとめた追跡
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HandRuns {
    pub left: RunTracker,
    pub right: RunTracker,
}

/// 2 つの手の区間が「同じ時に終わった」とみなす時刻差(秒)。両手のサインを下ろしたとき、
/// 左右の検出が切れる時刻はサンプル 1〜2 個ずれることがあるため
const SAME_END_TOLERANCE_SECS: f64 = 0.5;

impl HandRuns {
    /// エンジンの `status.hands`("L" | "R" | "LR" | "none")を 1 サンプルとして入れる
    pub fn push_status(&mut self, t: f64, hands: &str) {
        let (l, r) = hands_flags(hands);
        self.left.push(t, l);
        self.right.push(t, r);
    }

    /// いま続いている区間の長さ(左右の良い方、秒)
    pub fn current_best(&self) -> f64 {
        self.left.current_secs().max(self.right.current_secs())
    }

    /// 直前に終わった区間の長さ(秒)。左右のうち後に終わった方。ほぼ同時に終わったなら長い方
    pub fn last_ended_best(&self) -> Option<f64> {
        match (self.left.last_ended(), self.right.last_ended()) {
            (None, None) => None,
            (Some(a), None) | (None, Some(a)) => Some(a.secs),
            (Some(a), Some(b)) => {
                if (a.ended_at - b.ended_at).abs() <= SAME_END_TOLERANCE_SECS {
                    Some(a.secs.max(b.secs))
                } else if a.ended_at > b.ended_at {
                    Some(a.secs)
                } else {
                    Some(b.secs)
                }
            }
        }
    }
}

/// "L" | "R" | "LR" | "none" → (左手あり, 右手あり)。知らない値は「なし」
pub fn hands_flags(hands: &str) -> (bool, bool) {
    match hands {
        "L" => (true, false),
        "R" => (false, true),
        "LR" => (true, true),
        _ => (false, false),
    }
}

/// 合格ライン以上か(色を変える判定)
pub fn passes(secs: f64) -> bool {
    secs >= MIN_HAND_RUN_SECONDS
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 0.1 秒間隔のサンプル列を流す(足し算の誤差が溜まらないよう、時刻は i × 0.1 で作る)。
    /// 戻り値は次のサンプルの時刻
    fn feed(tr: &mut RunTracker, start: f64, hits: &[bool]) -> f64 {
        for (i, &h) in hits.iter().enumerate() {
            tr.push(start + i as f64 * 0.1, h);
        }
        start + hits.len() as f64 * 0.1
    }

    #[test]
    fn continuous_hits_grow_and_cross_the_line() {
        let mut tr = RunTracker::default();
        // 11 サンプル連続(0.0〜1.0 秒)→ 最初から最後まで 1.0 秒 = 合格ライン
        feed(&mut tr, 0.0, &[true; 11]);
        assert!((tr.current_secs() - 1.0).abs() < 1e-9);
        assert!(passes(tr.current_secs()));
        // 10 サンプル(0.0〜0.9 秒)ではまだ届かない
        let mut tr = RunTracker::default();
        feed(&mut tr, 0.0, &[true; 10]);
        assert!(!passes(tr.current_secs()));
    }

    #[test]
    fn single_gap_is_bridged_two_gaps_split() {
        // 瞬断 1 サンプルは橋渡し: 0.0〜0.4 検出、0.5 欠落、0.6〜1.0 検出 → 0.0〜1.0 で 1.0 秒
        let mut tr = RunTracker::default();
        let mut hits = vec![true; 5];
        hits.push(false);
        hits.extend([true; 5]);
        feed(&mut tr, 0.0, &hits);
        assert!((tr.current_secs() - 1.0).abs() < 1e-9, "{}", tr.current_secs());
        assert!(tr.last_ended().is_none());

        // 2 サンプル続けて欠けたら別の区間: 前の区間は 0.0〜0.4(0.4 秒)で閉じ、新しい区間は 0.7 から
        let mut tr = RunTracker::default();
        let mut hits = vec![true; 5];
        hits.extend([false, false]);
        hits.extend([true; 3]);
        feed(&mut tr, 0.0, &hits);
        let ended = tr.last_ended().unwrap();
        assert!((ended.secs - 0.4).abs() < 1e-9);
        assert!((ended.ended_at - 0.4).abs() < 1e-9);
        assert!((tr.current_secs() - 0.2).abs() < 1e-9, "0.7〜0.9: {}", tr.current_secs());
    }

    #[test]
    fn during_a_bridged_gap_the_value_holds_then_drops_to_zero() {
        let mut tr = RunTracker::default();
        let t = feed(&mut tr, 0.0, &[true; 6]); // 0.0〜0.5
        tr.push(t, false); // 1 個目の欠落: まだ同じ区間(値は 0.5 のまま)
        assert!((tr.current_secs() - 0.5).abs() < 1e-9);
        tr.push(t + 0.1, false); // 2 個目: 途切れる
        assert_eq!(tr.current_secs(), 0.0);
        assert!((tr.last_ended().unwrap().secs - 0.5).abs() < 1e-9);
    }

    #[test]
    fn no_hits_means_zero_and_non_finite_time_is_ignored() {
        let mut tr = RunTracker::default();
        feed(&mut tr, 0.0, &[false; 20]);
        assert_eq!(tr.current_secs(), 0.0);
        assert!(tr.last_ended().is_none());
        tr.push(f64::NAN, true);
        assert_eq!(tr.current_secs(), 0.0);
    }

    // progress.rs の longest_run_seconds のテスト(1124〜1158 行)と同じ並びで、
    // 「橋渡しされるか・分かれるか」の判定が一致することを確かめる(秒数の数え方は上記のとおり違う)
    #[test]
    fn bridging_decisions_match_progress_rs_cases() {
        // 1 サンプルの谷は橋渡し → 区間は 1 つ
        let mut tr = RunTracker::default();
        feed(&mut tr, 0.0, &[true, true, false, true, true]);
        assert!(tr.last_ended().is_none());
        assert!(tr.current_secs() > 0.0);
        // 2 サンプルの谷は分割 → 区間は 2 つ(前の区間が閉じている)
        let mut tr = RunTracker::default();
        feed(&mut tr, 0.0, &[true, true, false, false, true]);
        assert!(tr.last_ended().is_some());
    }

    #[test]
    fn hand_runs_use_the_better_hand_and_hand_flags() {
        assert_eq!(hands_flags("L"), (true, false));
        assert_eq!(hands_flags("R"), (false, true));
        assert_eq!(hands_flags("LR"), (true, true));
        assert_eq!(hands_flags("none"), (false, false));
        assert_eq!(hands_flags("??"), (false, false));

        let mut hr = HandRuns::default();
        // 右手だけ 0.0〜1.2 秒、左手は 0.6〜1.2 秒
        let mut t = 0.0;
        for i in 0..13 {
            hr.push_status(t, if i >= 6 { "LR" } else { "R" });
            t += 0.1;
        }
        assert!((hr.current_best() - 1.2).abs() < 1e-9, "{}", hr.current_best());
        assert!((hr.left.current_secs() - 0.6).abs() < 1e-9);
        // 両手を下ろす → 両方の区間がほぼ同時に閉じ、直前の値は長い方(右手 1.2 秒)
        hr.push_status(t, "none");
        hr.push_status(t + 0.1, "none");
        assert_eq!(hr.current_best(), 0.0);
        assert!((hr.last_ended_best().unwrap() - 1.2).abs() < 1e-9);
    }

    #[test]
    fn last_ended_prefers_the_later_run_when_far_apart() {
        let mut hr = HandRuns::default();
        // 左手が 0.0〜2.0 秒で終わる
        let mut t = 0.0;
        for _ in 0..21 {
            hr.push_status(t, "L");
            t += 0.1;
        }
        hr.push_status(t, "none");
        hr.push_status(t + 0.1, "none");
        // 10 秒後に右手が 0.3 秒だけ出て終わる → 直前は右手の 0.3 秒(左手の古い 2.0 秒ではない)
        let mut t = 12.0;
        for _ in 0..4 {
            hr.push_status(t, "R");
            t += 0.1;
        }
        hr.push_status(t, "none");
        hr.push_status(t + 0.1, "none");
        assert!((hr.last_ended_best().unwrap() - 0.3).abs() < 1e-9, "{:?}", hr.last_ended_best());
    }
}
