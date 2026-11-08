//! ライブ経路の手検出の診断(`#[ignore]` テスト。本番コードからは使われない)。
//!
//! 動画ファイルを「カメラの代わり」に、ライブ経路と同じ Palm → Hand の処理で全フレームを流し、
//! フレームごとの観測(Palm の最大スコア、Hand の presence、handedness、骨格の幾何)を記録して、
//! 次の3つに分類する。**正解ラベルは無いので、(b) は教師なしの代理指標**(目視で確かめる前提):
//!
//! - (a) 手が検出されない(特に、手が写っている区間の途中で抜ける「途切れ」)
//! - (b) 検出されるが骨格がずれる疑い(時間方向のスパイク / 手の大きさの急変 /
//!   Palm 枠と骨格の中心の乖離 / 骨格がクロップ枠に接する)
//! - (c) 左右の枠が入れ替わる(同じ手が前フレームと別のスロットに入る / 2本が同じスロットを取り合う)
//!
//! 実行: `cargo test --release live_diag_report -- --ignored --nocapture`
//! 環境変数:
//!   LIVE_DIAG_VIDEOS  対象動画(':' 区切り。既定は raw_jsl の ok テイク 16 本 = L0 と同じ)
//!   LIVE_DIAG_ROI     full(既定。ライブの既定モデルの ROI) | center
//!   LIVE_DIAG_OUT     指定すると、動画ごとのフレーム別 TSV をこのディレクトリに書く
//!   LIVE_DIAG_OVERLAY 1 なら、骨格と分類フラグを描いたオーバーレイ mp4 も LIVE_DIAG_OUT に書く

use crate::*;
use std::fmt::Write as _;

/// Palm の診断用の低い閾値(本番は 0.5)。0.1 未満は「何も見えていない」として 0 扱い
const PALM_DIAG_FLOOR: f32 = 0.1;
/// 「Palm はほぼ届いていた」とみなす下限。0.3〜0.5 は閾値で惜しくも落ちたもの
const PALM_NEAR: f32 = 0.3;

// ---- 判定の閾値(すべて仮置き。分布を併記して、閾値の妥当性を読めるようにする)----
/// 途切れとみなす手なし連続の最長(秒)。これより長い手なしは「手を下げた」とみなす
const DROPOUT_MAX_SECS: f32 = 0.5;
/// スパイク: 前後フレームの手首の中点からの外れ(手の大きさ hs 単位)
const SPIKE_HS: f32 = 0.6;
/// 手の大きさの急変: 前フレームとの比がこの範囲の外
const SIZE_RATIO_RANGE: (f32, f32) = (0.6, 1.6);
/// Palm 枠の中心と骨格の重心の乖離(hs 単位)
const MISMATCH_HS: f32 = 1.0;
/// 骨格がクロップ枠に接する(クロップ中心からの正規化距離。0.5 が枠の縁)
const BORDER_EXTENT: f32 = 0.47;
/// presence が低めで採用された手
const LOW_CONF: f32 = 0.7;
/// 同じ手とみなす手首間の距離(hs 単位)
const MATCH_HS: f32 = 1.5;

const F_NONE: u32 = 1 << 0; // 手なし(途切れでないもの)
const F_DROP_PALM0: u32 = 1 << 1; // 途切れ: Palm 最大 < 0.3
const F_DROP_PALM_LOW: u32 = 1 << 2; // 途切れ: Palm 最大 0.3〜0.5
const F_DROP_HAND_REJ: u32 = 1 << 3; // 途切れ: Palm は出たが Hand の presence < 0.5
const F_SPIKE: u32 = 1 << 4;
const F_SIZE: u32 = 1 << 5;
const F_MISMATCH: u32 = 1 << 6;
const F_BORDER: u32 = 1 << 7;
const F_LOWCONF: u32 = 1 << 8;
const F_SWAP: u32 = 1 << 9;
const F_COLLIDE: u32 = 1 << 10;

const FLAG_NAMES: &[(u32, &str)] = &[
    (F_NONE, "none"),
    (F_DROP_PALM0, "drop_palm0"),
    (F_DROP_PALM_LOW, "drop_palmlow"),
    (F_DROP_HAND_REJ, "drop_handrej"),
    (F_SPIKE, "spike"),
    (F_SIZE, "size"),
    (F_MISMATCH, "mismatch"),
    (F_BORDER, "border"),
    (F_LOWCONF, "lowconf"),
    (F_SWAP, "swap"),
    (F_COLLIDE, "collide"),
];

fn flag_str(f: u32) -> String {
    let v: Vec<&str> = FLAG_NAMES.iter().filter(|(b, _)| f & b != 0).map(|(_, n)| *n).collect();
    if v.is_empty() {
        "-".into()
    } else {
        v.join(",")
    }
}

#[derive(Clone, Debug)]
struct HandObs {
    conf: f32,
    handedness: f32,
    accepted: bool,
    wrist: (f32, f32),
    /// 手の大きさ(手首〜中指付け根の距離。px)
    hs: f32,
    /// 手のひらの幅 / 長さ(小指側付け根〜人差し指側付け根 ÷ hs)。横向き・縦向きの目安
    palm_ratio: f32,
    /// Palm keypoint から求めた手の回転角(rad)
    rot: f32,
    center_mismatch: f32,
    /// クロップ中心からの最大の正規化距離(0.5 = 枠の縁)
    extent: f32,
    slot: Option<usize>,
    lms: Vec<[f32; 2]>,
    palm_box: [f32; 4],
}

#[derive(Clone, Debug)]
struct FrameObs {
    n_palm: usize,
    palm_max: f32,
    hands: Vec<HandObs>,
}

impl FrameObs {
    fn accepted(&self) -> Vec<&HandObs> {
        self.hands.iter().filter(|h| h.accepted).collect()
    }
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

fn observe(
    palm: &mut Session,
    hand: &mut Session,
    anchors: &[[f32; 2]],
    roi: PalmRoiMode,
    frame: &Array3<u8>,
) -> Result<FrameObs> {
    assert!(!roi.is_multi_roi() && !matches!(roi, PalmRoiMode::Person { .. }), "診断は単一 ROI のみ");
    let h = frame.shape()[0] as u32;
    let w = frame.shape()[1] as u32;
    let mut cands = Vec::new();
    for r in palm_rois(roi, w, h) {
        cands.extend(palm_candidates_above(palm, frame, anchors, r, PALM_DIAG_FLOOR)?);
    }
    // 低い閾値で NMS しても、0.5 以上の生き残りは本番(0.5 で切ってから NMS)と一致する
    // (0.5 以上の候補を落とせるのはより高スコア = やはり 0.5 以上の候補だけなので)
    let boxes = nms_palm_candidates(cands);
    let palm_max = boxes.first().map_or(0.0, |b| b.score);
    let prod: Vec<&PalmBox> = boxes.iter().filter(|b| b.score >= PALM_SCORE_THRESHOLD).collect();

    let mut raw: Vec<(Hand, &PalmBox)> = Vec::new();
    for p in &prod {
        raw.push((run_hand_landmark_unthresholded(hand, frame, p)?, *p));
    }
    let accepted_hands: Vec<Hand> = raw
        .iter()
        .filter(|(hd, _)| hd.confidence >= HAND_CONF_THRESHOLD)
        .map(|(hd, _)| hd.clone())
        .collect();
    let (feat, _, _) = frame_hand_feature(&accepted_hands, w as f32, h as f32);

    let mut hands = Vec::new();
    for (hd, pb) in &raw {
        let lm = &hd.landmarks;
        let p = |i: usize| (lm[i].x, lm[i].y);
        let hs = dist(p(0), p(9)).max(1.0);
        let palm_ratio = dist(p(5), p(17)) / hs;
        let cen = (
            lm.iter().map(|l| l.x).sum::<f32>() / lm.len() as f32,
            lm.iter().map(|l| l.y).sum::<f32>() / lm.len() as f32,
        );
        let pcx = (pb.x1 + pb.x2) / 2.0;
        let pcy = (pb.y1 + pb.y2) / 2.0;
        let (pw, ph) = (pb.x2 - pb.x1, pb.y2 - pb.y1);
        // 手ランドマークのクロップ幾何(preprocess_for_hand_landmark と同じ式)の逆変換で、
        // 骨格がクロップのどこまで広がっているかを測る
        let theta = palm_rotation(pb);
        let (cs, sn) = (theta.cos(), theta.sin());
        let crop = (pw.max(ph) * HAND_CROP_ENLARGE).max(1.0);
        let shift = HAND_CROP_SHIFT_Y * ph;
        let (cx, cy) = (pcx - sn * shift, pcy + cs * shift);
        let extent = lm
            .iter()
            .map(|l| {
                let (dx, dy) = (l.x - cx, l.y - cy);
                let u = cs * dx + sn * dy;
                let v = -sn * dx + cs * dy;
                u.abs().max(v.abs()) / crop
            })
            .fold(0.0f32, f32::max);
        let accepted = hd.confidence >= HAND_CONF_THRESHOLD;
        let slot = if accepted {
            (0..2).find(|s| (feat[s * HAND_POINTS * 3] - lm[0].x / w as f32).abs() < 1e-6
                && (feat[s * HAND_POINTS * 3 + 1] - lm[0].y / h as f32).abs() < 1e-6)
        } else {
            None
        };
        hands.push(HandObs {
            conf: hd.confidence,
            handedness: hd.handedness,
            accepted,
            wrist: p(0),
            hs,
            palm_ratio,
            rot: theta,
            center_mismatch: dist((pcx, pcy), cen) / hs,
            extent,
            slot,
            lms: lm.iter().map(|l| [l.x, l.y]).collect(),
            palm_box: [pb.x1, pb.y1, pb.x2, pb.y2],
        });
    }
    Ok(FrameObs { n_palm: prod.len(), palm_max, hands })
}

// ---- 分析 ----

#[derive(Default, Clone)]
struct Agg {
    frames: usize,
    none_frames: usize, // 手なしフレーム(途切れも含む)
    hand_frames: usize,
    two_hand_frames: usize,
    dropout_runs: usize,
    dropout_frames: usize,
    drop_palm0: usize,
    drop_palm_low: usize,
    drop_hand_rej: usize,
    // 採用された手ごとの指標
    hands: usize,
    spike: usize,
    spike_eligible: usize,
    size: usize,
    size_eligible: usize,
    mismatch: usize,
    border: usize,
    lowconf: usize,
    swap: usize,
    swap_eligible: usize,
    collide: usize,
    two_hand_eligible: usize,
    // 分布
    speeds: Vec<f32>,
    rot_speeds: Vec<f32>,
    flank_speeds: Vec<f32>,
    flank_rot_speeds: Vec<f32>,
    flank_ratio: Vec<f32>,
    all_ratio: Vec<f32>,
    mismatch_vals: Vec<f32>,
    extent_vals: Vec<f32>,
    conf_vals: Vec<f32>,
    palm_max_none: Vec<f32>,
}

impl Agg {
    fn merge(&mut self, o: &Agg) {
        macro_rules! add { ($($f:ident),*) => { $( self.$f += o.$f; )* } }
        add!(frames, none_frames, hand_frames, two_hand_frames, dropout_runs, dropout_frames,
             drop_palm0, drop_palm_low, drop_hand_rej, hands, spike, spike_eligible, size,
             size_eligible, mismatch, border, lowconf, swap, swap_eligible, collide,
             two_hand_eligible);
        macro_rules! ext { ($($f:ident),*) => { $( self.$f.extend_from_slice(&o.$f); )* } }
        ext!(speeds, rot_speeds, flank_speeds, flank_rot_speeds, flank_ratio, all_ratio,
             mismatch_vals, extent_vals, conf_vals, palm_max_none);
    }
}

fn normalize_rad(a: f32) -> f32 {
    crate::normalize_radians(a)
}

/// 前フレームの採用手のうち、`h` と同じ手とみなせる最寄りのもの
fn match_prev<'a>(h: &HandObs, prev: &[&'a HandObs]) -> Option<&'a HandObs> {
    prev.iter()
        .copied()
        .map(|p| (dist(h.wrist, p.wrist), p))
        .filter(|(d, p)| *d < MATCH_HS * h.hs.max(p.hs))
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .map(|(_, p)| p)
}

/// `obs` を `stride` 枚おきに間引いた列(= その fps で処理した場合)を分析する。
/// 返り値の flags は間引いた列の添字順。`fps` は元動画の fps。
fn analyze(obs: &[&FrameObs], stride: usize, fps: f32) -> (Agg, Vec<u32>) {
    let eff_fps = fps / stride as f32;
    let max_gap = (DROPOUT_MAX_SECS * eff_fps).ceil() as usize;
    let n = obs.len();
    let acc: Vec<Vec<&HandObs>> = obs.iter().map(|o| o.accepted()).collect();
    let mut flags = vec![0u32; n];
    let mut a = Agg { frames: n, ..Default::default() };

    // (a) 手なしの連続区間を調べ、両側を手ありに挟まれた短い区間を「途切れ」にする
    let mut i = 0;
    while i < n {
        if !acc[i].is_empty() {
            i += 1;
            continue;
        }
        let start = i;
        while i < n && acc[i].is_empty() {
            i += 1;
        }
        let end = i; // [start, end) が手なし
        let flanked = start > 0 && end < n;
        let is_dropout = flanked && (end - start) <= max_gap;
        if is_dropout {
            a.dropout_runs += 1;
            // 両側の手から、途切れ前後の動きの速さ・向きの変化・手のひらの向きを見る
            let before = acc[start - 1].iter().copied().max_by(|x, y| x.conf.partial_cmp(&y.conf).unwrap());
            let after = acc[end].iter().copied().max_by(|x, y| x.conf.partial_cmp(&y.conf).unwrap());
            if let (Some(b), Some(af)) = (before, after) {
                let dt = (end - start + 1) as f32 / eff_fps;
                let d = dist(b.wrist, af.wrist) / ((b.hs + af.hs) / 2.0);
                a.flank_speeds.push(d / dt);
                a.flank_rot_speeds.push(normalize_rad(af.rot - b.rot).abs() / dt);
                a.flank_ratio.push(b.palm_ratio);
                a.flank_ratio.push(af.palm_ratio);
            }
        }
        for k in start..end {
            let o = obs[k];
            let cause = if o.palm_max < PALM_SCORE_THRESHOLD {
                a.palm_max_none.push(o.palm_max);
                if o.palm_max < PALM_NEAR { F_DROP_PALM0 } else { F_DROP_PALM_LOW }
            } else {
                F_DROP_HAND_REJ
            };
            if is_dropout {
                flags[k] |= cause;
                a.dropout_frames += 1;
                match cause {
                    F_DROP_PALM0 => a.drop_palm0 += 1,
                    F_DROP_PALM_LOW => a.drop_palm_low += 1,
                    _ => a.drop_hand_rej += 1,
                }
            } else {
                flags[k] |= F_NONE;
            }
        }
        a.none_frames += end - start;
    }

    // 手ありフレームの指標
    for i in 0..n {
        let hs_here = &acc[i];
        if hs_here.is_empty() {
            continue;
        }
        a.hand_frames += 1;
        if hs_here.len() >= 2 {
            a.two_hand_frames += 1;
            a.two_hand_eligible += 1;
            // 2本が同じスロットを取り合った(希望スロットが同じ)
            let pref = |h: &HandObs| h.handedness >= 0.5;
            if pref(hs_here[0]) == pref(hs_here[1]) {
                a.collide += 1;
                flags[i] |= F_COLLIDE;
            }
        }
        for h in hs_here {
            a.hands += 1;
            a.all_ratio.push(h.palm_ratio);
            a.mismatch_vals.push(h.center_mismatch);
            a.extent_vals.push(h.extent);
            a.conf_vals.push(h.conf);
            if h.center_mismatch > MISMATCH_HS {
                a.mismatch += 1;
                flags[i] |= F_MISMATCH;
            }
            if h.extent > BORDER_EXTENT {
                a.border += 1;
                flags[i] |= F_BORDER;
            }
            if h.conf < LOW_CONF {
                a.lowconf += 1;
                flags[i] |= F_LOWCONF;
            }
            let prev = if i > 0 { match_prev(h, &acc[i - 1]) } else { None };
            if let Some(p) = prev {
                let dt = 1.0 / eff_fps;
                a.speeds.push(dist(h.wrist, p.wrist) / ((h.hs + p.hs) / 2.0) / dt);
                a.rot_speeds.push(normalize_rad(h.rot - p.rot).abs() / dt);
                a.size_eligible += 1;
                let r = h.hs / p.hs;
                if r < SIZE_RATIO_RANGE.0 || r > SIZE_RATIO_RANGE.1 {
                    a.size += 1;
                    flags[i] |= F_SIZE;
                }
                // 入れ替わり: 両フレームとも手が1本だけで、同じ手なのにスロットが違う
                if acc[i - 1].len() == 1 && hs_here.len() == 1 {
                    a.swap_eligible += 1;
                    if p.slot != h.slot {
                        a.swap += 1;
                        flags[i] |= F_SWAP;
                    }
                }
                // スパイク: 前・現・次が同じ手で追えるとき、現が前後の中点から大きく外れる
                if i + 1 < n {
                    if let Some(nx) = acc[i + 1]
                        .iter()
                        .copied()
                        .filter(|c| match_prev(c, &[h]).is_some())
                        .min_by(|x, y| dist(x.wrist, h.wrist).partial_cmp(&dist(y.wrist, h.wrist)).unwrap())
                    {
                        a.spike_eligible += 1;
                        let mid = ((p.wrist.0 + nx.wrist.0) / 2.0, (p.wrist.1 + nx.wrist.1) / 2.0);
                        if dist(h.wrist, mid) / h.hs > SPIKE_HS {
                            a.spike += 1;
                            flags[i] |= F_SPIKE;
                        }
                    }
                }
            }
        }
    }
    (a, flags)
}

fn pct(v: &[f32], q: f32) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = (q * s.len() as f32).ceil() as usize;
    s[rank.clamp(1, s.len()) - 1]
}

fn dist_line(name: &str, v: &[f32]) -> String {
    format!(
        "  {:<28} n={:<5} p10 {:6.2} / 中央 {:6.2} / p90 {:6.2} / p99 {:6.2}\n",
        name,
        v.len(),
        pct(v, 0.1),
        pct(v, 0.5),
        pct(v, 0.9),
        pct(v, 0.99)
    )
}

fn rate(n: usize, d: usize) -> String {
    if d == 0 {
        "n/a".into()
    } else {
        format!("{} / {} ({:.1}%)", n, d, n as f64 * 100.0 / d as f64)
    }
}

fn report(title: &str, a: &Agg) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "== {} ==", title);
    let _ = writeln!(s, "フレーム {} / 手あり {} (うち2本 {}) / 手なし {}", a.frames, a.hand_frames, a.two_hand_frames, a.none_frames);
    let _ = writeln!(s, "(a) 途切れ(手あり区間の途中で、{:.1}秒以内だけ手なし): {} 区間 / {} フレーム", DROPOUT_MAX_SECS, a.dropout_runs, a.dropout_frames);
    let _ = writeln!(
        s,
        "    原因内訳: Palm 最大<0.3 {} / Palm 0.3〜0.5(惜しい) {} / Palm は出たが Hand presence<0.5 {}",
        a.drop_palm0, a.drop_palm_low, a.drop_hand_rej
    );
    let _ = writeln!(s, "(b) 骨格ずれの疑い(採用された手 {} 件。代理指標):", a.hands);
    let _ = writeln!(s, "    スパイク(前後の中点から>{:.1}hs): {}", SPIKE_HS, rate(a.spike, a.spike_eligible));
    let _ = writeln!(s, "    手の大きさの急変(比が{:.1}〜{:.1}の外): {}", SIZE_RATIO_RANGE.0, SIZE_RATIO_RANGE.1, rate(a.size, a.size_eligible));
    let _ = writeln!(s, "    Palm枠と骨格の中心の乖離(>{:.1}hs): {}", MISMATCH_HS, rate(a.mismatch, a.hands));
    let _ = writeln!(s, "    骨格がクロップ枠に接する(>{:.2}): {}", BORDER_EXTENT, rate(a.border, a.hands));
    let _ = writeln!(s, "    presence が低め(<{:.1}で採用): {}", LOW_CONF, rate(a.lowconf, a.hands));
    let _ = writeln!(s, "(c) 左右枠: 手1本どうしで同じ手のスロットが変わった: {} / 2本が同じスロットを取り合った: {}", rate(a.swap, a.swap_eligible), rate(a.collide, a.two_hand_eligible));
    let _ = writeln!(s, "分布(hs = 手首〜中指付け根の距離。速さは hs/秒):");
    s.push_str(&dist_line("手首の速さ(全体)", &a.speeds));
    s.push_str(&dist_line("手首の速さ(途切れの前後)", &a.flank_speeds));
    s.push_str(&dist_line("回転の速さ(全体 rad/s)", &a.rot_speeds));
    s.push_str(&dist_line("回転の速さ(途切れの前後)", &a.flank_rot_speeds));
    s.push_str(&dist_line("手のひら幅/長さ(全体)", &a.all_ratio));
    s.push_str(&dist_line("手のひら幅/長さ(途切れの前後)", &a.flank_ratio));
    s.push_str(&dist_line("Palm枠-骨格の乖離(hs)", &a.mismatch_vals));
    s.push_str(&dist_line("クロップ内の広がり(0.5=縁)", &a.extent_vals));
    s.push_str(&dist_line("Hand presence(採用)", &a.conf_vals));
    s.push_str(&dist_line("手なし中の Palm 最大", &a.palm_max_none));
    s
}

fn write_tsv(path: &Path, obs: &[FrameObs], flags: &[u32], fps: f32) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "frame\tt_sec\tpalm_max\tn_palm\tn_acc\th0_conf\th0_handed\th0_slot\th0_wx\th0_wy\th0_hs\th0_ratio\th0_extent\th0_mismatch\th1_conf\th1_handed\th1_slot\tflags")?;
    for (i, o) in obs.iter().enumerate() {
        let acc = o.accepted();
        let c = |h: Option<&&HandObs>, f: &dyn Fn(&HandObs) -> String| h.map_or("-".to_string(), |h| f(h));
        let (h0, h1) = (acc.first(), acc.get(1));
        let slot = |h: &HandObs| h.slot.map_or("-".into(), |s| if s == 0 { "L".to_string() } else { "R".to_string() });
        writeln!(
            f,
            "{}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            i,
            i as f32 / fps,
            o.palm_max,
            o.n_palm,
            acc.len(),
            c(h0, &|h| format!("{:.3}", h.conf)),
            c(h0, &|h| format!("{:.3}", h.handedness)),
            c(h0, &slot),
            c(h0, &|h| format!("{:.0}", h.wrist.0)),
            c(h0, &|h| format!("{:.0}", h.wrist.1)),
            c(h0, &|h| format!("{:.0}", h.hs)),
            c(h0, &|h| format!("{:.2}", h.palm_ratio)),
            c(h0, &|h| format!("{:.2}", h.extent)),
            c(h0, &|h| format!("{:.2}", h.center_mismatch)),
            c(h1, &|h| format!("{:.3}", h.conf)),
            c(h1, &|h| format!("{:.3}", h.handedness)),
            c(h1, &slot),
            flag_str(flags[i]),
        )?;
    }
    Ok(())
}

/// 骨格(左右スロット別の色)と Palm 枠を描き、分類フラグを枠の色で示す
fn draw_diag_overlay(frame: &Array3<u8>, o: &FrameObs, flags: u32, out: &Path) -> Result<()> {
    let (h, w) = (frame.shape()[0] as u32, frame.shape()[1] as u32);
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(w, h, frame.iter().copied().collect()).context("frame→image")?;
    for hd in &o.hands {
        let color = match (hd.accepted, hd.slot) {
            (false, _) => Rgb([150u8, 150, 150]),
            (true, Some(0)) => Rgb([60u8, 120, 255]),
            (true, _) => Rgb([255u8, 0, 200]),
        };
        let b = hd.palm_box;
        draw_rect_outline(&mut img, b[0], b[1], b[2], b[3], Rgb([0u8, 200, 255]), 1);
        for &(a, c) in HAND_CONNECTIONS {
            draw_line(&mut img, hd.lms[a][0], hd.lms[a][1], hd.lms[c][0], hd.lms[c][1], color, 1);
        }
        for l in &hd.lms {
            draw_dot(&mut img, l[0], l[1], 3, color);
        }
    }
    // フラグごとの色の帯(画面上端)。途切れ=黄 / スパイク・大きさ急変=赤 / 乖離・接触=橙 / 入れ替わり=緑
    let bands = [
        (F_DROP_PALM0 | F_DROP_PALM_LOW | F_DROP_HAND_REJ, Rgb([255u8, 220, 0])),
        (F_SPIKE | F_SIZE, Rgb([255u8, 0, 0])),
        (F_MISMATCH | F_BORDER, Rgb([255u8, 140, 0])),
        (F_SWAP | F_COLLIDE, Rgb([0u8, 220, 0])),
    ];
    for (k, (mask, col)) in bands.iter().enumerate() {
        if flags & mask != 0 {
            for y in 0..30u32 {
                for x in (k as u32 * 120)..(k as u32 * 120 + 110) {
                    if x < w && y < h {
                        img.put_pixel(x, y, *col);
                    }
                }
            }
        }
    }
    img.save(out)?;
    Ok(())
}

fn fps_of(v: &Path) -> (VideoInfo, f32) {
    let info = probe_video(&v.to_path_buf()).unwrap();
    let fps = if info.fps > 0.0 { info.fps as f32 } else { 30.0 };
    (info, fps)
}

#[test]
#[ignore]
fn live_diag_report() {
    for p in [Path::new(PALM_MODEL), Path::new(HAND_MODEL)] {
        if !p.exists() {
            eprintln!("skip: fixture not found: {}", p.display());
            return;
        }
    }
    let roi = match std::env::var("LIVE_DIAG_ROI").as_deref() {
        Ok("center") => PalmRoiMode::CenterSquare,
        _ => PalmRoiMode::FullFrame,
    };
    let videos: Vec<PathBuf> = match std::env::var("LIVE_DIAG_VIDEOS") {
        Ok(v) => v.split(':').map(PathBuf::from).collect(),
        Err(_) => crate::tests::sample_raw_jsl_takes(16),
    };
    if videos.is_empty() {
        eprintln!("skip: 動画が無い");
        return;
    }
    let out_dir = std::env::var("LIVE_DIAG_OUT").ok().map(PathBuf::from);
    let overlay = std::env::var("LIVE_DIAG_OVERLAY").as_deref() == Ok("1");
    if let Some(d) = &out_dir {
        std::fs::create_dir_all(d).unwrap();
    }
    eprintln!("roi={} / {} 本", roi.label(), videos.len());

    let anchors = load_palm_anchors();
    let mut palm = Session::builder().unwrap().commit_from_file(PALM_MODEL).unwrap();
    let mut hand = Session::builder().unwrap().commit_from_file(HAND_MODEL).unwrap();

    let mut total: Vec<Agg> = vec![Agg::default(), Agg::default()]; // stride 1, 2
    for video in &videos {
        let (info, fps) = fps_of(video);
        let mut obs: Vec<FrameObs> = Vec::new();
        extract_frames(&video.to_path_buf(), info.width, info.height, |_i, frame| {
            obs.push(observe(&mut palm, &mut hand, &anchors, roi, &frame)?);
            Ok(())
        })
        .unwrap();
        let stem = video.file_stem().unwrap().to_string_lossy().to_string();
        // raw_jsl は <word_id>_<romaji>/<rep>.mp4 なので、親ディレクトリ名を付けて出力の衝突を避ける
        let name = match video.parent().and_then(|p| p.file_name()) {
            Some(d) => format!("{}_{}", d.to_string_lossy(), stem),
            None => stem,
        };
        let refs: Vec<&FrameObs> = obs.iter().collect();
        let mut line = format!("{:<28} {:>4} フレーム {:.1}fps", name, obs.len(), fps);
        let mut flags1 = Vec::new();
        for (k, stride) in [1usize, 2].into_iter().enumerate() {
            let sub: Vec<&FrameObs> = refs.iter().copied().step_by(stride).collect();
            let (a, fl) = analyze(&sub, stride, fps);
            let _ = write!(
                line,
                " | s{}: 手{} 途切れ{}区間/{}f spike{} swap{} collide{}",
                stride, a.hand_frames, a.dropout_runs, a.dropout_frames, a.spike, a.swap, a.collide
            );
            total[k].merge(&a);
            if stride == 1 {
                flags1 = fl;
            }
        }
        eprintln!("{}", line);
        if let Some(d) = &out_dir {
            write_tsv(&d.join(format!("{}.tsv", name)), &obs, &flags1, fps).unwrap();
            if overlay {
                let tmp = d.join(format!("{}_frames", name));
                std::fs::create_dir_all(&tmp).unwrap();
                extract_frames(&video.to_path_buf(), info.width, info.height, |i, frame| {
                    draw_diag_overlay(&frame, &obs[i], flags1[i], &tmp.join(format!("frame_{:05}.png", i)))
                })
                .unwrap();
                encode_overlay_video(&tmp, &d.join(format!("{}_overlay.mp4", name)), fps as f64).unwrap();
                std::fs::remove_dir_all(&tmp).unwrap();
            }
        }
    }
    eprintln!();
    eprint!("{}", report("全体 stride=1(動画の fps のまま)", &total[0]));
    eprintln!();
    eprint!("{}", report("全体 stride=2(半分に間引き。カメラ約15fps 相当)", &total[1]));
}

/// Palm / Hand 1 回あたりの時間を「前処理」と「ONNX 推論」に分けて測る(追跡で何が省けるかの見積り用)。
/// 環境変数 LIVE_DIAG_VIDEOS / LIVE_DIAG_ROI は `live_diag_report` と同じ。
/// 実行: `cargo test --release live_diag_timing -- --ignored --nocapture`
#[test]
#[ignore]
fn live_diag_timing() {
    let roi = match std::env::var("LIVE_DIAG_ROI").as_deref() {
        Ok("center") => PalmRoiMode::CenterSquare,
        _ => PalmRoiMode::FullFrame,
    };
    let videos: Vec<PathBuf> = match std::env::var("LIVE_DIAG_VIDEOS") {
        Ok(v) => v.split(':').map(PathBuf::from).collect(),
        Err(_) => crate::tests::sample_raw_jsl_takes(16),
    };
    let anchors = load_palm_anchors();
    let mut palm = Session::builder().unwrap().commit_from_file(PALM_MODEL).unwrap();
    let mut hand = Session::builder().unwrap().commit_from_file(HAND_MODEL).unwrap();
    let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1000.0;
    // (palm 前処理, palm 推論+出力取り出し+候補, NMS, hand 前処理, hand 推論)
    let (mut pp, mut pi, mut nm, mut hp, mut hi) = (vec![], vec![], vec![], vec![], vec![]);
    let mut warm = 5;
    for video in &videos {
        let (info, _) = fps_of(video);
        extract_frames(&video.to_path_buf(), info.width, info.height, |_i, frame| {
            let (w, h) = (frame.shape()[1] as u32, frame.shape()[0] as u32);
            let r = palm_rois(roi, w, h)[0];
            let t = std::time::Instant::now();
            let pre = preprocess_for_palm_roi(&frame, r)?;
            let t_pre = ms(t);
            let t = std::time::Instant::now();
            let v = Value::from_array((pre.shape.clone(), pre.data.clone()))?;
            let outs = palm.run(ort::inputs!["input_1" => v])?;
            let _ = outs["Identity"].try_extract_tensor::<f32>()?;
            let _ = outs["Identity_1"].try_extract_tensor::<f32>()?;
            drop(outs);
            let t_run = ms(t);
            let t = std::time::Instant::now();
            let cands = palm_candidates(&mut palm, &frame, &anchors, r)?; // 推論をもう一度含むので NMS 分だけ後で引く
            let t_cand_total = ms(t);
            let t = std::time::Instant::now();
            let boxes = nms_palm_candidates(cands);
            let t_nms = ms(t);
            let mut hand_pre = 0.0;
            let mut hand_run = 0.0;
            if let Some(b) = boxes.first() {
                let t = std::time::Instant::now();
                let hp_ = preprocess_for_hand_landmark(&frame, b)?;
                hand_pre = ms(t);
                let t = std::time::Instant::now();
                let v = Value::from_array((hp_.shape, hp_.data))?;
                let outs = hand.run(ort::inputs!["input_1" => v])?;
                let _ = outs["Identity"].try_extract_tensor::<f32>()?;
                hand_run = ms(t);
            }
            let _ = t_cand_total;
            if warm > 0 {
                warm -= 1;
            } else {
                pp.push(t_pre as f32);
                pi.push(t_run as f32);
                nm.push(t_nms as f32);
                if hand_pre > 0.0 {
                    hp.push(hand_pre as f32);
                    hi.push(hand_run as f32);
                }
            }
            Ok(())
        })
        .unwrap();
    }
    eprintln!("roi={}  (ms。Palm 前処理=クロップ+リサイズ+正規化 / Palm 推論=ONNX run+出力取り出し)", roi.label());
    for (n, v) in [("Palm 前処理", &pp), ("Palm 推論", &pi), ("Palm NMS", &nm), ("Hand 前処理(手あり)", &hp), ("Hand 推論(手あり)", &hi)] {
        eprint!("{}", dist_line(n, v));
    }
}

#[cfg(test)]
mod unit {
    use super::*;

    fn hand(wrist: (f32, f32), slot: usize) -> HandObs {
        HandObs {
            conf: 0.9,
            handedness: if slot == 1 { 0.9 } else { 0.1 },
            accepted: true,
            wrist,
            hs: 100.0,
            palm_ratio: 0.8,
            rot: 0.0,
            center_mismatch: 0.2,
            extent: 0.2,
            slot: Some(slot),
            lms: vec![],
            palm_box: [0.0; 4],
        }
    }

    fn frame(hands: Vec<HandObs>, palm_max: f32) -> FrameObs {
        FrameObs { n_palm: hands.len(), palm_max, hands }
    }

    #[test]
    fn dropout_in_the_middle_is_flagged_but_edges_are_not() {
        // なし, あり, なし, なし, あり, なし  → 真ん中の 2 枚だけが途切れ。両端は対象外
        let f = [
            frame(vec![], 0.0),
            frame(vec![hand((100.0, 100.0), 1)], 0.9),
            frame(vec![], 0.2),
            frame(vec![], 0.4),
            frame(vec![hand((105.0, 100.0), 1)], 0.9),
            frame(vec![], 0.0),
        ];
        let refs: Vec<&FrameObs> = f.iter().collect();
        let (a, fl) = analyze(&refs, 1, 30.0);
        assert_eq!(a.dropout_runs, 1);
        assert_eq!(a.dropout_frames, 2);
        assert_eq!((a.drop_palm0, a.drop_palm_low, a.drop_hand_rej), (1, 1, 0));
        assert_eq!(fl[2], F_DROP_PALM0);
        assert_eq!(fl[3], F_DROP_PALM_LOW);
        assert_eq!(fl[0], F_NONE);
        assert_eq!(fl[5], F_NONE);
    }

    #[test]
    fn long_gap_is_not_a_dropout() {
        let mut f = vec![frame(vec![hand((100.0, 100.0), 1)], 0.9)];
        f.extend((0..30).map(|_| frame(vec![], 0.0))); // 1 秒 = 手を下げた
        f.push(frame(vec![hand((100.0, 100.0), 1)], 0.9));
        let refs: Vec<&FrameObs> = f.iter().collect();
        let (a, _) = analyze(&refs, 1, 30.0);
        assert_eq!(a.dropout_runs, 0);
    }

    #[test]
    fn slot_swap_of_a_single_tracked_hand_is_flagged() {
        let f = [
            frame(vec![hand((100.0, 100.0), 1)], 0.9),
            frame(vec![hand((110.0, 100.0), 0)], 0.9), // ほぼ同じ位置なのにスロットが変わった
            frame(vec![hand((120.0, 100.0), 0)], 0.9),
        ];
        let refs: Vec<&FrameObs> = f.iter().collect();
        let (a, fl) = analyze(&refs, 1, 30.0);
        assert_eq!((a.swap, a.swap_eligible), (1, 2));
        assert!(fl[1] & F_SWAP != 0 && fl[2] & F_SWAP == 0);
    }

    #[test]
    fn spike_is_flagged_only_on_the_outlier_frame() {
        let f = [
            frame(vec![hand((100.0, 100.0), 1)], 0.9),
            frame(vec![hand((100.0, 190.0), 1)], 0.9), // 前後の中点(100,100)から 0.9hs
            frame(vec![hand((100.0, 100.0), 1)], 0.9),
        ];
        let refs: Vec<&FrameObs> = f.iter().collect();
        let (a, fl) = analyze(&refs, 1, 30.0);
        assert_eq!(a.spike, 1);
        assert!(fl[1] & F_SPIKE != 0 && fl[0] & F_SPIKE == 0 && fl[2] & F_SPIKE == 0);
    }

    #[test]
    fn two_hands_preferring_the_same_slot_collide() {
        let mut h2 = hand((400.0, 100.0), 1);
        h2.handedness = 0.8;
        let f = [frame(vec![hand((100.0, 100.0), 1), h2], 0.9)];
        let refs: Vec<&FrameObs> = f.iter().collect();
        let (a, _) = analyze(&refs, 1, 30.0);
        assert_eq!((a.collide, a.two_hand_eligible), (1, 1));
    }

    #[test]
    fn stride_changes_dropout_window_in_frames() {
        // 0.5 秒 = stride1 で 15 枚、stride2 で 8 枚。10 枚の手なしは stride1 では途切れ、stride2(5 枚)でも途切れ
        let mk = |gap: usize| {
            let mut f = vec![frame(vec![hand((100.0, 100.0), 1)], 0.9)];
            f.extend((0..gap).map(|_| frame(vec![], 0.0)));
            f.push(frame(vec![hand((100.0, 100.0), 1)], 0.9));
            f
        };
        let f = mk(16); // 16 枚 > 15 枚: stride1 では対象外
        let refs: Vec<&FrameObs> = f.iter().collect();
        assert_eq!(analyze(&refs, 1, 30.0).0.dropout_runs, 0);
        let f = mk(15);
        let refs: Vec<&FrameObs> = f.iter().collect();
        assert_eq!(analyze(&refs, 1, 30.0).0.dropout_runs, 1);
    }
}
