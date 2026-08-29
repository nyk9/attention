//! 手形記述子: 生の 126 次元ハンドランドマークを、幾何的な記述子ベクトルに変換する
//!
//! # なぜこれを作るのか
//!
//! 認識モデルの入力は現在 `pose_extractor build-dict` が出す生の 126 次元
//! (左手 21 点 × xyz + 右手 21 点 × xyz、画像座標を width/height で正規化したもの)。
//! この表現には、学習にとって不都合な性質が2つある:
//!
//! 1. **左右の枠が撮影時の使用手で入れ替わる**([[stage1-hand-switching-finding]])。
//!    同じサインでも右手で撮ったテイクと左手で撮ったテイクで、値が入る枠がまるごと変わる
//! 2. **画像内の絶対位置・手の大きさ(カメラ距離)に依存する**。指の形が同じでも、
//!    立ち位置が少しずれるだけでベクトルが大きく動く
//!
//! 手形記述子は「手首を原点にしてスケール正規化した上での関節角と指先間距離」なので、
//! 平行移動・スケール・**左右反転**のいずれにも不変になる。事前調査では、この記述子 +
//! 学習なしの最近傍探索が、学習済みモデル(top-1 27-29%)を大きく上回る成績を出した。
//!
//! # 1 手あたり 33 次元の内訳
//!
//! | 範囲 | 内容 | 不変性 |
//! |---|---|---|
//! | 0..15 | 各指の関節角 15 個(親指〜小指の 5 指 × 3 関節) | 平行移動・スケール・左右反転に不変 |
//! | 15..20 | 指先 - 手首の距離 5 個(スケールで割る) | 同上 |
//! | 20..30 | 指先どうしの距離 10 個(スケールで割る) | 同上 |
//! | 30..32 | 手首の画像内座標 xy | 位置・動きの情報(不変ではない) |
//! | 32..33 | 検出フラグ | そのフレームで手が取れたか |
//!
//! スケールは「手首 → 中指 MCP(landmark 9)」の距離。手の大きさ・カメラ距離の代理。
//!
//! 1 フレーム = 主たる手 33 次元 + 非主たる手 33 次元 = **66 次元**。
//!
//! # 主たる手の決め方と座標系の統一
//!
//! 「主たる手」はテイク全体で決める(フレーム単位ではない): 検出できたフレーム数が
//! 多い方の枠を主たる手とする(同数なら右手枠)。主たる手が左手枠だったテイクは、
//! `pose_data::mirror_features` でフレームごと左右反転してから読む。これで
//! 「主たる手は常に右手枠」に揃う。
//!
//! なお関節角と正規化距離(0..30)は左右反転で値が変わらないため、反転が実際に効くのは
//! 手首座標 xy だけである(この性質はテスト `mirror_invariance` で担保している)。

use crate::config::POSE_FEATURE_DIM;
use crate::pose_data::mirror_features;
use serde::{Deserialize, Serialize};

/// 1 手あたりの記述子次元(関節角15 + 指先-手首距離5 + 指先間距離10 + 手首xy 2 + 検出フラグ1)
pub const HAND_BLOCK_DIM: usize = 33;

/// 1 フレームあたりの手形記述子の次元(主たる手 + 非主たる手)
pub const HANDSHAPE_FEATURE_DIM: usize = HAND_BLOCK_DIM * 2;

/// 幾何量だけの部分(関節角 + 距離)の次元。左右反転に完全不変な範囲
const GEOMETRY_DIM: usize = 30;

/// 片手枠の次元(21 点 × xyz)
const HAND_SLOT_DIM: usize = POSE_FEATURE_DIM / 2;

/// 指の付け根(MCP)のランドマーク番号。親指・人差し指・中指・薬指・小指
const FINGER_BASES: [usize; 5] = [1, 5, 9, 13, 17];

/// 指先のランドマーク番号
const FINGER_TIPS: [usize; 5] = [4, 8, 12, 16, 20];

/// スケール基準に使うランドマーク(中指 MCP)。手首との距離を「手の大きさ」とみなす
const SCALE_LANDMARK: usize = 9;

/// 記述子のどの成分を残すか。アブレーション(どの成分が効いているかの切り分け)用。
///
/// 既定は全部残す(`all()`)。CLI `--drop-descriptor` で成分を落とす。
/// 検出フラグは常に残す(その手が取れたかどうかは、どの成分を落としても意味が変わらない)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorParts {
    /// 関節角 15 個
    pub angles: bool,
    /// 指先 - 手首の距離 5 個
    pub tip_wrist: bool,
    /// 指先どうしの距離 10 個
    pub tip_pairs: bool,
    /// 手首の画像内座標 xy
    pub wrist_xy: bool,
    /// 非主たる手のブロックまるごと
    pub off_hand: bool,
}

impl Default for DescriptorParts {
    fn default() -> Self {
        Self::all()
    }
}

impl DescriptorParts {
    pub fn all() -> Self {
        Self {
            angles: true,
            tip_wrist: true,
            tip_pairs: true,
            wrist_xy: true,
            off_hand: true,
        }
    }

    /// CLI の `--drop-descriptor` 指定(カンマ区切り)を解決する。
    /// 空文字・未指定は `all()`。例: "wrist,offhand"
    pub fn parse_drop(spec: &str) -> Result<Self, String> {
        let mut parts = Self::all();
        for raw in spec.split(',') {
            let name = raw.trim();
            if name.is_empty() {
                continue;
            }
            match name {
                "angles" => parts.angles = false,
                "tipwrist" => parts.tip_wrist = false,
                "tippairs" => parts.tip_pairs = false,
                "wrist" => parts.wrist_xy = false,
                "offhand" => parts.off_hand = false,
                _ => {
                    return Err(format!(
                        "未知の記述子成分です: \"{}\"(angles / tipwrist / tippairs / wrist / offhand から選んでください)",
                        name
                    ))
                }
            }
        }
        if !parts.angles && !parts.tip_wrist && !parts.tip_pairs && !parts.wrist_xy {
            return Err(
                "主たる手の成分を全部落とすと検出フラグしか残りません。1 つ以上残してください"
                    .to_string(),
            );
        }
        Ok(parts)
    }

    /// 全部残しているか(既定経路の判定に使う)
    pub fn is_full(self) -> bool {
        self == Self::all()
    }

    /// 66 次元のうち残す次元の添字。手ブロックの順(主たる手 → 非主たる手)を保つ
    pub fn kept_indices(self) -> Vec<usize> {
        let mut kept = Vec::new();
        let hands = if self.off_hand { 2 } else { 1 };
        for s in 0..hands {
            let base = s * HAND_BLOCK_DIM;
            if self.angles {
                kept.extend(base..base + 15);
            }
            if self.tip_wrist {
                kept.extend(base + 15..base + 20);
            }
            if self.tip_pairs {
                kept.extend(base + 20..base + 30);
            }
            if self.wrist_xy {
                kept.extend(base + 30..base + 32);
            }
            // 検出フラグは常に残す
            kept.push(base + 32);
        }
        kept
    }

    /// この成分構成での 1 フレームの次元
    pub fn feature_dim(self) -> usize {
        self.kept_indices().len()
    }

    /// 落とした成分の一覧(表示・記録用)。全部残していれば "なし"
    pub fn dropped_names(self) -> String {
        let mut dropped = Vec::new();
        if !self.angles {
            dropped.push("angles");
        }
        if !self.tip_wrist {
            dropped.push("tipwrist");
        }
        if !self.tip_pairs {
            dropped.push("tippairs");
        }
        if !self.wrist_xy {
            dropped.push("wrist");
        }
        if !self.off_hand {
            dropped.push("offhand");
        }
        if dropped.is_empty() {
            "なし".to_string()
        } else {
            dropped.join(",")
        }
    }
}

/// 認識モデルへ与える入力特徴量の種類。CLI `--input-features` に対応する。
/// `model_config.json` にも保存するので、保存済みモデルを読むときに
/// 「このモデルはどの入力表現で学習されたか」が分かる
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InputFeatures {
    /// 従来通り、生の 126 次元ハンドランドマークをそのまま使う(既定)
    #[default]
    Raw,
    /// フレームごとに手形記述子を計算する(時間方向の情報が残る)
    Handshape,
    /// テイク全体の平均記述子を全フレームに複製する(事前調査の最近傍探索と同じ内容)
    HandshapeMean,
}

impl InputFeatures {
    /// CLI 文字列から解決する。未知の値は理由付きのエラーにする
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "raw" => Ok(Self::Raw),
            "handshape" => Ok(Self::Handshape),
            "handshape-mean" => Ok(Self::HandshapeMean),
            _ => Err(format!(
                "未知の入力特徴量です: \"{}\"(raw / handshape / handshape-mean から選んでください)",
                name
            )),
        }
    }

    /// この入力特徴量での 1 フレームあたりの次元(= PoseEncoder の入力次元)
    pub fn feature_dim(self) -> usize {
        match self {
            Self::Raw => POSE_FEATURE_DIM,
            Self::Handshape | Self::HandshapeMean => HANDSHAPE_FEATURE_DIM,
        }
    }

    /// 生の 126 次元のままかどうか(既定経路の判定に使う)
    pub fn is_raw(self) -> bool {
        self == Self::Raw
    }

    /// CLI 表示・記録用の名前(`parse` の逆)
    pub fn name(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Handshape => "handshape",
            Self::HandshapeMean => "handshape-mean",
        }
    }
}

/// 3 次元ベクトルの差
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// b を挟む角 a-b-c(ラジアン、0..π)。退化したベクトルがあれば 0 を返す
fn joint_angle(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let u = sub(a, b);
    let v = sub(c, b);
    let (nu, nv) = (norm(u), norm(v));
    if nu < 1e-9 || nv < 1e-9 {
        return 0.0;
    }
    let cos = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv);
    cos.clamp(-1.0, 1.0).acos()
}

/// 片手枠(63 次元)から 21 点を取り出す。全ゼロ(= 未検出)なら None
fn hand_points(frame: &[f32], offset: usize) -> Option<[[f32; 3]; 21]> {
    let slot = &frame[offset..offset + HAND_SLOT_DIM];
    if slot.iter().all(|&v| v == 0.0) {
        return None;
    }
    let mut pts = [[0.0f32; 3]; 21];
    for (i, p) in pts.iter_mut().enumerate() {
        *p = [slot[i * 3], slot[i * 3 + 1], slot[i * 3 + 2]];
    }
    Some(pts)
}

/// 21 点から幾何記述子 30 次元 + 手首 xy を作る。
/// 戻り値は (幾何 30 次元, 手首xy)
fn hand_descriptor(p: &[[f32; 3]; 21]) -> ([f32; GEOMETRY_DIM], [f32; 2]) {
    let mut out = [0.0f32; GEOMETRY_DIM];
    let mut idx = 0;

    // (a) 関節角 15 個: 各指について [手首, MCP, PIP, DIP, TIP] の中間 3 点で角を測る
    for base in FINGER_BASES {
        let chain = [0, base, base + 1, base + 2, base + 3];
        for j in 1..4 {
            out[idx] = joint_angle(p[chain[j - 1]], p[chain[j]], p[chain[j + 1]]);
            idx += 1;
        }
    }

    // スケール = 手首 → 中指 MCP の距離。0 割りを避けるため下限を敷く
    let scale = norm(sub(p[SCALE_LANDMARK], p[0])).max(1e-6);

    // (b) 指先 - 手首の距離 5 個
    for t in FINGER_TIPS {
        out[idx] = norm(sub(p[t], p[0])) / scale;
        idx += 1;
    }

    // (c) 指先どうしの距離 10 個
    for i in 0..FINGER_TIPS.len() {
        for j in (i + 1)..FINGER_TIPS.len() {
            out[idx] = norm(sub(p[FINGER_TIPS[i]], p[FINGER_TIPS[j]])) / scale;
            idx += 1;
        }
    }

    debug_assert_eq!(idx, GEOMETRY_DIM);
    (out, [p[0][0], p[0][1]])
}

/// テイク内で左手枠 / 右手枠が検出できたフレーム数を数える
fn detection_counts(features: &[f32], frames: usize) -> (usize, usize) {
    let mut left = 0;
    let mut right = 0;
    for t in 0..frames {
        let frame = &features[t * POSE_FEATURE_DIM..(t + 1) * POSE_FEATURE_DIM];
        if hand_points(frame, 0).is_some() {
            left += 1;
        }
        if hand_points(frame, HAND_SLOT_DIM).is_some() {
            right += 1;
        }
    }
    (left, right)
}

/// 主たる手が左手枠だったテイクを反転して「主たる手 = 右手枠」に揃える。
/// 反転が必要なかった場合は入力をそのまま返す(無駄なコピーを避ける)
fn align_to_right_dominant(features: &[f32], frames: usize) -> std::borrow::Cow<'_, [f32]> {
    let (left, right) = detection_counts(features, frames);
    if left > right {
        std::borrow::Cow::Owned(mirror_features(features))
    } else {
        // 同数のときは右手枠を主たる手とする(事前調査の実装と同じ規則)
        std::borrow::Cow::Borrowed(features)
    }
}

/// 手ポーズ列(生 126 次元 × frames)を手形記述子列(66 次元 × frames)に変換する。
///
/// - `features`: `[frames * POSE_FEATURE_DIM]` のフラット列
/// - 戻り値: `[frames * HANDSHAPE_FEATURE_DIM]` のフラット列
///
/// # 手が 1 フレームも検出されないテイクの扱い
///
/// 全ゼロのベクトルを返す(生特徴が全ゼロだったのと同じ状態)。撮影失敗テイクを
/// 学習から自動で落とすことはしない(データの取捨選択は `quality_flag` の役目で、
/// ここで暗黙に間引くと件数が合わなくなり原因追跡が難しくなるため)。
///
/// # 未検出フレームの扱い
///
/// - `Handshape`: そのフレームの当該手の幾何 30 次元と手首 xy は「そのテイクで検出できた
///   フレームの平均」で埋め、検出フラグだけ 0 にする。ゼロ埋めにすると「全関節が 0 ラジアン
///   = 指が完全に折り畳まれた手」という実在しない手形をモデルに見せてしまうため
/// - `HandshapeMean`: そもそも全フレームが平均値なので区別はない。検出フラグには
///   検出できたフレームの割合(カバレッジ)が入る
pub fn handshape_sequence(features: &[f32], frames: usize, mode: InputFeatures) -> Vec<f32> {
    handshape_sequence_with_parts(features, frames, mode, DescriptorParts::all())
}

/// `handshape_sequence` の成分選択版(アブレーション用)。
/// まず 66 次元をすべて作り、そのあと残す次元だけを抜き出す。
/// 「作ってから間引く」ので、成分を落としても残る次元の値は完全に同じになる
pub fn handshape_sequence_with_parts(
    features: &[f32],
    frames: usize,
    mode: InputFeatures,
    parts: DescriptorParts,
) -> Vec<f32> {
    let full = handshape_sequence_full(features, frames, mode);
    if parts.is_full() {
        return full;
    }
    let kept = parts.kept_indices();
    let mut out = Vec::with_capacity(frames * kept.len());
    for t in 0..frames {
        let base = t * HANDSHAPE_FEATURE_DIM;
        for &j in &kept {
            out.push(full[base + j]);
        }
    }
    out
}

fn handshape_sequence_full(features: &[f32], frames: usize, mode: InputFeatures) -> Vec<f32> {
    assert!(
        !mode.is_raw(),
        "handshape_sequence は Raw(生 126 次元)には使えません"
    );
    assert_eq!(
        features.len(),
        frames * POSE_FEATURE_DIM,
        "features の長さがフレーム数 × {} と一致しません",
        POSE_FEATURE_DIM
    );

    let aligned = align_to_right_dominant(features, frames);
    // 主たる手 = 右手枠(オフセット 63)、非主たる手 = 左手枠(オフセット 0)
    let slots = [HAND_SLOT_DIM, 0];

    // まず全フレーム分の生記述子を作る(未検出は None)
    // per_frame[slot_index][frame] = Option<(幾何30, 手首xy)>
    let mut per_frame: [Vec<Option<([f32; GEOMETRY_DIM], [f32; 2])>>; 2] =
        [Vec::with_capacity(frames), Vec::with_capacity(frames)];
    for t in 0..frames {
        let frame = &aligned[t * POSE_FEATURE_DIM..(t + 1) * POSE_FEATURE_DIM];
        for (s, &offset) in slots.iter().enumerate() {
            per_frame[s].push(hand_points(frame, offset).map(|p| hand_descriptor(&p)));
        }
    }

    // 手ごとに「検出できたフレームの平均」を出す(未検出フレームの補完と TakeMean 用)
    let mut means: [Option<([f32; GEOMETRY_DIM], [f32; 2])>; 2] = [None, None];
    let mut coverage = [0.0f32; 2];
    for s in 0..2 {
        let detected: Vec<&([f32; GEOMETRY_DIM], [f32; 2])> =
            per_frame[s].iter().filter_map(|d| d.as_ref()).collect();
        coverage[s] = if frames == 0 {
            0.0
        } else {
            detected.len() as f32 / frames as f32
        };
        if detected.is_empty() {
            continue;
        }
        let n = detected.len() as f32;
        let mut geo = [0.0f32; GEOMETRY_DIM];
        let mut wrist = [0.0f32; 2];
        for d in &detected {
            for k in 0..GEOMETRY_DIM {
                geo[k] += d.0[k];
            }
            wrist[0] += d.1[0];
            wrist[1] += d.1[1];
        }
        for g in geo.iter_mut() {
            *g /= n;
        }
        wrist[0] /= n;
        wrist[1] /= n;
        means[s] = Some((geo, wrist));
    }

    let mut out = vec![0.0f32; frames * HANDSHAPE_FEATURE_DIM];
    for t in 0..frames {
        for s in 0..2 {
            let base = t * HANDSHAPE_FEATURE_DIM + s * HAND_BLOCK_DIM;
            let (source, flag) = match mode {
                InputFeatures::HandshapeMean => (means[s], coverage[s]),
                InputFeatures::Handshape => match per_frame[s][t] {
                    // 検出できたフレームはその値、できなかったフレームはテイク平均で補完
                    Some(d) => (Some(d), 1.0),
                    None => (means[s], 0.0),
                },
                InputFeatures::Raw => unreachable!("冒頭の assert で弾いている"),
            };
            let Some((geo, wrist)) = source else {
                // その手はテイク中 1 フレームも検出されなかった → ブロックごと全ゼロのまま
                continue;
            };
            out[base..base + GEOMETRY_DIM].copy_from_slice(&geo);
            out[base + GEOMETRY_DIM] = wrist[0];
            out[base + GEOMETRY_DIM + 1] = wrist[1];
            out[base + GEOMETRY_DIM + 2] = flag;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用に「指をまっすぐ伸ばした手」を作る。
    /// 手首を (x0, y0) に置き、各指を y 方向に等間隔で伸ばす
    fn straight_hand(x0: f32, y0: f32, scale: f32) -> [f32; HAND_SLOT_DIM] {
        let mut slot = [0.0f32; HAND_SLOT_DIM];
        // 手首
        slot[0] = x0;
        slot[1] = y0;
        slot[2] = 0.0;
        for (fi, base) in FINGER_BASES.iter().enumerate() {
            for j in 0..4 {
                let idx = base + j;
                // 指ごとに x を少しずらし、関節は y 方向にまっすぐ並べる
                slot[idx * 3] = x0 + (fi as f32 - 2.0) * 0.02 * scale;
                slot[idx * 3 + 1] = y0 - (j as f32 + 1.0) * 0.05 * scale;
                slot[idx * 3 + 2] = 0.0;
            }
        }
        slot
    }

    /// 片手だけが入ったフレーム列を作る。`right` が true なら右手枠に入れる
    fn take_with_one_hand(frames: usize, right: bool, x0: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; frames * POSE_FEATURE_DIM];
        for t in 0..frames {
            let hand = straight_hand(x0 + t as f32 * 0.01, 0.5, 1.0);
            let offset = t * POSE_FEATURE_DIM + if right { HAND_SLOT_DIM } else { 0 };
            out[offset..offset + HAND_SLOT_DIM].copy_from_slice(&hand);
        }
        out
    }

    // [TEST向き] 次元の取り決めが崩れていないこと(モデルの入力次元が直接ここに依存する)
    #[test]
    fn dimensions_are_consistent() {
        assert_eq!(HAND_BLOCK_DIM, GEOMETRY_DIM + 3);
        assert_eq!(HANDSHAPE_FEATURE_DIM, 66);
        assert_eq!(InputFeatures::Raw.feature_dim(), POSE_FEATURE_DIM);
        assert_eq!(
            InputFeatures::Handshape.feature_dim(),
            HANDSHAPE_FEATURE_DIM
        );
        assert_eq!(
            InputFeatures::HandshapeMean.feature_dim(),
            HANDSHAPE_FEATURE_DIM
        );
    }

    // [TEST向き] name() と parse() が往復すること + serde が同じ表記を使うこと
    #[test]
    fn name_parse_and_serde_roundtrip() {
        for mode in [
            InputFeatures::Raw,
            InputFeatures::Handshape,
            InputFeatures::HandshapeMean,
        ] {
            assert_eq!(InputFeatures::parse(mode.name()).unwrap(), mode);
            let json = serde_json::to_string(&mode).unwrap();
            assert_eq!(json, format!("\"{}\"", mode.name()));
            assert_eq!(
                serde_json::from_str::<InputFeatures>(&json).unwrap(),
                mode
            );
        }
        assert_eq!(InputFeatures::default(), InputFeatures::Raw);
    }

    // [TEST向き] CLI 文字列の解決。未知の値はエラーになること
    #[test]
    fn parse_input_features() {
        assert_eq!(InputFeatures::parse("raw").unwrap(), InputFeatures::Raw);
        assert_eq!(
            InputFeatures::parse("handshape").unwrap(),
            InputFeatures::Handshape
        );
        assert_eq!(
            InputFeatures::parse("handshape-mean").unwrap(),
            InputFeatures::HandshapeMean
        );
        assert!(InputFeatures::parse("angles").is_err());
    }

    // [TEST向き] 出力の形。frames は保たれ、1 フレーム 66 次元になること
    #[test]
    fn output_shape() {
        let frames = 10;
        let take = take_with_one_hand(frames, true, 0.4);
        let out = handshape_sequence(&take, frames, InputFeatures::Handshape);
        assert_eq!(out.len(), frames * HANDSHAPE_FEATURE_DIM);
    }

    // [TEST向き] 幾何の妥当性: まっすぐ伸ばした指の PIP/DIP 関節角は π に近いこと
    #[test]
    fn straight_finger_has_flat_joint_angles() {
        let frames = 1;
        let take = take_with_one_hand(frames, true, 0.4);
        let out = handshape_sequence(&take, frames, InputFeatures::Handshape);
        // 主たる手ブロックの先頭 15 個が関節角。各指の 2 番目・3 番目(PIP/DIP)は
        // 完全に一直線なので π。1 番目(MCP)は手首→MCP と MCP→PIP の角で、
        // 指ごとに x をずらしている分 π からずれる
        for finger in 0..5 {
            for j in 1..3 {
                let angle = out[finger * 3 + j];
                assert!(
                    (angle - std::f32::consts::PI).abs() < 1e-3,
                    "指{} 関節{} の角度が π から離れています: {}",
                    finger,
                    j,
                    angle
                );
            }
        }
    }

    // [TEST向き] 左右反転不変性。これがミラー拡張を記述子空間で行わない根拠になる
    #[test]
    fn mirror_invariance() {
        let frames = 6;
        // 右手枠だけのテイクと、それを左右反転したテイク(= 左手枠だけになる)
        let take = take_with_one_hand(frames, true, 0.4);
        let mirrored = mirror_features(&take);

        for mode in [InputFeatures::Handshape, InputFeatures::HandshapeMean] {
            let a = handshape_sequence(&take, frames, mode);
            let b = handshape_sequence(&mirrored, frames, mode);
            assert_eq!(a.len(), b.len());
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (x - y).abs() < 1e-4,
                    "{:?}: 次元{} が左右反転で変わりました: {} vs {}",
                    mode,
                    i,
                    x,
                    y
                );
            }
        }
    }

    // [TEST向き] 主たる手が常に前半ブロックへ来ること(左手だけのテイクでも)
    #[test]
    fn dominant_hand_goes_to_first_block() {
        let frames = 4;
        let left_only = take_with_one_hand(frames, false, 0.4);
        let out = handshape_sequence(&left_only, frames, InputFeatures::Handshape);
        // 前半ブロック(主たる手)は非ゼロ、後半ブロック(非主たる手)は全ゼロ
        let first: f32 = out[0..HAND_BLOCK_DIM].iter().map(|v| v.abs()).sum();
        let second: f32 = out[HAND_BLOCK_DIM..HANDSHAPE_FEATURE_DIM]
            .iter()
            .map(|v| v.abs())
            .sum();
        assert!(first > 0.0, "主たる手ブロックが空です");
        assert_eq!(second, 0.0, "非主たる手ブロックが埋まっています");
        // 検出フラグは 1.0
        assert_eq!(out[GEOMETRY_DIM + 2], 1.0);
    }

    // [TEST向き] 手が 1 フレームも検出されないテイク → 全ゼロ(仕様として明示)
    #[test]
    fn take_without_any_hand_is_all_zero() {
        let frames = 3;
        let empty = vec![0.0f32; frames * POSE_FEATURE_DIM];
        for mode in [InputFeatures::Handshape, InputFeatures::HandshapeMean] {
            let out = handshape_sequence(&empty, frames, mode);
            assert_eq!(out.len(), frames * HANDSHAPE_FEATURE_DIM);
            assert!(out.iter().all(|&v| v == 0.0), "{:?} で全ゼロになりません", mode);
        }
    }

    // [TEST向き] 未検出フレームはテイク平均で補完され、検出フラグだけ 0 になること
    #[test]
    fn undetected_frame_is_imputed_with_take_mean() {
        let frames = 4;
        let mut take = take_with_one_hand(frames, true, 0.4);
        // フレーム 2 の右手枠を消す(未検出にする)
        let offset = 2 * POSE_FEATURE_DIM + HAND_SLOT_DIM;
        for v in take[offset..offset + HAND_SLOT_DIM].iter_mut() {
            *v = 0.0;
        }

        let out = handshape_sequence(&take, frames, InputFeatures::Handshape);
        let block = |t: usize| &out[t * HANDSHAPE_FEATURE_DIM..t * HANDSHAPE_FEATURE_DIM + HAND_BLOCK_DIM];

        // 検出フラグ: フレーム2 だけ 0、他は 1
        assert_eq!(block(2)[GEOMETRY_DIM + 2], 0.0);
        assert_eq!(block(1)[GEOMETRY_DIM + 2], 1.0);
        // 幾何部分はゼロ埋めではなく平均で埋まっている(π 近傍の関節角が残る)
        assert!(
            block(2)[1] > 3.0,
            "未検出フレームがゼロ埋めされています: {}",
            block(2)[1]
        );
        // 手首 x はフレームごとに動かしているので、平均値は他フレームと一致しない
        assert!(block(2)[GEOMETRY_DIM] > 0.0);
    }

    // [TEST向き] 成分の指定と次元数
    #[test]
    fn descriptor_parts_parse_and_dim() {
        assert_eq!(DescriptorParts::parse_drop("").unwrap(), DescriptorParts::all());
        assert_eq!(DescriptorParts::all().feature_dim(), HANDSHAPE_FEATURE_DIM);

        let no_wrist = DescriptorParts::parse_drop("wrist").unwrap();
        assert!(!no_wrist.wrist_xy);
        assert_eq!(no_wrist.feature_dim(), (33 - 2) * 2); // 62

        let no_off = DescriptorParts::parse_drop("offhand").unwrap();
        assert_eq!(no_off.feature_dim(), 33);

        let both = DescriptorParts::parse_drop("wrist,offhand").unwrap();
        assert_eq!(both.feature_dim(), 31);
        assert_eq!(both.dropped_names(), "wrist,offhand");

        assert!(DescriptorParts::parse_drop("bogus").is_err());
        // 主たる手の成分を全部落とすのは禁止
        assert!(DescriptorParts::parse_drop("angles,tipwrist,tippairs,wrist").is_err());
    }

    // [TEST向き] 成分を落としても、残る次元の値は落とさない場合と完全に一致すること
    #[test]
    fn dropping_parts_keeps_remaining_values_identical() {
        let frames = 5;
        let take = take_with_one_hand(frames, true, 0.4);
        let full = handshape_sequence(&take, frames, InputFeatures::Handshape);

        let parts = DescriptorParts::parse_drop("wrist").unwrap();
        let reduced = handshape_sequence_with_parts(&take, frames, InputFeatures::Handshape, parts);
        let kept = parts.kept_indices();
        assert_eq!(reduced.len(), frames * kept.len());
        for t in 0..frames {
            for (i, &j) in kept.iter().enumerate() {
                assert_eq!(
                    reduced[t * kept.len() + i],
                    full[t * HANDSHAPE_FEATURE_DIM + j],
                    "フレーム{} 次元{} が一致しません",
                    t,
                    j
                );
            }
        }
    }

    // [TEST向き] TakeMean は全フレーム同じ値になり、フラグにカバレッジが入ること
    #[test]
    fn take_mean_broadcasts_identical_frames() {
        let frames = 4;
        let mut take = take_with_one_hand(frames, true, 0.4);
        let offset = 1 * POSE_FEATURE_DIM + HAND_SLOT_DIM;
        for v in take[offset..offset + HAND_SLOT_DIM].iter_mut() {
            *v = 0.0;
        }

        let out = handshape_sequence(&take, frames, InputFeatures::HandshapeMean);
        let first = &out[0..HANDSHAPE_FEATURE_DIM];
        for t in 1..frames {
            let cur = &out[t * HANDSHAPE_FEATURE_DIM..(t + 1) * HANDSHAPE_FEATURE_DIM];
            assert_eq!(first, cur, "フレーム{} が先頭と違います", t);
        }
        // 4 フレーム中 3 フレームで検出 → カバレッジ 0.75
        assert!((first[GEOMETRY_DIM + 2] - 0.75).abs() < 1e-6);
    }
}
