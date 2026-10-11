//! 検出範囲(CenterSquare の枠、エンジンの `preview.crop`)の縁に手の点が近づいたかの判定(純関数)。
//!
//! 背景(CLAUDE.md 2026-08-22 実験5): CenterSquare の範囲は余裕がほぼ無く、手の 21 点のどれかが範囲外に
//! 出る割合は 3.7%。方向・移動・指示を表す語(iku / are-sore / ni など)で、利き手を外側へ伸ばしたときに切れやすい。
//! 撮影中にそれを見えるようにするため、縁から枠幅の `EDGE_MARGIN_RATIO` 以内(または枠の外)に
//! 点が 1 つでもあれば警告にする。
//!
//! 座標系: `preview.hands` と `preview.crop` はどちらも元フレーム(1620x1080)に対する正規化座標
//! (x は幅・y は高さで割った値)で、反転していない。判定は画面に見えている向き(鏡像)で行い、
//! 「画面の右の縁」のように、利用者が見ている向きで知らせる。

use crate::overlay::{mirror_crop, mirror_x};
use crate::protocol::{CropRect, HandsPoints};

/// 縁からの距離がこの割合(枠の幅・高さに対する比)以内なら「縁に近い」。
/// CenterSquare は画素で正方形なので、縦横とも同じ画素数の帯になる
pub const EDGE_MARGIN_RATIO: f64 = 0.03;

/// どの縁に近いか(画面に見えている鏡像の向き)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EdgeStatus {
    pub left: bool,
    pub right: bool,
    pub top: bool,
    pub bottom: bool,
}

impl EdgeStatus {
    pub fn any(&self) -> bool {
        self.left || self.right || self.top || self.bottom
    }

    /// 画面に出す文("画面の右の縁に手が近づいています")。近くなければ None
    pub fn message(&self) -> Option<String> {
        let mut sides = Vec::new();
        if self.left {
            sides.push("左");
        }
        if self.right {
            sides.push("右");
        }
        if self.top {
            sides.push("上");
        }
        if self.bottom {
            sides.push("下");
        }
        if sides.is_empty() {
            None
        } else {
            Some(format!("検出範囲の{}の縁に手が近づいています", sides.join("・")))
        }
    }
}

/// 1 点の判定(鏡像にした座標と枠で)。枠の外の点も「近い」に含める
fn point_edges(x: f64, y: f64, c: CropRect, st: &mut EdgeStatus) {
    if !x.is_finite() || !y.is_finite() {
        return;
    }
    let mx = EDGE_MARGIN_RATIO * c.w;
    let my = EDGE_MARGIN_RATIO * c.h;
    if x < c.x + mx {
        st.left = true;
    }
    if x > c.x + c.w - mx {
        st.right = true;
    }
    if y < c.y + my {
        st.top = true;
    }
    if y > c.y + c.h - my {
        st.bottom = true;
    }
}

/// 検出された手(左右とも)の全点について、検出範囲の縁に近いかを調べる。
/// 手が検出されていなければ「近くない」
pub fn edge_proximity(hands: &HandsPoints, crop: CropRect) -> EdgeStatus {
    let mut st = EdgeStatus::default();
    if !(crop.w > 0.0 && crop.h > 0.0) {
        return st;
    }
    let c = mirror_crop(crop);
    for pts in [&hands.left, &hands.right].into_iter().flatten() {
        for p in pts {
            point_edges(mirror_x(p[0]), p[1], c, &mut st);
        }
    }
    st
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1620x1080 の中央正方(エンジンの crop と同じ値)
    const CENTER: CropRect = CropRect { x: 0.1667, y: 0.0, w: 0.6667, h: 1.0 };

    fn hand_at(x: f64, y: f64) -> Vec<[f64; 2]> {
        vec![[x, y]; 21]
    }

    #[test]
    fn hand_in_the_middle_is_not_near() {
        let h = HandsPoints { left: None, right: Some(hand_at(0.5, 0.5)) };
        let st = edge_proximity(&h, CENTER);
        assert!(!st.any());
        assert_eq!(st.message(), None);
    }

    #[test]
    fn no_hands_is_not_near() {
        assert!(!edge_proximity(&HandsPoints::default(), CENTER).any());
    }

    #[test]
    fn margin_boundary_and_mirroring() {
        // 元の座標で枠の左端(x=0.1667)から 3% 以内 → 画面(鏡像)では右の縁
        let margin = EDGE_MARGIN_RATIO * CENTER.w; // 0.02
        let inside_band = CENTER.x + margin * 0.5;
        let h = HandsPoints { left: None, right: Some(hand_at(inside_band, 0.5)) };
        let st = edge_proximity(&h, CENTER);
        assert_eq!(st, EdgeStatus { right: true, ..Default::default() });
        assert_eq!(st.message().unwrap(), "検出範囲の右の縁に手が近づいています");

        // 帯のすぐ内側(3% より少し内)なら近くない
        let just_inside = CENTER.x + margin * 1.5;
        let h = HandsPoints { left: None, right: Some(hand_at(just_inside, 0.5)) };
        assert!(!edge_proximity(&h, CENTER).any());

        // 元の座標で右端付近 → 画面では左の縁
        let h = HandsPoints { left: Some(hand_at(CENTER.x + CENTER.w - margin * 0.5, 0.5)), right: None };
        assert_eq!(edge_proximity(&h, CENTER), EdgeStatus { left: true, ..Default::default() });
    }

    #[test]
    fn outside_the_crop_counts_and_one_point_is_enough() {
        // 枠の外(元の座標で x=0.05)に指先が 1 点だけ出ている
        let mut pts = hand_at(0.4, 0.5);
        pts[8] = [0.05, 0.5];
        let h = HandsPoints { left: None, right: Some(pts) };
        assert!(edge_proximity(&h, CENTER).right);
    }

    #[test]
    fn top_and_bottom_edges() {
        let h = HandsPoints { left: Some(hand_at(0.5, 0.985)), right: Some(hand_at(0.5, 0.01)) };
        let st = edge_proximity(&h, CENTER);
        assert!(st.top && st.bottom && !st.left && !st.right);
        assert_eq!(st.message().unwrap(), "検出範囲の上・下の縁に手が近づいています");
    }

    #[test]
    fn degenerate_crop_and_non_finite_points_are_ignored() {
        let h = HandsPoints { left: None, right: Some(hand_at(f64::NAN, 0.5)) };
        assert!(!edge_proximity(&h, CENTER).any());
        let zero = CropRect { x: 0.0, y: 0.0, w: 0.0, h: 0.0 };
        assert!(!edge_proximity(&HandsPoints { left: Some(hand_at(0.0, 0.0)), right: None }, zero).any());
    }
}
