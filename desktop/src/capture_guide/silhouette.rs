//! 立ち位置ガイド: 鏡像プレビューの上に、頭と肩の輪郭の半透明の枠を描く。
//!
//! # 枠の基準位置(定数)の出典
//!
//! 既存 153 テイクの人物の鼻・両肩の位置の中央値。
//! - 元データ: `transformer_burn/data/pose_dict_full_center_body.json`(gitignore 対象)の各テイクの
//!   `body_sequence`(1 フレーム 8 次元 = [鼻x, 鼻y, 左肩x, 左肩y, 右肩x, 右肩y, 有効, aspect])。
//!   x は幅・y は高さで割った、**元フレーム全体(1620x1080)に対する正規化座標**(metadata の `body_features`)
//! - 対象: 有効(8 次元目の 7 番目)> 0.5 のフレームだけ。153 テイク × 10 フレーム = 1530 フレームのうち
//!   1530 フレームすべてが有効(全テイクで aspect = 1.5 = 1620/1080)
//! - 計算方法: 全有効フレームをひとまとめにして(テイクごとの代表値は取らずに)、各量の中央値
//!   - 肩中心 = ((左肩x + 右肩x)/2, (左肩y + 右肩y)/2)
//!   - 肩幅 = |左肩x − 右肩x|(幅で正規化した値)
//!   - 鼻 = (鼻x, 鼻y)
//! - 再計算: このファイルの `#[ignore]` テスト `guide_constants_match_body_dict` が同じ計算をして、
//!   下の定数と一致するか確かめる(実行方法はテストのコメント)
//!
//! # 座標系(overlay.rs と同じ)
//!
//! エンジンのプレビュー画像は、元フレーム全体を縦横比を保って縮小したもの(json_engine.rs の
//! `preview_size` / `downscale_rgb`)で、手の点(`preview.hands`)も検出範囲(`preview.crop`)も
//! 元フレームに対する正規化座標。`body_sequence` も同じ正規化なので、プレビュー画像の幅・高さを掛ければ
//! そのまま画素になる。鏡像表示なので x は `mirror_x`(1 − x)で反転してから描く。
//!
//! # 枠の形(目安。厳密さは不要)
//!
//! 頭は楕円、首から肩、肩から下は画面の下端へ向かう線。頭の大きさ・首の長さは人体の一般的な比率から
//! 決めた見た目の調整値で、データから測ったものではない(下の `HEAD_*` / `NECK_*` 定数)。

use image::RgbaImage;

// ===== データから決めた定数(出典は冒頭)=====

/// 肩中心の x(元フレームの幅で正規化、反転前)。中央値、n = 1530 フレーム / 153 テイク
pub const SHOULDER_CENTER_X: f64 = 0.5136;
/// 肩中心の y(高さで正規化)。中央値、n = 1530
pub const SHOULDER_CENTER_Y: f64 = 0.7354;
/// 肩幅(幅で正規化)。中央値、n = 1530
pub const SHOULDER_WIDTH: f64 = 0.3064;
/// 鼻の x(幅で正規化、反転前)。中央値、n = 1530
pub const NOSE_X: f64 = 0.5112;
/// 鼻の y(高さで正規化)。中央値、n = 1530
pub const NOSE_Y: f64 = 0.4558;

// ===== 見た目の調整値(データ由来ではない)=====

/// 頭の幅 = 肩幅 × この値。最初は一般的な比率(肩峰間の幅は顔の幅のおよそ 2.4 倍)から 0.42 にしたが、
/// 実映像(iku の 02.mp4)に重ねて目視したら顔だけを囲み髪を含む頭より小さかったので、0.55 に広げた。
/// 下の首・肩の係数も同じ目視で、実際の肩の輪郭に近づくよう調整した(どれも目安で、測定値ではない)
const HEAD_WIDTH_PER_SHOULDER: f64 = 0.55;
/// 頭の高さ = 頭の幅 × この値
const HEAD_HEIGHT_PER_WIDTH: f64 = 1.3;
/// 楕円の中心は鼻より少し上(鼻先は頭のてっぺんから約 6 割の位置)。頭の高さに対する比
const HEAD_CENTER_ABOVE_NOSE: f64 = 0.12;

/// 枠の色(白)と不透明度。映像が透けて見えるように半透明
const GUIDE_COLOR: [u8; 3] = [255, 255, 255];
const GUIDE_ALPHA: f32 = 0.6;

/// ガイド枠の形(鏡像にした画素座標)
#[derive(Debug, Clone, PartialEq)]
pub struct GuideShape {
    /// 頭の楕円の中心
    pub head_center: (f32, f32),
    /// 楕円の横・縦の半径
    pub head_radius: (f32, f32),
    /// 左側の輪郭(首 → 肩 → 下端)。画面の左に見える側
    pub left_outline: Vec<(f32, f32)>,
    /// 右側の輪郭
    pub right_outline: Vec<(f32, f32)>,
    /// 肩中心(鏡像)。テスト・診断用
    pub shoulder_center: (f32, f32),
}

/// 画像の大きさ(画素)から、鏡像のガイド枠の形を作る(純関数)
pub fn guide_shape(width: u32, height: u32) -> GuideShape {
    let (w, h) = (width as f64, height as f64);
    // 画素へ(x は鏡像)
    let to_px = |x: f64, y: f64| (((1.0 - x) * w) as f32, (y * h) as f32);

    let shoulder_w_px = SHOULDER_WIDTH * w;
    let head_w = HEAD_WIDTH_PER_SHOULDER * shoulder_w_px;
    let head_h = HEAD_HEIGHT_PER_WIDTH * head_w;
    let (nose_px_x, nose_px_y) = to_px(NOSE_X, NOSE_Y);
    let head_cy = nose_px_y as f64 - HEAD_CENTER_ABOVE_NOSE * head_h;
    let chin_y = head_cy + head_h / 2.0;

    let (scx, scy) = to_px(SHOULDER_CENTER_X, SHOULDER_CENTER_Y);
    let (scx, scy) = (scx as f64, scy as f64);
    let rise = (scy - chin_y).max(0.0); // あごから肩の高さまで

    // 片側の輪郭。sign = -1 で画面の左側、+1 で右側
    let side = |sign: f64| -> Vec<(f32, f32)> {
        let neck_x = nose_px_x as f64 + sign * 0.24 * head_w;
        vec![
            // 首の上端(あごの少し上)
            (neck_x, chin_y - 0.10 * head_h),
            // 首の付け根
            (nose_px_x as f64 + sign * 0.28 * head_w, scy - 0.45 * rise),
            // 肩の上(僧帽筋の斜面)
            (scx + sign * 0.46 * shoulder_w_px, scy - 0.12 * rise),
            // 肩の外側(肩の点は関節の中心なので、輪郭はそれより外)
            (scx + sign * 0.66 * shoulder_w_px, scy + 0.10 * shoulder_w_px),
            // 腕に沿って画面の下端へ
            (scx + sign * 0.70 * shoulder_w_px, h),
        ]
        .into_iter()
        .map(|(x, y)| (x as f32, y as f32))
        .collect()
    };

    GuideShape {
        head_center: (nose_px_x, head_cy as f32),
        head_radius: ((head_w / 2.0) as f32, (head_h / 2.0) as f32),
        left_outline: side(-1.0),
        right_outline: side(1.0),
        shoulder_center: (scx as f32, scy as f32),
    }
}

/// 半透明で 1 画素を塗る(範囲外は何もしない)
fn blend(img: &mut RgbaImage, x: i32, y: i32, color: [u8; 3], alpha: f32) {
    if x < 0 || y < 0 || x as u32 >= img.width() || y as u32 >= img.height() {
        return;
    }
    let p = img.get_pixel_mut(x as u32, y as u32);
    for (i, c) in color.iter().enumerate() {
        p[i] = (p[i] as f32 * (1.0 - alpha) + *c as f32 * alpha).round() as u8;
    }
    p[3] = 255;
}

/// 太さ `thick` の点(正方形)の画素を集める。半透明は同じ画素を重ねて塗ると濃くなるので、
/// 線の画素をいったん集合に集めてから 1 回ずつ塗る
fn stamp(set: &mut std::collections::HashSet<(i32, i32)>, x: f32, y: f32, thick: i32) {
    if !x.is_finite() || !y.is_finite() {
        return;
    }
    let r = thick / 2;
    let (cx, cy) = (x.round() as i32, y.round() as i32);
    for oy in -r..=r {
        for ox in -r..=r {
            set.insert((cx + ox, cy + oy));
        }
    }
}

fn line_pixels(set: &mut std::collections::HashSet<(i32, i32)>, a: (f32, f32), b: (f32, f32), thick: i32, lim: f32) {
    if [a.0, a.1, b.0, b.1].iter().any(|v| !v.is_finite() || v.abs() > lim) {
        return;
    }
    let steps = (b.0 - a.0).abs().max((b.1 - a.1).abs()).ceil().max(1.0) as i32;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        stamp(set, a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t, thick);
    }
}

/// 線の太さ(画素)。プレビューの幅 640 で 3 画素
pub fn guide_thickness(width: u32) -> i32 {
    ((width as f32 / 213.0).round() as i32).max(2)
}

/// 鏡像にした画像にガイド枠を描く
pub fn draw_guide(img: &mut RgbaImage) {
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return;
    }
    let shape = guide_shape(w, h);
    let thick = guide_thickness(w);
    let lim = 4.0 * w.max(h) as f32;
    let mut set = std::collections::HashSet::new();

    // 頭の楕円: 周を細かく区切った折れ線
    let (hx, hy) = shape.head_center;
    let (rx, ry) = shape.head_radius;
    let n = 96;
    let pt = |k: i32| {
        let a = k as f32 / n as f32 * std::f32::consts::TAU;
        (hx + rx * a.cos(), hy + ry * a.sin())
    };
    for k in 0..n {
        line_pixels(&mut set, pt(k), pt(k + 1), thick, lim);
    }
    for outline in [&shape.left_outline, &shape.right_outline] {
        for seg in outline.windows(2) {
            line_pixels(&mut set, seg[0], seg[1], thick, lim);
        }
    }
    for (x, y) in set {
        blend(img, x, y, GUIDE_COLOR, GUIDE_ALPHA);
    }
}

/// 画面の説明に使う一文
pub const GUIDE_LEGEND: &str = "白い枠: 既存の撮影データでの頭と肩のだいたいの位置(目安)";

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn shape_is_mirrored_and_scaled_to_the_image() {
        // 1620x1080 そのままの画素で: 肩中心 x = (1 - 0.5136) * 1620 = 787.97、y = 0.7354 * 1080 = 794.2
        let s = guide_shape(1620, 1080);
        assert!((s.shoulder_center.0 - 787.97).abs() < 0.1, "{:?}", s.shoulder_center);
        assert!((s.shoulder_center.1 - 794.23).abs() < 0.1);
        // 頭の中心 x は鼻 x の鏡像 = (1 - 0.5112) * 1620 = 791.86
        assert!((s.head_center.0 - 791.86).abs() < 0.1);
        // 頭は肩より上、あご(中心 + 縦半径)も肩より上
        assert!(s.head_center.1 + s.head_radius.1 < s.shoulder_center.1);
        // プレビュー(640x427)では同じ割合の位置になる
        let p = guide_shape(640, 427);
        assert!((p.shoulder_center.0 / 640.0 - s.shoulder_center.0 / 1620.0).abs() < 1e-3);
        assert!((p.shoulder_center.1 / 427.0 - s.shoulder_center.1 / 1080.0).abs() < 1e-3);
    }

    #[test]
    fn outlines_are_left_right_symmetric_around_the_body_and_reach_the_bottom() {
        let s = guide_shape(640, 427);
        assert_eq!(s.left_outline.len(), s.right_outline.len());
        // 画面の左側の輪郭は肩中心より左、右側は右
        for (l, r) in s.left_outline.iter().zip(&s.right_outline) {
            assert!(l.0 < r.0, "{l:?} {r:?}");
            assert!((l.1 - r.1).abs() < 1e-3, "同じ高さ");
        }
        // 最後の点は画面の下端
        assert_eq!(s.left_outline.last().unwrap().1, 427.0);
        // 肩の外側の点どうしの間隔は肩幅より広い(輪郭は関節より外)
        let lo = s.left_outline[3];
        let ro = s.right_outline[3];
        assert!(ro.0 - lo.0 > (SHOULDER_WIDTH * 640.0) as f32);
    }

    #[test]
    fn mirroring_moves_a_right_leaning_person_to_the_left() {
        // 反転前の肩中心 x は 0.5136(中央より少し右)→ 鏡像では中央より少し左
        let s = guide_shape(1000, 1000);
        assert!(s.shoulder_center.0 < 500.0);
    }

    #[test]
    fn draw_guide_blends_without_hiding_the_video() {
        let mut img = RgbaImage::from_pixel(640, 427, Rgba([0, 0, 0, 255]));
        draw_guide(&mut img);
        let s = guide_shape(640, 427);
        // 頭の楕円の左端(中心 − 横半径)の画素は、白と黒の間(半透明)
        let x = (s.head_center.0 - s.head_radius.0).round() as u32;
        let y = s.head_center.1.round() as u32;
        let p = img.get_pixel(x, y);
        assert!(p[0] > 100 && p[0] < 200, "半透明のはず: {p:?}");
        // 枠から離れた所(左上隅)は元のまま
        assert_eq!(*img.get_pixel(2, 2), Rgba([0, 0, 0, 255]));
        // 線が重なる所でも濃さは同じ(二重に塗らない)
        let painted: Vec<u8> = img.pixels().filter(|p| p[0] > 0).map(|p| p[0]).collect();
        assert!(!painted.is_empty());
        assert!(painted.iter().all(|&v| v == painted[0]), "濃さが一定でない");
    }

    #[test]
    fn thickness_scales_with_width() {
        assert_eq!(guide_thickness(640), 3);
        assert_eq!(guide_thickness(100), 2);
        assert!(guide_thickness(1620) > 3);
    }

    /// ガイド定数の再計算(手動実行)。元データは gitignore 対象なので、既定のパス
    /// (リポジトリ直下の transformer_burn/data/pose_dict_full_center_body.json)か、
    /// 環境変数 ATTENTION_BODY_DICT で場所を渡す。例(メインのチェックアウトのデータを使う):
    ///   ATTENTION_BODY_DICT=/path/to/attention/transformer_burn/data/pose_dict_full_center_body.json \
    ///     cargo test --release guide_constants_match_body_dict -- --ignored --nocapture
    /// ファイルを読むだけで、何も書かない
    #[test]
    #[ignore]
    fn guide_constants_match_body_dict() {
        let path = std::env::var("ATTENTION_BODY_DICT").map(std::path::PathBuf::from).unwrap_or_else(|_| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("transformer_burn/data/pose_dict_full_center_body.json")
        });
        let Ok(text) = std::fs::read_to_string(&path) else {
            return eprintln!("skip: {} が読めない", path.display());
        };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let tags = v["tags"].as_object().unwrap();
        let (mut cx, mut cy, mut sw, mut nx, mut ny) = (vec![], vec![], vec![], vec![], vec![]);
        let (mut total, mut takes_with_valid) = (0usize, 0usize);
        for t in tags.values() {
            let mut any = false;
            for f in t["body_sequence"].as_array().unwrap() {
                let f: Vec<f64> = f.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
                total += 1;
                if f[6] > 0.5 {
                    any = true;
                    cx.push((f[2] + f[4]) / 2.0);
                    cy.push((f[3] + f[5]) / 2.0);
                    sw.push((f[2] - f[4]).abs());
                    nx.push(f[0]);
                    ny.push(f[1]);
                }
            }
            takes_with_valid += any as usize;
        }
        fn median(v: &mut [f64]) -> f64 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = v.len();
            if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
        }
        let got = [
            ("SHOULDER_CENTER_X", median(&mut cx), SHOULDER_CENTER_X),
            ("SHOULDER_CENTER_Y", median(&mut cy), SHOULDER_CENTER_Y),
            ("SHOULDER_WIDTH", median(&mut sw), SHOULDER_WIDTH),
            ("NOSE_X", median(&mut nx), NOSE_X),
            ("NOSE_Y", median(&mut ny), NOSE_Y),
        ];
        eprintln!(
            "テイク {} 本(有効フレームのあるもの {} 本)、有効フレーム {} / {}",
            tags.len(),
            takes_with_valid,
            cx.len(),
            total
        );
        for (name, m, c) in got {
            eprintln!("{name}: 中央値 {m:.6} / 定数 {c}");
            assert!((m - c).abs() < 0.00051, "{name} が定数とずれている: {m} vs {c}");
        }
    }
}
