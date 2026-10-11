//! プレビュー画像の鏡像化と、手の骨格・検出範囲の重ね描き(すべて CPU で RGBA バッファに描く)。
//!
//! 鏡のように表示するため、エンジンが送ってくる反転していない映像と座標のうち、
//! 表示するものだけを左右反転する(認識に使う映像はエンジン側にあり、ここでは触れない)。
//!   - 画像: 水平反転
//!   - 点の座標: x を 1 - x に
//!   - 検出範囲の枠: x を 1 - (x + w) に(右端が新しい左端になる)
//!
//! GPUI の描画 API(paint_path など)ではなくバッファに直接描くのは、実装が単純で
//! 画像と重ね描きの位置ずれが原理的に起きないため。

use crate::protocol::{CropRect, HandsPoints};
use base64::Engine as _;
use image::{imageops, Rgba, RgbaImage};

/// MediaPipe Hands の 21 点を結ぶ 21 本のボーン(pose_extractor/src/main.rs の `HAND_CONNECTIONS` と同じ)
pub const HAND_CONNECTIONS: &[(usize, usize)] = &[
    // 親指
    (0, 1), (1, 2), (2, 3), (3, 4),
    // 人差し指
    (0, 5), (5, 6), (6, 7), (7, 8),
    // 中指
    (5, 9), (9, 10), (10, 11), (11, 12),
    // 薬指
    (9, 13), (13, 14), (14, 15), (15, 16),
    // 小指
    (13, 17), (17, 18), (18, 19), (19, 20),
    // 手のひら基部
    (0, 17),
];

/// 左手枠・右手枠の色(暗い映像でも明るい映像でも見えるよう彩度の高い色)
const LEFT_COLOR: Rgba<u8> = Rgba([0, 229, 255, 255]); // シアン
const RIGHT_COLOR: Rgba<u8> = Rgba([255, 171, 0, 255]); // 橙
const CROP_COLOR: Rgba<u8> = Rgba([255, 235, 59, 255]); // 黄
const DOT_CORE: Rgba<u8> = Rgba([255, 255, 255, 255]);

/// 座標の x を鏡像にする
pub fn mirror_x(x: f64) -> f64 {
    1.0 - x
}

/// 検出範囲の枠の鏡像(左端 = 1 - (x + w))
pub fn mirror_crop(c: CropRect) -> CropRect {
    CropRect {
        x: 1.0 - (c.x + c.w),
        ..c
    }
}

/// 正規化座標 → 画素座標(小数)。範囲外もそのまま返す(描画側で切り取る)
pub fn to_pixel(p: [f64; 2], width: u32, height: u32) -> (f32, f32) {
    ((p[0] * width as f64) as f32, (p[1] * height as f64) as f32)
}

/// 1 つの手の線分(ボーン)を、鏡像にした画素座標で返す。21 点に満たなければ空
pub fn hand_segments(points: &[[f64; 2]], width: u32, height: u32) -> Vec<((f32, f32), (f32, f32))> {
    if points.len() < 21 {
        return Vec::new();
    }
    let px: Vec<(f32, f32)> = points
        .iter()
        .map(|p| to_pixel([mirror_x(p[0]), p[1]], width, height))
        .collect();
    HAND_CONNECTIONS.iter().map(|&(a, b)| (px[a], px[b])).collect()
}

/// base64 の JPEG をデコードして水平反転する
pub fn decode_mirrored(jpeg_b64: &str) -> Result<RgbaImage, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(jpeg_b64.trim())
        .map_err(|e| format!("base64 の復号に失敗: {e}"))?;
    let img = image::load_from_memory_with_format(&bytes, image::ImageFormat::Jpeg)
        .map_err(|e| format!("JPEG のデコードに失敗: {e}"))?;
    Ok(imageops::flip_horizontal(&img.to_rgba8()))
}

fn put(img: &mut RgbaImage, x: i32, y: i32, c: Rgba<u8>) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, c);
    }
}

/// 太さ `thick`(画素)の線。端点が画像の外でもよい(範囲外の画素は書かない)
pub fn draw_line(img: &mut RgbaImage, a: (f32, f32), b: (f32, f32), thick: i32, c: Rgba<u8>) {
    // 極端に大きい座標(異常値)で長時間ループしないよう、描画対象を画像の周囲に制限する
    let lim = 4.0 * img.width().max(img.height()) as f32;
    if [a.0, a.1, b.0, b.1].iter().any(|v| !v.is_finite() || v.abs() > lim) {
        return;
    }
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let steps = dx.abs().max(dy.abs()).ceil().max(1.0) as i32;
    let r = (thick - 1).max(0) / 2;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let (x, y) = ((a.0 + dx * t).round() as i32, (a.1 + dy * t).round() as i32);
        for oy in -r..=(r + (thick - 1) % 2) {
            for ox in -r..=(r + (thick - 1) % 2) {
                put(img, x + ox, y + oy, c);
            }
        }
    }
}

/// 中心 (cx, cy)、半径 r の塗りつぶしの円
pub fn draw_dot(img: &mut RgbaImage, center: (f32, f32), r: i32, c: Rgba<u8>) {
    if !center.0.is_finite() || !center.1.is_finite() {
        return;
    }
    let (cx, cy) = (center.0.round() as i32, center.1.round() as i32);
    for oy in -r..=r {
        for ox in -r..=r {
            if ox * ox + oy * oy <= r * r {
                put(img, cx + ox, cy + oy, c);
            }
        }
    }
}

/// 矩形の枠線(画素座標)
pub fn draw_rect(img: &mut RgbaImage, x0: f32, y0: f32, x1: f32, y1: f32, thick: i32, c: Rgba<u8>) {
    draw_line(img, (x0, y0), (x1, y0), thick, c);
    draw_line(img, (x1, y0), (x1, y1), thick, c);
    draw_line(img, (x1, y1), (x0, y1), thick, c);
    draw_line(img, (x0, y1), (x0, y0), thick, c);
}

fn draw_hand(img: &mut RgbaImage, points: &[[f64; 2]], color: Rgba<u8>) {
    let (w, h) = (img.width(), img.height());
    for (a, b) in hand_segments(points, w, h) {
        draw_line(img, a, b, 3, color);
    }
    for p in points {
        let px = to_pixel([mirror_x(p[0]), p[1]], w, h);
        draw_dot(img, px, 4, color);
        draw_dot(img, px, 2, DOT_CORE);
    }
}

/// 反転済みの画像に、検出範囲の枠と手の骨格を描く(ライブ認識の画面。枠は黄・太さ 2)
pub fn draw_overlay(img: &mut RgbaImage, hands: &HandsPoints, crop: CropRect) {
    draw_overlay_with(img, hands, crop, CROP_COLOR, 2);
}

/// `draw_overlay` の、検出範囲の枠の色と太さを指定できる版(撮影ガイドで、縁に近づいたら枠を赤く太くする)
pub fn draw_overlay_with(img: &mut RgbaImage, hands: &HandsPoints, crop: CropRect, crop_color: Rgba<u8>, crop_thick: i32) {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let c = mirror_crop(crop);
    // 全体を指す枠(0,0,1,1)は画像の縁に張り付くので描かない(人物追従・複数タイルのとき)
    let is_full = crop.x <= 0.001 && crop.y <= 0.001 && crop.w >= 0.999 && crop.h >= 0.999;
    if !is_full {
        draw_rect(
            img,
            c.x as f32 * w,
            c.y as f32 * h,
            (c.x + c.w) as f32 * w - 1.0,
            (c.y + c.h) as f32 * h - 1.0,
            crop_thick,
            crop_color,
        );
    }
    if let Some(p) = &hands.left {
        draw_hand(img, p, LEFT_COLOR);
    }
    if let Some(p) = &hands.right {
        draw_hand(img, p, RIGHT_COLOR);
    }
}

/// GPUI の RenderImage は BGRA 順を期待する。RGBA → BGRA(R と B を入れ替える)
pub fn rgba_to_bgra(img: &mut RgbaImage) {
    for px in img.chunks_exact_mut(4) {
        px.swap(0, 2);
    }
}

/// プレビュー 1 枚の処理: デコード → 鏡像 → 重ね描き
pub fn compose_preview(
    jpeg_b64: &str,
    hands: &HandsPoints,
    crop: CropRect,
) -> Result<RgbaImage, String> {
    let mut img = decode_mirrored(jpeg_b64)?;
    draw_overlay(&mut img, hands, crop);
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::jpeg::JpegEncoder;

    #[test]
    fn mirror_x_flips_about_center() {
        assert_eq!(mirror_x(0.0), 1.0);
        assert_eq!(mirror_x(1.0), 0.0);
        assert!((mirror_x(0.25) - 0.75).abs() < 1e-12);
        // 2 回で元に戻る
        assert!((mirror_x(mirror_x(0.3)) - 0.3).abs() < 1e-12);
    }

    #[test]
    fn mirror_crop_keeps_width_and_flips_left_edge() {
        // 1620x1080 の中央正方: x=0.1667, w=0.6667 → 左右対称なので反転しても同じ
        let c = mirror_crop(CropRect { x: 0.1667, y: 0.0, w: 0.6667, h: 1.0 });
        assert!((c.x - (1.0 - 0.1667 - 0.6667)).abs() < 1e-12);
        assert!((c.x - 0.1666).abs() < 1e-3);
        assert_eq!((c.y, c.w, c.h), (0.0, 0.6667, 1.0));
        // 左に寄った枠は右に寄る
        let c = mirror_crop(CropRect { x: 0.0, y: 0.1, w: 0.4, h: 0.5 });
        assert!((c.x - 0.6).abs() < 1e-12);
    }

    #[test]
    fn to_pixel_scales_and_keeps_out_of_range() {
        assert_eq!(to_pixel([0.5, 0.25], 640, 400), (320.0, 100.0));
        assert_eq!(to_pixel([-0.1, 1.2], 100, 100), (-10.0, 120.0));
    }

    fn hand_with_wrist_at(x: f64, y: f64) -> Vec<[f64; 2]> {
        let mut v = vec![[0.5, 0.5]; 21];
        v[0] = [x, y];
        v
    }

    #[test]
    fn hand_segments_are_mirrored_and_complete() {
        let pts = hand_with_wrist_at(0.2, 0.4);
        let segs = hand_segments(&pts, 100, 50);
        assert_eq!(segs.len(), HAND_CONNECTIONS.len());
        assert_eq!(segs.len(), 21);
        // 手首(0)から出る最初のボーン (0,1): 始点は x=1-0.2=0.8 → 80 画素、y=0.4 → 20 画素
        assert_eq!(segs[0].0, (80.0, 20.0));
        // 終点(点 1)は (0.5,0.5) → x=1-0.5=0.5 → 50、y=25
        assert_eq!(segs[0].1, (50.0, 25.0));
        // 21 点に満たなければ描かない
        assert!(hand_segments(&pts[..20], 100, 50).is_empty());
    }

    #[test]
    fn connections_topology() {
        assert_eq!(HAND_CONNECTIONS.len(), 21);
        for &(a, b) in HAND_CONNECTIONS {
            assert!(a < 21 && b < 21 && a != b);
        }
        // 21 点すべてがどれかのボーンに含まれる
        for i in 0..21 {
            assert!(HAND_CONNECTIONS.iter().any(|&(a, b)| a == i || b == i), "点 {i}");
        }
    }

    fn solid_jpeg_b64(w: u32, h: u32, left: [u8; 3], right: [u8; 3]) -> String {
        // 左半分と右半分で色を変えた合成画像(JPEG の色ずれを許容して比較するため大きく違う色にする)
        let mut img = image::RgbImage::new(w, h);
        for (x, _y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb(if x < w / 2 { left } else { right });
        }
        let mut buf = Vec::new();
        JpegEncoder::new_with_quality(&mut buf, 95)
            .encode_image(&img)
            .unwrap();
        base64::engine::general_purpose::STANDARD.encode(buf)
    }

    #[test]
    fn decode_flips_horizontally() {
        // 元: 左が赤、右が青 → 反転後は左が青、右が赤
        let b64 = solid_jpeg_b64(32, 16, [255, 0, 0], [0, 0, 255]);
        let img = decode_mirrored(&b64).unwrap();
        assert_eq!((img.width(), img.height()), (32, 16));
        let l = img.get_pixel(3, 8);
        let r = img.get_pixel(28, 8);
        assert!(l[2] > 200 && l[0] < 60, "左は青のはず: {l:?}");
        assert!(r[0] > 200 && r[2] < 60, "右は赤のはず: {r:?}");
    }

    #[test]
    fn decode_rejects_garbage() {
        assert!(decode_mirrored("!!!not base64!!!").is_err());
        assert!(decode_mirrored("AAAA").is_err()); // base64 としては正しいが JPEG ではない
    }

    #[test]
    fn draw_line_stays_in_bounds_and_handles_outliers() {
        let mut img = RgbaImage::new(20, 20);
        // 端点が画像の外でもパニックしない
        draw_line(&mut img, (-5.0, -5.0), (30.0, 30.0), 3, Rgba([1, 2, 3, 255]));
        assert_eq!(*img.get_pixel(10, 10), Rgba([1, 2, 3, 255]));
        // 非有限・極端な値は何も描かない(長時間ループしない)
        let before = img.clone();
        draw_line(&mut img, (f32::NAN, 0.0), (5.0, 5.0), 2, Rgba([9, 9, 9, 255]));
        draw_line(&mut img, (0.0, 0.0), (1.0e9, 5.0), 2, Rgba([9, 9, 9, 255]));
        assert_eq!(img, before);
    }

    #[test]
    fn draw_overlay_places_dot_at_mirrored_position() {
        let mut img = RgbaImage::from_pixel(100, 100, Rgba([0, 0, 0, 255]));
        let mut pts = vec![[0.5, 0.5]; 21];
        pts[8] = [0.1, 0.2]; // 反転後は x=0.9 → 画素 90、y=20
        let hands = HandsPoints { left: None, right: Some(pts) };
        let full = CropRect { x: 0.0, y: 0.0, w: 1.0, h: 1.0 };
        draw_overlay(&mut img, &hands, full);
        // 点の中心は白(DOT_CORE)
        assert_eq!(*img.get_pixel(90, 20), DOT_CORE);
        // 反転前の位置(10,20)には何も描かれない
        assert_eq!(*img.get_pixel(10, 20), Rgba([0, 0, 0, 255]));
        // 全体の枠は描かない: 左上隅は黒のまま
        assert_eq!(*img.get_pixel(0, 0), Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn draw_overlay_draws_crop_frame_on_mirrored_side() {
        let mut img = RgbaImage::from_pixel(100, 100, Rgba([0, 0, 0, 255]));
        // 左に寄った枠(x 0〜0.4)→ 鏡像では右に寄る(x 0.6〜1.0)
        let crop = CropRect { x: 0.0, y: 0.1, w: 0.4, h: 0.5 };
        draw_overlay(&mut img, &HandsPoints::default(), crop);
        // 鏡像の左端 x=60 の縦線上(y=30)に枠色
        assert_eq!(*img.get_pixel(60, 30), CROP_COLOR);
        // 元の位置(x=0 付近)には描かれない
        assert_eq!(*img.get_pixel(1, 30), Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn bgra_swap() {
        let mut img = RgbaImage::from_pixel(2, 1, Rgba([10, 20, 30, 255]));
        rgba_to_bgra(&mut img);
        assert_eq!(*img.get_pixel(0, 0), Rgba([30, 20, 10, 255]));
    }

    /// 目視確認用(手動実行): エンジンの出力 JSONL(ATTENTION_APP_PREVIEW_JSONL)の preview の N 番目
    /// (ATTENTION_APP_PREVIEW_N、既定 25)に鏡像+重ね描きをして PNG(ATTENTION_APP_PREVIEW_OUT)に保存する。
    /// 実映像が写るので、出力先はスクラッチパッドなどリポジトリ外にすること
    #[test]
    #[ignore]
    fn visual_check_real_preview() {
        let (Ok(src), Ok(out)) = (
            std::env::var("ATTENTION_APP_PREVIEW_JSONL"),
            std::env::var("ATTENTION_APP_PREVIEW_OUT"),
        ) else {
            eprintln!("skip: 環境変数が無い");
            return;
        };
        let n: usize = std::env::var("ATTENTION_APP_PREVIEW_N").ok().and_then(|v| v.parse().ok()).unwrap_or(25);
        let mut k = 0;
        for line in std::fs::read_to_string(src).unwrap().lines() {
            if let Ok(Some(crate::protocol::EngineEvent::Preview { jpeg_b64, hands, crop, .. })) =
                crate::protocol::parse_event(line)
            {
                k += 1;
                if k == n {
                    let img = compose_preview(&jpeg_b64, &hands, crop).unwrap();
                    img.save(&out).unwrap();
                    eprintln!("saved {out} {}x{}", img.width(), img.height());
                    return;
                }
            }
        }
        panic!("preview が {n} 件に満たない");
    }
}
