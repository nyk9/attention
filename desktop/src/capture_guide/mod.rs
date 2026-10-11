//! 撮影ガイドモード(R0、docs/desktop-capture-design.md §4.3〜§4.5・§5)。
//!
//! Photo Booth などで撮影している横で、立ち位置と手の検出を確かめるための画面。
//! **録画はしない・何も保存しない**。このモジュールはファイルを読むだけで、書き込み・作成・削除の API を
//! 呼ばない(テスト `capture_guide_tests.rs` でソースの文字列とディレクトリの中身の両方から固定している)。
//!
//! - `silhouette`: 立ち位置ガイド(頭と肩の半透明の枠)
//! - `hand_run`: 連続して手が検出できている秒数
//! - `edge`: 検出範囲の縁への接近
//! - `progress`: お題(次の語)と進捗の読み取り
//!
//! # 経過時間の基準(なぜ「受信時刻」か)
//!
//! エンジンの `status` / `preview` の JSON には時刻の項目が無い(json_engine.rs の `Event::Status` は
//! state・hands・fps・rec_secs だけ)。`rec_secs` は記録中の秒数で、撮影ガイドでは開始操作を送らない
//! (記録しない)ので 0 のまま使えない。そこで、**読み取りスレッドが status の行を受け取った時刻**を使う。
//! - カメラ入力では、エンジンは届いたフレームを順に処理してすぐ status を出す(最大 10 回/秒)ので、
//!   受信時刻はカメラの時刻とほぼ一定の遅れで進み、間隔の測定に使える
//! - UI スレッドが Inbox を見に行く時刻(約 33ms ごと)ではなく、読み取りスレッドで行を読んだ時刻にするのは、
//!   (1) 見に行く間隔の揺れを入れないため、(2) Inbox の status は最新の 1 件だけを残す作りで、
//!   UI が遅れると途中のサンプルが消え、瞬断の判定(1 サンプル)が狂うため。サンプルは別のキューに全部積む
//! - 動画ファイル入力(開発・検証用)では、エンジンは動画を最速で処理するので、受信時刻は動画の時刻と
//!   一致しない(この Mac では処理 fps が動画の 30fps 前後なので近いが、保証はない)

pub mod edge;
pub mod hand_run;
pub mod progress;
pub mod silhouette;

use crate::overlay;
use crate::protocol::{CropRect, HandsPoints};
use edge::EdgeStatus;
use hand_run::HandRuns;
use image::{Rgba, RgbaImage};
use progress::ProgressView;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// 縁に近いときの検出範囲の枠の色(赤)と太さ
const EDGE_WARN_COLOR: Rgba<u8> = Rgba([255, 40, 40, 255]);
const EDGE_WARN_THICK: i32 = 5;
/// 縁に近くないときの枠(ライブ認識と同じ黄・太さ 2。overlay.rs の CROP_COLOR と同じ値)
const CROP_COLOR: Rgba<u8> = Rgba([255, 235, 59, 255]);
const CROP_THICK: i32 = 2;

/// 進捗を読み直す間隔(撮影中に別のツールが index.tsv や動画を足すので、定期的に読み直す。読むだけ)
pub const PROGRESS_RELOAD_INTERVAL: Duration = Duration::from_secs(3);

/// エンジンの status 1 件ぶんの手の有無と、それを受け取った時刻
#[derive(Debug, Clone, PartialEq)]
pub struct HandSample {
    pub at: Instant,
    /// "L" | "R" | "LR" | "none"
    pub hands: String,
}

/// 撮影ガイド用のプレビュー 1 枚: デコード → 鏡像 → ガイド枠 → 検出範囲の枠(縁に近ければ赤)→ 手の骨格。
/// 縁の判定結果も返す(画面の文字表示に使う)
pub fn compose_guide_preview(
    jpeg_b64: &str,
    hands: &HandsPoints,
    crop: CropRect,
) -> Result<(RgbaImage, EdgeStatus), String> {
    let mut img = overlay::decode_mirrored(jpeg_b64)?;
    let edge = edge::edge_proximity(hands, crop);
    // ガイド枠を先に描き、その上に検出範囲の枠と骨格を描く(骨格が枠に隠れないように)
    silhouette::draw_guide(&mut img);
    let (color, thick) = crop_style(&edge);
    overlay::draw_overlay_with(&mut img, hands, crop, color, thick);
    Ok((img, edge))
}

/// 検出範囲の枠の色と太さ(純関数)。縁に近ければ赤く太く
pub fn crop_style(edge: &EdgeStatus) -> (Rgba<u8>, i32) {
    if edge.any() {
        (EDGE_WARN_COLOR, EDGE_WARN_THICK)
    } else {
        (CROP_COLOR, CROP_THICK)
    }
}

/// 撮影ガイド画面の状態(UI 非依存)
pub struct GuideState {
    /// 秒数の起点(このセッションで最初に受け取ったサンプルの時刻)
    epoch: Option<Instant>,
    pub runs: HandRuns,
    /// 最新の手の有無("L" | "R" | "LR" | "none")。エンジンが動いていなければ None
    pub hands: Option<String>,
    /// 最新のプレビューでの縁の判定
    pub edge: EdgeStatus,
    /// 進捗(読めないときは理由)。まだ読んでいなければ None
    pub progress: Option<Result<ProgressView, String>>,
    pub data_dir: PathBuf,
    /// データディレクトリの決め方と探した場所(エラー文言用)。`new` で作った場合は既定の場所扱い
    pub data_dir_choice: progress::DataDirChoice,
    last_reload: Option<Instant>,
}

impl GuideState {
    #[cfg(test)]
    pub fn new(data_dir: PathBuf) -> Self {
        let choice = progress::DataDirChoice {
            tried: vec![data_dir.clone()],
            path: data_dir,
            origin: progress::DataDirOrigin::Default,
        };
        Self::from_choice(choice)
    }

    pub fn from_choice(choice: progress::DataDirChoice) -> Self {
        Self {
            epoch: None,
            runs: HandRuns::default(),
            hands: None,
            edge: EdgeStatus::default(),
            progress: None,
            data_dir: choice.path.clone(),
            data_dir_choice: choice,
            last_reload: None,
        }
    }

    /// エンジンのセッションが始まる・終わるときに、手の検出の状態を空にする(進捗は残す)
    pub fn reset_session(&mut self) {
        self.epoch = None;
        self.runs = HandRuns::default();
        self.hands = None;
        self.edge = EdgeStatus::default();
    }

    /// status のサンプルを時刻順に入れる
    pub fn push_sample(&mut self, s: &HandSample) {
        let epoch = *self.epoch.get_or_insert(s.at);
        let t = s.at.saturating_duration_since(epoch).as_secs_f64();
        self.runs.push_status(t, &s.hands);
        self.hands = Some(s.hands.clone());
    }

    /// 読み直す時刻か
    pub fn reload_due(&self, now: Instant) -> bool {
        match self.last_reload {
            None => true,
            Some(t) => now.saturating_duration_since(t) >= PROGRESS_RELOAD_INTERVAL,
        }
    }

    /// 進捗を読み直す(読むだけ)
    pub fn reload_progress(&mut self, now: Instant) {
        self.progress = Some(progress::load_progress_for(&self.data_dir_choice));
        self.last_reload = Some(now);
    }
}

/// 手の有無 → 大きく出す短い文字
pub fn hands_big_text(hands: Option<&str>) -> &'static str {
    match hands {
        Some("L") => "左手",
        Some("R") => "右手",
        Some("LR") => "両手",
        Some(_) => "手なし",
        None => "-",
    }
}

/// 画面の左右の向きの補足。エンジンの L/R は特徴量の左手枠/右手枠(手の左右の判定)で、映像の位置ではない。
/// 鏡像表示では、利用者の右手は画面の右側に見える
pub const HANDS_NOTE: &str = "左右は手の判定(利用者から見た左手・右手)。鏡のように表示しているので、右手は画面の右側に見えます";

/// 連続検出秒数の表示文字("1.3 秒")
pub fn run_text(secs: f64) -> String {
    format!("{:.1} 秒", if secs.is_finite() { secs.max(0.0) } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;
    use image::codecs::jpeg::JpegEncoder;

    fn jpeg_b64(w: u32, h: u32) -> String {
        let img = image::RgbImage::from_pixel(w, h, image::Rgb([0, 0, 0]));
        let mut buf = Vec::new();
        JpegEncoder::new_with_quality(&mut buf, 95).encode_image(&img).unwrap();
        base64::engine::general_purpose::STANDARD.encode(buf)
    }

    const CENTER: CropRect = CropRect { x: 0.1667, y: 0.0, w: 0.6667, h: 1.0 };

    fn is_red(p: &Rgba<u8>) -> bool {
        p[0] > 200 && p[1] < 90 && p[2] < 90
    }
    fn is_yellow(p: &Rgba<u8>) -> bool {
        p[0] > 200 && p[1] > 200 && p[2] < 120
    }

    #[test]
    fn crop_turns_red_when_a_hand_nears_the_edge() {
        let b64 = jpeg_b64(640, 427);
        // 鏡像の枠の左端 x = (1 - 0.8334) * 640 = 106.6 画素。縦の中ほどで枠線の色を見る
        let edge_x = ((1.0 - (CENTER.x + CENTER.w)) * 640.0).round() as u32 + 1;

        // 手が中央 → 黄色の枠、縁の判定なし
        let center = HandsPoints { left: None, right: Some(vec![[0.5, 0.5]; 21]) };
        let (img, edge) = compose_guide_preview(&b64, &center, CENTER).unwrap();
        assert!(!edge.any());
        assert!(is_yellow(img.get_pixel(edge_x, 200)), "{:?}", img.get_pixel(edge_x, 200));

        // 手が元の座標の枠の右端付近(画面では左の縁)→ 赤い枠
        let near = HandsPoints { left: None, right: Some(vec![[0.825, 0.5]; 21]) };
        let (img, edge) = compose_guide_preview(&b64, &near, CENTER).unwrap();
        assert!(edge.left && !edge.right, "{edge:?}");
        assert!(is_red(img.get_pixel(edge_x, 200)), "{:?}", img.get_pixel(edge_x, 200));
    }

    #[test]
    fn crop_style_values() {
        assert_eq!(crop_style(&EdgeStatus::default()), (CROP_COLOR, CROP_THICK));
        let near = EdgeStatus { right: true, ..Default::default() };
        assert_eq!(crop_style(&near), (EDGE_WARN_COLOR, EDGE_WARN_THICK));
    }

    #[test]
    fn guide_preview_contains_the_guide_frame() {
        let b64 = jpeg_b64(640, 427);
        let (img, _) = compose_guide_preview(&b64, &HandsPoints::default(), CENTER).unwrap();
        let s = silhouette::guide_shape(640, 427);
        // 頭の楕円の上端の画素は、黒より明るい(半透明の白)
        let p = img.get_pixel(s.head_center.0.round() as u32, (s.head_center.1 - s.head_radius.1).round() as u32);
        assert!(p[0] > 80, "{p:?}");
    }

    #[test]
    fn guide_state_measures_time_from_receive_timestamps() {
        let mut g = GuideState::new(PathBuf::from("/nowhere"));
        let t0 = Instant::now();
        for i in 0..12 {
            g.push_sample(&HandSample { at: t0 + Duration::from_millis(100 * i), hands: "R".into() });
        }
        assert!((g.runs.current_best() - 1.1).abs() < 1e-6, "{}", g.runs.current_best());
        assert!(hand_run::passes(g.runs.current_best()));
        assert_eq!(hands_big_text(g.hands.as_deref()), "右手");
        g.reset_session();
        assert_eq!(g.runs.current_best(), 0.0);
        assert_eq!(hands_big_text(g.hands.as_deref()), "-");
    }

    #[test]
    fn reload_is_periodic() {
        let mut g = GuideState::new(PathBuf::from("/このフォルダは無い"));
        let t0 = Instant::now();
        assert!(g.reload_due(t0));
        g.reload_progress(t0);
        // 読めないフォルダ → 理由が入る(パニックしない)
        assert!(matches!(&g.progress, Some(Err(e)) if e.contains("見つかりません")));
        assert!(!g.reload_due(t0 + Duration::from_secs(1)));
        assert!(g.reload_due(t0 + PROGRESS_RELOAD_INTERVAL));
    }

    #[test]
    fn text_helpers() {
        assert_eq!(hands_big_text(Some("L")), "左手");
        assert_eq!(hands_big_text(Some("LR")), "両手");
        assert_eq!(hands_big_text(Some("none")), "手なし");
        assert_eq!(run_text(1.26), "1.3 秒");
        assert_eq!(run_text(f64::NAN), "0.0 秒");
    }
}
