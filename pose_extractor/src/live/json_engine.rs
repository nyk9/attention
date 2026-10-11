//! アプリ(GUI)専用の内部入口: 標準入力の JSON コマンドで操作し、標準出力へ JSON イベントを流す。
//! 設計: docs/desktop-app-design.md §3.2。起動は `pose-extract --engine-json`(メニュー・ヘルプには出さない)。
//!
//! 画面は別プロセス(別言語)なので、**この JSON の会話がエンジンと画面の唯一の境界**になる。
//! ライブ認識のループ(`run_session`)・区切り・推論は変えず、`LiveUi`(出力)と `KeySource`(入力)の
//! 別実装として足している。
//!
//! # 会話(1 行 1 メッセージの JSON。UTF-8)
//!
//! - コマンド(標準入力): `start` / `toggle` / `set_segmenter` / `quit`。標準入力が閉じられたら quit と同じ
//! - イベント(標準出力): `ready` / `status` / `preview` / `result` / `message` / `error` / `stopped`
//!   と、実装で足した `segmenter`(区切り方の切り替えが実際に反映されたとき)
//! - ログ・診断は標準エラー。標準出力は JSON 行だけ(迷い込み対策は `take_json_sink`)
//!
//! # スレッド(Rust 初心者向けのメモ)
//!
//! ```text
//! [stdin 読み取りスレッド] ──Input──▶ (チャンネル) ──▶ [メインスレッド: run_session ループ]
//!                                                         │ JsonKeys.poll で受け取る
//!                                                         │ JsonUi / SwitchableSegmenter がイベントを作る
//!                                                         ▼
//!                                     (チャンネル) ──Outgoing──▶ [出力スレッド] ─▶ 標準出力
//! ```
//!
//! - 出力を 1 本のスレッドに集約するのは、(1) イベントの順序を保つ、(2) JPEG 化と base64 化という
//!   重い処理を推論ループから外す、(3) 画面が読んでいないときに推論ループが止まらないようにする
//!   (間引いてよい `status` / `preview` は、キューが詰まっていたら捨てる)ため。
//! - `Arc` = 参照カウント付きの共有所有、`mpsc` = スレッド間のメッセージ箱(送信側は clone できる)
//!
//! # 守っていること
//!
//! - 映像・特徴量をディスクに書かない(既存のオプトイン `LIVE_DEBUG_DUMP_DIR` の挙動は live.rs のまま)
//! - 認識経路の生フレームには触れない。プレビューは参照から縮小コピーを作る

use super::capture;
use super::*;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::io::BufRead;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, RecvTimeoutError, SyncSender, TrySendError};
use std::time::Duration;

/// プレビューの既定(仕様: max_fps=10, width=640)
const DEFAULT_PREVIEW_FPS: f64 = 10.0;
const MAX_PREVIEW_FPS: f64 = 30.0;
const DEFAULT_PREVIEW_WIDTH: u32 = 640;
const MIN_PREVIEW_WIDTH: u32 = 16;
const MAX_PREVIEW_WIDTH: u32 = 4096;
/// `status` の最大頻度(回/秒)。状態(待機/記録中)が変わったときは即時に出す
const STATUS_MAX_HZ: f64 = 10.0;
const JPEG_QUALITY: u8 = 70;
/// 出力スレッドへのキューの長さ。詰まったら `status` / `preview` を捨てる
const OUT_QUEUE: usize = 64;

// ===== コマンド(GUI → エンジン)=====

/// 区切り方の指定("auto" | "manual")
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum SegKind {
    #[default]
    Auto,
    Manual,
}

impl SegKind {
    fn mode(self) -> SegmentMode {
        match self {
            SegKind::Auto => SegmentMode::Auto,
            SegKind::Manual => SegmentMode::Manual,
        }
    }
}

fn mode_name(mode: SegmentMode) -> &'static str {
    match mode {
        SegmentMode::Auto => "auto",
        SegmentMode::Manual => "manual",
    }
}

/// カメラのデバイス番号。文字列("0")でも数値(0)でも受ける
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(untagged)]
pub(super) enum DeviceId {
    Text(String),
    Number(i64),
}

impl DeviceId {
    pub(super) fn as_string(&self) -> String {
        match self {
            DeviceId::Text(s) => s.trim().to_string(),
            DeviceId::Number(n) => n.to_string(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(super) enum SourceSpec {
    /// 省略時のデバイスは "0"(FaceTime HD カメラ)
    Camera {
        #[serde(default)]
        device: Option<DeviceId>,
    },
    File {
        path: String,
    },
}

/// プレビューの設定。省略した項目は既定値
#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
#[serde(default)]
pub(super) struct PreviewCfg {
    pub(super) max_fps: f64,
    pub(super) width: u32,
}

impl Default for PreviewCfg {
    fn default() -> Self {
        Self {
            max_fps: DEFAULT_PREVIEW_FPS,
            width: DEFAULT_PREVIEW_WIDTH,
        }
    }
}

/// `"preview"` の 3 状態を区別する: 省略 = None(既定のプレビュー)/ `null` = Some(None)(プレビュー無し)/
/// オブジェクト = Some(Some(設定))。serde は `Option<Option<T>>` の null と省略を区別しないので自前で読む
fn double_option<'de, D, T>(d: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(d).map(Some)
}

/// `start` の動作モード。省略は "live"(従来どおりのライブ認識。何も保存しない)。
/// "capture" は撮影モード(録画・判定・登録。live/capture.rs に閉じている。docs/desktop-capture-design.md §4.2)
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum StartMode {
    #[default]
    Live,
    Capture,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub(super) struct StartCmd {
    pub(super) source: SourceSpec,
    /// 省略時は既定のデモ用モデル
    #[serde(default)]
    pub(super) model_dir: Option<String>,
    #[serde(default)]
    pub(super) segmenter: SegKind,
    #[serde(default, deserialize_with = "double_option")]
    pub(super) preview: Option<Option<PreviewCfg>>,
    /// 省略時は "live"
    #[serde(default)]
    pub(super) mode: StartMode,
    /// 撮影モードの登録先(`raw_jsl` のフォルダ)。撮影モードでは必須。ライブ認識では使わない
    #[serde(default)]
    pub(super) data_dir: Option<String>,
    /// 撮影モードのカウントダウン秒数(省略 = 3 秒。0〜10 に丸める)。ライブ認識では使わない
    #[serde(default)]
    pub(super) countdown_secs: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub(super) enum Command {
    Start(StartCmd),
    /// Enter 相当(手動の開始/停止。`Key::Toggle`)
    Toggle,
    SetSegmenter {
        mode: SegKind,
    },
    Quit,
    /// 撮影モードだけ: 録画開始(カウントダウンのあと録画)。`word_id` は画面が出しているお題の語
    /// (省略時はエンジンが「次の語」を決める)
    CaptureRecord {
        #[serde(default)]
        word_id: Option<String>,
    },
    /// 撮影モードだけ: 録画停止(カウントダウン中なら取り消し)
    CaptureStop,
    /// 撮影モードだけ: 採用(登録先へ移して index.tsv に記帳)。使えないと判定されたテイクは
    /// `confirm_ng: true` のときだけ、NG フラグ付きで記録する
    CaptureAccept {
        #[serde(default)]
        confirm_ng: bool,
    },
    /// 撮影モードだけ: 破棄(一時ファイルを消す。撮り直しは破棄してから次の録画)
    CaptureDiscard,
}

/// 1 行をコマンドとして読む。空行は Ok(None)(無視)。解釈できなければ理由つきの Err
// [TEST向き] 純粋関数。正常・不正な JSON・未知のコマンド・必須項目の欠け
fn parse_command(line: &str) -> Result<Option<Command>, String> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }
    serde_json::from_str::<Command>(line)
        .map(Some)
        .map_err(|e| e.to_string())
}

// ===== イベント(エンジン → GUI)=====

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(super) struct VocabEntry {
    tag: String,
    label: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(super) struct RankedItem {
    tag: String,
    label: String,
    p: f64,
}

/// 左右の手の 21 点の (x, y)。画像の幅・高さで正規化したまま(0〜1 を少し外れることがある)。
/// 手が未検出の枠は null
#[derive(Debug, Clone, Serialize, PartialEq)]
pub(super) struct HandsPoints {
    left: Option<Vec<[f64; 2]>>,
    right: Option<Vec<[f64; 2]>>,
}

/// 元フレームに対する正規化座標の矩形(Palm 検出に渡す範囲)
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub(super) struct CropRect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum Event {
    Ready {
        vocab: Vec<VocabEntry>,
        model_dir: String,
        palm_roi: String,
        frame_select: String,
        frames: usize,
        segmenter: &'static str,
        /// 前処理をどこから決めたか: "preprocess.json"(モデル)| "default"(live_wizard と同じ既定)
        preprocess: &'static str,
    },
    Status {
        state: &'static str,
        hands: &'static str,
        fps: f64,
        rec_secs: f64,
    },
    Preview {
        width: u32,
        height: u32,
        jpeg_b64: String,
        hands: HandsPoints,
        crop: CropRect,
    },
    Result {
        tag: String,
        label: String,
        p: f64,
        ranked: Vec<RankedItem>,
        low_confidence: bool,
        span: Option<[usize; 2]>,
        buffered: usize,
        duration_secs: f64,
        selected_with_hand: usize,
        selected_total: usize,
    },
    Message {
        level: &'static str,
        text: String,
    },
    Error {
        text: String,
    },
    Stopped {
        dropped_frames: u64,
    },
    /// 区切り方の切り替えが実際に反映されたとき(`set_segmenter` への応答)
    Segmenter {
        mode: &'static str,
    },
    /// 撮影モードだけ: 撮影の状態(待機・カウントダウン・録画中・判定中・判定済み)。
    /// 状態が変わったときは必ず届く。カウントダウン/録画中の秒数は最大 10 回/秒で更新される
    CaptureState {
        /// "idle" | "countdown" | "recording" | "judging" | "judged"
        phase: &'static str,
        /// カウントダウンの残り秒数(countdown 以外は 0)
        countdown_left: f64,
        /// 録画した長さ(秒。録画に書いたコマ数 ÷ fps)
        rec_secs: f64,
        /// 自動停止する長さ(秒)
        max_secs: f64,
        /// 録画に使う(使った)fps。countdown のあいだは実測値、録画が始まるまで不明なら null
        fps: Option<f64>,
        /// 録画キューがあふれて書けなかったコマ数(0 でなければ再生が少し速くなる)
        dropped: u64,
        /// 録画する(した)語と、登録先になる見込みのテイク番号。録画を始める前は null
        word: Option<CaptureWord>,
        /// 登録先(`raw_jsl`)。常に画面に出す
        data_dir: String,
        /// 判定(judged のときだけ)
        verdict: Option<VerdictView>,
        /// 画面に出す注意(fps が出ていない、自動停止した、など)
        notes: Vec<String>,
    },
    /// 撮影モードだけ: 採用して登録した結果(登録先・判定・長さ)
    CaptureSaved {
        /// `data_dir` からの相対パス("011_kare/04.mp4")
        file_path: String,
        abs_path: String,
        word_id: String,
        label: String,
        rep_idx: u32,
        /// index.tsv に書いた値("ok" | "ng_hands" | "ng_duration" | "")
        quality_flag: String,
        usable: bool,
        longest_run_secs: f64,
        duration_secs: f64,
        data_dir: String,
    },
}

/// 撮影で録画する語
#[derive(Debug, Clone, Serialize, PartialEq)]
pub(super) struct CaptureWord {
    pub(super) word_id: String,
    pub(super) romaji: String,
    pub(super) label: String,
    /// 登録先になる見込みのテイク番号(採用の瞬間に決め直すので、他のツールが先に足していれば変わる)
    pub(super) rep_idx: u32,
}

/// 判定の結果(画面に出す)
#[derive(Debug, Clone, Serialize, PartialEq)]
pub(super) struct VerdictView {
    pub(super) usable: bool,
    /// 使えない理由の機械可読な符号: "hands_short" | "too_short" | "too_long" | "judge_failed"
    pub(super) reasons: Vec<&'static str>,
    /// 画面にそのまま出す 1 文("使えそう: ..." / "使えない: ...")
    pub(super) summary: String,
    /// 手が連続して検出できた最長区間(左右の良い方、秒)
    pub(super) longest_run_secs: f64,
    pub(super) left_run_secs: f64,
    pub(super) right_run_secs: f64,
    pub(super) duration_secs: f64,
    /// 採用したときに index.tsv へ書く値("ok" | "ng_hands" | "ng_duration" | 判定できなかったときは "")
    pub(super) quality_flag: String,
}

/// 数値は有限であること(NaN / Inf を JSON に出さない)。非有限は 0 にする
fn finite(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

pub(super) fn round_to(x: f64, decimals: i32) -> f64 {
    let k = 10f64.powi(decimals);
    finite((finite(x) * k).round() / k)
}

pub(super) fn r4(x: f64) -> f64 {
    round_to(x, 4)
}

/// 手の有無 → "L" | "R" | "LR" | "none"
fn hands_name(left: bool, right: bool) -> &'static str {
    match (left, right) {
        (true, true) => "LR",
        (true, false) => "L",
        (false, true) => "R",
        (false, false) => "none",
    }
}

/// 126 次元の特徴量から左右の手の (x, y) を取り出す。
/// レイアウトは `frame_hand_feature`(main.rs)と `mirror_features`(transformer_burn/src/pose_data.rs)の通り:
/// 左手枠 [0..63] + 右手枠 [63..126]、各枠は 21 点 × (x, y, z)、x は幅・y は高さで正規化。
/// 全ゼロの枠は未検出(null)。非有限の値を含む枠も null にする(JSON に NaN を出さない)
// [TEST向き] 純粋関数。片手だけ・両手・未検出・非有限
fn hands_from_feature(feature: &[f32; DICT_FEATURE_DIM]) -> HandsPoints {
    fn slot(values: &[f32]) -> Option<Vec<[f64; 2]>> {
        if values.iter().all(|v| *v == 0.0) || values.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(
            values
                .chunks_exact(3)
                .map(|p| [r4(p[0] as f64), r4(p[1] as f64)])
                .collect(),
        )
    }
    let half = DICT_FEATURE_DIM / 2;
    HandsPoints {
        left: slot(&feature[..half]),
        right: slot(&feature[half..]),
    }
}

/// Palm 検出に渡す範囲を、元フレームに対する正規化座標で返す。
/// 中央正方クロップ・相対矩形は固定の矩形。FullFrame は全体。人物追従(フレームごとに変わる)と
/// 複数タイルは 1 つの矩形で表せないので全体を返す
// [TEST向き] 1620x1080 の中央正方 → x=0.1667, w=0.6667
pub(super) fn crop_rect(roi: PalmRoiMode, w: usize, h: usize) -> CropRect {
    let full = CropRect {
        x: 0.0,
        y: 0.0,
        w: 1.0,
        h: 1.0,
    };
    if w == 0 || h == 0 || matches!(roi, PalmRoiMode::Person { .. }) || roi.is_multi_roi() {
        return full;
    }
    match crate::palm_rois(roi, w as u32, h as u32).first() {
        Some(r) => CropRect {
            x: r4(r.x as f64 / w as f64),
            y: r4(r.y as f64 / h as f64),
            w: r4(r.w as f64 / w as f64),
            h: r4(r.h as f64 / h as f64),
        },
        None => full,
    }
}

/// 認識結果 → `result` イベント。候補が空なら None
fn result_event(r: &LiveResult, labels: &Labels) -> Option<Event> {
    let (top_tag, top_p) = r.ranked.first()?;
    let ranked: Vec<RankedItem> = r
        .ranked
        .iter()
        .take(SHOW_TOPK)
        .map(|(tag, p)| RankedItem {
            tag: tag.clone(),
            label: labels.display(tag),
            p: r4(*p as f64),
        })
        .collect();
    Some(Event::Result {
        tag: top_tag.clone(),
        label: labels.display(top_tag),
        p: r4(*top_p as f64),
        ranked,
        low_confidence: *top_p < LOW_CONFIDENCE,
        span: r.selection.span.map(|(s, e)| [s, e]),
        buffered: r.buffered,
        duration_secs: round_to(r.duration_secs, 2),
        selected_with_hand: r.selected_with_hand,
        selected_total: r.selection.indices.len(),
    })
}

/// `ui.message` の文 → (level, text)。「短すぎる動きを無視」は通常動作なので info(外側の括弧は外す)、
/// 「手が検出されませんでした」などは warn
// [TEST向き] 純粋関数
fn classify_message(msg: &str) -> (&'static str, String) {
    if msg == DISCARD_MESSAGE {
        let text = msg
            .trim_start_matches(['(', '('])
            .trim_end_matches([')', ')']);
        return ("info", text.to_string());
    }
    ("warn", msg.to_string())
}

/// `ready` の語彙一覧。words.tsv の並び順で、モデルの語彙に含まれるものだけ。
/// words.tsv に無いモデルのタグは末尾にタグ名のまま足す(`result` で出うるので、画面が引けるように)
// [TEST向き] 純粋関数
fn build_vocab(tags: &[String], labels: &Labels) -> Vec<VocabEntry> {
    let mut out: Vec<VocabEntry> = Vec::new();
    for (romaji, label) in &labels.rows {
        if tags.contains(romaji) && !out.iter().any(|v| &v.tag == romaji) {
            out.push(VocabEntry {
                tag: romaji.clone(),
                label: label.clone(),
            });
        }
    }
    for t in tags {
        if !out.iter().any(|v| &v.tag == t) {
            out.push(VocabEntry {
                tag: t.clone(),
                label: t.clone(),
            });
        }
    }
    out
}

// ===== プレビューの縮小・符号化 =====

/// 縮小後のサイズ。希望幅を元の幅以下に抑え(拡大はしない)、縦横比を保つ
// [TEST向き] 純粋関数
fn preview_size(src_w: usize, src_h: usize, want_w: u32) -> (usize, usize) {
    let w = (want_w as usize).min(src_w).max(1);
    let h = ((src_h as f64 * w as f64 / src_w.max(1) as f64).round() as usize).max(1);
    (w, h)
}

/// RGB の (高さ, 幅, 3) 配列を `ow` x `oh` に縮小して、行優先の RGB バイト列にする(箱形の平均)。
/// 元の配列は読むだけで書き換えない(認識経路のフレームを壊さない)。
/// 縮小率が約 2.5 倍なので、出力 1 画素あたり元の 6 画素ほどを平均する
// [TEST向き] 純粋関数。平均・端の画素・1x1
fn downscale_rgb(src: &Array3<u8>, ow: usize, oh: usize) -> Vec<u8> {
    let (sh, sw) = (src.shape()[0], src.shape()[1]);
    let mut out = vec![0u8; ow * oh * 3];
    // 標準レイアウト(行優先)なら借用のまま、そうでなければここで並べ直す
    let cow = src.as_standard_layout();
    let Some(data) = cow.as_slice() else {
        return out;
    };
    if sh == 0 || sw == 0 || ow == 0 || oh == 0 || data.len() != sh * sw * 3 {
        return out;
    }
    for oy in 0..oh {
        let y0 = oy * sh / oh;
        let y1 = ((oy + 1) * sh / oh).clamp(y0 + 1, sh);
        for ox in 0..ow {
            let x0 = ox * sw / ow;
            let x1 = ((ox + 1) * sw / ow).clamp(x0 + 1, sw);
            let mut sum = [0u32; 3];
            for y in y0..y1 {
                for px in data[(y * sw + x0) * 3..(y * sw + x1) * 3].chunks_exact(3) {
                    sum[0] += px[0] as u32;
                    sum[1] += px[1] as u32;
                    sum[2] += px[2] as u32;
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as u32;
            let o = (oy * ow + ox) * 3;
            for c in 0..3 {
                out[o + c] = ((sum[c] + n / 2) / n) as u8;
            }
        }
    }
    out
}

fn encode_jpeg(rgb: &[u8], w: u32, h: u32) -> Result<Vec<u8>, String> {
    let mut buf = Vec::with_capacity(rgb.len() / 8);
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, JPEG_QUALITY)
        .encode(rgb, w, h, image::ExtendedColorType::Rgb8)
        .map_err(|e| e.to_string())?;
    Ok(buf)
}

/// 標準の base64(パディングあり)。依存クレートを増やさないための小さな実装
// [TEST向き] 純粋関数。RFC 4648 のテストベクタ
fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[(n >> 18 & 63) as usize] as char);
        out.push(TABLE[(n >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(n >> 6 & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// 出力スレッドに渡す、まだ JPEG 化していないプレビュー
struct PreviewJob {
    width: u32,
    height: u32,
    rgb: Vec<u8>,
    hands: HandsPoints,
    crop: CropRect,
}

fn preview_event(job: PreviewJob) -> Result<Event, String> {
    let jpeg = encode_jpeg(&job.rgb, job.width, job.height)?;
    Ok(Event::Preview {
        width: job.width,
        height: job.height,
        jpeg_b64: base64_encode(&jpeg),
        hands: job.hands,
        crop: job.crop,
    })
}

// ===== 出力スレッド =====

enum Outgoing {
    Event(Event),
    Preview(PreviewJob),
    Shutdown,
}

/// 出力スレッドへの送り口。clone して複数の場所で持てる
#[derive(Clone)]
pub(super) struct Emitter {
    tx: SyncSender<Outgoing>,
    /// 標準出力に書けなくなったら true(画面側が落ちた = 閉じたパイプ)。ループはこれを見て終了する
    broken: Arc<AtomicBool>,
}

impl Emitter {
    /// 必ず届けたいイベント(キューが一杯なら空くまで待つ)
    pub(super) fn send(&self, e: Event) {
        let _ = self.tx.send(Outgoing::Event(e));
    }

    /// 間引いてよいイベント。キューが一杯なら捨てて false を返す
    pub(super) fn try_send(&self, e: Event) -> bool {
        !matches!(
            self.tx.try_send(Outgoing::Event(e)),
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_))
        )
    }

    fn try_send_preview(&self, job: PreviewJob) -> bool {
        self.tx.try_send(Outgoing::Preview(job)).is_ok()
    }

    pub(super) fn message(&self, level: &'static str, text: impl Into<String>) {
        self.send(Event::Message {
            level,
            text: text.into(),
        });
    }

    pub(super) fn is_broken(&self) -> bool {
        self.broken.load(Ordering::SeqCst)
    }
}

/// 出力スレッドを止めて待つための持ち手
pub(super) struct WriterHandle {
    emitter: Emitter,
    handle: std::thread::JoinHandle<()>,
}

impl WriterHandle {
    /// 溜まっているイベントをすべて書き出してから、出力スレッドを止める
    pub(super) fn finish(self) {
        let _ = self.emitter.tx.send(Outgoing::Shutdown);
        let _ = self.handle.join();
    }
}

fn write_line(out: &mut impl Write, ev: &Event) -> std::io::Result<()> {
    let mut line = serde_json::to_vec(ev).map_err(std::io::Error::other)?;
    line.push(b'\n');
    out.write_all(&line)?;
    out.flush()
}

/// 出力スレッドを起こす。`out` へ 1 イベント = 1 行で書く。
/// 書き込みに失敗したら(画面が閉じた)`broken` を立てて、以降は読み捨てる(送り手を詰まらせない)
pub(super) fn spawn_writer(mut out: impl Write + Send + 'static) -> (Emitter, WriterHandle) {
    let (tx, rx) = sync_channel::<Outgoing>(OUT_QUEUE);
    let broken = Arc::new(AtomicBool::new(false));
    let emitter = Emitter {
        tx,
        broken: Arc::clone(&broken),
    };
    let handle = std::thread::spawn(move || {
        while let Ok(msg) = rx.recv() {
            let ev = match msg {
                Outgoing::Shutdown => break,
                Outgoing::Event(e) => e,
                Outgoing::Preview(job) => match preview_event(job) {
                    Ok(e) => e,
                    Err(why) => {
                        eprintln!("警告: プレビューの JPEG 化に失敗: {}", why);
                        continue;
                    }
                },
            };
            if broken.load(Ordering::SeqCst) {
                continue;
            }
            if let Err(e) = write_line(&mut out, &ev) {
                eprintln!("標準出力に書けません(画面側が閉じたとみなして終了します): {}", e);
                broken.store(true, Ordering::SeqCst);
            }
        }
    });
    let writer = WriterHandle {
        emitter: emitter.clone(),
        handle,
    };
    (emitter, writer)
}

// ===== 終了要求 =====

/// 終了要求の旗(quit・標準入力の EOF・SIGTERM/SIGINT のどれかで立つ)。clone して共有する
#[derive(Clone, Default)]
pub(super) struct StopFlag(Arc<AtomicBool>);

impl StopFlag {
    pub(super) fn set(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub(super) fn is_set(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

// ===== 標準入力の読み取り =====

/// 標準入力スレッドからメインスレッドへ渡すもの
#[derive(Debug)]
pub(super) enum Input {
    Cmd(Command),
    /// 標準入力が閉じられた(quit と同じ扱い)
    Eof,
}

/// 標準入力を 1 行ずつ読み、コマンドに直してチャンネルへ流すスレッドを起こす。
/// 不正な行は `message`(warn)を出して無視する(落ちない)。`quit` と EOF では終了要求の旗も立てる
/// (推論ループがカメラの 1 枚目を待っているだけの間でも、監視スレッドがカメラを止められるように)
fn spawn_command_reader(
    mut reader: impl BufRead + Send + 'static,
    tx: mpsc::Sender<Input>,
    emitter: Emitter,
    stop: StopFlag,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    // 不正な UTF-8 が混ざっても落ちない(置換文字になり、JSON としてのエラーで報告される)
                    let line = String::from_utf8_lossy(&buf);
                    match parse_command(&line) {
                        Ok(Some(cmd)) => {
                            let quit = matches!(cmd, Command::Quit);
                            if tx.send(Input::Cmd(cmd)).is_err() {
                                return;
                            }
                            if quit {
                                stop.set();
                                return;
                            }
                        }
                        Ok(None) => {}
                        Err(why) => {
                            let why: String = why.chars().take(200).collect();
                            emitter.message(
                                "warn",
                                format!("コマンドを解釈できないため無視しました: {}", why),
                            );
                        }
                    }
                }
            }
        }
        let _ = tx.send(Input::Eof);
        stop.set();
    })
}

/// `KeySource` の JSON 版。標準入力から来たコマンドを `Key` に直す。
/// `start` の 2 回目以降は無視、`set_segmenter` は `SwitchableSegmenter` への依頼箱に入れる
struct JsonKeys {
    rx: Receiver<Input>,
    /// 最初に押したことにするキー(動画ファイル + 手動のとき「動画全体を 1 回の録画にする」ため)
    initial: VecDeque<Key>,
    /// 区切り方の切り替え依頼(メインスレッドだけで使うので Rc<Cell>)
    pending: Rc<Cell<Option<SegmentMode>>>,
    emitter: Emitter,
    stop: StopFlag,
}

impl KeySource for JsonKeys {
    fn poll(&mut self) -> Option<Key> {
        if let Some(k) = self.initial.pop_front() {
            return Some(k);
        }
        // 画面が閉じた(出力が壊れた)・終了要求(SIGTERM など)でも終了する
        if self.emitter.is_broken() || self.stop.is_set() {
            return Some(Key::Quit);
        }
        loop {
            match self.rx.try_recv() {
                Ok(Input::Cmd(Command::Toggle)) => return Some(Key::Toggle),
                Ok(Input::Cmd(Command::Quit)) => return Some(Key::Quit),
                Ok(Input::Cmd(Command::SetSegmenter { mode })) => {
                    self.pending.set(Some(mode.mode()));
                }
                Ok(Input::Cmd(Command::Start(_))) => {
                    self.emitter
                        .message("warn", "start は受け付け済みです(2 回目以降は無視します)");
                }
                // 撮影モード専用のコマンド。ライブ認識では受け付けない(録画・保存の経路に入らせない)
                Ok(Input::Cmd(
                    Command::CaptureRecord { .. }
                    | Command::CaptureStop
                    | Command::CaptureAccept { .. }
                    | Command::CaptureDiscard,
                )) => {
                    self.emitter.message(
                        "warn",
                        "撮影用のコマンドです。ライブ認識では録画も保存もしないので無視しました",
                    );
                }
                Ok(Input::Eof) | Err(TryRecvError::Disconnected) => return Some(Key::Quit),
                Err(TryRecvError::Empty) => return None,
            }
        }
    }
}

/// 実行中に区切り方(自動/手動)を切り替えられる `Segmenter`。
/// 依頼は `JsonKeys` が箱に入れ、ここで反映する。記録中は切り替えず、記録が終わってから反映する
/// (記録の途中で止め方が変わると、止まらない/二重に止まる記録が出うるため)
struct SwitchableSegmenter {
    current: Box<dyn Segmenter>,
    mode: SegmentMode,
    pending: Rc<Cell<Option<SegmentMode>>>,
    emitter: Emitter,
    deferred_notified: bool,
}

impl SwitchableSegmenter {
    fn new(mode: SegmentMode, pending: Rc<Cell<Option<SegmentMode>>>, emitter: Emitter) -> Self {
        Self {
            current: make_segmenter(mode),
            mode,
            pending,
            emitter,
            deferred_notified: false,
        }
    }

    fn apply_pending(&mut self, recording: bool) {
        let Some(want) = self.pending.get() else {
            return;
        };
        if recording {
            if !self.deferred_notified {
                self.deferred_notified = true;
                self.emitter
                    .message("info", "区切り方の切り替えは、今の記録が終わってから反映します");
            }
            return;
        }
        self.pending.set(None);
        self.deferred_notified = false;
        if want != self.mode {
            self.current = make_segmenter(want);
            self.mode = want;
        }
        self.emitter.send(Event::Segmenter {
            mode: mode_name(want),
        });
    }
}

impl Segmenter for SwitchableSegmenter {
    fn on_toggle(&mut self, recording: bool) -> Option<Boundary> {
        self.apply_pending(recording);
        self.current.on_toggle(recording)
    }

    fn on_frame(&mut self, frame: &LiveFrame, recording: bool) -> Option<Boundary> {
        self.apply_pending(recording);
        self.current.on_frame(frame, recording)
    }
}

// ===== 画面への出力(LiveUi)=====

/// 最大頻度の間引き。`allow(now)` が true を返したら通す。
///
/// 基準は「通した時刻」ではなく「次に通してよい時刻(`next`)」。通すたびに `next` を 1 間隔ずつ進める。
/// 「前回通した時刻から 1 間隔以上」で判定すると、間隔が 0.1 秒のところへ 0.067 秒間隔
/// (約 15fps のカメラ)のフレームが来たとき、1 枚おき(約 7.5 回/秒)にしか通らない。
/// 一方、止まったあとに時刻が大きく進んだときは `next` を現在時刻に合わせ直し、溜まった分を
/// 連続して通さない(バーストしない)
#[derive(Debug)]
struct Throttle {
    min_interval: f64,
    /// 次に通してよい時刻(まだ何も通していなければ None = 最初は必ず通る)
    next: Option<f64>,
}

impl Throttle {
    fn new(max_hz: f64) -> Self {
        Self {
            min_interval: if max_hz > 0.0 { 1.0 / max_hz } else { f64::INFINITY },
            next: None,
        }
    }

    // [TEST向き] 境界: 最初は必ず通る / 間隔ちょうどは通る / それ未満は通らない / 15fps でも約 10 回/秒 / バーストしない
    fn allow(&mut self, now: f64) -> bool {
        let ok = match self.next {
            None => true,
            // 1e-9 は i/10 のような時刻の引き算で 0.0999999.. になる誤差の吸収
            Some(n) => now + 1e-9 >= n,
        };
        if ok {
            self.advance(now);
        }
        ok
    }

    /// 通したことにして `next` を進める。格子(前回の基準 + 間隔)を保つが、
    /// 進めた先がもう過去(1 間隔以上遅れている)なら現在時刻から数え直す
    fn advance(&mut self, now: f64) {
        let on_grid = self.next.map(|n| n + self.min_interval);
        self.next = Some(match on_grid {
            Some(g) if g > now => g,
            _ => now + self.min_interval,
        });
    }

    /// 間引き判定を通さずに「今出した」ことにする(状態変化で即時に出したとき)。格子は今に合わせ直す
    fn mark(&mut self, now: f64) {
        self.next = Some(now + self.min_interval);
    }
}

struct PreviewState {
    cfg: PreviewCfg,
    gate: Throttle,
}

/// `LiveUi` の JSON 版。イベントを作って出力スレッドへ送る
pub(super) struct JsonUi {
    emitter: Emitter,
    labels: Labels,
    roi: PalmRoiMode,
    preview: Option<PreviewState>,
    status_gate: Throttle,
    last_recording: Option<bool>,
    /// 間引きに使う時計(秒)。本番は起動からの経過時間、テストでは差し替える
    clock: Box<dyn FnMut() -> f64>,
}

impl JsonUi {
    pub(super) fn new(
        emitter: Emitter,
        labels: Labels,
        roi: PalmRoiMode,
        preview: Option<PreviewCfg>,
        clock: Box<dyn FnMut() -> f64>,
    ) -> Self {
        Self {
            emitter,
            labels,
            roi,
            preview: preview.map(|cfg| PreviewState {
                cfg,
                gate: Throttle::new(cfg.max_fps),
            }),
            status_gate: Throttle::new(STATUS_MAX_HZ),
            last_recording: None,
            clock,
        }
    }
}

impl LiveUi for JsonUi {
    fn status(&mut self, s: &StatusLine) {
        let now = (self.clock)();
        // 待機 ⇔ 記録中が変わったときは間引かずに即時、それ以外は最大 10 回/秒
        let changed = self.last_recording != Some(s.recording);
        if changed {
            self.status_gate.mark(now);
        } else if !self.status_gate.allow(now) {
            return;
        }
        self.last_recording = Some(s.recording);
        let ev = Event::Status {
            state: if s.recording { "recording" } else { "idle" },
            hands: hands_name(s.left, s.right),
            fps: round_to(s.fps, 1),
            rec_secs: round_to(s.rec_secs, 2),
        };
        if changed {
            self.emitter.send(ev);
        } else {
            self.emitter.try_send(ev);
        }
    }

    fn message(&mut self, msg: &str) {
        let (level, text) = classify_message(msg);
        self.emitter.message(level, text);
    }

    fn result(&mut self, r: &LiveResult) {
        match result_event(r, &self.labels) {
            Some(ev) => self.emitter.send(ev),
            None => self.emitter.message("warn", "認識結果が空でした"),
        }
    }

    fn preview(&mut self, raw: &Array3<u8>, frame: &LiveFrame) {
        let Some(ps) = self.preview.as_mut() else {
            return;
        };
        // 間引きの判定を最初にして、捨てるフレームでは縮小もしない(推論ループを遅くしない)
        if !ps.gate.allow((self.clock)()) {
            return;
        }
        let (sh, sw) = (raw.shape()[0], raw.shape()[1]);
        if sh == 0 || sw == 0 {
            return;
        }
        let (ow, oh) = preview_size(sw, sh, ps.cfg.width);
        let job = PreviewJob {
            width: ow as u32,
            height: oh as u32,
            // 縮小は生フレームを読むだけ(JPEG 化と base64 化は出力スレッドで行う)
            rgb: downscale_rgb(raw, ow, oh),
            hands: hands_from_feature(&frame.feature),
            crop: crop_rect(self.roi, sw, sh),
        };
        self.emitter.try_send_preview(job);
    }
}

// ===== 起動と全体の流れ =====

/// 入力元に「捨てたフレーム数」を足したもの(`stopped` で報告する)
trait JsonSource: FrameSource {
    fn dropped_frames(&self) -> u64 {
        0
    }
}

impl JsonSource for CameraSource {
    fn dropped_frames(&self) -> u64 {
        CameraSource::dropped_frames(self)
    }
}

impl JsonSource for VideoFileSource {}

/// `Box<dyn JsonSource>` をそのまま `run_session` に渡せるようにする
impl FrameSource for Box<dyn JsonSource> {
    fn next_frame(&mut self) -> Result<Option<SourceFrame>> {
        (**self).next_frame()
    }
}

/// `start` を受けて作った、実行に必要な部品一式
struct Built {
    rec: Box<dyn Recognizer>,
    source: Box<dyn JsonSource>,
    tags: Vec<String>,
    labels: Labels,
    roi: PalmRoiMode,
    select: FrameSelectMode,
    frames: usize,
    model_dir: String,
    /// 前処理を preprocess.json から決めたか(false = live_wizard と同じ既定)
    preprocess_from_model: bool,
}

/// プレビュー設定の確定。省略 → 既定 / null → 無し / 値 → 範囲に丸める(丸めたら warn)
// [TEST向き] 純粋関数に近い(警告文を返す)
pub(super) fn resolve_preview(req: &Option<Option<PreviewCfg>>) -> (Option<PreviewCfg>, Vec<String>) {
    let mut warns = Vec::new();
    let cfg = match req {
        None => PreviewCfg::default(),
        Some(None) => return (None, warns),
        Some(Some(c)) => *c,
    };
    let mut fps = cfg.max_fps;
    if !fps.is_finite() || fps <= 0.0 {
        warns.push(format!(
            "preview.max_fps が不正なので {} にしました",
            DEFAULT_PREVIEW_FPS
        ));
        fps = DEFAULT_PREVIEW_FPS;
    } else if fps > MAX_PREVIEW_FPS {
        warns.push(format!("preview.max_fps は最大 {} です", MAX_PREVIEW_FPS));
        fps = MAX_PREVIEW_FPS;
    }
    let width = cfg.width.clamp(MIN_PREVIEW_WIDTH, MAX_PREVIEW_WIDTH);
    if width != cfg.width {
        warns.push(format!(
            "preview.width は {}〜{} に丸めました",
            MIN_PREVIEW_WIDTH, MAX_PREVIEW_WIDTH
        ));
    }
    (Some(PreviewCfg { max_fps: fps, width }), warns)
}

/// `start` を待つ。EOF / quit / 終了要求なら None(何も始めずに終わる)
fn wait_for_start(rx: &Receiver<Input>, emitter: &Emitter, stop: &StopFlag) -> Option<StartCmd> {
    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Input::Cmd(Command::Start(s))) => return Some(s),
            Ok(Input::Cmd(Command::Quit)) | Ok(Input::Eof) => return None,
            Ok(Input::Cmd(_)) => {
                emitter.message("warn", "start より前のコマンドは無視します");
            }
            Err(RecvTimeoutError::Timeout) => {
                if stop.is_set() || emitter.is_broken() {
                    return None;
                }
            }
            Err(RecvTimeoutError::Disconnected) => return None,
        }
    }
}

/// 全体の流れ: start を待つ → 部品を作る → ready → ループ → 後始末 → stopped。
/// `build` は部品の作り方(本番は ONNX とカメラ/動画、テストでは偽物)。
/// 回復不能な失敗は Err で返す(呼び出し側が `error` イベントにして非 0 で終わる)
#[cfg(test)]
fn serve(
    rx: Receiver<Input>,
    emitter: &Emitter,
    stop: &StopFlag,
    build: impl FnOnce(&StartCmd) -> Result<Built>,
) -> Result<()> {
    let Some(start) = wait_for_start(&rx, emitter, stop) else {
        emitter.send(Event::Stopped { dropped_frames: 0 });
        return Ok(());
    };
    serve_live(rx, emitter, stop, start, build)
}

/// `start` を受けたあとの入口。`mode` で撮影モード(live/capture.rs)かライブ認識かを分ける。
/// **ライブ認識と撮影モードが分かれるのはここだけ**: 撮影モードの録画・保存のコードは `capture` に閉じていて、
/// ライブ認識の経路(`serve_live` 以降、`run_session`、`JsonUi`)から呼ぶ箇所は無い
/// (live/capture.rs のテストで、この 1 か所以外から呼ばれていないことを固定している)
fn serve_dispatch(rx: Receiver<Input>, emitter: &Emitter, stop: &StopFlag) -> Result<()> {
    let Some(start) = wait_for_start(&rx, emitter, stop) else {
        emitter.send(Event::Stopped { dropped_frames: 0 });
        return Ok(());
    };
    match start.mode {
        StartMode::Capture => capture::serve_capture_real(rx, emitter, stop, &start),
        StartMode::Live => serve_live(rx, emitter, stop, start, |s| build_real(s, stop, emitter)),
    }
}

/// ライブ認識の本体(`start` を受け取ったあと)。何も保存しない
fn serve_live(
    rx: Receiver<Input>,
    emitter: &Emitter,
    stop: &StopFlag,
    start: StartCmd,
    build: impl FnOnce(&StartCmd) -> Result<Built>,
) -> Result<()> {
    let built = match build(&start) {
        Ok(b) => b,
        // 部品を作っている最中に終了要求が来て、入力元を開かずに引き返した(カメラを触っていない)
        Err(e) if e.downcast_ref::<StopRequested>().is_some() => {
            emitter.send(Event::Stopped { dropped_frames: 0 });
            return Ok(());
        }
        Err(e) => return Err(e),
    };
    let Built {
        mut rec,
        mut source,
        tags,
        labels,
        roi,
        select,
        frames,
        model_dir,
        preprocess_from_model,
    } = built;

    let seg_mode = start.segmenter.mode();
    let (preview_cfg, preview_warns) = resolve_preview(&start.preview);
    emitter.send(Event::Ready {
        vocab: build_vocab(&tags, &labels),
        model_dir,
        palm_roi: roi.label(),
        frame_select: select.label(),
        frames,
        segmenter: mode_name(seg_mode),
        preprocess: if preprocess_from_model {
            "preprocess.json"
        } else {
            "default"
        },
    });
    if !preprocess_from_model {
        emitter.message(
            "info",
            "モデルに preprocess.json が無いため、既定の前処理(中央正方クロップ + サイン区間)を使います",
        );
    }
    for w in preview_warns {
        emitter.message("warn", w);
    }

    let t0 = Instant::now();
    let mut ui = JsonUi::new(
        emitter.clone(),
        labels,
        roi,
        preview_cfg,
        Box::new(move || t0.elapsed().as_secs_f64()),
    );
    let pending = Rc::new(Cell::new(None));
    let mut segmenter = SwitchableSegmenter::new(seg_mode, Rc::clone(&pending), emitter.clone());
    // 動画ファイル + 手動は、live_wizard と同じく「再生開始と同時に録画を始め、終端で認識」
    let initial = match (&start.source, seg_mode) {
        (SourceSpec::File { .. }, SegmentMode::Manual) => VecDeque::from([Key::Toggle]),
        _ => VecDeque::new(),
    };
    let mut keys = JsonKeys {
        rx,
        initial,
        pending,
        emitter: emitter.clone(),
        stop: stop.clone(),
    };
    let settings = Settings {
        select,
        frames,
        topk: SHOW_TOPK,
    };

    let outcome = run_session(&mut *rec, &mut source, &mut keys, &mut segmenter, &mut ui, &settings);

    // カメラ用 ffmpeg を止めてから stopped を出す(画面が stopped を見たときは、もうカメラは離している)
    let dropped = source.dropped_frames();
    clear_child_pid();
    drop(source);
    match outcome {
        Ok(_) => {}
        // 終了要求の最中に ffmpeg を止めたことによる入力断は、異常ではない
        Err(_) if stop.is_set() => {}
        Err(e) => return Err(e),
    }
    emitter.send(Event::Stopped {
        dropped_frames: dropped,
    });
    Ok(())
}

/// 部品を作る途中で終了要求(quit・標準入力の EOF・SIGTERM)に気づいて引き返したことを表すエラー。
/// `serve` がこれを受けたら、異常ではなく `stopped` だけを出して正常に終わる
#[derive(Debug)]
pub(super) struct StopRequested;

impl std::fmt::Display for StopRequested {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "終了要求を受けたため、入力元を開かずに終わります")
    }
}

impl std::error::Error for StopRequested {}

/// 本番の部品の作り方: モデルと ONNX を読み、入力元(カメラ / 動画)を開く。
/// `stop` は終了要求の旗。モデル読み込みには時間がかかるので、その間に quit などが来ていたら、
/// カメラを起動する前に引き返す。`emitter` は画面への告知(warn)用
fn build_real(start: &StartCmd, stop: &StopFlag, emitter: &Emitter) -> Result<Built> {
    let model_dir = PathBuf::from(start.model_dir.as_deref().unwrap_or(DEFAULT_MODEL_DIR));
    // 前処理(ROI・フレーム選択)は preprocess.json が最優先、無ければ live_wizard と同じ既定
    let (roi, select, frames, from_model) = match load_model_preprocess(&model_dir)? {
        Some(p) => (p.roi, p.select, p.frames.unwrap_or(SEQ_LEN), true),
        None => (PalmRoiMode::CenterSquare, SIGN_SPAN, SEQ_LEN, false),
    };
    let mut engine = Engine::load(&model_dir, roi)?;
    if let Some(dump) = DebugDump::from_env()? {
        let notice = format!(
            "注意: {} が指定されているため、推論に渡したフレームを {} に書き出します(確認後に削除してください)",
            DEBUG_DUMP_ENV,
            dump.dir.display()
        );
        eprintln!("{}", notice);
        // 標準エラーは画面から見えないので、画面にも警告として出す
        emitter.message("warn", notice);
        engine.enable_debug_dump(dump);
    }
    let labels = load_labels();
    let tags = engine.predictor.tags().to_vec();
    // ここまでで時間が経っている。終了要求がもう来ていたら、カメラ(ffmpeg)を起動せずに引き返す
    if stop.is_set() {
        return Err(StopRequested.into());
    }
    let source: Box<dyn JsonSource> = match &start.source {
        SourceSpec::Camera { device } => {
            let device = device
                .as_ref()
                .map(DeviceId::as_string)
                .unwrap_or_else(|| "0".into());
            if device.is_empty() {
                anyhow::bail!("camera の device が空です");
            }
            // 注意: ここでカメラが起動する
            let mut cam = CameraSource::open(&device)?;
            register_child_pid(cam.child.id());
            // ffmpeg が自分から止まって `failure_reason` が回収するときは、回収の前に PID の登録を外す
            // (回収済みの PID に監視スレッドが kill を送らないように)
            cam.set_before_reap(clear_child_pid);
            // 確認から起動までの間に終了要求が来ていたら、監視スレッドはもう PID が無いまま終わっている
            // 可能性があるので、ここで止める
            if stop.is_set() {
                kill_registered_child();
            }
            Box::new(cam)
        }
        SourceSpec::File { path } => {
            let path = PathBuf::from(path);
            if !path.is_file() {
                anyhow::bail!("動画ファイルが見つかりません: {}", path.display());
            }
            let video = VideoFileSource::open(&path, 1)?;
            register_child_pid(video.child.id());
            if stop.is_set() {
                kill_registered_child();
            }
            Box::new(video)
        }
    };
    Ok(Built {
        rec: Box::new(engine),
        source,
        tags,
        labels,
        roi,
        select,
        frames,
        model_dir: model_dir.display().to_string(),
        preprocess_from_model: from_model,
    })
}

/// アプリ専用の入口(`pose-extract --engine-json`)。TTY 不要。設定は標準入力の `start` で受ける
pub fn run_engine_json() -> Result<()> {
    // 標準出力を JSON 専用にする(以降の println! は標準エラーに出る)。シグナルは旗を立てるだけ
    let sink = take_json_sink();
    install_signal_handlers();

    let (emitter, writer) = spawn_writer(sink);
    let stop = StopFlag::default();
    let (tx, rx) = mpsc::channel();
    spawn_command_reader(
        std::io::BufReader::new(std::io::stdin()),
        tx,
        emitter.clone(),
        stop.clone(),
    );
    spawn_stop_watcher(stop.clone());

    let result = serve_dispatch(rx, &emitter, &stop);
    if let Err(e) = &result {
        emitter.send(Event::Error {
            text: format!("{:#}", e),
        });
    }
    writer.finish();
    result
}

// ===== 後始末(子プロセス・シグナル・標準出力)=====
//
// 標準ライブラリだけで済ませるため、必要な少数の C 関数(dup / dup2 / fcntl / signal / kill)は
// `extern "C"` で直接宣言している(これらは std がすでにリンクしている libc のもの。クレートは増やさない)。
// `unsafe` はこの節に閉じ込めている。

#[cfg(unix)]
mod sys {
    use std::os::raw::c_int;
    extern "C" {
        pub fn dup(fd: c_int) -> c_int;
        pub fn dup2(old: c_int, new: c_int) -> c_int;
        /// 注意: C の `fcntl(int, int, ...)` は**可変長引数**。必ず `...` 付きで宣言する。
        /// Apple Silicon の macOS では、可変長引数の渡し方が固定引数と違う(スタックに積む)ので、
        /// 3 つ目の引数付きで固定引数として宣言すると値が正しく届かない。Rust は `...` 付きの宣言なら
        /// 正しい呼び出し規約で呼ぶ
        pub fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
        pub fn signal(signum: c_int, handler: usize) -> usize;
        pub fn kill(pid: c_int, sig: c_int) -> c_int;
    }
    // fcntl のコマンド値。macOS の <sys/fcntl.h> と Linux の <fcntl.h> で同じ値
    // (F_GETFD = 1 / F_SETFD = 2 / FD_CLOEXEC = 1)
    #[cfg(test)]
    pub const F_GETFD: c_int = 1;
    pub const F_SETFD: c_int = 2;
    pub const FD_CLOEXEC: c_int = 1;
}

/// `fd` を複製し、複製に close-on-exec(FD_CLOEXEC)を付けて返す。失敗したら負の値。
///
/// Rust メモ: fd = ファイルディスクリプタ(OS が開いているファイルやパイプに振る番号)。
/// 子プロセス(ffmpeg など)を起動すると、既定では開いている fd がそのまま子に引き継がれる。
/// FD_CLOEXEC を付けた fd は、子プロセスが起動する(exec)瞬間に自動で閉じられるので引き継がれない。
/// `dup` の複製には FD_CLOEXEC が付かないため、JSON 用の書き込み口を ffmpeg が掴んだままになり、
/// エンジンが後始末なしで死んでも画面側に EOF(終わり)が届かなくなる。
/// 複製して付けるまでの間に別スレッドが子プロセスを起動する隙は、起動直後の単一スレッドで呼ぶので無い
#[cfg(unix)]
fn dup_cloexec(fd: std::os::raw::c_int) -> std::os::raw::c_int {
    use std::os::unix::io::FromRawFd;
    // SAFETY: dup / fcntl は fd 番号と整数だけを受け取る。F_SETFD の 3 つ目の引数は int で渡す
    unsafe {
        let new = sys::dup(fd);
        if new >= 0 && sys::fcntl(new, sys::F_SETFD, sys::FD_CLOEXEC) < 0 {
            // 付けられなかった複製は使わない(引き継ぎが残るので)。呼び出し側はフォールバックする
            // 所有権を File に渡して drop し、fd を閉じる
            drop(std::fs::File::from_raw_fd(new));
            return -1;
        }
        new
    }
}

/// `fd` に FD_CLOEXEC が付いているか(テスト用)
#[cfg(all(unix, test))]
fn fd_is_cloexec(fd: std::os::raw::c_int) -> bool {
    // SAFETY: F_GETFD は fd のフラグを読むだけ(第 3 引数なし)
    let flags = unsafe { sys::fcntl(fd, sys::F_GETFD) };
    flags >= 0 && flags & sys::FD_CLOEXEC != 0
}

/// シグナルハンドラが立てる旗(ハンドラの中では旗を立てる以外のことはしない)
static SIGNAL_RECEIVED: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn on_signal(_sig: std::os::raw::c_int) {
    SIGNAL_RECEIVED.store(true, Ordering::SeqCst);
}

/// SIGINT(Ctrl+C)と SIGTERM を「終了要求」として扱う。受けたら監視スレッドが ffmpeg を止め、
/// ループが quit と同じ後始末をして終わる(既定のままだと即死して、Drop が走らない)
fn install_signal_handlers() {
    #[cfg(unix)]
    {
        const SIGINT: std::os::raw::c_int = 2;
        const SIGTERM: std::os::raw::c_int = 15;
        // SAFETY: ハンドラは AtomicBool に store するだけ(非同期シグナルから呼んでよい操作)
        unsafe {
            sys::signal(SIGINT, on_signal as usize);
            sys::signal(SIGTERM, on_signal as usize);
        }
    }
}

/// その PID のプロセスが生きているか(`kill(pid, 0)`: 信号は送らず、存在と権限だけ確かめる)。
/// 撮影モードが、強制終了で残った一時フォルダの持ち主が生きているかを調べるのに使う
pub(super) fn pid_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        // SAFETY: シグナル 0 は何も送らず、プロセスの有無を返すだけ
        let r = unsafe { sys::kill(pid as std::os::raw::c_int, 0) };
        // 権限が無い(EPERM)なら存在はしている
        r == 0 || std::io::Error::last_os_error().raw_os_error() == Some(1)
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        true
    }
}

/// いま動いている入力元(ffmpeg)の PID。終了要求のとき、ループが 1 枚目のフレームを待っている最中でも
/// 止められるように、監視スレッドが使う
static CHILD_PID: Mutex<Option<u32>> = Mutex::new(None);

pub(super) fn register_child_pid(pid: u32) {
    *lock(&CHILD_PID) = Some(pid);
}

/// 入力元を drop する**前**に呼ぶ。以降は監視スレッドが PID に触れない
/// (drop で ffmpeg が回収されたあとに、別プロセスが同じ PID を使い回していても誤って止めないため)
pub(super) fn clear_child_pid() {
    *lock(&CHILD_PID) = None;
}

/// 登録済みの入力元(ffmpeg)を強制終了する。PID はロックの中で使うので、`clear_child_pid` 後には届かない
pub(super) fn kill_registered_child() {
    let guard = lock(&CHILD_PID);
    if let Some(pid) = *guard {
        #[cfg(unix)]
        {
            const SIGKILL: std::os::raw::c_int = 9;
            // SAFETY: まだ回収(wait)されていない子プロセスの PID。clear_child_pid より前にしか届かない
            unsafe {
                sys::kill(pid as std::os::raw::c_int, SIGKILL);
            }
        }
        #[cfg(not(unix))]
        let _ = pid;
    }
}

/// シグナルを終了要求の旗に写し、旗が立ったらカメラ(ffmpeg)を止める監視スレッド。
/// ffmpeg を止めると入力が途切れ、ループは次の `next_frame` で抜ける
fn spawn_stop_watcher(stop: StopFlag) {
    std::thread::spawn(move || loop {
        if SIGNAL_RECEIVED.load(Ordering::SeqCst) {
            stop.set();
        }
        if stop.is_set() {
            kill_registered_child();
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    });
}

/// JSON を書く先を作る。標準出力(fd 1)を複製して JSON 専用にし、fd 1 自体は標準エラーに付け替える。
/// 依存ライブラリや既存コードの `println!`(例: モデル読み込み時の構成表示)が標準出力に混ざって
/// JSON 行を壊すのを、根元から防ぐ。失敗したら普通の標準出力にフォールバックする
fn take_json_sink() -> Box<dyn Write + Send> {
    #[cfg(unix)]
    {
        use std::os::unix::io::FromRawFd;
        // SAFETY: dup_cloexec が返した新しい fd の所有権を File が持つ。dup2 は fd 1 を fd 2 の複製に差し替えるだけ
        // (JSON 用の複製は FD_CLOEXEC 付き。あとで起動する ffmpeg / ffprobe に引き継がれない)
        unsafe {
            let fd = dup_cloexec(1);
            if fd >= 0 {
                if sys::dup2(2, 1) >= 0 {
                    return Box::new(std::fs::File::from_raw_fd(fd));
                }
                drop(std::fs::File::from_raw_fd(fd));
            }
        }
    }
    eprintln!("警告: 標準出力を JSON 専用にできませんでした(println! が混ざる可能性があります)");
    Box::new(std::io::stdout())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use std::sync::atomic::AtomicUsize;

    // ===== ヘルパ =====

    /// 出力スレッドの書き先(テストでは Vec<u8> に溜める)
    #[derive(Clone, Default)]
    struct SharedBuf(Arc<Mutex<Vec<u8>>>);
    impl Write for SharedBuf {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            lock(&self.0).extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl SharedBuf {
        fn events(&self) -> Vec<Value> {
            let text = String::from_utf8(lock(&self.0).clone()).unwrap();
            text.lines()
                .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("JSON でない行: {l:?} ({e})")))
                .collect()
        }
    }

    fn types(ev: &[Value]) -> Vec<String> {
        ev.iter()
            .map(|e| e["type"].as_str().unwrap().to_string())
            .collect()
    }

    fn to_json(ev: &Event) -> Value {
        serde_json::to_value(ev).unwrap()
    }

    // ===== JSON の形(golden)=====

    #[test]
    fn golden_ready() {
        let ev = Event::Ready {
            vocab: vec![VocabEntry {
                tag: "konnichiwa".into(),
                label: "こんにちは".into(),
            }],
            model_dir: "../transformer_burn/models/rec_demo_center_signspan".into(),
            palm_roi: "center-square".into(),
            frame_select: "sign-span(samples=16,max_gap=1)".into(),
            frames: 10,
            segmenter: "auto",
            preprocess: "default",
        };
        assert_eq!(
            to_json(&ev),
            json!({
                "type": "ready",
                "vocab": [{"tag": "konnichiwa", "label": "こんにちは"}],
                "model_dir": "../transformer_burn/models/rec_demo_center_signspan",
                "palm_roi": "center-square",
                "frame_select": "sign-span(samples=16,max_gap=1)",
                "frames": 10,
                "segmenter": "auto",
                "preprocess": "default",
            })
        );
    }

    #[test]
    fn golden_status_exact_text() {
        let ev = Event::Status {
            state: "recording",
            hands: "LR",
            fps: 15.2,
            rec_secs: 1.3,
        };
        assert_eq!(
            serde_json::to_string(&ev).unwrap(),
            r#"{"type":"status","state":"recording","hands":"LR","fps":15.2,"rec_secs":1.3}"#
        );
    }

    #[test]
    fn golden_preview() {
        let ev = Event::Preview {
            width: 640,
            height: 427,
            jpeg_b64: "AAAA".into(),
            hands: HandsPoints {
                left: None,
                right: Some(vec![[0.1, 0.2]; 21]),
            },
            crop: CropRect {
                x: 0.1667,
                y: 0.0,
                w: 0.6667,
                h: 1.0,
            },
        };
        let v = to_json(&ev);
        assert_eq!(v["type"], "preview");
        assert_eq!(v["width"], 640);
        assert_eq!(v["height"], 427);
        assert_eq!(v["jpeg_b64"], "AAAA");
        assert!(v["hands"]["left"].is_null());
        assert_eq!(v["hands"]["right"].as_array().unwrap().len(), 21);
        assert_eq!(v["hands"]["right"][0], json!([0.1, 0.2]));
        assert_eq!(v["crop"], json!({"x": 0.1667, "y": 0.0, "w": 0.6667, "h": 1.0}));
        // キー集合が仕様と一致する(余計なキーが無い)
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(|s| s.as_str()).collect();
        keys.sort();
        assert_eq!(keys, vec!["crop", "hands", "height", "jpeg_b64", "type", "width"]);
    }

    fn live_result() -> LiveResult {
        LiveResult {
            buffered: 71,
            duration_secs: 4.6,
            selection: Selection {
                indices: (0..10).collect(),
                span: Some((28, 59)),
                fps: 15.0,
            },
            selected_with_hand: 7,
            ranked: vec![
                ("konnichiwa".into(), 0.99),
                ("ohayou".into(), 0.006),
                ("arigatou".into(), 0.002),
                ("sumimasen".into(), 0.001),
            ],
        }
    }

    fn test_labels() -> Labels {
        Labels {
            rows: vec![
                ("konnichiwa".into(), "こんにちは".into()),
                ("ohayou".into(), "おはよう".into()),
                ("arigatou".into(), "ありがとう".into()),
            ],
        }
    }

    #[test]
    fn golden_result() {
        let ev = result_event(&live_result(), &test_labels()).unwrap();
        let v = to_json(&ev);
        assert_eq!(
            v,
            json!({
                "type": "result",
                "tag": "konnichiwa",
                "label": "こんにちは",
                "p": 0.99,
                "ranked": [
                    {"tag": "konnichiwa", "label": "こんにちは", "p": 0.99},
                    {"tag": "ohayou", "label": "おはよう", "p": 0.006},
                    {"tag": "arigatou", "label": "ありがとう", "p": 0.002},
                ],
                "low_confidence": false,
                "span": [28, 59],
                "buffered": 71,
                "duration_secs": 4.6,
                "selected_with_hand": 7,
                "selected_total": 10,
            })
        );
    }

    #[test]
    fn result_low_confidence_boundary_and_null_span() {
        let mut r = live_result();
        r.selection.span = None;
        // 閾値ちょうど(0.5)は自信あり(format_result と同じ判定)
        r.ranked = vec![("unknown_tag".into(), 0.5)];
        let v = to_json(&result_event(&r, &test_labels()).unwrap());
        assert_eq!(v["low_confidence"], false);
        assert!(v["span"].is_null());
        // 語彙表に無いタグは label がタグ名のまま
        assert_eq!(v["label"], "unknown_tag");
        r.ranked = vec![("konnichiwa".into(), 0.4999)];
        let v = to_json(&result_event(&r, &test_labels()).unwrap());
        assert_eq!(v["low_confidence"], true);
        // 候補が空なら None
        r.ranked.clear();
        assert!(result_event(&r, &test_labels()).is_none());
    }

    #[test]
    fn result_numbers_are_finite() {
        let mut r = live_result();
        r.duration_secs = f64::NAN;
        r.ranked = vec![("konnichiwa".into(), f32::NAN), ("ohayou".into(), f32::INFINITY)];
        let text = serde_json::to_string(&result_event(&r, &test_labels()).unwrap()).unwrap();
        // serde_json は NaN / Inf を `null` にしてしまう(エラーにならない)ので、文字列に NaN が無いかだけでは足りない。
        // 数値であるべき項目が `null` になっていない(= 有限な数に直されている)ことを確かめる
        assert!(!text.contains("NaN") && !text.contains("Infinity"), "{text}");
        let v: Value = serde_json::from_str(&text).unwrap();
        for (name, x) in [
            ("p", &v["p"]),
            ("duration_secs", &v["duration_secs"]),
            ("ranked[0].p", &v["ranked"][0]["p"]),
            ("ranked[1].p", &v["ranked"][1]["p"]),
        ] {
            let n = x.as_f64().unwrap_or_else(|| panic!("{name} が数値でない(null になった?): {text}"));
            assert!(n.is_finite(), "{name}: {text}");
        }
        assert_eq!(v["p"], 0.0);
        assert_eq!(v["duration_secs"], 0.0);
        assert_eq!(v["ranked"][1]["p"], 0.0);
    }

    #[test]
    fn golden_message_error_stopped_segmenter() {
        assert_eq!(
            serde_json::to_string(&Event::Message {
                level: "warn",
                text: "手が検出されませんでした".into()
            })
            .unwrap(),
            r#"{"type":"message","level":"warn","text":"手が検出されませんでした"}"#
        );
        assert_eq!(
            serde_json::to_string(&Event::Error { text: "x".into() }).unwrap(),
            r#"{"type":"error","text":"x"}"#
        );
        assert_eq!(
            serde_json::to_string(&Event::Stopped { dropped_frames: 764 }).unwrap(),
            r#"{"type":"stopped","dropped_frames":764}"#
        );
        assert_eq!(
            serde_json::to_string(&Event::Segmenter { mode: "manual" }).unwrap(),
            r#"{"type":"segmenter","mode":"manual"}"#
        );
    }

    // 撮影モードのイベントの形(golden)。画面(desktop/src/protocol.rs)の parse_capture_state_and_saved と同じ形
    #[test]
    fn golden_capture_state() {
        let ev = Event::CaptureState {
            phase: "judged",
            countdown_left: 0.0,
            rec_secs: 4.2,
            max_secs: 10.0,
            fps: Some(30.0),
            dropped: 0,
            word: Some(CaptureWord {
                word_id: "011".into(),
                romaji: "kare".into(),
                label: "かれ".into(),
                rep_idx: 4,
            }),
            data_dir: "/data/raw_jsl".into(),
            verdict: Some(VerdictView {
                usable: false,
                reasons: vec!["hands_short"],
                summary: "使えない: 手が 1.0 秒続いて検出されていません(最長 0.67 秒)".into(),
                longest_run_secs: 0.67,
                left_run_secs: 0.0,
                right_run_secs: 0.67,
                duration_secs: 4.2,
                quality_flag: "ng_hands".into(),
            }),
            notes: vec!["注意".into()],
        };
        assert_eq!(
            to_json(&ev),
            json!({
                "type": "capture_state",
                "phase": "judged",
                "countdown_left": 0.0,
                "rec_secs": 4.2,
                "max_secs": 10.0,
                "fps": 30.0,
                "dropped": 0,
                "word": {"word_id": "011", "romaji": "kare", "label": "かれ", "rep_idx": 4},
                "data_dir": "/data/raw_jsl",
                "verdict": {
                    "usable": false,
                    "reasons": ["hands_short"],
                    "summary": "使えない: 手が 1.0 秒続いて検出されていません(最長 0.67 秒)",
                    "longest_run_secs": 0.67,
                    "left_run_secs": 0.0,
                    "right_run_secs": 0.67,
                    "duration_secs": 4.2,
                    "quality_flag": "ng_hands"
                },
                "notes": ["注意"]
            })
        );
        // 待機: word / verdict / fps は null
        let idle = Event::CaptureState {
            phase: "idle",
            countdown_left: 0.0,
            rec_secs: 0.0,
            max_secs: 10.0,
            fps: None,
            dropped: 0,
            word: None,
            data_dir: "/data/raw_jsl".into(),
            verdict: None,
            notes: vec![],
        };
        let v = to_json(&idle);
        assert!(v["word"].is_null() && v["verdict"].is_null() && v["fps"].is_null(), "{v}");
        assert_eq!(v["phase"], "idle");
        assert_eq!(v["notes"], json!([]));
    }

    #[test]
    fn golden_capture_saved() {
        let ev = Event::CaptureSaved {
            file_path: "011_kare/04.mp4".into(),
            abs_path: "/data/raw_jsl/011_kare/04.mp4".into(),
            word_id: "011".into(),
            label: "かれ".into(),
            rep_idx: 4,
            quality_flag: "ok".into(),
            usable: true,
            longest_run_secs: 1.67,
            duration_secs: 4.2,
            data_dir: "/data/raw_jsl".into(),
        };
        assert_eq!(
            to_json(&ev),
            json!({
                "type": "capture_saved",
                "file_path": "011_kare/04.mp4",
                "abs_path": "/data/raw_jsl/011_kare/04.mp4",
                "word_id": "011",
                "label": "かれ",
                "rep_idx": 4,
                "quality_flag": "ok",
                "usable": true,
                "longest_run_secs": 1.67,
                "duration_secs": 4.2,
                "data_dir": "/data/raw_jsl"
            })
        );
    }

    // 撮影モードのコマンドと start のモード指定(画面の capture_command_lines と同じ形を読める)
    #[test]
    fn parse_capture_commands_and_start_mode() {
        assert_eq!(
            parse_command(r#"{"cmd":"capture_record","word_id":"011"}"#).unwrap().unwrap(),
            Command::CaptureRecord { word_id: Some("011".into()) }
        );
        assert_eq!(
            parse_command(r#"{"cmd":"capture_record"}"#).unwrap().unwrap(),
            Command::CaptureRecord { word_id: None }
        );
        assert_eq!(parse_command(r#"{"cmd":"capture_stop"}"#).unwrap().unwrap(), Command::CaptureStop);
        assert_eq!(
            parse_command(r#"{"cmd":"capture_accept","confirm_ng":true}"#).unwrap().unwrap(),
            Command::CaptureAccept { confirm_ng: true }
        );
        assert_eq!(
            parse_command(r#"{"cmd":"capture_accept"}"#).unwrap().unwrap(),
            Command::CaptureAccept { confirm_ng: false },
            "confirm_ng 省略 = false(使えないテイクは確認なしでは記録しない)"
        );
        assert_eq!(parse_command(r#"{"cmd":"capture_discard"}"#).unwrap().unwrap(), Command::CaptureDiscard);
        // start: mode 省略 = live。capture では data_dir / countdown_secs を受ける
        let Command::Start(s) = parse_command(r#"{"cmd":"start","source":{"kind":"camera"}}"#).unwrap().unwrap() else {
            panic!()
        };
        assert_eq!((s.mode, s.data_dir, s.countdown_secs), (StartMode::Live, None, None));
        let Command::Start(s) = parse_command(
            r#"{"cmd":"start","mode":"capture","source":{"kind":"file","path":"/v.mp4"},"data_dir":"/d","countdown_secs":0}"#,
        )
        .unwrap()
        .unwrap() else {
            panic!()
        };
        assert_eq!(s.mode, StartMode::Capture);
        assert_eq!(s.data_dir.as_deref(), Some("/d"));
        assert_eq!(s.countdown_secs, Some(0.0));
        assert!(parse_command(r#"{"cmd":"start","mode":"nope","source":{"kind":"camera"}}"#).is_err());
    }

    #[test]
    fn message_classification() {
        assert_eq!(
            classify_message(DISCARD_MESSAGE),
            ("info", "短すぎる動きを無視しました".to_string())
        );
        let (lv, text) = classify_message("手が検出されませんでした(認識せず)。もう一度どうぞ");
        assert_eq!(lv, "warn");
        assert!(text.starts_with("手が検出されませんでした"));
    }

    #[test]
    fn status_hands_names() {
        assert_eq!(hands_name(true, true), "LR");
        assert_eq!(hands_name(true, false), "L");
        assert_eq!(hands_name(false, true), "R");
        assert_eq!(hands_name(false, false), "none");
    }

    // ===== コマンドのパース =====

    #[test]
    fn parse_start_camera_and_file() {
        let c = parse_command(
            r#"{"cmd":"start","source":{"kind":"camera","device":"0"},"model_dir":"m","segmenter":"manual","preview":{"max_fps":5,"width":320}}"#,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            c,
            Command::Start(StartCmd {
            mode: StartMode::Live,
            data_dir: None,
            countdown_secs: None,
                source: SourceSpec::Camera {
                    device: Some(DeviceId::Text("0".into()))
                },
                model_dir: Some("m".into()),
                segmenter: SegKind::Manual,
                preview: Some(Some(PreviewCfg {
                    max_fps: 5.0,
                    width: 320
                })),
            })
        );
        let c = parse_command(r#"{"cmd":"start","source":{"kind":"file","path":"/a/b.mp4"},"segmenter":"auto"}"#)
            .unwrap()
            .unwrap();
        let Command::Start(s) = c else { panic!() };
        assert_eq!(s.source, SourceSpec::File { path: "/a/b.mp4".into() });
        assert_eq!(s.model_dir, None);
        assert_eq!(s.preview, None, "省略 = 既定のプレビュー");
    }

    #[test]
    fn parse_start_defaults_and_preview_three_states() {
        // segmenter 省略 = auto、device 数値も受ける
        let Command::Start(s) = parse_command(r#"{"cmd":"start","source":{"kind":"camera","device":1}}"#)
            .unwrap()
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(s.segmenter, SegKind::Auto);
        assert_eq!(
            s.source,
            SourceSpec::Camera {
                device: Some(DeviceId::Number(1))
            }
        );
        assert_eq!(DeviceId::Number(1).as_string(), "1");
        // preview: null = プレビュー無し
        let Command::Start(s) = parse_command(r#"{"cmd":"start","source":{"kind":"camera"},"preview":null}"#)
            .unwrap()
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(s.preview, Some(None));
        // preview: {} = 項目は既定値
        let Command::Start(s) = parse_command(r#"{"cmd":"start","source":{"kind":"camera"},"preview":{}}"#)
            .unwrap()
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(s.preview, Some(Some(PreviewCfg::default())));
        // 知らない項目は無視(将来の拡張に強い)
        assert!(parse_command(r#"{"cmd":"start","source":{"kind":"camera"},"future":1}"#).is_ok());
    }

    #[test]
    fn parse_simple_commands() {
        assert_eq!(parse_command(r#"{"cmd":"toggle"}"#).unwrap(), Some(Command::Toggle));
        assert_eq!(parse_command(r#"{"cmd":"quit"}"#).unwrap(), Some(Command::Quit));
        assert_eq!(
            parse_command(r#" {"cmd":"set_segmenter","mode":"manual"} "#).unwrap(),
            Some(Command::SetSegmenter {
                mode: SegKind::Manual
            })
        );
        // 空行は無視
        assert_eq!(parse_command("").unwrap(), None);
        assert_eq!(parse_command("  \r\n").unwrap(), None);
    }

    #[test]
    fn parse_rejects_bad_input() {
        for bad in [
            "not json",
            "{",
            "[]",
            "42",
            r#"{"cmd":"explode"}"#,
            r#"{"nocmd":1}"#,
            r#"{"cmd":"set_segmenter"}"#,
            r#"{"cmd":"set_segmenter","mode":"turbo"}"#,
            r#"{"cmd":"start"}"#,
            r#"{"cmd":"start","source":{"kind":"satellite"}}"#,
            r#"{"cmd":"start","source":{"kind":"file"}}"#,
        ] {
            assert!(parse_command(bad).is_err(), "拒否されるべき: {bad}");
        }
    }

    // ===== プレビュー設定の確定 =====

    #[test]
    fn resolve_preview_cases() {
        assert_eq!(resolve_preview(&None).0, Some(PreviewCfg::default()));
        assert_eq!(resolve_preview(&Some(None)).0, None);
        let (cfg, warns) = resolve_preview(&Some(Some(PreviewCfg { max_fps: 5.0, width: 320 })));
        assert_eq!(cfg, Some(PreviewCfg { max_fps: 5.0, width: 320 }));
        assert!(warns.is_empty());
        for bad_fps in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let (cfg, warns) = resolve_preview(&Some(Some(PreviewCfg { max_fps: bad_fps, width: 640 })));
            assert_eq!(cfg.unwrap().max_fps, DEFAULT_PREVIEW_FPS);
            assert_eq!(warns.len(), 1);
        }
        let (cfg, _) = resolve_preview(&Some(Some(PreviewCfg { max_fps: 1000.0, width: 0 })));
        let cfg = cfg.unwrap();
        assert_eq!(cfg.max_fps, MAX_PREVIEW_FPS);
        assert_eq!(cfg.width, MIN_PREVIEW_WIDTH);
    }

    // ===== 間引き =====

    #[test]
    fn throttle_boundaries() {
        let mut t = Throttle::new(10.0);
        assert!(t.allow(0.0), "最初は必ず通る");
        assert!(!t.allow(0.05));
        assert!(!t.allow(0.0999));
        assert!(t.allow(0.1), "間隔ちょうどは通る");
        assert!(!t.allow(0.15));
        assert!(t.allow(0.2));
        // 浮動小数の誤差: 0.3 - 0.2 = 0.09999999999999998 でも 0.1 秒として扱う
        assert!(t.allow(0.3));
        // 時刻が巻き戻っても壊れない(通らないだけ)
        assert!(!t.allow(0.0));
        // 無効な頻度(0 以下)は 2 回目以降を通さない
        let mut off = Throttle::new(0.0);
        assert!(off.allow(0.0));
        assert!(!off.allow(1e9));
        // mark は間引き判定を通さず「出した」ことにする
        let mut m = Throttle::new(10.0);
        m.mark(1.0);
        assert!(!m.allow(1.05));
        assert!(m.allow(1.1));
    }

    /// 一定のフレームレート `fps` で `secs` 秒分の時刻を流して、通った回数を数える
    fn throttle_passes(max_hz: f64, fps: f64, secs: f64) -> usize {
        let mut t = Throttle::new(max_hz);
        let n = (fps * secs) as usize;
        (0..n).filter(|i| t.allow(*i as f64 / fps)).count()
    }

    // 約 15fps のカメラで 10 回/秒を指定しても、1 枚おき(約 7.5 回/秒)にならず約 10 回/秒になる。
    // 「前回通した時刻から 1 間隔」で判定する旧方式では 75 回(7.5 回/秒)だった
    #[test]
    fn throttle_15fps_source_gets_about_10_per_second() {
        let n = throttle_passes(10.0, 15.0, 10.0);
        assert!((98..=102).contains(&n), "10 秒で {n} 回(期待は約 100)");
    }

    // 約 30fps(3 枚に 1 枚)・60fps(6 枚に 1 枚)でも約 10 回/秒
    #[test]
    fn throttle_30fps_and_60fps_sources_get_10_per_second() {
        let n30 = throttle_passes(10.0, 30.0, 10.0);
        assert!((99..=101).contains(&n30), "30fps: 10 秒で {n30} 回");
        let n60 = throttle_passes(10.0, 60.0, 10.0);
        assert!((99..=101).contains(&n60), "60fps: 10 秒で {n60} 回");
        // 指定より遅い入力(8fps)は、あるだけ全部通る
        assert_eq!(throttle_passes(10.0, 8.0, 10.0), 80);
    }

    // 長く止まったあとに、溜まった分を連続して通さない(バーストしない)
    #[test]
    fn throttle_does_not_burst_after_a_stall() {
        let mut t = Throttle::new(10.0);
        for i in 0..30 {
            t.allow(i as f64 / 30.0);
        }
        // 100 秒止まったあと、30fps が再開。最初の 0.1 秒未満(3 枚)のうち通るのは 1 枚だけ
        let base = 100.0;
        let first: Vec<bool> = (0..3).map(|i| t.allow(base + i as f64 / 30.0)).collect();
        assert_eq!(first, vec![true, false, false], "止まった直後に連続して通った");
        // その後も 1 秒あたり約 10 回
        let n = (3..33).filter(|i| t.allow(base + *i as f64 / 30.0)).count();
        assert!((9..=11).contains(&n), "再開後 1 秒で {n} 回");
        // 1 間隔ちょうど遅れて来ても、2 回続けて通らない
        let mut u = Throttle::new(10.0);
        assert!(u.allow(0.0));
        assert!(u.allow(0.35));
        assert!(!u.allow(0.36));
        assert!(!u.allow(0.44));
        assert!(u.allow(0.45));
    }

    #[test]
    fn json_ui_status_throttles_but_state_change_is_immediate() {
        let buf = SharedBuf::default();
        let (emitter, writer) = spawn_writer(buf.clone());
        // 1 フレーム 0.02 秒(50fps)で進む偽の時計
        let now = Rc::new(Cell::new(0.0));
        let clock = {
            let now = Rc::clone(&now);
            Box::new(move || now.get())
        };
        let mut ui = JsonUi::new(emitter.clone(), test_labels(), PalmRoiMode::CenterSquare, None, clock);
        let st = |recording: bool| StatusLine {
            recording,
            rec_secs: 0.0,
            left: false,
            right: true,
            fps: 15.0,
        };
        // 1 秒ぶん(50 フレーム)待機 → 通るのは 1 + 10 回(0.00, 0.10, ..., 0.98 まで)
        for i in 0..50 {
            now.set(i as f64 * 0.02);
            ui.status(&st(false));
        }
        // 直後(0.02 秒後)に記録中へ変わる → 間引かれずに即時
        now.set(0.98 + 0.001);
        ui.status(&st(true));
        drop(ui);
        drop(emitter);
        writer.finish();
        let ev = buf.events();
        let idle = ev.iter().filter(|e| e["state"] == "idle").count();
        let rec = ev.iter().filter(|e| e["state"] == "recording").count();
        assert_eq!(idle, 10, "{ev:?}");
        assert_eq!(rec, 1);
        assert_eq!(ev.last().unwrap()["state"], "recording");
    }

    // ===== 手の取り出し・クロップ・縮小 =====

    fn feature_with(left: Option<f32>, right: Option<f32>) -> [f32; DICT_FEATURE_DIM] {
        let mut f = [0.0f32; DICT_FEATURE_DIM];
        if let Some(base) = left {
            for k in 0..21 {
                f[k * 3] = base + k as f32 * 0.01; // x
                f[k * 3 + 1] = 0.5; // y
                f[k * 3 + 2] = 9.9; // z は出力に含めない
            }
        }
        if let Some(base) = right {
            for k in 0..21 {
                f[63 + k * 3] = base + k as f32 * 0.01;
                f[63 + k * 3 + 1] = 0.25;
                f[63 + k * 3 + 2] = 9.9;
            }
        }
        f
    }

    #[test]
    fn hands_extraction_layout() {
        // 左だけ
        let h = hands_from_feature(&feature_with(Some(0.1), None));
        assert!(h.right.is_none());
        let l = h.left.unwrap();
        assert_eq!(l.len(), 21);
        assert_eq!(l[0], [0.1, 0.5]);
        assert_eq!(l[20], [0.3, 0.5]);
        // 右だけ
        let h = hands_from_feature(&feature_with(None, Some(0.6)));
        assert!(h.left.is_none());
        assert_eq!(h.right.unwrap()[1], [0.61, 0.25]);
        // 両手
        let h = hands_from_feature(&feature_with(Some(0.1), Some(0.6)));
        assert!(h.left.is_some() && h.right.is_some());
        // 未検出(全ゼロ)
        let h = hands_from_feature(&[0.0; DICT_FEATURE_DIM]);
        assert!(h.left.is_none() && h.right.is_none());
        // 非有限を含む枠は null(NaN を JSON に出さない)
        let mut f = feature_with(Some(0.1), Some(0.6));
        f[5] = f32::NAN;
        f[70] = f32::INFINITY;
        let h = hands_from_feature(&f);
        assert!(h.left.is_none() && h.right.is_none());
        // JSON にしても NaN / Infinity が出ない
        let text = serde_json::to_string(&h).unwrap();
        assert_eq!(text, r#"{"left":null,"right":null}"#);
    }

    #[test]
    fn crop_rect_cases() {
        // 1620x1080 の中央正方クロップ(270,0,1080,1080) → 仕様の例の値
        let c = crop_rect(PalmRoiMode::CenterSquare, 1620, 1080);
        assert_eq!(
            c,
            CropRect {
                x: 0.1667,
                y: 0.0,
                w: 0.6667,
                h: 1.0
            }
        );
        let full = CropRect {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
        };
        assert_eq!(crop_rect(PalmRoiMode::FullFrame, 1620, 1080), full);
        // 複数タイル・ゼロサイズは全体
        assert_eq!(crop_rect(PalmRoiMode::Tiles { count: 3 }, 1620, 1080), full);
        assert_eq!(crop_rect(PalmRoiMode::CenterSquare, 0, 0), full);
        // 相対矩形はそのまま
        let c = crop_rect(
            PalmRoiMode::Relative {
                x: 0.25,
                y: 0.0,
                w: 0.5,
                h: 1.0,
            },
            1000,
            500,
        );
        assert_eq!(c, CropRect { x: 0.25, y: 0.0, w: 0.5, h: 1.0 });
    }

    #[test]
    fn preview_size_keeps_aspect_and_never_upscales() {
        assert_eq!(preview_size(1620, 1080, 640), (640, 427));
        assert_eq!(preview_size(1620, 1080, 4096), (1620, 1080));
        assert_eq!(preview_size(4, 2, 640), (4, 2));
        assert_eq!(preview_size(1, 1, 16), (1, 1));
    }

    #[test]
    fn downscale_averages_boxes_and_does_not_touch_source() {
        // 4x4 の画素値 = 列番号 * 10(全チャンネル同じ)。2x2 に縮小すると各箱 (0,10)→5 と (20,30)→25
        let mut src = Array3::<u8>::zeros((4, 4, 3));
        for y in 0..4 {
            for x in 0..4 {
                for c in 0..3 {
                    src[[y, x, c]] = (x * 10) as u8;
                }
            }
        }
        let before = src.clone();
        let out = downscale_rgb(&src, 2, 2);
        assert_eq!(out.len(), 2 * 2 * 3);
        assert_eq!(&out[0..3], &[5, 5, 5]);
        assert_eq!(&out[3..6], &[25, 25, 25]);
        assert_eq!(&out[6..9], &[5, 5, 5]);
        assert_eq!(src, before, "元の配列は変わらない");
        // 同サイズ = そのまま、1x1 = 全体の平均
        assert_eq!(downscale_rgb(&src, 4, 4)[3..6], [10, 10, 10]);
        assert_eq!(downscale_rgb(&src, 1, 1), vec![15, 15, 15]);
        // 割り切れない縮小(5x3 → 2x2)でも全画素が範囲内で落ちない
        let odd = Array3::<u8>::from_elem((3, 5, 3), 200);
        assert!(downscale_rgb(&odd, 2, 2).iter().all(|v| *v == 200));
    }

    #[test]
    fn base64_matches_rfc4648_vectors() {
        for (input, expect) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64_encode(input.as_bytes()), expect);
        }
        // 全バイト値で '+' '/' が使われる
        let all: Vec<u8> = (0..=255u8).collect();
        let enc = base64_encode(&all);
        assert!(enc.contains('+') && enc.contains('/'));
    }

    #[test]
    fn jpeg_encoding_roundtrips_dimensions() {
        let (w, h) = (32u32, 18u32);
        let rgb: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 251) as u8).collect();
        let jpeg = encode_jpeg(&rgb, w, h).unwrap();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "JPEG の先頭");
        let img = image::load_from_memory(&jpeg).unwrap();
        assert_eq!((img.width(), img.height()), (w, h));
        // preview_event は base64 に包む
        let ev = preview_event(PreviewJob {
            width: w,
            height: h,
            rgb,
            hands: HandsPoints { left: None, right: None },
            crop: CropRect { x: 0.0, y: 0.0, w: 1.0, h: 1.0 },
        })
        .unwrap();
        let v = to_json(&ev);
        assert_eq!(v["jpeg_b64"].as_str().unwrap(), base64_encode(&jpeg));
    }

    #[test]
    fn vocab_order_follows_words_tsv_and_filters_by_model() {
        let labels = test_labels();
        // モデルは ohayou, konnichiwa, extra だけ。words.tsv の並び(konnichiwa → ohayou)で、
        // words.tsv に無い extra は末尾にタグ名のまま。arigatou はモデルに無いので出ない
        let tags: Vec<String> = ["ohayou", "extra", "konnichiwa"].map(String::from).to_vec();
        let v = build_vocab(&tags, &labels);
        assert_eq!(
            v,
            vec![
                VocabEntry { tag: "konnichiwa".into(), label: "こんにちは".into() },
                VocabEntry { tag: "ohayou".into(), label: "おはよう".into() },
                VocabEntry { tag: "extra".into(), label: "extra".into() },
            ]
        );
    }

    // ===== 標準入力の読み取りスレッド =====

    fn recv_all(rx: &Receiver<Input>) -> Vec<String> {
        let mut out = Vec::new();
        while let Ok(i) = rx.recv_timeout(Duration::from_secs(5)) {
            let end = matches!(i, Input::Eof);
            out.push(format!("{i:?}"));
            if end {
                break;
            }
        }
        out
    }

    #[test]
    fn command_reader_forwards_commands_warns_on_bad_lines_and_ends_with_eof() {
        let input: &[u8] = b"{\"cmd\":\"toggle\"}\nthis is not json\n\n{\"cmd\":\"boom\"}\n\xff\xfe broken utf8\n{\"cmd\":\"set_segmenter\",\"mode\":\"manual\"}\n";
        let buf = SharedBuf::default();
        let (emitter, writer) = spawn_writer(buf.clone());
        let (tx, rx) = mpsc::channel();
        let stop = StopFlag::default();
        let h = spawn_command_reader(
            std::io::Cursor::new(input.to_vec()),
            tx,
            emitter.clone(),
            stop.clone(),
        );
        let got = recv_all(&rx);
        h.join().unwrap();
        assert_eq!(got.len(), 3, "toggle, set_segmenter, Eof: {got:?}");
        assert!(got[0].contains("Toggle"));
        assert!(got[1].contains("SetSegmenter"));
        assert!(got[2].contains("Eof"));
        assert!(stop.is_set(), "EOF で終了要求の旗が立つ");
        drop(emitter);
        writer.finish();
        let ev = buf.events();
        // 不正な 3 行(not json / boom / 壊れた UTF-8)が warn になり、空行は黙って無視される
        assert_eq!(ev.len(), 3, "{ev:?}");
        assert!(ev.iter().all(|e| e["type"] == "message" && e["level"] == "warn"));
    }

    #[test]
    fn command_reader_stops_reading_after_quit() {
        let input = "{\"cmd\":\"quit\"}\n{\"cmd\":\"toggle\"}\n";
        let (emitter, writer) = spawn_writer(SharedBuf::default());
        let (tx, rx) = mpsc::channel();
        let stop = StopFlag::default();
        spawn_command_reader(std::io::Cursor::new(input.as_bytes().to_vec()), tx, emitter, stop.clone())
            .join()
            .unwrap();
        assert!(stop.is_set());
        assert!(matches!(rx.try_recv(), Ok(Input::Cmd(Command::Quit))));
        assert!(rx.try_recv().is_err(), "quit の後のコマンドは読まない");
        writer.finish();
    }

    // ===== ループ全体(偽の入力元・認識器。ONNX・カメラ不要)=====

    /// 左上画素が 0 以外のフレームを「右手あり」とみなす認識器
    struct FakeRec;
    impl Recognizer for FakeRec {
        fn observe(&mut self, frame: &Array3<u8>) -> Result<Observation> {
            let has = frame[[0, 0, 0]] > 0;
            let mut feature = [0.0f32; DICT_FEATURE_DIM];
            if has {
                for k in 0..21 {
                    feature[63 + k * 3] = 0.5;
                    feature[63 + k * 3 + 1] = 0.5;
                }
            }
            Ok(Observation {
                feature,
                left: false,
                right: has,
            })
        }
        fn predict(&self, flat: &[f32], frames: usize, k: usize) -> Vec<(String, f32)> {
            assert_eq!(flat.len(), frames * DICT_FEATURE_DIM);
            vec![
                ("konnichiwa".to_string(), 0.9),
                ("ohayou".to_string(), 0.07),
                ("arigatou".to_string(), 0.03),
            ]
            .into_iter()
            .take(k)
            .collect()
        }
    }

    /// 値の並びを 1 フレームずつ返す入力元(フレームは 8x6 画素)。10fps。
    /// `endless` なら尽きず、最後の値を繰り返す(EOF で止まることの確認用)
    struct FakeSrc {
        values: VecDeque<u8>,
        last: u8,
        endless: bool,
        idx: usize,
        dropped_probe: Arc<AtomicBool>,
        frames_served: Arc<AtomicUsize>,
    }
    impl FrameSource for FakeSrc {
        fn next_frame(&mut self) -> Result<Option<SourceFrame>> {
            let v = match self.values.pop_front() {
                Some(v) => {
                    self.last = v;
                    v
                }
                None if self.endless => {
                    std::thread::sleep(Duration::from_millis(1));
                    self.last
                }
                None => return Ok(None),
            };
            let t = self.idx as f64 / 10.0;
            self.idx += 1;
            self.frames_served.fetch_add(1, Ordering::SeqCst);
            Ok(Some(SourceFrame {
                frame: Array3::from_elem((6, 8, 3), v),
                t,
            }))
        }
    }
    impl JsonSource for FakeSrc {
        fn dropped_frames(&self) -> u64 {
            764
        }
    }
    impl Drop for FakeSrc {
        fn drop(&mut self) {
            self.dropped_probe.store(true, Ordering::SeqCst);
        }
    }

    struct Probes {
        dropped: Arc<AtomicBool>,
        served: Arc<AtomicUsize>,
    }

    fn fake_built(values: Vec<u8>, endless: bool, roi: PalmRoiMode) -> (Built, Probes) {
        let probes = Probes {
            dropped: Arc::new(AtomicBool::new(false)),
            served: Arc::new(AtomicUsize::new(0)),
        };
        let src = FakeSrc {
            values: values.into(),
            last: 0,
            endless,
            idx: 0,
            dropped_probe: Arc::clone(&probes.dropped),
            frames_served: Arc::clone(&probes.served),
        };
        let built = Built {
            rec: Box::new(FakeRec),
            source: Box::new(src),
            tags: ["ohayou", "konnichiwa"].map(String::from).to_vec(),
            labels: test_labels(),
            roi,
            select: SIGN_SPAN,
            frames: 10,
            model_dir: "fake_model".into(),
            preprocess_from_model: true,
        };
        (built, probes)
    }

    fn frames_from(parts: &[(u8, usize)]) -> Vec<u8> {
        parts.iter().flat_map(|&(v, n)| vec![v; n]).collect()
    }

    fn start_file(seg: SegKind, preview: Option<Option<PreviewCfg>>) -> StartCmd {
        StartCmd {
            mode: StartMode::Live,
            data_dir: None,
            countdown_secs: None,
            source: SourceSpec::File { path: "dummy.mp4".into() },
            model_dir: None,
            segmenter: seg,
            preview,
        }
    }

    /// serve を 1 回走らせ、(出力イベント, 結果) を返す。`tx` は返り値で生かしておく(EOF にならないように)
    fn run_fake(
        inputs: Vec<Input>,
        built: Built,
        keep_open: bool,
    ) -> (Vec<Value>, Result<()>, Option<mpsc::Sender<Input>>) {
        let buf = SharedBuf::default();
        let (emitter, writer) = spawn_writer(buf.clone());
        let (tx, rx) = mpsc::channel();
        for i in inputs {
            tx.send(i).unwrap();
        }
        let stop = StopFlag::default();
        let res = serve(rx, &emitter, &stop, |_| Ok(built));
        drop(emitter);
        writer.finish();
        (buf.events(), res, keep_open.then_some(tx))
    }

    // ready → status/preview/message → result → stopped の順序
    #[test]
    fn full_loop_event_order_and_content() {
        let (built, probes) = fake_built(
            frames_from(&[(0, 10), (1, 15), (0, 15)]),
            false,
            PalmRoiMode::CenterSquare,
        );
        let preview = Some(Some(PreviewCfg { max_fps: 30.0, width: 8 }));
        let (ev, res, _tx) = run_fake(
            vec![Input::Cmd(Command::Start(start_file(SegKind::Auto, preview)))],
            built,
            true,
        );
        res.unwrap();
        let t = types(&ev);
        assert_eq!(t.first().unwrap(), "ready", "{t:?}");
        assert_eq!(t.last().unwrap(), "stopped", "{t:?}");
        assert_eq!(t.iter().filter(|x| *x == "ready").count(), 1);
        assert_eq!(t.iter().filter(|x| *x == "stopped").count(), 1);
        assert_eq!(t.iter().filter(|x| *x == "result").count(), 1, "{t:?}");
        assert!(t.iter().any(|x| x == "status"));
        assert!(t.iter().any(|x| x == "preview"));
        // result は ready の後、stopped の前
        let ri = t.iter().position(|x| x == "result").unwrap();
        assert!(ri > 0 && ri < t.len() - 1);
        // 内容
        let ready = &ev[0];
        assert_eq!(ready["segmenter"], "auto");
        assert_eq!(ready["palm_roi"], "center-square");
        assert_eq!(ready["frames"], 10);
        assert_eq!(ready["model_dir"], "fake_model");
        assert_eq!(ready["vocab"].as_array().unwrap().len(), 2);
        assert_eq!(ready["vocab"][0]["tag"], "konnichiwa", "words.tsv の並び");
        let result = &ev[ri];
        assert_eq!(result["tag"], "konnichiwa");
        assert_eq!(result["label"], "こんにちは");
        assert_eq!(result["low_confidence"], false);
        assert_eq!(result["selected_total"], 10);
        assert!(result["span"].is_array());
        assert_eq!(ev.last().unwrap()["dropped_frames"], 764);
        // プレビュー: 幅 8 に縮小・JPEG として読め、手(右)が入り、crop は 8x6 の中央正方
        let pv = ev.iter().find(|e| e["type"] == "preview").unwrap();
        assert_eq!(pv["width"], 8);
        assert_eq!(pv["height"], 6);
        assert_eq!(pv["crop"]["w"], 0.75);
        assert_eq!(pv["crop"]["h"], 1.0);
        // status は期待どおりの値域
        for s in ev.iter().filter(|e| e["type"] == "status") {
            assert!(s["state"] == "idle" || s["state"] == "recording");
            assert!(s["fps"].as_f64().unwrap().is_finite());
        }
        assert!(ev.iter().any(|e| e["type"] == "status" && e["state"] == "recording"));
        assert!(probes.dropped.load(Ordering::SeqCst), "入力元は drop されている");
    }

    // プレビュー無し(preview: null)なら preview イベントは一切出ない
    #[test]
    fn preview_null_emits_no_preview_events() {
        let (built, _p) = fake_built(frames_from(&[(1, 20)]), false, PalmRoiMode::FullFrame);
        let (ev, res, _tx) = run_fake(
            vec![Input::Cmd(Command::Start(start_file(SegKind::Auto, Some(None))))],
            built,
            true,
        );
        res.unwrap();
        assert!(!types(&ev).contains(&"preview".to_string()));
    }

    // 受け取った preview の JPEG が実際に読め、縮小サイズと一致する
    #[test]
    fn preview_events_carry_decodable_jpeg() {
        let (built, _p) = fake_built(frames_from(&[(200, 30)]), false, PalmRoiMode::FullFrame);
        let preview = Some(Some(PreviewCfg { max_fps: 30.0, width: 16 }));
        let (ev, res, _tx) = run_fake(
            vec![Input::Cmd(Command::Start(start_file(SegKind::Auto, preview)))],
            built,
            true,
        );
        res.unwrap();
        let pv = ev.iter().find(|e| e["type"] == "preview").expect("preview が出ている");
        // 元は 8x6 なので width=16 でも拡大しない
        assert_eq!((pv["width"].as_u64().unwrap(), pv["height"].as_u64().unwrap()), (8, 6));
        let b64 = pv["jpeg_b64"].as_str().unwrap();
        // 簡易 base64 デコード(テスト専用)
        let mut bytes = Vec::new();
        let val = |c: u8| -> u32 {
            match c {
                b'A'..=b'Z' => (c - b'A') as u32,
                b'a'..=b'z' => (c - b'a') as u32 + 26,
                b'0'..=b'9' => (c - b'0') as u32 + 52,
                b'+' => 62,
                b'/' => 63,
                _ => 0,
            }
        };
        for chunk in b64.as_bytes().chunks(4) {
            let n = chunk.iter().fold(0u32, |a, c| (a << 6) | val(*c));
            let pad = chunk.iter().filter(|c| **c == b'=').count();
            bytes.extend_from_slice(&n.to_be_bytes()[1..4 - pad]);
        }
        let img = image::load_from_memory(&bytes).unwrap();
        assert_eq!((img.width(), img.height()), (8, 6));
        // 右手の枠に値があるので hands.right が 21 点、left は null
        assert_eq!(pv["hands"]["right"].as_array().unwrap().len(), 21);
        assert!(pv["hands"]["left"].is_null());
    }

    // 標準入力 EOF で終了する(尽きない入力元でも止まる)。入力元の後始末が走り、stopped が最後
    #[test]
    fn eof_stops_endless_source_and_cleans_up() {
        let (built, probes) = fake_built(vec![1; 3], true, PalmRoiMode::FullFrame);
        let buf = SharedBuf::default();
        let (emitter, writer) = spawn_writer(buf.clone());
        let (tx, rx) = mpsc::channel();
        tx.send(Input::Cmd(Command::Start(start_file(SegKind::Manual, Some(None)))))
            .unwrap();
        let stop = StopFlag::default();
        let served = Arc::clone(&probes.served);
        let tx2 = tx.clone();
        // 数フレーム進んでから EOF を送る別スレッド
        let h = std::thread::spawn(move || {
            while served.load(Ordering::SeqCst) < 5 {
                std::thread::sleep(Duration::from_millis(1));
            }
            tx2.send(Input::Eof).unwrap();
        });
        let t0 = Instant::now();
        let res = serve(rx, &emitter, &stop, |_| Ok(built));
        h.join().unwrap();
        res.unwrap();
        assert!(t0.elapsed() < Duration::from_secs(5), "EOF で止まらない");
        assert!(probes.dropped.load(Ordering::SeqCst));
        drop(emitter);
        writer.finish();
        let ev = buf.events();
        let t = types(&ev);
        assert_eq!(t.first().unwrap(), "ready");
        assert_eq!(t.last().unwrap(), "stopped");
        // 手動 + 動画ファイルは開始と同時に録画。EOF(= quit)で捨てられ、結果は出ない
        assert!(!t.contains(&"result".to_string()));
        drop(tx);
    }

    // quit コマンドでも同様に終了する
    #[test]
    fn quit_command_stops_loop() {
        let (built, probes) = fake_built(vec![1; 3], true, PalmRoiMode::FullFrame);
        let (ev, res, _tx) = run_fake(
            vec![
                Input::Cmd(Command::Start(start_file(SegKind::Auto, Some(None)))),
                Input::Cmd(Command::Quit),
            ],
            built,
            true,
        );
        res.unwrap();
        assert_eq!(types(&ev).last().unwrap(), "stopped");
        assert!(probes.dropped.load(Ordering::SeqCst));
    }

    // start の前に EOF / quit なら、何も始めずに stopped だけ出して正常終了する
    #[test]
    fn eof_before_start_stops_quietly() {
        let buf = SharedBuf::default();
        let (emitter, writer) = spawn_writer(buf.clone());
        let (tx, rx) = mpsc::channel();
        tx.send(Input::Eof).unwrap();
        let res = serve(rx, &emitter, &StopFlag::default(), |_| -> Result<Built> {
            panic!("start が無いのに部品を作ろうとした")
        });
        res.unwrap();
        drop(emitter);
        writer.finish();
        assert_eq!(types(&buf.events()), vec!["stopped"]);
        drop(tx);
    }

    // start 前のコマンドは warn で無視して、その後の start を受け付ける
    #[test]
    fn commands_before_start_are_ignored_with_warning() {
        let (built, _p) = fake_built(frames_from(&[(0, 5)]), false, PalmRoiMode::FullFrame);
        let (ev, res, _tx) = run_fake(
            vec![
                Input::Cmd(Command::Toggle),
                Input::Cmd(Command::Start(start_file(SegKind::Auto, Some(None)))),
            ],
            built,
            true,
        );
        res.unwrap();
        let t = types(&ev);
        assert_eq!(t[0], "message");
        assert_eq!(ev[0]["level"], "warn");
        assert!(t.contains(&"ready".to_string()));
    }

    // 2 回目の start は無視される(warn)
    #[test]
    fn second_start_is_ignored() {
        let (built, _p) = fake_built(frames_from(&[(0, 5)]), false, PalmRoiMode::FullFrame);
        let s = Command::Start(start_file(SegKind::Auto, Some(None)));
        let (ev, res, _tx) = run_fake(
            vec![Input::Cmd(s.clone()), Input::Cmd(s)],
            built,
            true,
        );
        res.unwrap();
        assert_eq!(types(&ev).iter().filter(|x| *x == "ready").count(), 1);
        assert!(ev.iter().any(|e| e["type"] == "message" && e["level"] == "warn"));
    }

    // ライブ認識(mode 省略)に撮影用のコマンドが来ても、録画も保存も始まらず、警告して無視する
    // (撮影モードのイベントは一切出ない)
    #[test]
    fn live_mode_ignores_capture_commands() {
        let (built, probes) = fake_built(vec![1; 3], true, PalmRoiMode::FullFrame);
        let (ev, res, _tx) = run_fake(
            vec![
                Input::Cmd(Command::Start(start_file(SegKind::Auto, Some(None)))),
                Input::Cmd(Command::CaptureRecord { word_id: Some("001".into()) }),
                Input::Cmd(Command::CaptureStop),
                Input::Cmd(Command::CaptureAccept { confirm_ng: true }),
                Input::Cmd(Command::CaptureDiscard),
                Input::Cmd(Command::Quit),
            ],
            built,
            true,
        );
        res.unwrap();
        let t = types(&ev);
        assert!(!t.iter().any(|x| x.starts_with("capture_")), "{t:?}");
        let warns: Vec<&Value> = ev
            .iter()
            .filter(|e| e["type"] == "message" && e["text"].as_str().is_some_and(|t| t.contains("撮影用のコマンド")))
            .collect();
        assert_eq!(warns.len(), 4, "4 つのコマンドそれぞれに警告: {ev:?}");
        assert_eq!(t.last().unwrap(), "stopped");
        assert!(probes.dropped.load(Ordering::SeqCst));
    }

    // 手動 + 動画ファイル: 開始と同時に録画し、終端で認識(live_wizard と同じ)
    #[test]
    fn manual_file_records_whole_video() {
        let (built, _p) = fake_built(frames_from(&[(0, 5), (1, 15), (0, 5)]), false, PalmRoiMode::FullFrame);
        let (ev, res, _tx) = run_fake(
            vec![Input::Cmd(Command::Start(start_file(SegKind::Manual, Some(None))))],
            built,
            true,
        );
        res.unwrap();
        let r = ev.iter().find(|e| e["type"] == "result").expect("終端で認識される");
        assert_eq!(r["buffered"], 25);
    }

    // toggle コマンドで手動の開始/停止(カメラ + 手動)
    #[test]
    fn toggle_command_starts_and_stops_manual_recording() {
        // 尽きない入力 + 手あり。別スレッドで toggle(開始)→ しばらく後に toggle(停止)
        let (built, probes) = fake_built(vec![1; 2], true, PalmRoiMode::FullFrame);
        let buf = SharedBuf::default();
        let (emitter, writer) = spawn_writer(buf.clone());
        let (tx, rx) = mpsc::channel();
        let start = StartCmd {
            mode: StartMode::Live,
            data_dir: None,
            countdown_secs: None,
            source: SourceSpec::Camera { device: None },
            model_dir: None,
            segmenter: SegKind::Manual,
            preview: Some(None),
        };
        tx.send(Input::Cmd(Command::Start(start))).unwrap();
        let served = Arc::clone(&probes.served);
        let tx2 = tx.clone();
        let h = std::thread::spawn(move || {
            while served.load(Ordering::SeqCst) < 3 {
                std::thread::sleep(Duration::from_millis(1));
            }
            tx2.send(Input::Cmd(Command::Toggle)).unwrap();
            let at = served.load(Ordering::SeqCst);
            while served.load(Ordering::SeqCst) < at + 15 {
                std::thread::sleep(Duration::from_millis(1));
            }
            tx2.send(Input::Cmd(Command::Toggle)).unwrap();
            let at = served.load(Ordering::SeqCst);
            while served.load(Ordering::SeqCst) < at + 3 {
                std::thread::sleep(Duration::from_millis(1));
            }
            tx2.send(Input::Cmd(Command::Quit)).unwrap();
        });
        let stop = StopFlag::default();
        serve(rx, &emitter, &stop, |_| Ok(built)).unwrap();
        h.join().unwrap();
        drop(emitter);
        writer.finish();
        let ev = buf.events();
        let results: Vec<&Value> = ev.iter().filter(|e| e["type"] == "result").collect();
        assert_eq!(results.len(), 1, "{:?}", types(&ev));
        assert!(results[0]["buffered"].as_u64().unwrap() >= 10);
        assert!(ev.iter().any(|e| e["type"] == "status" && e["state"] == "recording"));
        drop(tx);
    }

    // set_segmenter: 待機中なら反映され segmenter イベントが出る。反映後は自動区切りで結果が出る
    #[test]
    fn set_segmenter_switches_when_idle() {
        let (built, _p) = fake_built(
            frames_from(&[(0, 10), (1, 15), (0, 15)]),
            false,
            PalmRoiMode::FullFrame,
        );
        let start = StartCmd {
            mode: StartMode::Live,
            data_dir: None,
            countdown_secs: None,
            source: SourceSpec::Camera { device: None },
            model_dir: None,
            segmenter: SegKind::Manual,
            preview: Some(None),
        };
        let (ev, res, _tx) = run_fake(
            vec![
                Input::Cmd(Command::Start(start)),
                Input::Cmd(Command::SetSegmenter { mode: SegKind::Auto }),
            ],
            built,
            true,
        );
        res.unwrap();
        let seg = ev.iter().find(|e| e["type"] == "segmenter").expect("segmenter イベント");
        assert_eq!(seg["mode"], "auto");
        assert_eq!(ev[0]["segmenter"], "manual", "ready は開始時の区切り方");
        assert_eq!(ev.iter().filter(|e| e["type"] == "result").count(), 1, "自動区切りに切り替わった");
    }

    // 記録中の set_segmenter は記録が終わってから反映(その旨を info で知らせる)
    #[test]
    fn set_segmenter_is_deferred_while_recording() {
        // 動画 + 手動 = 開始と同時に録画中。切り替えは終端まで反映されない
        let (built, _p) = fake_built(frames_from(&[(1, 20)]), false, PalmRoiMode::FullFrame);
        let (ev, res, _tx) = run_fake(
            vec![
                Input::Cmd(Command::Start(start_file(SegKind::Manual, Some(None)))),
                Input::Cmd(Command::SetSegmenter { mode: SegKind::Auto }),
            ],
            built,
            true,
        );
        res.unwrap();
        assert!(!types(&ev).contains(&"segmenter".to_string()));
        let m = ev
            .iter()
            .find(|e| e["type"] == "message" && e["level"] == "info")
            .expect("延期の通知");
        assert!(m["text"].as_str().unwrap().contains("切り替え"));
    }

    // 入力元の失敗(Err)は、終了要求が無ければ呼び出し側へ Err で返る(→ error イベント + 非 0)
    #[test]
    fn source_failure_is_an_error_unless_stopping() {
        struct Failing(Arc<AtomicBool>);
        impl FrameSource for Failing {
            fn next_frame(&mut self) -> Result<Option<SourceFrame>> {
                anyhow::bail!("カメラ入力が止まりました")
            }
        }
        impl JsonSource for Failing {}
        impl Drop for Failing {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        for stopping in [false, true] {
            let dropped = Arc::new(AtomicBool::new(false));
            let (mut built, _p) = fake_built(vec![], false, PalmRoiMode::FullFrame);
            built.source = Box::new(Failing(Arc::clone(&dropped)));
            let buf = SharedBuf::default();
            let (emitter, writer) = spawn_writer(buf.clone());
            let (tx, rx) = mpsc::channel();
            tx.send(Input::Cmd(Command::Start(start_file(SegKind::Auto, Some(None))))).unwrap();
            let stop = StopFlag::default();
            if stopping {
                stop.set();
            }
            let res = serve(rx, &emitter, &stop, |_| Ok(built));
            drop(emitter);
            writer.finish();
            assert!(dropped.load(Ordering::SeqCst), "失敗でも入力元は drop される");
            let t = types(&buf.events());
            if stopping {
                assert!(res.is_ok());
                assert_eq!(t.last().unwrap(), "stopped");
            } else {
                assert!(res.is_err());
                assert!(!t.contains(&"stopped".to_string()), "異常終了では stopped を出さない");
            }
            drop(tx);
        }
    }

    // build の失敗(モデル読み込み失敗など)は Err。ready は出ない
    #[test]
    fn build_failure_returns_error_without_ready() {
        let buf = SharedBuf::default();
        let (emitter, writer) = spawn_writer(buf.clone());
        let (tx, rx) = mpsc::channel();
        tx.send(Input::Cmd(Command::Start(start_file(SegKind::Auto, None)))).unwrap();
        let stop = StopFlag::default();
        let res = serve(rx, &emitter, &stop, |_| anyhow::bail!("認識モデルの読み込みに失敗"));
        assert!(res.unwrap_err().to_string().contains("認識モデル"));
        drop(emitter);
        writer.finish();
        assert!(buf.events().is_empty());
        drop(tx);
    }

    /// 最初の 1 回の書き込みは成功し、その後は壊れたパイプのように失敗する書き先
    struct BreaksAfter {
        ok: usize,
    }
    impl Write for BreaksAfter {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            if self.ok == 0 {
                return Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe));
            }
            self.ok -= 1;
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    // 標準出力が壊れた(画面が落ちた)ら、ループは自分で終了する
    #[test]
    fn broken_output_pipe_ends_the_loop() {
        let (built, probes) = fake_built(vec![1; 3], true, PalmRoiMode::FullFrame);
        let (emitter, writer) = spawn_writer(BreaksAfter { ok: 1 });
        let (tx, rx) = mpsc::channel();
        tx.send(Input::Cmd(Command::Start(start_file(SegKind::Auto, Some(None))))).unwrap();
        let stop = StopFlag::default();
        let t0 = Instant::now();
        serve(rx, &emitter, &stop, |_| Ok(built)).unwrap();
        assert!(t0.elapsed() < Duration::from_secs(5));
        assert!(emitter.is_broken());
        assert!(probes.dropped.load(Ordering::SeqCst));
        drop(emitter);
        writer.finish();
        drop(tx);
    }

    // 出力キューが詰まっても(読み手が遅い)、間引いてよいイベントは捨てられ、ループは止まらない
    #[test]
    fn droppable_events_are_dropped_when_queue_is_full() {
        struct Slow;
        impl Write for Slow {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                std::thread::sleep(Duration::from_millis(30));
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let (emitter, writer) = spawn_writer(Slow);
        let mut dropped = 0;
        for _ in 0..(OUT_QUEUE * 3) {
            if !emitter.try_send(Event::Stopped { dropped_frames: 0 }) {
                dropped += 1;
            }
        }
        assert!(dropped > 0, "詰まったのに 1 件も捨てていない");
        // 後始末はキューが空くまで待つので、全部流れるのを待たないよう素早く閉じる
        drop(emitter);
        writer.finish();
    }

    // ===== 実モデル + 実動画のスモーク(#[ignore])=====

    /// 書き出された行の時刻を記録する書き先(ready〜stopped の所要時間を測るため)
    #[derive(Clone, Default)]
    struct TimedSink {
        buf: SharedBuf,
        marks: Arc<Mutex<Vec<(String, Instant)>>>,
    }
    impl Write for TimedSink {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            let text = String::from_utf8_lossy(b);
            for kind in ["ready", "stopped"] {
                if text.contains(&format!("\"type\":\"{kind}\"")) {
                    lock(&self.marks).push((kind.to_string(), Instant::now()));
                }
            }
            self.buf.write(b)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// 実モデルで `serve` を 1 回回す(`start` を送り、尽きるまで)。(イベント, ready〜stopped の秒数, 処理フレーム数)
    fn run_real(video: &Path, preview: Option<Option<PreviewCfg>>, seg: SegKind) -> (Vec<Value>, f64, usize) {
        let sink = TimedSink::default();
        let (emitter, writer) = spawn_writer(sink.clone());
        let (tx, rx) = mpsc::channel();
        let start = StartCmd {
            mode: StartMode::Live,
            data_dir: None,
            countdown_secs: None,
            source: SourceSpec::File {
                path: video.display().to_string(),
            },
            model_dir: None,
            segmenter: seg,
            preview,
        };
        tx.send(Input::Cmd(Command::Start(start))).unwrap();
        let stop = StopFlag::default();
        let counted = Arc::new(AtomicUsize::new(0));
        let counted2 = Arc::clone(&counted);
        let emitter_for_build = emitter.clone();
        let stop_for_build = stop.clone();
        serve(rx, &emitter, &stop, move |s| {
            let mut b = build_real(s, &stop_for_build, &emitter_for_build)?;
            // フレーム数を数えるために入力元を包む
            struct Counting(Box<dyn JsonSource>, Arc<AtomicUsize>);
            impl FrameSource for Counting {
                fn next_frame(&mut self) -> Result<Option<SourceFrame>> {
                    let f = self.0.next_frame()?;
                    if f.is_some() {
                        self.1.fetch_add(1, Ordering::SeqCst);
                    }
                    Ok(f)
                }
            }
            impl JsonSource for Counting {
                fn dropped_frames(&self) -> u64 {
                    self.0.dropped_frames()
                }
            }
            b.source = Box::new(Counting(b.source, counted2));
            Ok(b)
        })
        .unwrap();
        drop(emitter);
        writer.finish();
        drop(tx);
        let marks = lock(&sink.marks).clone();
        let t_ready = marks.iter().find(|m| m.0 == "ready").unwrap().1;
        let t_stop = marks.iter().rev().find(|m| m.0 == "stopped").unwrap().1;
        (
            sink.buf.events(),
            t_stop.duration_since(t_ready).as_secs_f64(),
            counted.load(Ordering::SeqCst),
        )
    }

    fn median(v: &mut [f64]) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    /// 実モデル(rec_demo_center_signspan)+ raw_jsl の動画で `serve` を通し、受け取った JSON 行の種類と件数、
    /// result の top-1、プレビュー無し/有りの処理 fps を出す(カメラは使わない)。
    /// 環境変数: ENGINE_JSON_VIDEOS(':' 区切り。既定は raw_jsl から 2 本)/ ENGINE_JSON_ROUNDS(既定 3)
    /// 手動実行: `cargo test engine_json_smoke --release -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn engine_json_smoke() {
        for p in [DEFAULT_MODEL_DIR, PALM_MODEL, HAND_MODEL] {
            if !Path::new(p).exists() {
                eprintln!("skip: fixture not found: {p}");
                return;
            }
        }
        let videos: Vec<PathBuf> = match std::env::var("ENGINE_JSON_VIDEOS") {
            Ok(v) => v.split(':').map(PathBuf::from).collect(),
            Err(_) => crate::tests::sample_raw_jsl_takes(2),
        };
        if videos.is_empty() {
            eprintln!("skip: raw_jsl の動画が無い");
            return;
        }
        let rounds: usize = std::env::var("ENGINE_JSON_ROUNDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let default_preview = Some(Some(PreviewCfg::default()));
        let max_preview = Some(Some(PreviewCfg { max_fps: 30.0, width: 640 }));
        for video in &videos {
            let name = format!(
                "{}/{}",
                video.parent().and_then(|p| p.file_name()).unwrap_or_default().to_string_lossy(),
                video.file_name().unwrap_or_default().to_string_lossy()
            );
            // ONNX は最初の数回が遅いので 1 回捨てる
            let _ = run_real(video, Some(None), SegKind::Auto);
            let (mut off, mut def, mut max) = (Vec::new(), Vec::new(), Vec::new());
            let mut last_events = Vec::new();
            for _ in 0..rounds {
                for (cfg, sink) in [
                    (Some(None), &mut off),
                    (default_preview, &mut def),
                    (max_preview, &mut max),
                ] {
                    let (ev, secs, frames) = run_real(video, cfg, SegKind::Auto);
                    sink.push(frames as f64 / secs);
                    if cfg == default_preview {
                        last_events = ev;
                    }
                }
            }
            eprintln!("\n=== {name} ===");
            let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
            for e in &last_events {
                *counts.entry(e["type"].as_str().unwrap().to_string()).or_default() += 1;
            }
            eprintln!("受け取った行の種類と件数(プレビュー既定 10fps): {counts:?}");
            let ready = &last_events[0];
            eprintln!(
                "ready: palm_roi={} frame_select={} frames={} segmenter={} preprocess={} vocab={}語",
                ready["palm_roi"], ready["frame_select"], ready["frames"], ready["segmenter"],
                ready["preprocess"], ready["vocab"].as_array().unwrap().len()
            );
            for r in last_events.iter().filter(|e| e["type"] == "result") {
                eprintln!(
                    "result: top-1 = {} ({}) p={} low_confidence={} span={} buffered={} 手あり {}/{}",
                    r["tag"], r["label"], r["p"], r["low_confidence"], r["span"], r["buffered"],
                    r["selected_with_hand"], r["selected_total"]
                );
            }
            for e in last_events.iter().filter(|e| e["type"] == "message") {
                eprintln!("message: {} {}", e["level"], e["text"]);
            }
            if let Some(pv) = last_events.iter().find(|e| e["type"] == "preview") {
                eprintln!(
                    "preview 1 件目: {}x{} jpeg_b64 {} 文字 crop={} hands.left={} hands.right={}",
                    pv["width"], pv["height"], pv["jpeg_b64"].as_str().unwrap().len(), pv["crop"],
                    if pv["hands"]["left"].is_null() { "null" } else { "21点" },
                    if pv["hands"]["right"].is_null() { "null" } else { "21点" },
                );
            }
            let (off_m, def_m, max_m) = (median(&mut off), median(&mut def), median(&mut max));
            eprintln!(
                "処理 fps(中央値, {rounds} 回): プレビュー無し {:.2} / 10fps・640px {:.2} ({:+.1}%) / 30fps・640px {:.2} ({:+.1}%)",
                off_m, def_m, (def_m / off_m - 1.0) * 100.0, max_m, (max_m / off_m - 1.0) * 100.0
            );
            assert!(last_events.iter().any(|e| e["type"] == "result"), "{name}: result が出ない");
        }
    }

    // ===== 標準出力の複製(JSON 用 fd)=====

    // JSON 用に複製する fd には FD_CLOEXEC が付く。付いていないと、あとで起動する ffmpeg / ffprobe に
    // 書き込み口が引き継がれ、エンジンが後始末なしで死んでも画面に EOF が届かない
    #[cfg(unix)]
    #[test]
    fn json_sink_fd_is_close_on_exec() {
        use std::os::unix::io::FromRawFd;
        // 標準エラー(fd 2)を複製して確かめる(fd 1 を差し替える take_json_sink 自体は、
        // テストプロセスの標準出力を壊すので呼ばない)
        let fd = dup_cloexec(2);
        assert!(fd >= 0, "dup_cloexec に失敗");
        assert!(fd_is_cloexec(fd), "複製した fd に FD_CLOEXEC が付いていない");
        drop(unsafe { std::fs::File::from_raw_fd(fd) });
        // 対照: 素の dup には付かない(この性質が問題の原因)
        let plain = unsafe { sys::dup(2) };
        assert!(plain >= 0);
        assert!(!fd_is_cloexec(plain), "素の dup に FD_CLOEXEC が付いている(前提が変わった?)");
        drop(unsafe { std::fs::File::from_raw_fd(plain) });
        // 標準入出力そのもの(fd 0〜2)には付いていない(= 付けるのは複製だけ)
        assert!(!fd_is_cloexec(2));
    }

    // ===== 終了要求で引き返す =====

    // 部品を作る途中で終了要求に気づいて引き返したら、ready を出さずに stopped だけを出して正常に終わる
    #[test]
    fn serve_returns_stopped_only_when_build_aborts_for_stop() {
        let buf = SharedBuf::default();
        let (emitter, writer) = spawn_writer(buf.clone());
        let (tx, rx) = mpsc::channel();
        tx.send(Input::Cmd(Command::Start(start_file(SegKind::Auto, None)))).unwrap();
        let res = serve(rx, &emitter, &StopFlag::default(), |_| -> Result<Built> {
            Err(StopRequested.into())
        });
        writer.finish();
        assert!(res.is_ok(), "{res:?}");
        assert_eq!(types(&buf.events()), vec!["stopped"]);
    }

    // 別の失敗(StopRequested 以外)は、これまでどおり Err のまま
    #[test]
    fn serve_still_propagates_other_build_errors() {
        let buf = SharedBuf::default();
        let (emitter, writer) = spawn_writer(buf.clone());
        let (tx, rx) = mpsc::channel();
        tx.send(Input::Cmd(Command::Start(start_file(SegKind::Auto, None)))).unwrap();
        let res = serve(rx, &emitter, &StopFlag::default(), |_| -> Result<Built> {
            anyhow::bail!("モデルが読めない")
        });
        writer.finish();
        assert!(res.is_err());
        assert!(buf.events().is_empty());
    }
}
