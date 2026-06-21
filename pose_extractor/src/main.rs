mod progress;

use anyhow::{Context, Result};
use dialoguer::{theme::ColorfulTheme, Input, MultiSelect, Select};
use image::imageops::FilterType;
use image::{ImageBuffer, Rgb};
use ndarray::Array3;
use ort::session::Session;
use ort::value::Value;
use serde::Serialize;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const VIDEO_DIR: &str = "videos";
const DEFAULT_MODEL: &str = "models/blazepose_full.onnx";
const PALM_MODEL: &str = "models/palm_detection_mediapipe.onnx";
const HAND_MODEL: &str = "models/handpose_estimation_mediapipe.onnx";
const VIDEO_EXTS: &[&str] = &["mp4", "mov", "mkv", "avi", "webm", "m4v"];

const PALM_INPUT_SIZE: u32 = 192;
const PALM_SCORE_THRESHOLD: f32 = 0.5;
const PALM_NMS_THRESHOLD: f32 = 0.3;
const PALM_ANCHORS_BYTES: &[u8] = include_bytes!("../assets/palm_anchors.bin");

const HAND_INPUT_SIZE: u32 = 224;
const HAND_CROP_ENLARGE: f32 = 3.0;
const HAND_CROP_SHIFT_Y: f32 = -0.4;
const HAND_CONF_THRESHOLD: f32 = 0.5;

// build-dict: 1 フレームの特徴量 = 左手 21 点 + 右手 21 点、各 [x, y, z]
const HAND_POINTS: usize = 21;
const DICT_FEATURE_DIM: usize = HAND_POINTS * 3 * 2; // = 126

// MediaPipe Hands の 21 ランドマークを結ぶ骨格(21 本のボーン)。
// 各タプル (a, b) は landmarks[a]-landmarks[b] を線で結ぶ。回転アラインメント後に
// 指が解剖学的に正しく曲がっているかをオーバーレイ動画で目視するために使う。
const HAND_CONNECTIONS: &[(usize, usize)] = &[
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

// CLI はフラグを使わず、引数なし起動のトップレベルメニュー(質問形式)で全機能に入る。
// 設定値はすべて dialoguer の Select/Input/MultiSelect で対話的に尋ねる。
const RAW_JSL_DIR: &str = "../transformer_burn/data/raw_jsl";
// 認識モデル(手ポーズ列→タグ)の保存先。transformer_burn の --save 既定の置き場。
const REC_MODEL_DIR: &str = "../transformer_burn/models";

#[derive(Debug, Clone, Copy)]
enum OutputFormat {
    Json,
    Tsv,
}

#[derive(Debug)]
struct RunConfig {
    input: PathBuf,
    model: PathBuf,
    palm_model: Option<PathBuf>,
    hand_model: Option<PathBuf>,
    format: OutputFormat,
    apply_sigmoid: bool,
    save_overlay: Option<PathBuf>,
    overlay_count: usize,
    /// オーバーレイを全フレームの mp4 動画として書き出すか(false なら従来の PNG 数枚)
    overlay_video: bool,
    max_frames: Option<usize>,
    output: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct VideoInfo {
    width: u32,
    height: u32,
    fps: f64,
    frame_count: u64,
    duration: f64,
}

#[derive(Debug, Serialize, Clone)]
struct Landmark {
    x: f32,
    y: f32,
    z: f32,
    visibility: f32,
    presence: f32,
}

#[derive(Debug, Serialize)]
struct PoseFrame {
    frame_idx: usize,
    confidence: f32,
    landmarks: Vec<Landmark>,
}

#[derive(Debug, Serialize)]
struct PoseSequence {
    video: VideoInfo,
    model: String,
    sigmoid_applied: bool,
    frames: Vec<PoseFrame>,
    #[serde(skip_serializing_if = "Option::is_none")]
    palm_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    palm_frames: Option<Vec<PalmFrame>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hand_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hand_frames: Option<Vec<HandFrame>>,
}

#[derive(Debug, Serialize, Clone)]
struct PalmBox {
    score: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    keypoints: Vec<[f32; 2]>,
}

#[derive(Debug, Serialize)]
struct PalmFrame {
    frame_idx: usize,
    palms: Vec<PalmBox>,
}

#[derive(Debug, Serialize, Clone)]
struct HandLandmarkPoint {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Debug, Serialize, Clone)]
struct Hand {
    confidence: f32,
    handedness: f32,
    landmarks: Vec<HandLandmarkPoint>,
}

#[derive(Debug, Serialize)]
struct HandFrame {
    frame_idx: usize,
    hands: Vec<Hand>,
}

// === build-dict (S7) 出力スキーマ ===

#[derive(Debug, Serialize)]
struct PoseDict {
    metadata: PoseDictMeta,
    tags: BTreeMap<String, TagEntry>,
}

#[derive(Debug, Serialize)]
struct PoseDictMeta {
    /// 各タグのフレーム数 T
    frames: usize,
    /// 1 フレームの特徴量次元(= 126)
    feature_dim: usize,
    /// 特徴量の並び順の説明
    feature_layout: String,
    /// 座標正規化の説明
    normalization: String,
    /// 辞書に含めたタグ数
    tag_count: usize,
}

#[derive(Debug, Serialize)]
struct TagEntry {
    /// [T フレーム][feature_dim] の正規化済みハンドランドマーク列
    sequence: Vec<Vec<f32>>,
    /// 左手が検出されたフレームの割合 [0,1]
    left_hand_coverage: f32,
    /// 右手が検出されたフレームの割合 [0,1]
    right_hand_coverage: f32,
    /// 抽出元の動画ファイル名
    source: String,
}

fn main() -> Result<()> {
    let theme = ColorfulTheme::default();

    let actions = &[
        "動画から姿勢/手を抽出",
        "撮影セッション(録画フォルダを監視して自動取り込み)",
        "撮影進捗を確認",
        "タグ→ポーズ辞書を構築(build-dict)",
        "動画からタグを認識(推論)",
        "[dev] ONNXモデルの入出力を調査(inspect)",
        "[dev] ダミー推論で出力shapeを確認(test-infer)",
        "終了",
    ];

    let idx = Select::with_theme(&theme)
        .with_prompt("やることを選択")
        .items(actions)
        .default(0)
        .interact()?;

    match idx {
        0 => run_extraction_wizard(),
        1 => session_wizard(&theme),
        2 => progress_wizard(&theme),
        3 => build_dict_wizard(&theme),
        4 => recognize_wizard(&theme),
        5 => inspect_wizard(&theme),
        6 => test_infer_wizard(&theme),
        _ => {
            println!("終了します");
            Ok(())
        }
    }
}

/// 動画抽出ウィザード(従来の引数なし起動の挙動)
fn run_extraction_wizard() -> Result<()> {
    let configs = run_wizard()?;
    let total = configs.len();
    for (i, cfg) in configs.into_iter().enumerate() {
        if total > 1 {
            eprintln!("\n=== [{}/{}] {} ===", i + 1, total, cfg.input.display());
        }
        run_extraction(cfg)?;
    }
    if total > 1 {
        eprintln!("\ndone: {} videos processed", total);
    }
    Ok(())
}

/// 撮影進捗ウィザード
fn progress_wizard(theme: &ColorfulTheme) -> Result<()> {
    let data_dir: String = Input::with_theme(theme)
        .with_prompt("撮影データのルート")
        .default(RAW_JSL_DIR.into())
        .interact_text()?;

    let mode = Select::with_theme(theme)
        .with_prompt("動作")
        .items(&[
            "index.tsv を更新して進捗表示",
            "検証のみ(index.tsv を書き換えない)",
        ])
        .default(0)
        .interact()?;

    progress::run_progress(Path::new(data_dir.trim()), mode == 1)
}

/// 撮影セッションウィザード
fn session_wizard(theme: &ColorfulTheme) -> Result<()> {
    let data_dir: String = Input::with_theme(theme)
        .with_prompt("撮影データのルート(取り込み先)")
        .default(RAW_JSL_DIR.into())
        .interact_text()?;

    // OBS/QuickTime の既定保存先(~/Movies)を初期値として提示。
    // dialoguer の Input ビルダーは self を消費するため、default 有無で分岐する
    let home = std::env::var("HOME").unwrap_or_default();
    let prompt = "監視する録画フォルダ(OBS/QuickTime の保存先)";
    let watch: String = if home.is_empty() {
        Input::with_theme(theme).with_prompt(prompt).interact_text()?
    } else {
        Input::with_theme(theme)
            .with_prompt(prompt)
            .default(format!("{home}/Movies"))
            .interact_text()?
    };

    let hand = Select::with_theme(theme)
        .with_prompt("取り込み時の手検出チェック")
        .items(&[
            "有効(カバレッジが低いテイクを ng_hands フラグ+撮り直し提案)",
            "無効(取り込みが速い)",
        ])
        .default(0)
        .interact()?;

    progress::run_session(
        Path::new(data_dir.trim()),
        Path::new(watch.trim()),
        hand == 1,
    )
}

/// build-dict ウィザード
fn build_dict_wizard(theme: &ColorfulTheme) -> Result<()> {
    let input_dir: String = Input::with_theme(theme)
        .with_prompt("入力ディレクトリ(<タグ名>.mp4 を置いた場所)")
        .default(VIDEO_DIR.into())
        .interact_text()?;

    let output: String = Input::with_theme(theme)
        .with_prompt("出力 JSON パス")
        .default("tag_pose_dict.json".into())
        .interact_text()?;

    let frames: usize = Input::with_theme(theme)
        .with_prompt("ダウンサンプルするフレーム数(transformer_burn の SEQ_LEN と揃える)")
        .default(10)
        .interact_text()?;

    build_dict(
        Path::new(input_dir.trim()),
        Path::new(output.trim()),
        frames,
    )
}

/// 動画からタグを認識(推論)ウィザード。
/// 動画1本 → 手ポーズ列(126次元×frames)→ 認識モデル(CPU/NdArray)→ 上位 k タグ。
/// build-dict→train→predict の3手順を、判定だけワンショットに短縮する近道。
fn recognize_wizard(theme: &ColorfulTheme) -> Result<()> {
    // 1. 認識する動画を選ぶ
    let video = select_video(theme)?;

    // 2. 認識モデル(model.bin + tag_vocab.json を含むディレクトリ)を選ぶ
    let model_dir = select_model_dir(theme)?;

    // 3. フレーム数(学習時の SEQ_LEN と揃える)と表示件数
    let frames: usize = Input::with_theme(theme)
        .with_prompt("ダウンサンプルするフレーム数(学習時の SEQ_LEN と揃える)")
        .default(10)
        .interact_text()?;
    let topk: usize = Input::with_theme(theme)
        .with_prompt("上位何件のタグを表示するか")
        .default(5)
        .interact_text()?;

    // 4. Palm + Hand モデルをロード(build_dict と同じ手順)
    let palm_anchors = load_palm_anchors();
    let mut palm_session = Session::builder()?
        .commit_from_file(PALM_MODEL)
        .with_context(|| format!("failed to load palm model: {}", PALM_MODEL))?;
    let mut hand_session = Session::builder()?
        .commit_from_file(HAND_MODEL)
        .with_context(|| format!("failed to load hand model: {}", HAND_MODEL))?;

    // 5. 動画 → 手ポーズ列(TagEntry)を抽出
    eprintln!("\n=== 抽出: {} ===", video.display());
    let entry = extract_tag_features(
        &video,
        &mut palm_session,
        &mut hand_session,
        &palm_anchors,
        frames,
    )?;
    eprintln!(
        "left_hand_coverage={:.0}% right_hand_coverage={:.0}%",
        entry.left_hand_coverage * 100.0,
        entry.right_hand_coverage * 100.0
    );

    // 6. [T][126] を [frames*126] のフラット列にする
    let flat: Vec<f32> = entry.sequence.iter().flatten().copied().collect();

    // 7. CPU 推論(transformer_burn の薄いラッパー)。
    //    Box<dyn Error> は Send+Sync でないため anyhow へは文字列化して持ち上げる。
    let ranked = transformer_burn::recognition::predict_from_features(&model_dir, &flat, frames, topk)
        .map_err(|e| anyhow::anyhow!("認識推論に失敗: {}", e))?;

    // 8. 出力。S6 の教訓: 手の検出率が低いと、確率が高くても結果は信用できない
    if entry.left_hand_coverage < 0.5 && entry.right_hand_coverage < 0.5 {
        eprintln!(
            "警告: 両手とも検出率が低い(L={:.0}% R={:.0}%)。結果は不安定です",
            entry.left_hand_coverage * 100.0,
            entry.right_hand_coverage * 100.0
        );
    }
    println!("\n=== 認識結果(上位 {}) ===", ranked.len());
    for (rank, (tag, prob)) in ranked.iter().enumerate() {
        println!("{}) {}  {:.3}", rank + 1, tag, prob);
    }
    println!("\n注意: 学習に使った動画なら当たって当然(配線確認)。本判定は未見テイクで行うこと。");

    Ok(())
}

/// videos/ 内の動画を一覧から選ばせる(無ければ/その他はパス直接入力)
fn select_video(theme: &ColorfulTheme) -> Result<PathBuf> {
    let videos = list_videos(Path::new(VIDEO_DIR));
    let mut items: Vec<String> = videos
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .collect();
    items.push("[パスを直接入力]".into());
    let manual_idx = items.len() - 1;

    let idx = Select::with_theme(theme)
        .with_prompt("認識する動画を選択")
        .items(&items)
        .default(0)
        .interact()?;

    if idx == manual_idx {
        let s: String = Input::with_theme(theme)
            .with_prompt("動画パス")
            .interact_text()?;
        Ok(PathBuf::from(s.trim()))
    } else {
        Ok(videos[idx].clone())
    }
}

/// 認識モデルのディレクトリ(model.bin を含む)を一覧から選ばせる(無ければパス直接入力)
fn select_model_dir(theme: &ColorfulTheme) -> Result<PathBuf> {
    let dirs = list_model_dirs(Path::new(REC_MODEL_DIR));
    let mut items: Vec<String> = dirs
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .collect();
    items.push("[パスを直接入力]".into());
    let manual_idx = items.len() - 1;

    let idx = Select::with_theme(theme)
        .with_prompt("認識モデル(ディレクトリ)を選択")
        .items(&items)
        .default(0)
        .interact()?;

    if idx == manual_idx {
        let s: String = Input::with_theme(theme)
            .with_prompt("モデルディレクトリのパス")
            .interact_text()?;
        Ok(PathBuf::from(s.trim()))
    } else {
        Ok(dirs[idx].clone())
    }
}

/// REC_MODEL_DIR 直下で、認識モデルのサブディレクトリ一覧。
/// model.bin と tag_vocab.json の両方を持つものだけ(tag_vocab.json が
/// 認識モデルの目印。足場の Seq2Seq モデルには無いため誤選択を防げる)。
fn list_model_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_dir()
                    && p.join("model.bin").is_file()
                    && p.join("tag_vocab.json").is_file()
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    dirs.sort();
    dirs
}

/// inspect ウィザード(models/ の .onnx から選ぶ)
fn inspect_wizard(theme: &ColorfulTheme) -> Result<()> {
    let model = select_model(theme)?;
    inspect_onnx(&model)
}

/// test-infer ウィザード
fn test_infer_wizard(theme: &ColorfulTheme) -> Result<()> {
    let model = select_model(theme)?;
    test_inference(&model)
}

/// models/ 内の .onnx を一覧から選ばせる(無ければ直接入力)
fn select_model(theme: &ColorfulTheme) -> Result<PathBuf> {
    let models = list_models(Path::new("models"));
    let mut items: Vec<String> = models
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .collect();
    items.push("[パスを直接入力]".into());
    let manual_idx = items.len() - 1;

    let idx = Select::with_theme(theme)
        .with_prompt("ONNX モデルを選択")
        .items(&items)
        .default(0)
        .interact()?;

    if idx == manual_idx {
        let s: String = Input::with_theme(theme)
            .with_prompt("モデルパス")
            .interact_text()?;
        Ok(PathBuf::from(s.trim()))
    } else {
        Ok(models[idx].clone())
    }
}

/// 指定ディレクトリ直下の .onnx ファイル一覧
fn list_models(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s.eq_ignore_ascii_case("onnx"))
                    .unwrap_or(false)
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    files.sort();
    files
}

// [評価/スモーク] ffmpeg+ONNX統合の入口。回帰テスト化は不安定。「1本が落ちずに通る」程度のスモークまで
fn run_extraction(cfg: RunConfig) -> Result<()> {
    let info = probe_video(&cfg.input)?;
    eprintln!(
        "video: {}x{} @ {:.2}fps, {} frames, duration={:.2}s",
        info.width, info.height, info.fps, info.frame_count, info.duration
    );

    let mut session = Session::builder()?
        .commit_from_file(&cfg.model)
        .with_context(|| format!("failed to load model: {}", cfg.model.display()))?;
    eprintln!("loaded model: {}", cfg.model.display());

    let mut palm_session: Option<Session> = None;
    let palm_anchors: Vec<[f32; 2]> = load_palm_anchors();
    if let Some(palm_path) = &cfg.palm_model {
        let s = Session::builder()?
            .commit_from_file(palm_path)
            .with_context(|| format!("failed to load palm model: {}", palm_path.display()))?;
        eprintln!(
            "loaded palm detection model: {} (anchors: {})",
            palm_path.display(),
            palm_anchors.len()
        );
        palm_session = Some(s);
    }

    let mut hand_session: Option<Session> = None;
    if let Some(hand_path) = &cfg.hand_model {
        let s = Session::builder()?
            .commit_from_file(hand_path)
            .with_context(|| format!("failed to load hand model: {}", hand_path.display()))?;
        eprintln!("loaded hand landmark model: {}", hand_path.display());
        hand_session = Some(s);
    }

    let overlay_targets = overlay_target_indices(
        cfg.save_overlay.as_ref(),
        info.frame_count as usize,
        cfg.max_frames,
        cfg.overlay_count,
    );
    // オーバーレイ保存先を先に作る。動画モードは全フレームの PNG をここに連番で貯め、
    // ループ後に ffmpeg で mp4 へ束ねる(生フレームをメモリに溜めない)。
    if let Some(dir) = &cfg.save_overlay {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("create dir {}", dir.display()))?;
    }
    let mut frames: Vec<PoseFrame> = Vec::new();
    let mut palm_frames: Vec<PalmFrame> = Vec::new();
    let mut hand_frames: Vec<HandFrame> = Vec::new();

    extract_frames(&cfg.input, info.width, info.height, |idx, frame| {
        if let Some(m) = cfg.max_frames {
            if idx >= m {
                return Ok(());
            }
        }
        let (shape, data) = preprocess_frame(&frame)?;
        let input_value = Value::from_array((shape, data))?;
        let outputs = session.run(ort::inputs!["input_1" => input_value])?;

        let (_, ld_data) = outputs["Identity"].try_extract_tensor::<f32>()?;
        let (_, conf_data) = outputs["Identity_1"].try_extract_tensor::<f32>()?;
        let landmarks = parse_landmarks(ld_data)?;
        let pose_frame = PoseFrame {
            frame_idx: idx,
            confidence: conf_data[0],
            landmarks,
        };

        // palm / hand はこのフレームのローカル結果として持ち、オーバーレイ描画にも使う
        let mut palm_frame_local: Option<PalmFrame> = None;
        let mut hand_frame_local: Option<HandFrame> = None;
        if let Some(palm_s) = palm_session.as_mut() {
            let palms = run_palm_detection(palm_s, &frame, &palm_anchors)?;
            let mut hands_this_frame: Vec<Hand> = Vec::new();
            if let Some(hand_s) = hand_session.as_mut() {
                for palm in &palms {
                    if let Some(h) = run_hand_landmark(hand_s, &frame, palm)? {
                        hands_this_frame.push(h);
                    }
                }
            }
            palm_frame_local = Some(PalmFrame {
                frame_idx: idx,
                palms,
            });
            if hand_session.is_some() {
                hand_frame_local = Some(HandFrame {
                    frame_idx: idx,
                    hands: hands_this_frame,
                });
            }
        }

        // オーバーレイ描画。動画モードは全フレーム、PNG モードは overlay_targets のみ。
        // sigmoid は描画側で適用するため未適用(false)で渡す(可視性の色判定は sigmoid 適用後と同値)。
        if let Some(dir) = &cfg.save_overlay {
            if cfg.overlay_video || overlay_targets.contains(&idx) {
                let path = dir.join(format!("frame_{:05}.png", idx));
                draw_overlay(
                    &frame,
                    &pose_frame,
                    false,
                    palm_frame_local.as_ref(),
                    hand_frame_local.as_ref(),
                    &path,
                )?;
            }
        }

        frames.push(pose_frame);
        if let Some(pf) = palm_frame_local {
            palm_frames.push(pf);
        }
        if let Some(hf) = hand_frame_local {
            hand_frames.push(hf);
        }

        Ok(())
    })?;

    eprintln!("processed {} frames", frames.len());
    if !palm_frames.is_empty() {
        let total: usize = palm_frames.iter().map(|pf| pf.palms.len()).sum();
        let with_palm = palm_frames.iter().filter(|pf| !pf.palms.is_empty()).count();
        eprintln!(
            "palm detection: {} frames with palms, {} palms total",
            with_palm, total
        );
    }

    let palm_model_str = cfg.palm_model.as_ref().map(|p| p.to_string_lossy().to_string());
    let palm_frames_opt = if palm_session.is_some() {
        Some(palm_frames)
    } else {
        None
    };
    let hand_model_str = cfg.hand_model.as_ref().map(|p| p.to_string_lossy().to_string());
    let hand_frames_opt = if hand_session.is_some() {
        let total: usize = hand_frames.iter().map(|hf| hf.hands.len()).sum();
        let with_hand = hand_frames.iter().filter(|hf| !hf.hands.is_empty()).count();
        eprintln!(
            "hand landmark: {} frames with hands, {} hands total",
            with_hand, total
        );
        Some(hand_frames)
    } else {
        None
    };

    let mut sequence = PoseSequence {
        video: info,
        model: cfg.model.to_string_lossy().to_string(),
        sigmoid_applied: cfg.apply_sigmoid,
        frames,
        palm_model: palm_model_str,
        palm_frames: palm_frames_opt,
        hand_model: hand_model_str,
        hand_frames: hand_frames_opt,
    };

    if cfg.apply_sigmoid {
        apply_sigmoid_to_frames(&mut sequence.frames);
    }

    print_stats(&sequence);

    // オーバーレイは上のループ内で PNG として描画済み。動画モードなら束ねて mp4 にする。
    if let Some(dir) = &cfg.save_overlay {
        if cfg.overlay_video {
            let mp4 = dir.join("overlay.mp4");
            encode_overlay_video(dir, &mp4, sequence.video.fps)?;
            eprintln!("wrote overlay video: {}", mp4.display());
        } else {
            eprintln!("wrote overlay PNGs to: {}", dir.display());
        }
    }

    let stdout = std::io::stdout();
    let mut out: Box<dyn Write> = match &cfg.output {
        Some(p) => {
            if let Some(parent) = p.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("create dir {}", parent.display()))?;
                }
            }
            Box::new(std::fs::File::create(p)?)
        }
        None => Box::new(stdout.lock()),
    };

    match cfg.format {
        OutputFormat::Json => {
            serde_json::to_writer_pretty(&mut out, &sequence)?;
            writeln!(out)?;
        }
        OutputFormat::Tsv => write_tsv(&sequence, &mut out)?,
    }

    if let Some(p) = &cfg.output {
        eprintln!("wrote {}", p.display());
    }

    Ok(())
}

/// S7: `<tag>.mp4` 群からタグ→ポーズ辞書を構築する。
/// 各動画のハンドランドマークを `frames` フレームにダウンサンプルして JSON に書き出す。
// [評価/スモーク] ディレクトリ統合処理。中の純粋部品(downsample/frame_hand_feature)を個別にテストする方が筋が良い
fn build_dict(input_dir: &Path, output: &Path, frames: usize) -> Result<()> {
    if frames == 0 {
        anyhow::bail!("--frames must be >= 1");
    }

    let videos = list_videos(input_dir);
    if videos.is_empty() {
        anyhow::bail!(
            "no videos found in {} (place `<tag>.mp4` files there)",
            input_dir.display()
        );
    }
    eprintln!("found {} videos in {}", videos.len(), input_dir.display());

    // Palm + Hand モデルを 1 回だけロードして全動画で使い回す
    let palm_anchors = load_palm_anchors();
    let mut palm_session = Session::builder()?
        .commit_from_file(PALM_MODEL)
        .with_context(|| format!("failed to load palm model: {}", PALM_MODEL))?;
    let mut hand_session = Session::builder()?
        .commit_from_file(HAND_MODEL)
        .with_context(|| format!("failed to load hand model: {}", HAND_MODEL))?;
    eprintln!("loaded palm + hand models");

    let mut tags: BTreeMap<String, TagEntry> = BTreeMap::new();
    for video in &videos {
        let tag = video
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".into());
        eprintln!("\n=== {} ({}) ===", tag, video.display());

        let entry = extract_tag_features(
            video,
            &mut palm_session,
            &mut hand_session,
            &palm_anchors,
            frames,
        )?;
        eprintln!(
            "  left_hand_coverage={:.0}% right_hand_coverage={:.0}%",
            entry.left_hand_coverage * 100.0,
            entry.right_hand_coverage * 100.0
        );
        if tags.contains_key(&tag) {
            eprintln!("  warning: duplicate tag '{}', overwriting previous entry", tag);
        }
        tags.insert(tag, entry);
    }

    let dict = PoseDict {
        metadata: PoseDictMeta {
            frames,
            feature_dim: DICT_FEATURE_DIM,
            feature_layout: "left_hand[21*xyz] then right_hand[21*xyz]; missing hand = zeros"
                .into(),
            normalization: "x/=width, y/=height, z/=width (z is relative depth)".into(),
            tag_count: tags.len(),
        },
        tags,
    };

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create dir {}", parent.display()))?;
        }
    }
    let file = std::fs::File::create(output)
        .with_context(|| format!("create {}", output.display()))?;
    serde_json::to_writer_pretty(file, &dict)?;
    eprintln!(
        "\nwrote dictionary: {} ({} tags, {} frames x {} dims)",
        output.display(),
        dict.metadata.tag_count,
        frames,
        DICT_FEATURE_DIM
    );

    Ok(())
}

/// 1 動画からハンドランドマーク列を抽出し、`target` フレームにダウンサンプルした TagEntry を返す。
// [評価] 推論依存。出力の良し悪しは coverage 等の指標で評価(テストでなく計測)
fn extract_tag_features(
    video: &PathBuf,
    palm_session: &mut Session,
    hand_session: &mut Session,
    palm_anchors: &[[f32; 2]],
    target: usize,
) -> Result<TagEntry> {
    let info = probe_video(video)?;
    let width = info.width as f32;
    let height = info.height as f32;

    // 全フレームの (特徴量, 左手検出, 右手検出) を集める
    let mut per_frame: Vec<([f32; DICT_FEATURE_DIM], bool, bool)> = Vec::new();
    extract_frames(video, info.width, info.height, |_idx, frame| {
        let palms = run_palm_detection(palm_session, &frame, palm_anchors)?;
        let mut hands: Vec<Hand> = Vec::new();
        for palm in &palms {
            if let Some(h) = run_hand_landmark(hand_session, &frame, palm)? {
                hands.push(h);
            }
        }
        per_frame.push(frame_hand_feature(&hands, width, height));
        Ok(())
    })?;

    if per_frame.is_empty() {
        anyhow::bail!("no frames extracted from {}", video.display());
    }

    let left_cov =
        per_frame.iter().filter(|(_, l, _)| *l).count() as f32 / per_frame.len() as f32;
    let right_cov =
        per_frame.iter().filter(|(_, _, r)| *r).count() as f32 / per_frame.len() as f32;

    // ダウンサンプル: 均等間隔で target 個のインデックスを取る
    let indices = downsample_indices(per_frame.len(), target);
    let sequence: Vec<Vec<f32>> = indices
        .iter()
        .map(|&i| per_frame[i].0.to_vec())
        .collect();

    Ok(TagEntry {
        sequence,
        left_hand_coverage: left_cov,
        right_hand_coverage: right_cov,
        source: video
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default(),
    })
}

/// 検出されたハンドを左手スロット([0..63])と右手スロット([63..126])に割り当て、
/// 正規化済みの 126 次元特徴量を作る。検出されなかった手はゼロ埋め。
/// handedness >= 0.5 を右手とみなし、衝突時は confidence の高い方を優先する。
// [TEST向き] 126次元ベクトルの組み立て。片手だけ/両手なしフレームの欠損埋めが静かにバグる箇所
fn frame_hand_feature(
    hands: &[Hand],
    width: f32,
    height: f32,
) -> ([f32; DICT_FEATURE_DIM], bool, bool) {
    let mut feat = [0.0f32; DICT_FEATURE_DIM];
    let mut left_filled = false;
    let mut right_filled = false;

    // confidence の高い順に最大 2 手まで採用
    let mut sorted: Vec<&Hand> = hands.iter().collect();
    sorted.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted.truncate(2);

    for h in sorted {
        let prefer_right = h.handedness >= 0.5;
        // 希望スロットが埋まっていれば反対側へ(両手検出を優先して残す)
        let slot = if prefer_right {
            if !right_filled {
                1
            } else if !left_filled {
                0
            } else {
                continue;
            }
        } else if !left_filled {
            0
        } else if !right_filled {
            1
        } else {
            continue;
        };

        let base = slot * HAND_POINTS * 3;
        for (k, lm) in h.landmarks.iter().enumerate().take(HAND_POINTS) {
            feat[base + k * 3] = lm.x / width;
            feat[base + k * 3 + 1] = lm.y / height;
            feat[base + k * 3 + 2] = lm.z / width;
        }
        if slot == 0 {
            left_filled = true;
        } else {
            right_filled = true;
        }
    }

    (feat, left_filled, right_filled)
}

/// n 個の要素から target 個を均等間隔で選ぶインデックス列。
/// n < target の場合は一部のフレームが繰り返される。
// [TEST向き] nフレーム分割の核。純粋関数。境界(端数・target>n・target=1・n=0)とoff-by-one
fn downsample_indices(n: usize, target: usize) -> Vec<usize> {
    if n == 0 || target == 0 {
        return vec![];
    }
    if target == 1 {
        return vec![0];
    }
    (0..target)
        .map(|i| i * (n - 1) / (target - 1))
        .collect()
}

enum OutputTarget {
    Stdout,
    File(PathBuf),
    Dir(PathBuf),
}

fn run_wizard() -> Result<Vec<RunConfig>> {
    let theme = ColorfulTheme::default();

    let videos = list_videos(Path::new(VIDEO_DIR));
    let has_videos = !videos.is_empty();

    let mut video_items: Vec<String> = videos
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .collect();
    let batch_item_idx = if has_videos {
        video_items.push(format!("[{} 件すべて処理]", videos.len()));
        Some(video_items.len() - 1)
    } else {
        None
    };
    video_items.push("[パスを直接入力]".into());
    let manual_item_idx = video_items.len() - 1;

    if !has_videos {
        eprintln!(
            "note: {} に動画が見つかりませんでした。直接パスを入力してください。",
            VIDEO_DIR
        );
    }

    let video_idx = Select::with_theme(&theme)
        .with_prompt("動画を選択")
        .items(&video_items)
        .default(0)
        .interact()?;

    let batch_mode = Some(video_idx) == batch_item_idx;
    let is_manual = video_idx == manual_item_idx;

    let inputs: Vec<PathBuf> = if batch_mode {
        videos.clone()
    } else if is_manual {
        let s: String = Input::with_theme(&theme)
            .with_prompt("動画パス")
            .interact_text()?;
        vec![PathBuf::from(s.trim())]
    } else {
        vec![videos[video_idx].clone()]
    };

    let format_idx = Select::with_theme(&theme)
        .with_prompt("出力形式")
        .items(&["JSON", "TSV (1行=1フレーム)"])
        .default(0)
        .interact()?;
    let format = if format_idx == 0 {
        OutputFormat::Json
    } else {
        OutputFormat::Tsv
    };

    let feature_labels = &[
        "sigmoid を visibility/presence に適用",
        "ランドマーク オーバーレイ保存(動画 mp4 / PNG)",
        "フレーム数制限 (動作確認用)",
        "MediaPipe Hands (Palm + Landmark) を並走",
    ];
    let selected = MultiSelect::with_theme(&theme)
        .with_prompt("追加機能 (space で選択, enter で確定)")
        .items(feature_labels)
        .interact()?;
    let apply_sigmoid = selected.contains(&0);
    let want_overlay = selected.contains(&1);
    let want_limit = selected.contains(&2);
    let want_hands = selected.contains(&3);

    let (overlay_root, overlay_count, overlay_video) = if want_overlay {
        let prompt = if batch_mode {
            "オーバーレイ保存先(動画ごとにサブディレクトリを作成)"
        } else {
            "オーバーレイ保存先ディレクトリ"
        };
        let dir: String = Input::with_theme(&theme)
            .with_prompt(prompt)
            .default("/tmp/overlay".into())
            .interact_text()?;
        // 動画 mp4(全フレーム・手の骨格つき)か、サンプル PNG 数枚か
        let mode_idx = Select::with_theme(&theme)
            .with_prompt("オーバーレイの出力形式")
            .items(&[
                "動画 mp4(全フレーム・手の骨格つき/品質チェック向き)",
                "PNG 数枚(サンプル)",
            ])
            .default(0)
            .interact()?;
        if mode_idx == 0 {
            (Some(PathBuf::from(dir.trim())), 0, true)
        } else {
            let count: usize = Input::with_theme(&theme)
                .with_prompt("オーバーレイ枚数")
                .default(3)
                .interact_text()?;
            (Some(PathBuf::from(dir.trim())), count, false)
        }
    } else {
        (None, 3, false)
    };

    let max_frames = if want_limit {
        let n: usize = Input::with_theme(&theme)
            .with_prompt("最大フレーム数")
            .default(10)
            .interact_text()?;
        Some(n)
    } else {
        None
    };

    let output_target = if batch_mode {
        let dir: String = Input::with_theme(&theme)
            .with_prompt("出力ディレクトリ(動画名から自動で .json/.tsv を生成)")
            .default("results".into())
            .interact_text()?;
        OutputTarget::Dir(PathBuf::from(dir.trim()))
    } else {
        let s: String = Input::with_theme(&theme)
            .with_prompt("出力ファイル (空欄=stdout)")
            .allow_empty(true)
            .interact_text()?;
        if s.trim().is_empty() {
            OutputTarget::Stdout
        } else {
            OutputTarget::File(PathBuf::from(s.trim()))
        }
    };

    let ext = match format {
        OutputFormat::Json => "json",
        OutputFormat::Tsv => "tsv",
    };

    let configs: Vec<RunConfig> = inputs
        .iter()
        .map(|input| {
            let stem = input
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "out".into());

            let output = match &output_target {
                OutputTarget::Stdout => None,
                OutputTarget::File(p) => Some(p.clone()),
                OutputTarget::Dir(d) => Some(d.join(format!("{stem}.{ext}"))),
            };

            let save_overlay = overlay_root.as_ref().map(|root| {
                if batch_mode {
                    root.join(&stem)
                } else {
                    root.clone()
                }
            });

            RunConfig {
                input: input.clone(),
                model: PathBuf::from(DEFAULT_MODEL),
                palm_model: if want_hands {
                    Some(PathBuf::from(PALM_MODEL))
                } else {
                    None
                },
                hand_model: if want_hands {
                    Some(PathBuf::from(HAND_MODEL))
                } else {
                    None
                },
                format,
                apply_sigmoid,
                save_overlay,
                overlay_count,
                overlay_video,
                max_frames,
                output,
            }
        })
        .collect();

    Ok(configs)
}

fn list_videos(dir: &Path) -> Vec<PathBuf> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return vec![],
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|s| VIDEO_EXTS.contains(&s.to_lowercase().as_str()))
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    files
}

// [TEST向き] 純粋な座標計算。横長/縦長/正方形の3ケースを手計算オラクルで固定
fn crop_square_center(w: u32, h: u32) -> (u32, u32, u32) {
    let side = w.min(h);
    let crop_x = (w - side) / 2;
    let crop_y = (h - side) / 2;
    (crop_x, crop_y, side)
}

// [TEST向き] 決定論的な前処理。出力shape(=256*256*3)と正規化範囲[0,1]を検証
fn preprocess_frame(frame: &Array3<u8>) -> Result<(Vec<usize>, Vec<f32>)> {
    let h = frame.shape()[0] as u32;
    let w = frame.shape()[1] as u32;
    let raw: Vec<u8> = frame.iter().copied().collect();
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(w, h, raw)
        .context("failed to construct ImageBuffer from frame bytes")?;
    let (crop_x, crop_y, side) = crop_square_center(w, h);
    let cropped = image::imageops::crop_imm(&img, crop_x, crop_y, side, side).to_image();
    let resized = image::imageops::resize(&cropped, 256, 256, FilterType::Triangle);
    let data: Vec<f32> = resized
        .pixels()
        .flat_map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();
    Ok((vec![1usize, 256, 256, 3], data))
}

// [TEST向き] 不変条件チェック向き。anchors数=2016 を assert するだけで埋め込み崩れを検知
fn load_palm_anchors() -> Vec<[f32; 2]> {
    PALM_ANCHORS_BYTES
        .chunks_exact(8)
        .map(|c| {
            [
                f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                f32::from_le_bytes([c[4], c[5], c[6], c[7]]),
            ]
        })
        .collect()
}

struct PalmPreprocess {
    shape: Vec<usize>,
    data: Vec<f32>,
    pad_left_orig: f32,
    pad_top_orig: f32,
    scale: f32,
}

// [TEST向き] 決定論的なレターボックス変換。出力shapeとscale/pad値を既知入力で固定
fn preprocess_for_palm(frame: &Array3<u8>) -> Result<PalmPreprocess> {
    let h = frame.shape()[0] as u32;
    let w = frame.shape()[1] as u32;
    let raw: Vec<u8> = frame.iter().copied().collect();
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(w, h, raw)
        .context("failed to construct ImageBuffer for palm preprocess")?;
    let size = PALM_INPUT_SIZE as f32;
    let ratio = (size / w as f32).min(size / h as f32);
    let new_w = ((w as f32) * ratio).round() as u32;
    let new_h = ((h as f32) * ratio).round() as u32;
    let resized = image::imageops::resize(&img, new_w, new_h, FilterType::Triangle);
    let pad_w = PALM_INPUT_SIZE.saturating_sub(new_w);
    let pad_h = PALM_INPUT_SIZE.saturating_sub(new_h);
    let left = (pad_w / 2) as i64;
    let top = (pad_h / 2) as i64;
    let mut canvas: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_pixel(PALM_INPUT_SIZE, PALM_INPUT_SIZE, Rgb([0u8, 0, 0]));
    image::imageops::overlay(&mut canvas, &resized, left, top);
    let data: Vec<f32> = canvas
        .pixels()
        .flat_map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();
    Ok(PalmPreprocess {
        shape: vec![
            1usize,
            PALM_INPUT_SIZE as usize,
            PALM_INPUT_SIZE as usize,
            3,
        ],
        data,
        pad_left_orig: left as f32 / ratio,
        pad_top_orig: top as f32 / ratio,
        scale: w.max(h) as f32,
    })
}

// [評価] ONNX推論。非決定的・モデル依存。前処理(preprocess_for_palm)とIoU(iou_xyxy)を切り出してテスト
fn run_palm_detection(
    session: &mut Session,
    frame: &Array3<u8>,
    anchors: &[[f32; 2]],
) -> Result<Vec<PalmBox>> {
    let pre = preprocess_for_palm(frame)?;
    let input_value = Value::from_array((pre.shape, pre.data))?;
    let outputs = session.run(ort::inputs!["input_1" => input_value])?;
    let (_, box_data) = outputs["Identity"].try_extract_tensor::<f32>()?;
    let (_, score_data) = outputs["Identity_1"].try_extract_tensor::<f32>()?;

    if score_data.len() != anchors.len() {
        anyhow::bail!(
            "palm anchor mismatch: model output {} scores, anchors file has {}",
            score_data.len(),
            anchors.len()
        );
    }

    let input_size = PALM_INPUT_SIZE as f32;
    let mut candidates: Vec<(f32, [f32; 4], Vec<[f32; 2]>)> = Vec::new();

    for i in 0..anchors.len() {
        let raw = score_data[i] as f64;
        let s = (1.0 / (1.0 + (-raw).exp())) as f32;
        if s < PALM_SCORE_THRESHOLD {
            continue;
        }
        let off = i * 18;
        let ax = anchors[i][0];
        let ay = anchors[i][1];
        let cx_d = box_data[off] / input_size;
        let cy_d = box_data[off + 1] / input_size;
        let w_d = box_data[off + 2] / input_size;
        let h_d = box_data[off + 3] / input_size;
        let x1 = (cx_d - w_d / 2.0 + ax) * pre.scale - pre.pad_left_orig;
        let y1 = (cy_d - h_d / 2.0 + ay) * pre.scale - pre.pad_top_orig;
        let x2 = (cx_d + w_d / 2.0 + ax) * pre.scale - pre.pad_left_orig;
        let y2 = (cy_d + h_d / 2.0 + ay) * pre.scale - pre.pad_top_orig;
        let mut kps: Vec<[f32; 2]> = Vec::with_capacity(7);
        for k in 0..7 {
            let kx = box_data[off + 4 + k * 2] / input_size;
            let ky = box_data[off + 4 + k * 2 + 1] / input_size;
            kps.push([
                (kx + ax) * pre.scale - pre.pad_left_orig,
                (ky + ay) * pre.scale - pre.pad_top_orig,
            ]);
        }
        candidates.push((s, [x1, y1, x2, y2], kps));
    }

    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep_mask: Vec<bool> = vec![true; candidates.len()];
    for i in 0..candidates.len() {
        if !keep_mask[i] {
            continue;
        }
        for j in (i + 1)..candidates.len() {
            if !keep_mask[j] {
                continue;
            }
            if iou_xyxy(&candidates[i].1, &candidates[j].1) > PALM_NMS_THRESHOLD {
                keep_mask[j] = false;
            }
        }
    }

    let results: Vec<PalmBox> = candidates
        .into_iter()
        .zip(keep_mask.into_iter())
        .filter(|(_, k)| *k)
        .map(|((score, b, kps), _)| PalmBox {
            score,
            x1: b[0],
            y1: b[1],
            x2: b[2],
            y2: b[3],
            keypoints: kps,
        })
        .collect();

    Ok(results)
}

// [TEST向き] 純粋な数式。完全重なり=1/非重なり=0/半分重なり を手計算で固定(定番のテスト対象)
fn iou_xyxy(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let intersect = iw * ih;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - intersect;
    if union > 0.0 {
        intersect / union
    } else {
        0.0
    }
}

struct HandPreprocess {
    shape: Vec<usize>,
    data: Vec<f32>,
    cx_orig: f32,
    cy_orig: f32,
    crop_size: f32,
    /// 回転正規化の cos/sin(クロップ→元画像の逆変換に使う)。回転なしなら cos=1, sin=0。
    cos_r: f32,
    sin_r: f32,
}

// === Hand 回転アラインメント(MediaPipe 手パイプライン準拠) ===

/// 角度を (-π, π] に正規化する(MediaPipe NormalizeRadians 相当)。
#[inline]
fn normalize_radians(angle: f32) -> f32 {
    use std::f32::consts::PI;
    angle - 2.0 * PI * ((angle + PI) / (2.0 * PI)).floor()
}

/// Palm keypoint から手の回転正規化角(ラジアン)を求める。
/// keypoint[0]=手首中心, keypoint[2]=中指 MCP を結ぶベクトルが
/// 「上向き(target=90°)」になる回転角。MediaPipe DetectionsToRectsCalculator と同式。
/// keypoint が 3 点未満なら 0(回転なし=従来の軸並行クロップ)へフォールバック。
fn palm_rotation(palm: &PalmBox) -> f32 {
    if palm.keypoints.len() < 3 {
        return 0.0;
    }
    let (x1, y1) = (palm.keypoints[0][0], palm.keypoints[0][1]);
    let (x2, y2) = (palm.keypoints[2][0], palm.keypoints[2][1]);
    let target = std::f32::consts::FRAC_PI_2;
    // 画像座標は y 下向きなので atan2 の y を符号反転して数学的な向きへ揃える。
    let angle = target - (-(y2 - y1)).atan2(x2 - x1);
    normalize_radians(angle)
}

/// モデル入力(回転済み HAND_INPUT_SIZE 正方)座標 (mx,my) を元画像座標へ写すアフィン変換。
/// crop_size でスケール・(cos_r,sin_r) で回転・(cx,cy) で平行移動。
/// 回転なし(cos=1,sin=0)のとき (mx-center)*scale + c{x,y} に一致し従来挙動と後方互換。
#[inline]
fn hand_model_to_orig(
    mx: f32,
    my: f32,
    crop_size: f32,
    cos_r: f32,
    sin_r: f32,
    cx: f32,
    cy: f32,
) -> (f32, f32) {
    let dx = mx / HAND_INPUT_SIZE as f32 - 0.5;
    let dy = my / HAND_INPUT_SIZE as f32 - 0.5;
    (
        crop_size * (cos_r * dx - sin_r * dy) + cx,
        crop_size * (sin_r * dx + cos_r * dy) + cy,
    )
}

/// 元画像(Array3<u8>, [H,W,3])を実数座標 (x,y) で双線形サンプリングする。
/// 範囲外の隅は黒(0)として寄与させる(MediaPipe の border=ZERO 相当)。
#[inline]
fn sample_bilinear(frame: &Array3<u8>, x: f32, y: f32) -> [f32; 3] {
    let w = frame.shape()[1] as i32;
    let h = frame.shape()[0] as i32;
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let mut out = [0.0f32; 3];
    for (dy, wy) in [(0i32, 1.0 - fy), (1, fy)] {
        let yy = y0 + dy;
        if yy < 0 || yy >= h {
            continue;
        }
        for (dx, wx) in [(0i32, 1.0 - fx), (1, fx)] {
            let xx = x0 + dx;
            if xx < 0 || xx >= w {
                continue;
            }
            let wgt = wx * wy;
            for ch in 0..3 {
                out[ch] += frame[[yy as usize, xx as usize, ch]] as f32 * wgt;
            }
        }
    }
    out
}

// [TEST向き] 回転正規化付きクロップ変換。PalmBox の回転角でクロップを回し、双線形で
// HAND_INPUT_SIZE 正方へサンプリングする。回転 0 のとき従来の軸並行クロップに一致。
fn preprocess_for_hand_landmark(frame: &Array3<u8>, palm: &PalmBox) -> Result<HandPreprocess> {
    let palm_w = palm.x2 - palm.x1;
    let palm_h = palm.y2 - palm.y1;
    let palm_cx = (palm.x1 + palm.x2) / 2.0;
    let palm_cy = (palm.y1 + palm.y2) / 2.0;

    let palm_size = palm_w.max(palm_h);
    let crop_size = (palm_size * HAND_CROP_ENLARGE).max(1.0);

    // MediaPipe 手パイプライン準拠の回転正規化角(手首→中指MCP を上向きに揃える)。
    let theta = palm_rotation(palm);
    let cos_r = theta.cos();
    let sin_r = theta.sin();

    // クロップ中心シフトは回転後フレームの y 軸方向へ適用(MediaPipe RectTransformation 準拠)。
    // theta=0 では cy = palm_cy + HAND_CROP_SHIFT_Y*palm_h となり従来の shifted_cy に一致する。
    let shift = HAND_CROP_SHIFT_Y * palm_h;
    let cx = palm_cx - sin_r * shift;
    let cy = palm_cy + cos_r * shift;

    let size = HAND_INPUT_SIZE as usize;
    let mut data: Vec<f32> = Vec::with_capacity(size * size * 3);
    // NHWC: 行(y)優先で各画素 RGB を [0,1] 正規化して詰める。
    for my in 0..size {
        for mx in 0..size {
            // 画素中心でサンプリング。
            let (sx, sy) =
                hand_model_to_orig(mx as f32 + 0.5, my as f32 + 0.5, crop_size, cos_r, sin_r, cx, cy);
            let px = sample_bilinear(frame, sx, sy);
            data.push(px[0] / 255.0);
            data.push(px[1] / 255.0);
            data.push(px[2] / 255.0);
        }
    }

    Ok(HandPreprocess {
        shape: vec![1usize, size, size, 3],
        data,
        cx_orig: cx,
        cy_orig: cy,
        crop_size,
        cos_r,
        sin_r,
    })
}

// [評価] ONNX推論。前処理(preprocess_for_hand_landmark)とパース(parse_landmarks)を切り出してテスト
fn run_hand_landmark(
    session: &mut Session,
    frame: &Array3<u8>,
    palm: &PalmBox,
) -> Result<Option<Hand>> {
    let pre = preprocess_for_hand_landmark(frame, palm)?;
    let input_value = Value::from_array((pre.shape, pre.data))?;
    let outputs = session.run(ort::inputs!["input_1" => input_value])?;
    let (_, ld) = outputs["Identity"].try_extract_tensor::<f32>()?;
    let (_, conf) = outputs["Identity_1"].try_extract_tensor::<f32>()?;
    let (_, handedness) = outputs["Identity_2"].try_extract_tensor::<f32>()?;

    if ld.len() != 63 {
        anyhow::bail!(
            "hand landmark output size unexpected: got {} (expected 63)",
            ld.len()
        );
    }

    let confidence = conf[0];
    if confidence < HAND_CONF_THRESHOLD {
        return Ok(None);
    }

    let hand_score: f32 = handedness[0];
    // z はクロップ面内の深さ尺度なので回転不変。x,y は回転を含む逆アフィンで元画像へ戻す。
    let scale = pre.crop_size / HAND_INPUT_SIZE as f32;
    let mut landmarks: Vec<HandLandmarkPoint> = Vec::with_capacity(21);
    for i in 0..21 {
        let off = i * 3;
        let (x, y) = hand_model_to_orig(
            ld[off],
            ld[off + 1],
            pre.crop_size,
            pre.cos_r,
            pre.sin_r,
            pre.cx_orig,
            pre.cy_orig,
        );
        let z = ld[off + 2] * scale;
        landmarks.push(HandLandmarkPoint { x, y, z });
    }

    Ok(Some(Hand {
        confidence,
        handedness: hand_score,
        landmarks,
    }))
}

// [TEST向き] パース処理。195→39×[x,y,z,vis,pres] のインデックスずれを検知(壊れても静かに誤る型)
fn parse_landmarks(data: &[f32]) -> Result<Vec<Landmark>> {
    if data.len() != 195 {
        anyhow::bail!(
            "expected 195 floats for landmarks (39 x 5), got {}",
            data.len()
        );
    }
    let landmarks: Vec<Landmark> = data
        .chunks_exact(5)
        .map(|c| Landmark {
            x: c[0],
            y: c[1],
            z: c[2],
            visibility: c[3],
            presence: c[4],
        })
        .collect();
    Ok(landmarks)
}

// [TEST向き] 純粋数式。sigmoid(0)=0.5 など既知点で固定
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn apply_sigmoid_to_frames(frames: &mut [PoseFrame]) {
    for f in frames {
        for lm in &mut f.landmarks {
            lm.visibility = sigmoid(lm.visibility);
            lm.presence = sigmoid(lm.presence);
        }
    }
}

// [TEST向き] フォーマット/round-trip。197列・ヘッダ順・frame_idx連番。Vec<u8>に書いて読み直し一致
fn write_tsv<W: Write>(seq: &PoseSequence, w: &mut W) -> Result<()> {
    write!(w, "frame_idx\tconfidence")?;
    for i in 0..39 {
        write!(w, "\tx{i}\ty{i}\tz{i}\tvis{i}\tpres{i}")?;
    }
    writeln!(w)?;

    for f in &seq.frames {
        write!(w, "{}\t{}", f.frame_idx, f.confidence)?;
        for lm in &f.landmarks {
            write!(
                w,
                "\t{}\t{}\t{}\t{}\t{}",
                lm.x, lm.y, lm.z, lm.visibility, lm.presence
            )?;
        }
        writeln!(w)?;
    }
    Ok(())
}

// [TEST向き] 描画手前の純粋なインデックス選択ロジックなら固定可能(描画自体は目視)
fn overlay_target_indices(
    save_overlay: Option<&PathBuf>,
    frame_count: usize,
    max_frames: Option<usize>,
    overlay_count: usize,
) -> Vec<usize> {
    if save_overlay.is_none() {
        return vec![];
    }
    // 実際に処理されるフレーム数 = min(max_frames, frame_count)。
    // 旧実装 m.min(frame_count.max(m)) は max_frames > frame_count のとき常に
    // max_frames になり、存在しないフレーム番号を overlay 対象にしてしまうバグだった。
    let upper = match max_frames {
        Some(m) => m.min(frame_count),
        None => frame_count,
    };
    if upper == 0 {
        eprintln!("warning: frame_count is 0 (ffprobe nb_frames unavailable); overlay disabled");
        return vec![];
    }
    let count = overlay_count.max(1).min(upper);
    if count == 1 {
        return vec![0];
    }
    (0..count)
        .map(|i| i * (upper - 1) / (count - 1))
        .collect()
}

// [評価] 代理指標の出力。S6の教訓: ここが緑(conf=0.82-0.90)でも本体は破綻していた。数値テストの過信に注意
fn print_stats(seq: &PoseSequence) {
    if seq.frames.is_empty() {
        eprintln!("stats: (no frames)");
        return;
    }
    let n = seq.frames.len() as f32;
    let confs: Vec<f32> = seq.frames.iter().map(|f| f.confidence).collect();
    let conf_mean = confs.iter().sum::<f32>() / n;
    let conf_max = confs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let conf_min = confs.iter().cloned().fold(f32::INFINITY, f32::min);

    let vis_sum: f32 = seq
        .frames
        .iter()
        .flat_map(|f| f.landmarks.iter().map(|l| {
            if seq.sigmoid_applied {
                l.visibility
            } else {
                sigmoid(l.visibility)
            }
        }))
        .sum();
    let vis_mean = vis_sum / (n * 39.0);

    eprintln!(
        "stats: confidence mean={:.3} max={:.3} min={:.3} | visibility(sigmoid) mean={:.3}",
        conf_mean, conf_max, conf_min, vis_mean
    );
}

// [目視] オーバーレイ画像を出力。正しさは目で確認(S6はここで破綻を発見)。自動テスト不向き
fn draw_overlay(
    frame: &Array3<u8>,
    pose: &PoseFrame,
    sigmoid_applied: bool,
    palm: Option<&PalmFrame>,
    hand: Option<&HandFrame>,
    out: &std::path::Path,
) -> Result<()> {
    let h = frame.shape()[0] as u32;
    let w = frame.shape()[1] as u32;
    let raw: Vec<u8> = frame.iter().copied().collect();
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(w, h, raw)
        .context("failed to construct ImageBuffer from frame bytes for overlay")?;

    let radius: i32 = 3;
    let (crop_x, crop_y, side) = crop_square_center(w, h);
    let scale = side as f32 / 256.0;

    for lm in &pose.landmarks {
        let vis = if sigmoid_applied {
            lm.visibility
        } else {
            sigmoid(lm.visibility)
        };
        let color = if vis > 0.5 {
            Rgb([0u8, 255, 0])
        } else {
            Rgb([255u8, 0, 0])
        };
        let cx = (lm.x * scale + crop_x as f32).round() as i32;
        let cy = (lm.y * scale + crop_y as f32).round() as i32;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && py >= 0 && (px as u32) < w && (py as u32) < h {
                    img.put_pixel(px as u32, py as u32, color);
                }
            }
        }
    }

    if let Some(palm_frame) = palm {
        let cyan = Rgb([0u8, 200, 255]);
        let yellow = Rgb([255u8, 220, 0]);
        for pbox in &palm_frame.palms {
            draw_rect_outline(&mut img, pbox.x1, pbox.y1, pbox.x2, pbox.y2, cyan, 2);
            for kp in &pbox.keypoints {
                draw_dot(&mut img, kp[0], kp[1], 4, yellow);
            }
        }
    }

    if let Some(hand_frame) = hand {
        let magenta = Rgb([255u8, 0, 200]);
        let blue = Rgb([60u8, 120, 255]);
        for h in &hand_frame.hands {
            let color = if h.handedness > 0.5 { magenta } else { blue };
            // 骨格線(ボーン)を先に描き、その上に関節ドットを重ねる。
            // これで指の曲がり方(回転アラインメントの効果)が目視できる。
            if h.landmarks.len() >= HAND_POINTS {
                for &(a, b) in HAND_CONNECTIONS {
                    let p = &h.landmarks[a];
                    let q = &h.landmarks[b];
                    draw_line(&mut img, p.x, p.y, q.x, q.y, color, 1);
                }
            }
            for lm in &h.landmarks {
                draw_dot(&mut img, lm.x, lm.y, 3, color);
            }
        }
    }

    img.save(out)
        .with_context(|| format!("failed to save overlay PNG: {}", out.display()))?;
    Ok(())
}

fn draw_dot(
    img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    x: f32,
    y: f32,
    radius: i32,
    color: Rgb<u8>,
) {
    let cx = x.round() as i32;
    let cy = y.round() as i32;
    let w = img.width() as i32;
    let h = img.height() as i32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let px = cx + dx;
            let py = cy + dy;
            if px >= 0 && py >= 0 && px < w && py < h {
                img.put_pixel(px as u32, py as u32, color);
            }
        }
    }
}

/// 2 点間にブレゼンハム直線を引く。`thickness` は中心からの半径(0 で 1px、1 で 3px 幅)。
/// 手の骨格(指の曲がり)をオーバーレイ動画で目視するために使う。
fn draw_line(
    img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: Rgb<u8>,
    thickness: i32,
) {
    let mut x = x0.round() as i32;
    let mut y = y0.round() as i32;
    let x1 = x1.round() as i32;
    let y1 = y1.round() as i32;
    let dx = (x1 - x).abs();
    let dy = -(y1 - y).abs();
    let sx = if x < x1 { 1 } else { -1 };
    let sy = if y < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let w = img.width() as i32;
    let h = img.height() as i32;
    loop {
        // 太さ分の小さな矩形を塗る
        for oy in -thickness..=thickness {
            for ox in -thickness..=thickness {
                let px = x + ox;
                let py = y + oy;
                if px >= 0 && py >= 0 && px < w && py < h {
                    img.put_pixel(px as u32, py as u32, color);
                }
            }
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn draw_rect_outline(
    img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: Rgb<u8>,
    thickness: i32,
) {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let xa = x1.min(x2).round() as i32;
    let xb = x1.max(x2).round() as i32;
    let ya = y1.min(y2).round() as i32;
    let yb = y1.max(y2).round() as i32;
    for t in 0..thickness {
        for x in xa..=xb {
            for &y in &[ya + t, yb - t] {
                if x >= 0 && y >= 0 && x < w && y < h {
                    img.put_pixel(x as u32, y as u32, color);
                }
            }
        }
        for y in ya..=yb {
            for &x in &[xa + t, xb - t] {
                if x >= 0 && y >= 0 && x < w && y < h {
                    img.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }
}

// [dev/手動] 開発時にモデル構造を覗くためのサブコマンド実装。cargo test とは無関係
fn inspect_onnx(model_path: &PathBuf) -> Result<()> {
    let session = ort::session::Session::builder()?
        .commit_from_file(model_path)
        .with_context(|| format!("Failed to load ONNX model: {}", model_path.display()))?;

    println!("=== ONNX Model: {} ===", model_path.display());
    println!("inputs:");
    for input in session.inputs().iter() {
        println!("  name={}", input.name());
    }
    println!("outputs:");
    for output in session.outputs().iter() {
        println!("  name={}", output.name());
    }
    Ok(())
}

// [dev/手動] 名前に"test"が付くが cargo test ではない。`pose-extract test-infer` 用の手動スモーク。混同注意
fn test_inference(model_path: &PathBuf) -> Result<()> {
    let mut session = ort::session::Session::builder()?
        .commit_from_file(model_path)
        .with_context(|| format!("Failed to load ONNX model: {}", model_path.display()))?;

    // Dummy input: shape (1, 256, 256, 3), float32, all 0.5 (mid-gray)
    let shape = vec![1usize, 256, 256, 3];
    let data: Vec<f32> = vec![0.5; 1 * 256 * 256 * 3];
    println!("input shape: {:?}", shape);

    let input_value = ort::value::Value::from_array((shape, data))?;
    let outputs = session.run(ort::inputs!["input_1" => input_value])?;

    println!("=== Inference outputs ===");
    for (name, value) in outputs.iter() {
        let (shape, data) = value.try_extract_tensor::<f32>()?;
        println!(
            "  {}: shape={:?}, dtype=f32, len={}",
            name,
            shape,
            data.len()
        );
    }
    Ok(())
}

// [評価/スモーク] ffprobe外部依存。固定の小動画を1本同梱して「読めて妥当な値が返る」程度のスモーク
fn probe_video(path: &PathBuf) -> Result<VideoInfo> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            path.to_str().context("path must be UTF-8")?,
        ])
        .output()
        .context("Failed to run ffprobe (is ffmpeg installed?)")?;

    if !output.status.success() {
        anyhow::bail!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let json: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let stream = &json["streams"][0];

    let width = stream["width"].as_u64().context("width missing")? as u32;
    let height = stream["height"].as_u64().context("height missing")? as u32;
    let r_frame_rate = stream["r_frame_rate"]
        .as_str()
        .context("r_frame_rate missing")?;
    let fps = parse_frame_rate(r_frame_rate)?;
    let frame_count = stream["nb_frames"]
        .as_str()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    let duration = stream["duration"]
        .as_str()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);

    Ok(VideoInfo {
        width,
        height,
        fps,
        frame_count,
        duration,
    })
}

// [TEST向き] 純粋パース。"30/1"→30.0、"30000/1001"→29.97、0除算・異常文字列のエラー
fn parse_frame_rate(s: &str) -> Result<f64> {
    let parts: Vec<&str> = s.split('/').collect();
    match parts.as_slice() {
        [n, d] => {
            let num = n.parse::<f64>()?;
            let den = d.parse::<f64>()?;
            if den == 0.0 {
                anyhow::bail!("Invalid frame rate (division by zero): {}", s);
            }
            Ok(num / den)
        }
        [n] => Ok(n.parse::<f64>()?),
        _ => anyhow::bail!("Invalid frame rate format: {}", s),
    }
}

// [評価/スモーク] ffmpeg外部依存。「1本デコードしてフレーム数が想定通り」程度のスモークまで
fn extract_frames<F>(path: &PathBuf, width: u32, height: u32, mut callback: F) -> Result<()>
where
    F: FnMut(usize, Array3<u8>) -> Result<()>,
{
    let mut child = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-i",
            path.to_str().context("path must be UTF-8")?,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .stdout(Stdio::piped())
        .spawn()
        .context("Failed to spawn ffmpeg")?;

    let mut stdout = child.stdout.take().context("ffmpeg stdout not piped")?;
    let frame_size = (width * height * 3) as usize;
    let mut buf = vec![0u8; frame_size];
    let mut frame_idx = 0usize;

    loop {
        match stdout.read_exact(&mut buf) {
            Ok(()) => {
                let array = Array3::from_shape_vec(
                    (height as usize, width as usize, 3),
                    buf.clone(),
                )?;
                callback(frame_idx, array)?;
                frame_idx += 1;
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
    }

    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("ffmpeg exited with status: {}", status);
    }

    Ok(())
}

/// オーバーレイ PNG 群(frame_00000.png ...)を ffmpeg で 1 本の mp4 に束ねる。
/// フレーム番号は 0 からの連番なので image2 デマルチプレクサ(%05d)で読める。
fn encode_overlay_video(dir: &Path, out: &Path, fps: f64) -> Result<()> {
    // fps が不明(ffprobe で 0)なら 30 にフォールバック
    let fps = if fps > 0.0 { fps } else { 30.0 };
    let pattern = dir.join("frame_%05d.png");
    let status = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-framerate",
            &format!("{:.6}", fps),
            "-start_number",
            "0",
            "-i",
            pattern.to_str().context("overlay dir path must be UTF-8")?,
            // libx264 は偶数サイズ必須。奇数解像度でも安全側に丸める
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            out.to_str().context("overlay output path must be UTF-8")?,
        ])
        .status()
        .context("Failed to spawn ffmpeg for overlay video")?;
    if !status.success() {
        anyhow::bail!("ffmpeg (overlay video) exited with status: {}", status);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_indices_stay_within_frame_count() {
        let save = PathBuf::from("dummy.png");
        // max_frames > frame_count: 存在しないフレームを指してはいけない
        let idx = overlay_target_indices(Some(&save), 5, Some(10), 3);
        assert!(idx.iter().all(|&i| i < 5), "範囲外インデックス: {:?}", idx);
        assert_eq!(idx, vec![0, 2, 4]);
        // max_frames < frame_count: max_frames 側で頭打ち
        let idx = overlay_target_indices(Some(&save), 100, Some(4), 4);
        assert_eq!(idx, vec![0, 1, 2, 3]);
        // overlay 無効なら空
        assert!(overlay_target_indices(None, 5, None, 3).is_empty());
        // frame_count=0(nb_frames 不明)なら overlay 無効
        assert!(overlay_target_indices(Some(&save), 0, Some(10), 3).is_empty());
    }

    /// 実モデル(pose+palm+hand)で動画→オーバーレイ動画を生成する E2E スモーク。
    /// ONNX が重いので通常はスキップ。明示実行: `cargo test overlay_video_smoke -- --ignored`
    #[test]
    #[ignore]
    fn overlay_video_smoke() {
        let video = PathBuf::from("videos/0205-01.mp4");
        if !video.exists() {
            eprintln!("skip: {} が無い(撮影/モデル未配置)", video.display());
            return;
        }
        let dir = std::env::temp_dir().join("pose_overlay_smoke");
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = RunConfig {
            input: video,
            model: PathBuf::from(DEFAULT_MODEL),
            palm_model: Some(PathBuf::from(PALM_MODEL)),
            hand_model: Some(PathBuf::from(HAND_MODEL)),
            format: OutputFormat::Tsv,
            apply_sigmoid: false,
            save_overlay: Some(dir.clone()),
            overlay_count: 0,
            overlay_video: true,
            max_frames: Some(8), // 速度のため先頭 8 フレームのみ
            output: Some(dir.join("out.tsv")),
        };
        run_extraction(cfg).expect("run_extraction 失敗");

        // 全フレーム PNG と束ねた mp4 が出来ていること
        let mp4 = dir.join("overlay.mp4");
        assert!(mp4.exists(), "overlay.mp4 が無い");
        assert!(
            std::fs::metadata(&mp4).unwrap().len() > 0,
            "overlay.mp4 が空"
        );
        assert!(dir.join("frame_00000.png").exists(), "先頭フレーム PNG が無い");
    }

    #[test]
    fn hand_connections_cover_21_bones() {
        // MediaPipe Hands の骨格は 21 本。indices は 0..21 に収まり自己ループはない。
        assert_eq!(HAND_CONNECTIONS.len(), 21);
        for &(a, b) in HAND_CONNECTIONS {
            assert!(a < HAND_POINTS, "a={} が範囲外", a);
            assert!(b < HAND_POINTS, "b={} が範囲外", b);
            assert_ne!(a, b, "自己ループは無効: ({}, {})", a, b);
        }
    }

    #[test]
    fn draw_line_plots_endpoints_and_skips_oob() {
        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_pixel(10, 10, Rgb([0, 0, 0]));
        let red = Rgb([255u8, 0, 0]);
        // 水平線(太さ 0 = 1px)。両端と中間が塗られる。
        draw_line(&mut img, 1.0, 1.0, 8.0, 1.0, red, 0);
        assert_eq!(*img.get_pixel(1, 1), red);
        assert_eq!(*img.get_pixel(8, 1), red);
        assert_eq!(*img.get_pixel(5, 1), red);
        // 線から外れた点は黒のまま
        assert_eq!(*img.get_pixel(5, 5), Rgb([0u8, 0, 0]));
        // 画像外へ伸びる線でもパニックしない(クリップされる)
        draw_line(&mut img, -5.0, 5.0, 5.0, 5.0, red, 0);
        assert_eq!(*img.get_pixel(0, 5), red);
        assert_eq!(*img.get_pixel(5, 5), red);
    }

    /// テスト用 PalmBox を作る(回転は keypoints[0],[2] で決まるので bbox は適当でよい)。
    fn palm_with_kps(wrist: [f32; 2], middle_mcp: [f32; 2]) -> PalmBox {
        // index 1 はダミー。回転には index 0(手首)と 2(中指MCP)だけ使う。
        let keypoints = vec![wrist, [0.0, 0.0], middle_mcp];
        PalmBox {
            score: 1.0,
            x1: 0.0,
            y1: 0.0,
            x2: 100.0,
            y2: 100.0,
            keypoints,
        }
    }

    #[test]
    fn palm_rotation_basic_orientations() {
        use std::f32::consts::FRAC_PI_2;
        // 上向き(中指MCP が手首の真上=画像座標で y 小)→ 回転 0。
        let up = palm_rotation(&palm_with_kps([100.0, 200.0], [100.0, 100.0]));
        assert!(up.abs() < 1e-4, "上向きは回転0のはず: {up}");
        // 右向き(中指MCP が右)→ +90°。
        let right = palm_rotation(&palm_with_kps([100.0, 100.0], [200.0, 100.0]));
        assert!((right - FRAC_PI_2).abs() < 1e-4, "右向きは+π/2のはず: {right}");
        // 左向き(中指MCP が左)→ -90°。
        let left = palm_rotation(&palm_with_kps([100.0, 100.0], [0.0, 100.0]));
        assert!((left + FRAC_PI_2).abs() < 1e-4, "左向きは-π/2のはず: {left}");
        // keypoint 不足は 0 にフォールバック。
        let degenerate = PalmBox {
            score: 1.0,
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            keypoints: vec![[0.0, 0.0]],
        };
        assert_eq!(palm_rotation(&degenerate), 0.0);
    }

    #[test]
    fn hand_model_to_orig_center_and_rotation() {
        let center = HAND_INPUT_SIZE as f32 / 2.0;
        let crop = 300.0;
        let (cx, cy) = (500.0, 400.0);
        // モデル中心(112,112)は回転に関わらずクロップ中心へ写る。
        for &(c, s) in &[(1.0_f32, 0.0_f32), (0.0, 1.0), (-1.0, 0.0)] {
            let (ox, oy) = hand_model_to_orig(center, center, crop, c, s, cx, cy);
            assert!((ox - cx).abs() < 1e-3 && (oy - cy).abs() < 1e-3);
        }
        // 回転0なら従来式 (m-center)*scale + c に一致。
        let scale = crop / HAND_INPUT_SIZE as f32;
        let (ox, oy) = hand_model_to_orig(200.0, 50.0, crop, 1.0, 0.0, cx, cy);
        assert!((ox - ((200.0 - center) * scale + cx)).abs() < 1e-3);
        assert!((oy - ((50.0 - center) * scale + cy)).abs() < 1e-3);
        // +90°回転(c=0,s=1): モデル上端中央(112,0)は中心の右(+crop/2, 0)へ。
        let (ox, oy) = hand_model_to_orig(center, 0.0, crop, 0.0, 1.0, cx, cy);
        assert!((ox - (cx + crop / 2.0)).abs() < 1e-3, "ox={ox}");
        assert!((oy - cy).abs() < 1e-3, "oy={oy}");
    }

    #[test]
    fn sample_bilinear_exact_and_oob() {
        // 2x2 グレースケール風: (y,x) 値を R に入れる。
        let mut frame = Array3::<u8>::zeros((2, 2, 3));
        frame[[0, 0, 0]] = 10;
        frame[[0, 1, 0]] = 20;
        frame[[1, 0, 0]] = 30;
        frame[[1, 1, 0]] = 40;
        // 整数格子点は元値ぴったり。
        assert!((sample_bilinear(&frame, 0.0, 0.0)[0] - 10.0).abs() < 1e-4);
        assert!((sample_bilinear(&frame, 1.0, 1.0)[0] - 40.0).abs() < 1e-4);
        // 中点は4画素平均 = (10+20+30+40)/4 = 25。
        assert!((sample_bilinear(&frame, 0.5, 0.5)[0] - 25.0).abs() < 1e-4);
        // 範囲外は黒。
        assert_eq!(sample_bilinear(&frame, -5.0, -5.0), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn parse_frame_rate_cases() {
        assert!((parse_frame_rate("30/1").unwrap() - 30.0).abs() < 1e-9);
        assert!((parse_frame_rate("30000/1001").unwrap() - 29.97).abs() < 0.01);
        assert!((parse_frame_rate("60").unwrap() - 60.0).abs() < 1e-9);
        // 0除算は Ok(inf) ではなくエラー
        assert!(parse_frame_rate("30000/0").is_err());
        assert!(parse_frame_rate("abc").is_err());
    }

    /// 動画→タグ直結パスの E2E スモーク(in-sample)。
    /// ffmpeg + Palm/Hand ONNX + rec_smoke モデルの実体に依存するため #[ignore]。
    /// 手動実行: `cargo test recognize_smoke_in_sample -- --ignored --nocapture`
    /// 学習に使った動画なので top-5 に正解タグ "0205" が入るはず
    /// (これは配線確認であって精度の証拠ではない。本判定は未見テイクで行う)。
    #[test]
    #[ignore]
    fn recognize_smoke_in_sample() {
        let video = Path::new(VIDEO_DIR).join("0205-01.mp4");
        let model_dir = Path::new(REC_MODEL_DIR).join("rec_smoke");
        // フィクスチャが無い環境では明示してスキップ
        for p in [
            video.as_path(),
            model_dir.as_path(),
            Path::new(PALM_MODEL),
            Path::new(HAND_MODEL),
        ] {
            if !p.exists() {
                eprintln!("skip: fixture not found: {}", p.display());
                return;
            }
        }

        let frames = 10usize;
        let palm_anchors = load_palm_anchors();
        let mut palm = Session::builder()
            .unwrap()
            .commit_from_file(PALM_MODEL)
            .unwrap();
        let mut hand = Session::builder()
            .unwrap()
            .commit_from_file(HAND_MODEL)
            .unwrap();

        let entry =
            extract_tag_features(&video, &mut palm, &mut hand, &palm_anchors, frames).unwrap();
        let flat: Vec<f32> = entry.sequence.iter().flatten().copied().collect();
        assert_eq!(flat.len(), frames * DICT_FEATURE_DIM);

        let ranked =
            transformer_burn::recognition::predict_from_features(&model_dir, &flat, frames, 5)
                .unwrap();
        eprintln!("ranked = {:?}", ranked);
        assert!(
            ranked.iter().any(|(tag, _)| tag == "0205"),
            "top-5 に正解タグ 0205 が含まれない: {:?}",
            ranked
        );
    }
}
