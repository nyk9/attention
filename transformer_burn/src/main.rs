#![recursion_limit = "256"]
mod checkpoint;
mod config;
mod inference;
mod jsl_data;
mod jsl_vocabulary;
mod metrics;
mod model;
mod pose_data;
mod recognition;
mod recognition_training;
mod tag_vocabulary;
mod training;

use checkpoint::{TrainingBackend, load_model, save_model};
use inference::{predict_tags, run_inference};
use metrics::save_metrics;
use training::train_jsl;

use burn::backend::wgpu::WgpuDevice;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

/// Seq2Seq日本語→手話タグ翻訳モデル
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// 訓練モード（訓練を実行する場合に指定）
    #[arg(long)]
    train: bool,

    /// モデルを保存するディレクトリ
    #[arg(long)]
    save: Option<PathBuf>,

    /// モデルを読み込むディレクトリ
    #[arg(long)]
    load: Option<PathBuf>,

    /// 推論するテキスト
    #[arg(long)]
    predict: Option<String>,

    /// バックエンドの選択（auto, wgpu, ndarray）
    #[arg(long, default_value = "wgpu")]
    backend: String,

    /// Attention行列をCSVエクスポート（推論時のみ）
    #[arg(long)]
    export_attn: bool,

    /// 認識モデル訓練: build-dict が出力した pose dict JSON を指定
    #[arg(long)]
    train_pose: Option<PathBuf>,

    /// 認識モデル推論: pose dict JSON の各エントリを top-5 評価（--load 必須）
    #[arg(long)]
    predict_pose: Option<PathBuf>,

    /// 認識モデルのエポック数
    #[arg(long, default_value_t = config::POSE_EPOCHS)]
    epochs: usize,

    /// 未見テイク評価: pose dict をテイク単位で train/held-out に分割し、学習後に未見テイクで top-k を測る
    #[arg(long)]
    eval_holdout: Option<PathBuf>,

    /// 診断: pose dict の左右手カバレッジをラベル別に点検し、テイク間で使われた手が食い違う語を警告(学習なし)
    #[arg(long)]
    inspect_dict: Option<PathBuf>,

    /// held-out にするテイク数(ラベルごと、テイク番号の後ろから)。--eval-holdout 用
    #[arg(long, default_value_t = 1)]
    holdout_per_label: usize,

    /// ミラー拡張: 学習データに左右反転コピーを追加する(--train-pose / --eval-holdout の train 側のみ。
    /// held-out の未見テイクには適用しない)
    #[arg(long)]
    mirror_augment: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    let args = Args::parse();

    // 診断モード: モデルもバックエンドも使わない JSON 分析なのでここで完結させる
    if let Some(dict_path) = &args.inspect_dict {
        pose_data::inspect_dict(dict_path)?;
        return Ok(());
    }

    // 認識モデル(手ポーズ列→タグ)のモード。足場の Seq2Seq とはモデルも語彙も別物なので
    // ここで分岐して早期 return する
    if args.train_pose.is_some() || args.predict_pose.is_some() || args.eval_holdout.is_some() {
        run_recognition(&args)?;
        let duration = start_time.elapsed();
        println!("\n実行時間: {:.2}秒", duration.as_secs_f64());
        return Ok(());
    }

    let jsl_vocab = jsl_vocabulary::JslVocabulary::new();
    let training_device = WgpuDevice::default();

    // モデルの初期化または読み込み
    let mut model = if let Some(load_dir) = &args.load {
        load_model(load_dir, &training_device)?
    } else {
        model::Seq2SeqModel::<TrainingBackend>::new(&training_device)
    };

    // 訓練モード
    if args.train {
        println!("\n===== 訓練開始 =====");
        let jsl_data = jsl_data::JslTrainingData::load(&jsl_vocab, "data/training_data_jsl.txt");
        println!("訓練サンプル数: {}サンプル", jsl_data.len());

        let (trained_model, training_metrics) =
            train_jsl(model, &jsl_data, &jsl_vocab, &training_device);
        model = trained_model;
        println!("訓練完了！");

        // モデルとメトリクスを保存
        if let Some(save_dir) = &args.save {
            save_model(&model, save_dir)?;
            save_metrics(save_dir, &training_metrics)?;
        }
    }

    // 推論モード
    if let Some(predict_text) = &args.predict {
        println!("\n===== 推論テスト =====");

        if args.load.is_some() {
            // モデルが読み込まれている場合は指定されたバックエンドで推論
            let predicted_tags = run_inference(
                &args.backend,
                args.load.as_ref().unwrap(),
                predict_text,
                &jsl_vocab,
            )?;
            println!("入力: {} → 予測タグ: {}", predict_text, predicted_tags);
        } else {
            // 訓練直後の場合はTrainingBackendで推論
            let predicted_tags = predict_tags(&model, &jsl_vocab, predict_text, &training_device);
            println!("入力: {} → 予測タグ: {}", predict_text, predicted_tags);
        }
    }

    // デモモード（引数なし）
    if !args.train && args.predict.is_none() && args.load.is_none() {
        println!("===== デモモード =====");
        println!("使用方法:");
        println!("  訓練: cargo run --release -- --train --save models/test");
        println!("  推論: cargo run --release -- --load models/test --predict \"ありがとう\"");
        println!(
            "  継続訓練: cargo run --release -- --load models/test --train --save models/test2"
        );
    }

    let duration = start_time.elapsed();
    println!("\n実行時間: {:.2}秒", duration.as_secs_f64());

    Ok(())
}

/// 認識モデル(手ポーズ列→タグ)の訓練・推論を実行する
///
/// 使用例:
///   訓練: cargo run --release -- --train-pose data/pose_dict_smoke.json --save models/rec001
///   推論: cargo run --release -- --load models/rec001 --predict-pose data/pose_dict_smoke.json
///   未見テイク評価: cargo run --release -- --eval-holdout data/pose_dict_smoke.json [--holdout-per-label 1]
fn run_recognition(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let device = WgpuDevice::default();

    // --- 未見テイク評価モード(Phase 0a の合否判定) ---
    if let Some(dict_path) = &args.eval_holdout {
        // train から新規学習するモードなので、--load(既存モデル読み込み)は意味を持たない。
        // 黙って無視すると誤用に気づきにくいのでエラーにする
        if args.load.is_some() {
            return Err("--eval-holdout は train から新規学習するため --load とは併用できません".into());
        }
        println!("\n===== 未見テイク評価(手ポーズ列→タグ) =====");
        println!("pose dict: {}", dict_path.display());
        println!("holdout_per_label: {}", args.holdout_per_label);

        let split =
            pose_data::PoseTrainingData::load_holdout_split(dict_path, args.holdout_per_label)?;

        println!("\n--- テイク単位の分割 ---");
        for (label, tr, te) in &split.per_label {
            println!("  {:<12} train={} held-out={}", label, tr, te);
        }
        if !split.train_only.is_empty() {
            println!(
                "  注意: テイク不足で held-out を作れず train のみに入れたラベル: {:?}",
                split.train_only
            );
        }
        println!(
            "学習サンプル {} 件 / 未見テイク {} 件",
            split.train.len(),
            split.test.len()
        );

        // ミラー拡張は train 側のみに適用する。held-out(未見テイク)は実撮影データの
        // 分布のまま評価しないと「解けたことにする」測定になってしまうため対象外
        let mut train_data = split.train;
        if args.mirror_augment {
            let before = train_data.len();
            train_data = train_data.with_mirror_augmentation();
            println!(
                "ミラー拡張適用: 学習サンプル {} 件 → {} 件",
                before,
                train_data.len()
            );
        }

        // 語彙は学習データ(train)から構築。held-out のラベルは train の部分集合なので必ず含まれる
        let vocab = train_data.build_vocabulary();
        let model =
            recognition::RecognitionModel::<TrainingBackend>::new(vocab.vocab_size(), &device);
        let (model, loss_history) = recognition_training::train_recognition(
            model,
            &train_data,
            &vocab,
            &device,
            args.epochs,
        );
        println!(
            "訓練完了: Loss {:.6} → {:.6}",
            loss_history.first().unwrap_or(&0.0),
            loss_history.last().unwrap_or(&0.0)
        );

        println!("\n===== 未見テイクでの評価(これが Phase 0a の合否判定) =====");
        recognition_training::evaluate_topk(&model, &split.test, &vocab, &device);

        if let Some(save_dir) = &args.save {
            recognition::save_recognition_model(&model, &vocab, save_dir)?;
        }
        return Ok(());
    }

    // --- 訓練モード ---
    if let Some(dict_path) = &args.train_pose {
        println!("\n===== 認識モデル訓練(手ポーズ列→タグ) =====");
        println!("pose dict: {}", dict_path.display());

        if args.load.is_some() {
            // 継続訓練は語彙の一致確認が必要になるため未対応(将来課題)
            return Err("--train-pose と --load の併用(継続訓練)は未対応です".into());
        }

        let mut data = pose_data::PoseTrainingData::load(dict_path)?;
        if args.mirror_augment {
            let before = data.len();
            data = data.with_mirror_augmentation();
            println!("ミラー拡張適用: {} 件 → {} 件", before, data.len());
        }
        let vocab = data.build_vocabulary();
        println!(
            "サンプル数: {} / タグ数: {} ({:?})",
            data.len(),
            vocab.tags.len(),
            vocab.tags
        );

        let model =
            recognition::RecognitionModel::<TrainingBackend>::new(vocab.vocab_size(), &device);
        let (model, loss_history) =
            recognition_training::train_recognition(model, &data, &vocab, &device, args.epochs);

        println!(
            "訓練完了: Loss {:.6} → {:.6}",
            loss_history.first().unwrap_or(&0.0),
            loss_history.last().unwrap_or(&0.0)
        );

        // 学習データ自身での top-5 確認(ループが回っているかのスモーク)
        recognition_training::evaluate_topk(&model, &data, &vocab, &device);

        if let Some(save_dir) = &args.save {
            recognition::save_recognition_model(&model, &vocab, save_dir)?;
        }
        return Ok(());
    }

    // --- 推論モード ---
    if let Some(dict_path) = &args.predict_pose {
        println!("\n===== 認識モデル推論(手ポーズ列→タグ) =====");
        let load_dir = args
            .load
            .as_ref()
            .ok_or("--predict-pose には --load <モデルディレクトリ> が必要です")?;

        let (model, vocab) =
            recognition::load_recognition_model::<TrainingBackend>(load_dir, &device)?;
        let data = pose_data::PoseTrainingData::load(dict_path)?;
        recognition_training::evaluate_topk(&model, &data, &vocab, &device);
    }

    Ok(())
}
