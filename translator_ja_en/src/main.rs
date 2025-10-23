#![recursion_limit = "256"]
mod checkpoint;
mod config;
mod inference;
mod jsl_data;
mod jsl_vocabulary;
mod metrics;
mod model;
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    let args = Args::parse();

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
