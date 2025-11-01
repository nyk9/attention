#![recursion_limit = "256"]

use translator_ja_en::config;
use translator_ja_en::metrics::save_metrics;
use translator_ja_en::translation_checkpoint::save_model;
use translator_ja_en::translation_data::TranslationData;
use translator_ja_en::translation_inference::{predict_translation, run_translation_inference};
use translator_ja_en::translation_model::Seq2SeqModel;
use translator_ja_en::translation_training::{train_translation, TrainingBackend};
use translator_ja_en::translation_vocabulary::{SourceVocabulary, TargetVocabulary};

use burn::backend::wgpu::WgpuDevice;
use clap::Parser;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Seq2Seq日本語→英語翻訳モデル
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

    /// 推論するテキスト（日本語）
    #[arg(long)]
    predict: Option<String>,

    /// バックエンドの選択（auto, wgpu, ndarray）
    #[arg(long, default_value = "wgpu")]
    backend: String,

    /// Attention行列をCSVエクスポート（推論時のみ）
    #[arg(long)]
    export_attn: bool,

    /// ビーム探索のビーム幅（1=貪欲探索、5推奨）
    #[arg(long, default_value = "5")]
    beam_width: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let args = Args::parse();

    println!("\n===== 日英翻訳モデル =====");

    // データセットから語彙を構築
    println!("語彙を構築中...");
    let src_vocab = SourceVocabulary::new();

    // 英語語彙はデータセットから構築
    let data_content = fs::read_to_string("data/translation_data_ja_en.txt")
        .expect("データファイルが読み込めません");

    let english_sentences: Vec<&str> = data_content
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .filter_map(|line| line.split('\t').nth(1))
        .collect();

    let tgt_vocab = TargetVocabulary::from_dataset(&english_sentences);

    println!("日本語語彙サイズ: {}", src_vocab.vocab_size);
    println!("英語語彙サイズ: {}", tgt_vocab.vocab_size);

    let training_device = WgpuDevice::default();

    // モデルの初期化または読み込み
    let mut model = if let Some(load_dir) = &args.load {
        println!("モデルを読み込み中: {}", load_dir.display());
        translator_ja_en::translation_inference::load_model_generic::<TrainingBackend>(
            load_dir,
            &training_device,
            src_vocab.vocab_size,
            tgt_vocab.vocab_size,
        )?
    } else {
        Seq2SeqModel::<TrainingBackend>::new(
            &training_device,
            src_vocab.vocab_size,
            tgt_vocab.vocab_size,
        )
    };

    // 訓練モード
    if args.train {
        println!("\n===== 訓練開始 =====");
        let translation_data = TranslationData::load(
            &src_vocab,
            &tgt_vocab,
            "data/translation_data_ja_en.txt",
            config::SRC_SEQ_LEN,
        );
        println!("訓練サンプル数: {}サンプル", translation_data.len());

        let (trained_model, training_metrics) =
            train_translation(model, &translation_data, &tgt_vocab, &training_device);
        model = trained_model;
        println!("訓練完了！");

        // モデルとメトリクスを保存
        if let Some(save_dir) = &args.save {
            save_model(&model, save_dir)?;
            save_metrics(
                save_dir,
                &training_metrics,
                src_vocab.vocab_size,
                tgt_vocab.vocab_size,
            )?;
        }
    }

    // 推論モード
    if let Some(predict_text) = &args.predict {
        println!("\n===== 推論テスト =====");
        println!("入力: {}", predict_text);
        println!("ビーム幅: {}", args.beam_width);

        if args.load.is_some() {
            // モデルが読み込まれている場合は指定されたバックエンドで推論
            let predicted_translation = run_translation_inference(
                &args.backend,
                args.load.as_ref().unwrap(),
                predict_text,
                &src_vocab,
                &tgt_vocab,
                args.beam_width,
            )?;
            println!("翻訳: {}", predicted_translation);
        } else {
            // 訓練直後の場合はTrainingBackendで推論
            let predicted_translation = predict_translation(
                &model,
                &src_vocab,
                &tgt_vocab,
                predict_text,
                &training_device,
                args.beam_width,
            );
            println!("翻訳: {}", predicted_translation);
        }
    }

    // デモモード（引数なし）
    if !args.train && args.predict.is_none() && args.load.is_none() {
        println!("\n===== デモモード =====");
        println!("使用方法:");
        println!("  訓練: cargo run --release -- --train --save models/test");
        println!("  推論（ビーム探索）: cargo run --release -- --load models/test --predict \"こんにちは\" --beam-width 5");
        println!("  推論（貪欲探索）: cargo run --release -- --load models/test --predict \"こんにちは\" --beam-width 1");
        println!("  継続訓練: cargo run --release -- --load models/test --train --save models/test2");
    }

    let duration = start_time.elapsed();
    println!("\n実行時間: {:.2}秒", duration.as_secs_f64());

    Ok(())
}
