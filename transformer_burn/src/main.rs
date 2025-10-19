#![recursion_limit = "256"]
mod config;
mod jsl_data;
mod jsl_vocabulary;
mod model;

use crate::config::{BATCH_SIZE, SEQ_LEN};
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::Int;
use std::time::Instant;

// 訓練用バックエンド（GPU, 高速訓練）
type TrainingBackend = Autodiff<Wgpu>;

fn train_jsl(
    model: model::TransformerModel<TrainingBackend>,
    training_data: &jsl_data::JslTrainingData,
    device: &<TrainingBackend as Backend>::Device,
) -> model::TransformerModel<TrainingBackend> {
    use config::{EPOCHS, LEARNING_RATE};

    // Adamオプティマイザーの初期化
    let mut optimizer = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .init();

    let mut model = model;

    println!("訓練開始: {}エポック", EPOCHS);

    for epoch in 0..EPOCHS {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        // バッチごとに処理
        for (batch_inputs, batch_targets) in training_data.batches(BATCH_SIZE) {
            let batch_size = batch_inputs.len();

            // Vec<Vec<i32>> を flatten して Tensor に変換
            let flattened: Vec<i32> = batch_inputs.iter().flatten().copied().collect();
            let input_tensor =
                Tensor::<TrainingBackend, 1, Int>::from_data(flattened.as_slice(), device)
                    .reshape([batch_size, SEQ_LEN]);

            let target_tensor =
                Tensor::<TrainingBackend, 1, Int>::from_data(batch_targets.as_slice(), device);

            // フォワードパス
            let logits = model.forward(input_tensor);

            // logits: [batch_size, seq_len, vocab_size]
            // 最後のトークン位置を取得:[batch_size, vocab_size]
            let logits_last = logits
                .slice([0..batch_size, SEQ_LEN - 1..SEQ_LEN, 0..config::VOCAB_SIZE])
                .reshape([batch_size, config::VOCAB_SIZE]);

            // CrossEntropyLoss
            let loss = burn::nn::loss::CrossEntropyLoss::new(Some(config::PAD_TOKEN), device)
                .forward(logits_last, target_tensor);

            // バックプロパゲーション
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);

            // パラメータ更新
            model = optimizer.step(LEARNING_RATE, model, grads);

            total_loss += loss.into_scalar();
            batch_count += 1;
        }

        // エポックごとの損失を表示
        if epoch % 10 == 0 || epoch == EPOCHS - 1 {
            let avg_loss = total_loss / batch_count as f32;
            println!("Epoch {}/{}: Loss = {:.6}", epoch + 1, EPOCHS, avg_loss);
        }
    }

    model
}

fn predict_tags(
    model: &model::TransformerModel<TrainingBackend>,
    vocab: &jsl_vocabulary::JslVocabulary,
    input_text: &str,
    device: &<TrainingBackend as Backend>::Device,
) -> String {
    use config::SEQ_LEN;

    // 入力テキストをトークン化
    let tokens = vocab.encode(input_text);
    let tokens = vocab.pad_sequence(&tokens, SEQ_LEN);

    // Tensorに変換
    let input_tensor = Tensor::<TrainingBackend, 1, Int>::from_data(tokens.as_slice(), device)
        .reshape([1, SEQ_LEN]);

    // 予測
    let output = model.forward(input_tensor);
    let last_logits = output
        .slice([0..1, SEQ_LEN - 1..SEQ_LEN, 0..config::VOCAB_SIZE])
        .reshape([config::VOCAB_SIZE]);

    // 最も確率の高いトークンを取得
    let predicted_id = last_logits.argmax(0).into_scalar() as usize;

    // タグとして解釈（タグIDのみをデコード）
    if predicted_id >= vocab.tag_start_id {
        format!("<{}>", vocab.id_to_token[predicted_id])
    } else {
        vocab.id_to_token[predicted_id].clone()
    }
}


fn main() {
    let start_time = Instant::now();

    // ===== JSL（日本語→手話タグ翻訳）の動作確認 =====
    println!("===== JSL語彙とデータローダーの動作確認 =====");

    let jsl_vocab = jsl_vocabulary::JslVocabulary::new();
    println!("JSL語彙サイズ: {}", jsl_vocab.vocab_size);
    println!("タグ開始ID: {}", jsl_vocab.tag_start_id);

    // エンコード・デコードのテスト（複数ケース）
    let test_cases = vec![
        "わたしはたべます<私><食べる>",
        "ありがとう<ありがとう>",
        "わたしはあしたがっこうにいきます<明日><私><学校><行く>",
        "はい<はい>",
    ];

    println!("\n===== エンコード・デコードテスト =====");
    for test_combined in test_cases {
        let encoded = jsl_vocab.encode(test_combined);
        let decoded = jsl_vocab.decode(&encoded);
        let tags_only = jsl_vocab.decode_tags_only(&encoded);

        println!("\n入力: {}", test_combined);
        println!("エンコード: {:?}", encoded);
        println!("デコード: {}", decoded);
        println!("タグのみ: {}", tags_only);
    }

    // JSL訓練データのロード
    let jsl_data = jsl_data::JslTrainingData::load(&jsl_vocab, "data/training_data_jsl.txt");
    println!("JSL訓練サンプル数: {}サンプル", jsl_data.len());

    // ===== JSL訓練フェーズ =====
    println!("\n===== JSL訓練開始 =====");

    // 訓練用デバイス（GPU、Autodiff対応）
    let training_device = WgpuDevice::default();

    // 訓練用モデルを初期化
    let jsl_model = model::TransformerModel::<TrainingBackend>::new(&training_device);

    // 訓練実行
    let trained_jsl_model = train_jsl(jsl_model, &jsl_data, &training_device);

    println!("JSL訓練完了！");

    // ===== JSL推論テスト =====
    println!("\n===== JSL推論テスト =====");
    let test_sentences = vec![
        "わたしはたべます",
        "ありがとう",
        "おはようございます",
    ];

    for test_text in test_sentences {
        let predicted_tags = predict_tags(&trained_jsl_model, &jsl_vocab, test_text, &training_device);
        println!("入力: {} → 予測タグ: {}", test_text, predicted_tags);
    }

    let duration = start_time.elapsed();
    println!("実行時間: {:?}s", duration.as_secs_f64());
}
