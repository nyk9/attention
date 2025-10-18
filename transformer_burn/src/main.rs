#![recursion_limit = "256"]
mod config;
mod data;
mod model;
mod vocabulary;

use crate::config::{BATCH_SIZE, SEQ_LEN};
use burn::backend::Autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::optim::GradientsParams;
use burn::optim::{AdamConfig, Optimizer};
use burn::prelude::*;
use burn::tensor::Int;
use data::TrainingData;
use std::time::Instant;

// 推論用バックエンド（GPU）
type InferenceBackend = Wgpu;
// 訓練用バックエンド（GPU, 高速訓練）
type TrainingBackend = Autodiff<Wgpu>;

fn generate_text(
    model: &model::TransformerModel<TrainingBackend>,
    vocab: &vocabulary::Vocabulary,
    seed_text: &str,
    num_chars: usize,
    device: &<TrainingBackend as Backend>::Device,
) -> String {
    use config::SEQ_LEN;

    let mut current_text = seed_text.to_string();

    for _ in 0..num_chars {
        // 現在のテキストをトークン化
        let tokens = vocab.encode(&current_text);
        let tokens = vocab.pad_sequence(&tokens);

        // Tensorに変換
        let input_tensor = Tensor::<TrainingBackend, 1, Int>::from_data(tokens.as_slice(), device)
            .reshape([1, SEQ_LEN]);

        // 予測
        let output = model.forward(input_tensor);
        let last_logits = output
            .slice([0..1, SEQ_LEN - 1..SEQ_LEN, 0..config::VOCAB_SIZE])
            .reshape([config::VOCAB_SIZE]);

        // 最も確率の高いトークンを取得
        let predicted_id = last_logits.argmax(0).into_scalar();
        let predicted_char = vocab.id_to_char[predicted_id as usize];

        // 生成した文字を追加
        current_text.push(predicted_char);
    }

    current_text
}

fn train(
    model: model::TransformerModel<TrainingBackend>,
    training_data: &TrainingData,
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
        if epoch % 100 == 0 || epoch == EPOCHS - 1 {
            let avg_loss = total_loss / batch_count as f32;
            println!("Epoch {}/{}: Loss = {:.6}", epoch + 1, EPOCHS, avg_loss);
        }
    }

    model
}

fn main() {
    let start_time = Instant::now();

    // デバイス初期化
    // let device: WgpuDevice = Default::default();

    // Vocabulary作成
    let vocab = vocabulary::Vocabulary::new();

    // テスト文字列
    let test_text = "こんにちは";
    let tokens = vocab.encode(test_text);
    let tokens = vocab.pad_sequence(&tokens);

    println!("入力: {}", test_text);
    println!("トークン: {:?}", tokens);

    let training_data = TrainingData::load(&vocab, "data/training_data.txt");
    println!("訓練データ準備完了: {}サンプル", training_data.len());

    // ===== 訓練フェーズ =====
    println!("\n===== 訓練開始 =====");

    // 訓練用デバイス（CPU、Autodiff対応）
    let training_device = WgpuDevice::default();

    // 訓練用モデルを初期化
    let training_model = model::TransformerModel::<TrainingBackend>::new(&training_device);

    // 訓練実行
    let trained_model = train(training_model, &training_data, &training_device);

    println!("訓練完了！");

    // ===== 推論テスト =====
    println!("\n===== 推論テスト =====");
    println!("訓練済みモデルで「こんにちは」の次の文字を予測:");

    // テスト用のTensorを訓練バックエンドで作成
    let test_tokens_tensor =
        Tensor::<TrainingBackend, 1, Int>::from_data(tokens.as_slice(), &training_device)
            .reshape([1, SEQ_LEN]);

    // テスト推論（訓練済みモデルで）
    let test_output = trained_model.forward(test_tokens_tensor);
    let last_position_logits = test_output
        .slice([0..1, SEQ_LEN - 1..SEQ_LEN, 0..config::VOCAB_SIZE])
        .reshape([config::VOCAB_SIZE]);

    // 最も確率の高いトークンを取得
    let predicted_token_id = last_position_logits.argmax(0).into_scalar();
    let predicted_char = vocab.id_to_char[predicted_token_id as usize];

    println!("入力: {}", test_text);
    println!("予測された次の文字: {}", predicted_char);
    println!("予測トークンID: {}", predicted_token_id);

    // ===== テキスト生成デモ =====
    println!("\n===== テキスト生成デモ =====");
    let generated = generate_text(&trained_model, &vocab, test_text, 20, &training_device);
    println!("種: \"{}\"", test_text);
    println!("生成: \"{}\"", generated);

    let duration = start_time.elapsed();
    println!("実行時間: {:?}s", duration.as_secs_f64());
}
