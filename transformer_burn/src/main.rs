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
    model: model::Seq2SeqModel<TrainingBackend>,
    training_data: &jsl_data::JslTrainingData,
    vocab: &jsl_vocabulary::JslVocabulary,
    device: &<TrainingBackend as Backend>::Device,
) -> model::Seq2SeqModel<TrainingBackend> {
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
        for (batch_inputs, batch_targets) in training_data.batches(BATCH_SIZE, vocab.pad_id) {
            let batch_size = batch_inputs.len();
            let target_len = batch_targets[0].len();

            // 入力: Vec<Vec<i32>> を flatten して Tensor に変換
            let flattened_inputs: Vec<i32> = batch_inputs.iter().flatten().copied().collect();
            let src_tokens =
                Tensor::<TrainingBackend, 1, Int>::from_data(flattened_inputs.as_slice(), device)
                    .reshape([batch_size, SEQ_LEN]);

            // ターゲット: Vec<Vec<i32>> を flatten して Tensor に変換
            let flattened_targets: Vec<i32> = batch_targets.iter().flatten().copied().collect();
            let full_target_tensor =
                Tensor::<TrainingBackend, 1, Int>::from_data(flattened_targets.as_slice(), device)
                    .reshape([batch_size, target_len]);

            // Teacher Forcing: デコーダー入力は [SOS, tag1, ..., tagN]（EOSを除く）
            let tgt_input = full_target_tensor
                .clone()
                .slice([0..batch_size, 0..target_len - 1]);

            // ターゲット出力は [tag1, ..., tagN, EOS]（SOSを除く）
            let tgt_output = full_target_tensor
                .clone()
                .slice([0..batch_size, 1..target_len]);

            // フォワードパス
            let logits = model.forward(src_tokens, tgt_input, None, None);

            // logits: [batch_size, target_len-1, vocab_size]
            // tgt_output: [batch_size, target_len-1]

            // 各位置で損失を計算
            let mut total_position_loss = Tensor::<TrainingBackend, 1>::from_data([0.0], device);

            for pos in 0..target_len - 1 {
                // 各位置のlogits: [batch_size, vocab_size]
                let logits_at_pos = logits
                    .clone()
                    .slice([0..batch_size, pos..pos + 1, 0..config::VOCAB_SIZE])
                    .reshape([batch_size, config::VOCAB_SIZE]);

                // 各位置のターゲット: [batch_size]
                let targets_at_pos = tgt_output
                    .clone()
                    .slice([0..batch_size, pos..pos + 1])
                    .reshape([batch_size]);

                // CrossEntropyLoss（PADトークンは無視）
                let loss_at_pos =
                    burn::nn::loss::CrossEntropyLoss::new(Some(config::PAD_TOKEN), device)
                        .forward(logits_at_pos, targets_at_pos);

                total_position_loss = total_position_loss + loss_at_pos;
            }

            let loss = total_position_loss;

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
    model: &model::Seq2SeqModel<TrainingBackend>,
    vocab: &jsl_vocabulary::JslVocabulary,
    input_text: &str,
    device: &<TrainingBackend as Backend>::Device,
) -> String {
    use config::SEQ_LEN;

    // 入力テキストをトークン化
    let tokens = vocab.encode(input_text);
    let tokens = vocab.pad_sequence(&tokens, SEQ_LEN);

    // Tensorに変換 [1, SEQ_LEN]
    let src_tokens = Tensor::<TrainingBackend, 1, Int>::from_data(tokens.as_slice(), device)
        .reshape([1, SEQ_LEN]);

    // 自己回帰生成（最大10トークンまで）
    let generated_ids = model.generate(src_tokens, None, vocab.sos_id, vocab.eos_id, 10);

    // 生成されたトークンIDを取得 [1, generated_len]
    let generated_data: Vec<i32> = generated_ids.to_data().to_vec().unwrap();

    // タグのみを抽出してデコード
    let mut predicted_tags = Vec::new();
    for &id in &generated_data {
        let id_usize = id as usize;

        // SOS、EOS、PADをスキップ
        if id_usize == vocab.sos_id || id_usize == vocab.eos_id || id_usize == vocab.pad_id {
            continue;
        }

        // タグのみ抽出
        if id_usize >= vocab.tag_start_id && id_usize < vocab.vocab_size {
            predicted_tags.push(format!("<{}>", vocab.id_to_token[id_usize]));
        }
    }

    // タグをスペース区切りで結合
    predicted_tags.join(" ")
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

    // Seq2Seqモデルを初期化
    let jsl_model = model::Seq2SeqModel::<TrainingBackend>::new(&training_device);

    // 訓練実行
    let trained_jsl_model = train_jsl(jsl_model, &jsl_data, &jsl_vocab, &training_device);

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
