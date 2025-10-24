use crate::config::{BATCH_SIZE, EPOCHS, LEARNING_RATE, SRC_SEQ_LEN};
use crate::metrics::TrainingMetrics;
use crate::translation_data::TranslationData;
use crate::translation_model::Seq2SeqModel;
use crate::translation_vocabulary::TargetVocabulary;
use burn::backend::wgpu::Wgpu;
use burn::backend::Autodiff;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::Int;

pub type TrainingBackend = Autodiff<Wgpu>;

/// 日英翻訳の訓練実行
pub fn train_translation(
    model: Seq2SeqModel<TrainingBackend>,
    training_data: &TranslationData,
    tgt_vocab: &TargetVocabulary,
    device: &<TrainingBackend as Backend>::Device,
) -> (Seq2SeqModel<TrainingBackend>, TrainingMetrics) {
    let mut optimizer = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .init();

    let mut model = model;
    let mut loss_history = Vec::new();

    println!("訓練開始: {}エポック", EPOCHS);

    for epoch in 0..EPOCHS {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for (batch_inputs, batch_targets) in training_data.batches(BATCH_SIZE, tgt_vocab.pad_id) {
            let batch_size = batch_inputs.len();
            let target_len = batch_targets[0].len();

            // 入力テンソル作成（日本語）
            let flattened_inputs: Vec<i32> = batch_inputs.iter().flatten().copied().collect();
            let src_tokens =
                Tensor::<TrainingBackend, 1, Int>::from_data(flattened_inputs.as_slice(), device)
                    .reshape([batch_size, SRC_SEQ_LEN]);

            // ターゲットテンソル作成（英語）
            let flattened_targets: Vec<i32> = batch_targets.iter().flatten().copied().collect();
            let full_target_tensor =
                Tensor::<TrainingBackend, 1, Int>::from_data(flattened_targets.as_slice(), device)
                    .reshape([batch_size, target_len]);

            // Teacher Forcing: デコーダー入力は [SOS, word1, ..., wordN]
            let tgt_input = full_target_tensor
                .clone()
                .slice([0..batch_size, 0..target_len - 1]);

            // ターゲット出力は [word1, ..., wordN, EOS]
            let tgt_output = full_target_tensor
                .clone()
                .slice([0..batch_size, 1..target_len]);

            // フォワードパス
            let logits = model.forward(src_tokens, tgt_input, None, None);

            // 損失計算
            let loss = compute_loss(
                &logits,
                &tgt_output,
                batch_size,
                target_len,
                tgt_vocab.vocab_size,
                tgt_vocab.pad_id,
                device,
            );

            // バックプロパゲーション
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);

            // パラメータ更新
            model = optimizer.step(LEARNING_RATE, model, grads);

            total_loss += loss.into_scalar();
            batch_count += 1;
        }

        let avg_loss = total_loss / batch_count as f32;
        loss_history.push(avg_loss);

        // 進捗表示（10エポックごと、または最後）
        if epoch % 10 == 0 || epoch == EPOCHS - 1 {
            println!("Epoch {}/{}: Loss = {:.6}", epoch + 1, EPOCHS, avg_loss);
        }
    }

    let metrics = TrainingMetrics {
        loss_history: loss_history.clone(),
        final_loss: *loss_history.last().unwrap_or(&0.0),
        epochs: EPOCHS,
        learning_rate: LEARNING_RATE,
        batch_size: BATCH_SIZE,
    };

    (model, metrics)
}

/// 損失計算
fn compute_loss(
    logits: &Tensor<TrainingBackend, 3>,
    tgt_output: &Tensor<TrainingBackend, 2, Int>,
    batch_size: usize,
    target_len: usize,
    tgt_vocab_size: usize,
    pad_id: usize,
    device: &<TrainingBackend as Backend>::Device,
) -> Tensor<TrainingBackend, 1> {
    let mut total_position_loss = Tensor::<TrainingBackend, 1>::from_data([0.0], device);

    for pos in 0..target_len - 1 {
        let logits_at_pos = logits
            .clone()
            .slice([0..batch_size, pos..pos + 1, 0..tgt_vocab_size])
            .reshape([batch_size, tgt_vocab_size]);

        let targets_at_pos = tgt_output
            .clone()
            .slice([0..batch_size, pos..pos + 1])
            .reshape([batch_size]);

        let loss_at_pos = burn::nn::loss::CrossEntropyLoss::new(Some(pad_id), device)
            .forward(logits_at_pos, targets_at_pos);

        total_position_loss = total_position_loss + loss_at_pos;
    }

    total_position_loss
}
