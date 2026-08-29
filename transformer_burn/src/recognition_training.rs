//! 認識モデル(手ポーズ列→タグ)の学習ループと評価

use crate::config::BATCH_SIZE;
use crate::pose_data::PoseTrainingData;
use crate::recognition::RecognitionModel;
use crate::tag_vocabulary::TagVocabulary;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Int;

/// 認識モデルの学習を実行する。
/// training.rs の train_jsl と同じ構成(Adam + Teacher Forcing)で、
/// 入力がトークンID列ではなく手ポーズ列のテンソルになっている点だけが違う
pub fn train_recognition<B: AutodiffBackend>(
    model: RecognitionModel<B>,
    data: &PoseTrainingData,
    vocab: &TagVocabulary,
    device: &B::Device,
    epochs: usize,
    learning_rate: f64,
) -> (RecognitionModel<B>, Vec<f32>) {
    let mut optimizer = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .init();

    let mut model = model;
    let mut loss_history = Vec::new();

    println!(
        "認識モデル訓練開始: {}エポック, {}サンプル, {}タグ, lr={}",
        epochs,
        data.len(),
        vocab.tags.len(),
        learning_rate
    );

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for (features, targets, batch_size) in data.batches(BATCH_SIZE, vocab) {
            // 手ポーズ列テンソル [batch, frames, feature_dim]
            let pose = Tensor::<B, 1>::from_floats(features.as_slice(), device)
                .reshape([batch_size, data.frames, data.feature_dim]);

            // ターゲット [batch, 3] = [SOS, タグID, EOS]
            let target_len = targets[0].len();
            let flat_targets: Vec<i32> = targets.iter().flatten().copied().collect();
            let full_target = Tensor::<B, 1, Int>::from_data(flat_targets.as_slice(), device)
                .reshape([batch_size, target_len]);

            // Teacher Forcing: Decoder入力 [SOS, タグ] / 正解出力 [タグ, EOS]
            let tgt_input = full_target
                .clone()
                .slice([0..batch_size, 0..target_len - 1]);
            let tgt_output = full_target.clone().slice([0..batch_size, 1..target_len]);

            let logits = model.forward(pose, tgt_input);

            let loss = compute_loss(
                &logits,
                &tgt_output,
                batch_size,
                target_len,
                vocab.vocab_size(),
                vocab.pad_id(),
                device,
            );

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(learning_rate, model, grads);

            total_loss += loss.into_scalar().elem::<f32>();
            batch_count += 1;
        }

        let avg_loss = total_loss / batch_count as f32;
        loss_history.push(avg_loss);

        if epoch % 20 == 0 || epoch == epochs - 1 {
            println!("Epoch {}/{}: Loss = {:.6}", epoch + 1, epochs, avg_loss);
        }
    }

    (model, loss_history)
}

/// 位置ごとの CrossEntropy 損失の合計(training.rs の compute_loss の動的語彙版)
fn compute_loss<B: AutodiffBackend>(
    logits: &Tensor<B, 3>,
    tgt_output: &Tensor<B, 2, Int>,
    batch_size: usize,
    target_len: usize,
    vocab_size: usize,
    pad_id: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    let mut total_loss = Tensor::<B, 1>::from_data([0.0], device);

    for pos in 0..target_len - 1 {
        let logits_at_pos = logits
            .clone()
            .slice([0..batch_size, pos..pos + 1, 0..vocab_size])
            .reshape([batch_size, vocab_size]);

        let targets_at_pos = tgt_output
            .clone()
            .slice([0..batch_size, pos..pos + 1])
            .reshape([batch_size]);

        let loss_at_pos = burn::nn::loss::CrossEntropyLoss::new(Some(pad_id), device)
            .forward(logits_at_pos, targets_at_pos);

        total_loss = total_loss + loss_at_pos;
    }

    total_loss
}

/// データ全件に対して top-1 / top-5 を測って表示する。
/// 注意: 学習に使ったデータでの評価(in-sample)は「学習ループが正しく
/// 回っているか」の確認にしかならない。本当の精度は未見のテイクで測ること
pub fn evaluate_topk<B: Backend>(
    model: &RecognitionModel<B>,
    data: &PoseTrainingData,
    vocab: &TagVocabulary,
    device: &B::Device,
) {
    let k = 5;
    let mut top1_hits = 0;
    let mut topk_hits = 0;

    println!("\n--- 認識結果 (top-{}) ---", k);
    for sample in &data.samples {
        let preds = model.predict_topk(&sample.features, data.frames, vocab, k, device);

        let top1 = preds.first().map(|(t, _)| t.as_str()).unwrap_or("?");
        let in_topk = preds.iter().any(|(t, _)| t == &sample.label);
        if top1 == sample.label {
            top1_hits += 1;
        }
        if in_topk {
            topk_hits += 1;
        }

        let pred_str: Vec<String> = preds
            .iter()
            .map(|(t, p)| format!("{}({:.2})", t, p))
            .collect();
        println!(
            "  正解: {:<10} 予測: {} {}",
            sample.label,
            pred_str.join(" "),
            if in_topk { "" } else { "← top-5 外し" }
        );
    }

    let n = data.len();
    println!(
        "top-1: {}/{} ({:.0}%)  top-{}: {}/{} ({:.0}%)",
        top1_hits,
        n,
        top1_hits as f32 / n as f32 * 100.0,
        k,
        topk_hits,
        n,
        topk_hits as f32 / n as f32 * 100.0
    );
}
