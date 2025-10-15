mod attention;
mod matrix;

use attention::{
    create_embedding_matrix_public, create_output_weight, create_training_data, create_vocabulary,
    embed_tokens_with_matrix, plot_loss_curve, predict_next_token,
};
use proconio::input;
use std::time::Instant;

use crate::attention::{
    compute_multilayer_transformer, create_feedforward_w1_multilayer,
    create_feedforward_w2_multilayer, create_layer_norm_params_multilayer,
    create_multihead_output_projection_multilayer, create_multihead_qkv_weights_multilayer,
    generate_text_multilayer, pad_sequence, train_multilayer_transformer,
};

fn main() {
    // 文字列入力を受け取る
    input! {
        seed_text: String,
    }

    let start_time = Instant::now();

    // === Phase 9: 文字レベル自然言語処理 ===
    println!("=== Phase 9: 文字レベル自然言語処理  ===");
    println!("語彙サイズ: 51文字（ひらがな46 +  記号5）");
    println!("訓練データ:  日本語の簡単な挨拶・フレーズ");

    // Vocabularyを作成
    let vocab = create_vocabulary();
    println!(
        "\n語彙の一部: あ={}, い={}, う={}",
        vocab.char_to_id[&'あ'], vocab.char_to_id[&'い'], vocab.char_to_id[&'う']
    );

    // 入力文字列をトークンIDに変換
    let tokens = vocab.encode(&seed_text);
    let tokens = pad_sequence(&tokens);
    println!("\n入力文字列: \"{}\"", seed_text);
    println!("トークンID: {:?}", tokens);

    // 重み行列を初期化（複数層）
    let embedding_matrix = create_embedding_matrix_public();
    let (w_q_heads_layers, w_k_heads_layers, w_v_heads_layers) =
        create_multihead_qkv_weights_multilayer();
    let w_o_layers = create_multihead_output_projection_multilayer();
    let w1_layers = create_feedforward_w1_multilayer();
    let w2_layers = create_feedforward_w2_multilayer();
    let w_out = create_output_weight();
    let (gamma1_layers, beta1_layers) = create_layer_norm_params_multilayer();
    let (gamma2_layers, beta2_layers) = create_layer_norm_params_multilayer();

    let embedded = embed_tokens_with_matrix(&tokens, &embedding_matrix);
    let (final_output, _layer_outputs) = compute_multilayer_transformer(
        &embedded,
        &tokens,
        &w_q_heads_layers,
        &w_k_heads_layers,
        &w_v_heads_layers,
        &w_o_layers,
        &w1_layers,
        &w2_layers,
        &gamma1_layers,
        &beta1_layers,
        &gamma2_layers,
        &beta2_layers,
    );
    let predictions = predict_next_token(&final_output, &w_out);

    println!("\n【学習前の予測】");
    let char_predictions = vocab.decode_predictions(&predictions);
    println!("次文字予測（確率上位5つ）:");
    let mut sorted_preds = char_predictions.clone();
    sorted_preds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (c, prob) in sorted_preds.iter().take(5) {
        println!("  '{}': {:.1}%", c, prob * 100.0);
    }

    // 訓練の実行
    println!("\n=== Transformer学習開始 ===");
    let training_data = create_training_data();
    println!("訓練データ数: {}", training_data.len());

    let learning_rate = 0.0005;
    let epochs = 1000;

    println!("\n訓練中...");

    let (
        emb_trained,
        wq_heads_layers_trained,
        wk_heads_layers_trained,
        wv_heads_layers_trained,
        wo_layers_trained,
        w1_layers_trained,
        w2_layers_trained,
        wout_trained,
        gamma1_layers_trained,
        beta1_layers_trained,
        gamma2_layers_trained,
        beta2_layers_trained,
        loss_history,
    ) = train_multilayer_transformer(
        &training_data,
        embedding_matrix,
        w_q_heads_layers,
        w_k_heads_layers,
        w_v_heads_layers,
        w_o_layers,
        w1_layers,
        w2_layers,
        w_out,
        gamma1_layers,
        beta1_layers,
        gamma2_layers,
        beta2_layers,
        learning_rate,
        epochs,
    );

    println!("\n【学習後の予測】");
    let embedded_after = embed_tokens_with_matrix(&tokens, &emb_trained);
    let (final_output_after, _layer_outputs_after) = compute_multilayer_transformer(
        &embedded_after,
        &tokens,
        &wq_heads_layers_trained,
        &wk_heads_layers_trained,
        &wv_heads_layers_trained,
        &wo_layers_trained,
        &w1_layers_trained,
        &w2_layers_trained,
        &gamma1_layers_trained,
        &beta1_layers_trained,
        &gamma2_layers_trained,
        &beta2_layers_trained,
    );
    let predictions_after = predict_next_token(&final_output_after, &wout_trained);

    let char_predictions_after = vocab.decode_predictions(&predictions_after);
    println!("次文字予測（確率上位5つ）:");
    let mut sorted_preds = char_predictions_after.clone();
    sorted_preds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (c, prob) in sorted_preds.iter().take(5) {
        println!("  '{}': {:.1}%", c, prob * 100.0);
    }

    // テキスト生成デモ
    println!("\n【テキスト生成デモ】");
    let generated = generate_text_multilayer(
        &seed_text,
        20,
        &vocab,
        &emb_trained,
        &wq_heads_layers_trained,
        &wk_heads_layers_trained,
        &wv_heads_layers_trained,
        &wo_layers_trained,
        &w1_layers_trained,
        &w2_layers_trained,
        &gamma1_layers_trained,
        &beta1_layers_trained,
        &gamma2_layers_trained,
        &beta2_layers_trained,
        &wout_trained,
    );
    println!("種: \"{}\"", seed_text);
    println!("生成: \"{}\"", generated);

    let duration = start_time.elapsed();
    println!("\n=== 学習結果 ===");
    println!("訓練時間: {:.2}秒", duration.as_secs_f64());
    println!("初期損失: {:.4}", loss_history[0]);
    println!("最終損失: {:.4}", loss_history[loss_history.len() - 1]);

    // 学習曲線の表示
    plot_loss_curve(&loss_history);
}
