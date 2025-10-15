use std::collections::HashMap;

use std::fs;

use crate::matrix::{relu, relu_gradient, soft_max, Matrix};

pub struct Vocabulary {
    pub char_to_id: HashMap<char, usize>,
    pub id_to_char: Vec<char>,
    pub vocab_size: usize,
}

impl Vocabulary {
    pub fn new() -> Self {
        let mut chars = Vec::new();

        // ひらがな（あ〜ん）
        let hiragana =
 "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん";
        for c in hiragana.chars() {
            chars.push(c);
        }
        // 濁音
        let dakuon = "がぎぐげござじずぜぞだぢづでどばびぶべぼ";
        for c in dakuon.chars() {
            chars.push(c);
        }

        // 半濁音
        let handakuon = "ぱぴぷぺぽ";
        for c in handakuon.chars() {
            chars.push(c);
        }

        // 小文字（拗音用）
        let kogaki = "ぁぃぅぇぉゃゅょっ";
        for c in kogaki.chars() {
            chars.push(c);
        }

        // 記号
        chars.push('。');
        chars.push('、');
        chars.push('！');
        chars.push('？');
        chars.push(' ');
        chars.push('ー');
        chars.push('['); // [PAD] = 0
        chars.push(']'); // [UNK] = 1

        // char_to_id の構築
        let mut char_to_id = HashMap::new();
        for (id, &c) in chars.iter().enumerate() {
            char_to_id.insert(c, id);
        }

        let vocab_size = chars.len();

        Vocabulary {
            char_to_id,
            id_to_char: chars,
            vocab_size,
        }
    }

    // 文字列をトークンIDのベクトルに変換
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let mut token_ids = Vec::new();

        for c in text.chars() {
            if let Some(&id) = self.char_to_id.get(&c) {
                token_ids.push(id as i32);
            } else {
                // 未知の文字は [UNK] (id=1) に変換
                token_ids.push(1);
            }
        }

        token_ids
    }

    // トークンIDのベクトルを文字列に変換
    pub fn decode(&self, token_ids: &Vec<i32>) -> String {
        let mut text = String::new();

        for &id in token_ids {
            if id >= 0 && (id as usize) < self.vocab_size {
                text.push(self.id_to_char[id as usize]);
            } else {
                // 範囲外のIDは無視
                text.push('?');
            }
        }

        text
    }

    // 次文字予測の確率分布を文字と確率のペアに変換
    pub fn decode_predictions(&self, predictions: &Vec<(i32, f64)>) -> Vec<(char, f64)> {
        predictions
            .iter()
            .map(|(id, prob)| {
                let c = if *id >= 0 && (*id as usize) < self.vocab_size {
                    self.id_to_char[*id as usize]
                } else {
                    '?'
                };
                (c, *prob)
            })
            .collect()
    }
}

const D_MODEL: usize = 16; // 埋め込み次元
const VOCAB_SIZE: usize = 100; // 語彙サイズ（特殊トークン2 + ひらがな46 + 濁音20 + 半濁音5 + 小文字9 + 記号5 = 87、余裕を持たせて101）
const NUM_HEADS: usize = 2; // Multi-Headの数
const D_HEAD: usize = D_MODEL / NUM_HEADS; // 各ヘッドの埋め込み次元 ==4
const D_FF: usize = 32; // Feed-Forward中間層の次元（4 × D_MODEL）
const SEQ_LEN: usize = 10; // シーケンス長（入力として使う文字数）
const NUM_LAYERS: usize = 4; //Transformerブロックの層数
const PAD_TOKEN: i32 = 99; // パディング用の特殊トークンID

fn create_embedding_matrix() -> Matrix {
    let mut data = Vec::new();
    for i in 0..VOCAB_SIZE {
        let mut row = Vec::new();
        for j in 0..D_MODEL {
            let val = ((i + j) as f64 * 0.1) % 1.0;
            row.push(val);
        }
        data.push(row);
    }
    Matrix::from_vec(data)
}

// 埋め込み行列を引数として受け取るバージョン（学習用）
// Phase 7aでは埋め込み行列も学習対象なので、外部から渡す必要がある
pub fn embed_tokens_with_matrix(tokens: &Vec<i32>, embedding_matrix: &Matrix) -> Matrix {
    let mut embedded_data = Vec::new();
    for (pos, &token) in tokens.iter().enumerate() {
        let mut row = embedding_matrix.get_row(token as usize);

        // Position Encodingを追加
        for i in 0..D_MODEL {
            let pos_encoding = if i % 2 == 0 {
                (pos as f64 / 10000_f64.powf(i as f64 / D_MODEL as f64)).sin()
            } else {
                (pos as f64 / 10000_f64.powf((i - 1) as f64 / D_MODEL as f64)).cos()
            };
            row[i] += pos_encoding;
        }
        embedded_data.push(row);
    }
    Matrix::from_vec(embedded_data)
}

// 重み行列を外部から受け取ってQKV変換を適用（学習用）
// 既存のcreate_qkvは内部で重みを生成するが、学習時は外部から渡す必要がある
pub fn create_embedding_matrix_public() -> Matrix {
    create_embedding_matrix()
}
pub fn create_output_weight() -> Matrix {
    let mut data = Vec::new();
    for i in 0..D_MODEL {
        let mut row = Vec::new();
        for j in 0..VOCAB_SIZE {
            let val = ((i + j) as f64 * 0.15) % 1.0;
            row.push(val);
        }
        data.push(row);
    }
    Matrix::from_vec(data)
}

pub fn predict_next_token(output: &Matrix, w_out: &Matrix) -> Vec<(i32, f64)> {
    // 最後の位置の出力を取得
    let last_index = output.rows - 1;
    let last_output = output.get_row(last_index);

    // logitsを計算 (手動で行列とベクトルの積)
    let mut logits = Vec::new();
    for j in 0..VOCAB_SIZE {
        let mut sum = 0.0;
        for i in 0..D_MODEL {
            sum += last_output[i] * w_out.get_row(i)[j];
        }
        logits.push(sum);
    }

    // Softmaxを確立化
    let probs = soft_max(&logits);

    // 確率とトークンIDをペアにしたベクトルを返す
    let mut result = Vec::new();
    for (token_id, prob) in probs.iter().enumerate() {
        result.push((token_id as i32, *prob));
    }

    result
}

// === Phase 9: 文字レベル訓練データ ===

pub fn create_training_data() -> Vec<(Vec<i32>, i32)> {
    let vocab = Vocabulary::new();
    let mut data = Vec::new();

    // データファイルを読み込み
    let content = fs::read_to_string("data/training_data.txt")
        .expect("data/training_data.txtが見つかりません");

    // 1行1文として分割（空行は除外）
    let sentences: Vec<&str> = content
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect();

    // 各文章から訓練データを生成（窓スライド方式）
    for sentence in sentences {
        let token_ids = vocab.encode(sentence);

        for i in 0..(token_ids.len().saturating_sub(SEQ_LEN)) {
            let mut input = Vec::new();
            for j in 0..SEQ_LEN {
                input.push(token_ids[i + j]);
            }
            let label = token_ids[i + SEQ_LEN];
            data.push((input, label));
        }
    }

    data
}

pub fn plot_loss_curve(loss_history: &Vec<f64>) {
    if loss_history.is_empty() {
        return;
    }

    println!("\n=== 学習曲線 ===");

    let max_loss = loss_history
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_loss = loss_history.iter().copied().fold(f64::INFINITY, f64::min);
    let loss_range = max_loss - min_loss;

    let height = 10; // グラフの高さ（行数）
    let width = 50; // グラフの幅（文字数）

    // 各行を描画
    for i in 0..height {
        let threshold = max_loss - (i as f64 / height as f64) * loss_range;
        print!("{:.2} |", threshold);

        for j in 0..width {
            let epoch_index = (j * loss_history.len()) / width;
            if epoch_index < loss_history.len() {
                let loss = loss_history[epoch_index];
                if loss >= threshold {
                    print!("█");
                } else {
                    print!("░");
                }
            } else {
                print!(" ");
            }
        }
        println!();
    }

    // X軸
    print!("     +");
    for _ in 0..width {
        print!("-");
    }
    println!();

    // X軸ラベル
    print!("      0");
    for _ in 0..(width - 15) {
        print!(" ");
    }
    println!("{}  (epochs)", loss_history.len());
}

// === Phase 12: 複数層の重み初期化関数 ===

// 複数層のMulti-Head QKV重みを生成
// 各層で異なる初期値を持つように、layer_indexを使って初期化をずらす
pub fn create_multihead_qkv_weights_multilayer(
) -> (Vec<Vec<Matrix>>, Vec<Vec<Matrix>>, Vec<Vec<Matrix>>) {
    let mut w_q_layers = Vec::new();
    let mut w_k_layers = Vec::new();
    let mut w_v_layers = Vec::new();

    for layer in 0..NUM_LAYERS {
        let mut w_q_heads = Vec::new();
        let mut w_k_heads = Vec::new();
        let mut w_v_heads = Vec::new();

        for head in 0..NUM_HEADS {
            //layer_indexを使って各層で異なる初期値を生成
            w_q_heads.push(create_head_weight_matrix_for_layer(layer, head, 1));
            w_k_heads.push(create_head_weight_matrix_for_layer(layer, head, 2));
            w_v_heads.push(create_head_weight_matrix_for_layer(layer, head, 3));
        }

        w_q_layers.push(w_q_heads);
        w_k_layers.push(w_k_heads);
        w_v_layers.push(w_v_heads);
    }

    (w_q_layers, w_k_layers, w_v_layers)
}

// 層とヘッドの両方を考慮した重み行列生成
fn create_head_weight_matrix_for_layer(
    layer_index: usize,
    head_index: usize,
    weight_type: usize,
) -> Matrix {
    let mut data = Vec::new();
    for i in 0..D_MODEL {
        let mut row = Vec::new();
        for j in 0..D_HEAD {
            // 層とヘッドごとに異なる重みを生成
            let seed = weight_type * 100 + layer_index * 50 + head_index * 10;
            let val = if i == j {
                1.0
            } else {
                ((i + j + seed) as f64 * 0.05) % 0.5
            };
            row.push(val);
        }
        data.push(row);
    }
    Matrix::from_vec(data)
}

// 複数層の出力プロジェクション行列を生成
pub fn create_multihead_output_projection_multilayer() -> Vec<Matrix> {
    let mut w_o_layers = Vec::new();
    for layer in 0..NUM_LAYERS {
        let mut data = Vec::new();
        for i in 0..D_MODEL {
            let mut row = Vec::new();
            for j in 0..D_MODEL {
                // 各層で異なる初期値
                let val = if i == j {
                    1.0
                } else {
                    ((i + j + layer * 30) as f64 * 0.03) % 0.3
                };
                row.push(val);
            }
            data.push(row);
        }
        w_o_layers.push(Matrix::from_vec(data));
    }
    w_o_layers
}

// 複数層のFeed-Forward W1を生成
pub fn create_feedforward_w1_multilayer() -> Vec<Matrix> {
    let mut w1_layers = Vec::new();
    for layer in 0..NUM_LAYERS {
        let mut data = Vec::new();
        for i in 0..D_MODEL {
            let mut row = Vec::new();
            for j in 0..D_FF {
                let val = ((i + j + layer * 20) as f64 * 0.02) % 0.2;
                row.push(val);
            }
            data.push(row);
        }
        w1_layers.push(Matrix::from_vec(data));
    }
    w1_layers
}

// 複数層のFeed-Forward W2を生成
pub fn create_feedforward_w2_multilayer() -> Vec<Matrix> {
    let mut w2_layers = Vec::new();
    for layer in 0..NUM_LAYERS {
        let mut data = Vec::new();
        for i in 0..D_FF {
            let mut row = Vec::new();
            for j in 0..D_MODEL {
                let val = ((i * 2 + j + layer * 15) as f64 * 0.02) % 0.2;
                row.push(val);
            }
            data.push(row);
        }
        w2_layers.push(Matrix::from_vec(data));
    }
    w2_layers
}

// 複数層のLayer Normalizationパラメータを生成
pub fn create_layer_norm_params_multilayer() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut gamma_layers = Vec::new();
    let mut beta_layers = Vec::new();

    for _ in 0..NUM_LAYERS {
        gamma_layers.push(vec![1.0; D_MODEL]);
        beta_layers.push(vec![0.0; D_MODEL]);
    }

    (gamma_layers, beta_layers)
}

// Layer Normalizationの順伝播
// input: [seq_len, d_model]
// gamma, beta: 長さd_modelのベクトル
// 戻り値: 正規化された行列 [seq_len, d_model]
pub fn layer_normalization(input: &Matrix, gamma: &Vec<f64>, beta: &Vec<f64>) -> Matrix {
    let seq_len = input.rows;
    let mut normalized_data = Vec::new();
    let epsilon = 1e-6; // 数値安定性のための小さな値

    for i in 0..seq_len {
        let row = input.get_row(i);

        // 1. 平均を計算
        let mean: f64 = row.iter().sum::<f64>() / D_MODEL as f64;

        // 2. 分散を計算
        let variance: f64 = row.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / D_MODEL as f64;

        // 3. 正規化
        let std_dev = (variance + epsilon).sqrt();
        let mut normalized_row = Vec::new();
        for j in 0..D_MODEL {
            let normalized = (row[j] - mean) / std_dev;
            // 4. スケール・シフト
            let output = gamma[j] * normalized + beta[j];
            normalized_row.push(output);
        }

        normalized_data.push(normalized_row);
    }

    Matrix::from_vec(normalized_data)
}

// Layer Normalizationの逆伝播
// grad_output: 出力に対する勾配 [seq_len, d_model]
// input: 元の入力 [seq_len, d_model]
// gamma, beta: 学習可能パラメータ
// 戻り値: (入力への勾配, γへの勾配, βへの勾配)
pub fn backward_layer_normalization(
    grad_output: &Matrix,
    input: &Matrix,
    gamma: &Vec<f64>,
    _beta: &Vec<f64>,
) -> (Matrix, Vec<f64>, Vec<f64>) {
    let seq_len = input.rows;
    let epsilon = 1e-6;

    // γとβへの勾配を初期化
    let mut grad_gamma = vec![0.0; D_MODEL];
    let mut grad_beta = vec![0.0; D_MODEL];

    // 入力への勾配を格納する行列
    let mut grad_input_data = Vec::new();

    // 各行（各位置）に対して独立に逆伝播
    for i in 0..seq_len {
        let row = input.get_row(i);
        let grad_out_row = grad_output.get_row(i);

        // 順伝播の再計算（中間値が必要）
        let mean: f64 = row.iter().sum::<f64>() / D_MODEL as f64;
        let variance: f64 = row.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / D_MODEL as f64;
        let std_dev = (variance + epsilon).sqrt();

        // 正規化値を再計算
        let mut normalized = Vec::new();
        for j in 0..D_MODEL {
            normalized.push((row[j] - mean) / std_dev);
        }

        // 1. γとβへの勾配を累積
        for j in 0..D_MODEL {
            grad_gamma[j] += grad_out_row[j] * normalized[j];
            grad_beta[j] += grad_out_row[j];
        }

        // 2. 正規化値への勾配
        let mut grad_normalized = Vec::new();
        for j in 0..D_MODEL {
            grad_normalized.push(grad_out_row[j] * gamma[j]);
        }

        // 3. 分散への勾配
        let mut grad_variance = 0.0;
        for j in 0..D_MODEL {
            grad_variance +=
                grad_normalized[j] * (row[j] - mean) * (-0.5) * (variance + epsilon).powf(-1.5);
        }

        // 4. 平均への勾配
        let mut grad_mean = 0.0;
        for j in 0..D_MODEL {
            grad_mean += grad_normalized[j] * (-1.0 / std_dev);
        }
        grad_mean +=
            grad_variance * (-2.0 / D_MODEL as f64) * row.iter().map(|&x| x - mean).sum::<f64>();

        // 5. 入力への勾配
        let mut grad_input_row = Vec::new();
        for j in 0..D_MODEL {
            let grad = grad_normalized[j] / std_dev
                + grad_variance * 2.0 * (row[j] - mean) / D_MODEL as f64
                + grad_mean / D_MODEL as f64;
            grad_input_row.push(grad);
        }

        grad_input_data.push(grad_input_row);
    }

    let grad_input = Matrix::from_vec(grad_input_data);
    (grad_input, grad_gamma, grad_beta)
}

pub fn backward_transformer_block(
    grad_output: &Matrix,    //Transformerブロック出力への勾配 [seq_len, d_model]
    embedded: &Matrix,       // 元の入力
    tokens: &Vec<i32>,       // 入力トークン
    w_q_heads: &Vec<Matrix>, //重み行列群（勾配計算に必要）
    w_k_heads: &Vec<Matrix>,
    w_v_heads: &Vec<Matrix>,
    w_o: &Matrix,
    w1: &Matrix,
    w2: &Matrix,
    gamma1: &Vec<f64>,
    beta1: &Vec<f64>,
    gamma2: &Vec<f64>,
    beta2: &Vec<f64>,
) -> (
    Matrix,      // embeddedへの勾配
    Vec<Matrix>, // grad_w_q_heads
    Vec<Matrix>, // grad_w_k_heads
    Vec<Matrix>, // grad_w_v_heads
    Matrix,      // grad_w_o
    Matrix,      // grad_w1
    Matrix,      // grad_w2
    Vec<f64>,    // grad_gamma1
    Vec<f64>,    // grad_beta1
    Vec<f64>,    // grad_gamma2
    Vec<f64>,    // grad_beta2
) {
    // === 1. 順伝播の再計算（中間値が必要） ===
    let normalized1 = layer_normalization(embedded, gamma1, beta1);
    let attention_output =
        compute_multihead_attention(&normalized1, tokens, w_q_heads, w_k_heads, w_v_heads, w_o);
    let residual1 = embedded.add(&attention_output);

    let normalized2 = layer_normalization(&residual1, gamma2, beta2);
    let _ffn_output = compute_feedforward(&normalized2, w1, w2);

    // === 2. 逆伝播（残差接続2から開始） ===
    // grad_output = ∂L/∂residual2

    // 残差接続2の逆伝播: residual2 = residual1 + ffn_output
    // → 勾配は2つの経路に分岐
    let grad_residual1_from_residual = grad_output.clone(); // スキップ経路
    let grad_ffn_output = grad_output.clone();
    // FFN経路

    // Feed-Forward経路の逆伝播
    let (grad_w1, grad_w2, grad_normalized2) =
        backward_feedforward(&grad_ffn_output, &normalized2, w1, w2);

    // Layer Norm 2の逆伝播
    let (grad_residual1_from_ln2, grad_gamma2, grad_beta2) =
        backward_layer_normalization(&grad_normalized2, &residual1, gamma2, beta2);

    // residual1への勾配を合算
    let grad_residual1 = grad_residual1_from_residual.add(&grad_residual1_from_ln2);

    // === 3. 残差接続1の逆伝播 ===
    // residual1 = embedded + attention_output
    // → 勾配は2つの経路に分岐
    let grad_embedded_from_residual = grad_residual1.clone(); // スキップ経路
    let grad_attention_output = grad_residual1.clone(); // Attention経路

    // Multi-Head Attention経路の逆伝播
    let (grad_normalized1, grad_w_q_heads, grad_w_k_heads, grad_w_v_heads, grad_w_o) =
        backward_multihead_attention(
            &normalized1,
            &grad_attention_output,
            w_q_heads,
            w_k_heads,
            w_v_heads,
            w_o,
        );

    // Layer Norm 1の逆伝播
    let (grad_embedded_from_ln1, grad_gamma1, grad_beta1) =
        backward_layer_normalization(&grad_normalized1, embedded, gamma1, beta1);

    // embeddedへの勾配を合算
    let grad_embedded = grad_embedded_from_residual.add(&grad_embedded_from_ln1);

    // === 4. すべての勾配を返す ===
    (
        grad_embedded,
        grad_w_q_heads,
        grad_w_k_heads,
        grad_w_v_heads,
        grad_w_o,
        grad_w1,
        grad_w2,
        grad_gamma1,
        grad_beta1,
        grad_gamma2,
        grad_beta2,
    )
}

// Feed-Forward層の順伝播
// FFN(x) = W2 @ ReLU(W1 @ x)
//
// input: [seq_len, d_model]
// w1: [d_model, d_ff]
// w2: [d_ff, d_model]
// 戻り値: [seq_len, d_model]
pub fn compute_feedforward(input: &Matrix, w1: &Matrix, w2: &Matrix) -> Matrix {
    // 1. 線形変換（拡大）: input @ W1 → [seq_len, d_ff]
    let hidden = input.multiply(w1);

    // 2. ReLU活性化: ReLU(hidden) → [seq_len, d_ff]
    let activated = relu(&hidden);

    // 3. 線形変換（元の次元に戻す）: activated @ W2 →[seq_len, d_model]
    activated.multiply(w2)
}

// === Phase 12: 複数層の順伝播 ===

// 複数層のTransformerブロックを順番に適用
// embedded: 埋め込み層 [seq_len, d_model]
// w_q_heads_layers: 各層のMulti-Head Q重み [layer][head]
// ... (他の重み行列も同様)
// 戻り値: (最終出力, 各層の中間値)
pub fn compute_multilayer_transformer(
    embedded: &Matrix,
    tokens: &Vec<i32>,
    w_q_heads_layers: &Vec<Vec<Matrix>>,
    w_k_heads_layers: &Vec<Vec<Matrix>>,
    w_v_heads_layers: &Vec<Vec<Matrix>>,
    w_o_layers: &Vec<Matrix>,
    w1_layers: &Vec<Matrix>,
    w2_layers: &Vec<Matrix>,
    gamma1_layers: &Vec<Vec<f64>>,
    beta1_layers: &Vec<Vec<f64>>,
    gamma2_layers: &Vec<Vec<f64>>,
    beta2_layers: &Vec<Vec<f64>>,
) -> (Matrix, Vec<Matrix>) {
    // 各層の出力を保存（逆伝播で使用）
    let mut layer_outputs = Vec::new();

    // 第1層の入力は埋め込み層
    let mut current_input = embedded.clone();

    // 各層を順番に適用
    for layer in 0..NUM_LAYERS {
        let (output, _, _, _, _, _) = compute_transformer_block(
            &current_input,
            &tokens,
            &w_q_heads_layers[layer],
            &w_k_heads_layers[layer],
            &w_v_heads_layers[layer],
            &w_o_layers[layer],
            &w1_layers[layer],
            &w2_layers[layer],
            &gamma1_layers[layer],
            &beta1_layers[layer],
            &gamma2_layers[layer],
            &beta2_layers[layer],
        );

        // この層の出力を保存
        layer_outputs.push(output.clone());

        // 次の層の入力として使用
        current_input = output;
    }

    // 最終層の出力と各層の出力を返す
    let final_output = layer_outputs[NUM_LAYERS - 1].clone();
    (final_output, layer_outputs)
}

// === Phase 12: 複数層の逆伝播 ===

// 複数層のTransformerブロックの逆伝播
// grad_output: 最終層の出力への勾配 [seq_len,d_model]
// embedded: 元の埋め込み層 [seq_len, d_model]
// layer_outputs:各層の出力（順伝播で保存した中間値）
// w_*_layers: 各層の重みパラメータ
// 戻り値: (embeddedへの勾配,各層のパラメータ勾配)
pub fn backward_multilayer_transformer(
    grad_output: &Matrix,
    embedded: &Matrix,
    tokens: &Vec<i32>,
    layer_outputs: &Vec<Matrix>,
    w_q_heads_layers: &Vec<Vec<Matrix>>,
    w_k_heads_layers: &Vec<Vec<Matrix>>,
    w_v_heads_layers: &Vec<Vec<Matrix>>,
    w_o_layers: &Vec<Matrix>,
    w1_layers: &Vec<Matrix>,
    w2_layers: &Vec<Matrix>,
    gamma1_layers: &Vec<Vec<f64>>,
    beta1_layers: &Vec<Vec<f64>>,
    gamma2_layers: &Vec<Vec<f64>>,
    beta2_layers: &Vec<Vec<f64>>,
) -> (
    Matrix,           // grad_embedded
    Vec<Vec<Matrix>>, // grad_w_q_heads_layers
    Vec<Vec<Matrix>>, // grad_w_k_heads_layers
    Vec<Vec<Matrix>>, // grad_w_v_heads_layers
    Vec<Matrix>,      // grad_w_o_layers
    Vec<Matrix>,      // grad_w1_layers
    Vec<Matrix>,      // grad_w2_layers
    Vec<Vec<f64>>,    // grad_gamma1_layers
    Vec<Vec<f64>>,    // grad_beta1_layers
    Vec<Vec<f64>>,    // grad_gamma2_layers
    Vec<Vec<f64>>,    // grad_beta2_layers
) {
    //各層のパラメータ勾配を保存するベクトルを初期化
    let mut grad_w_q_heads_layers = Vec::new();
    let mut grad_w_k_heads_layers = Vec::new();
    let mut grad_w_v_heads_layers = Vec::new();
    let mut grad_w_o_layers = Vec::new();
    let mut grad_w1_layers = Vec::new();
    let mut grad_w2_layers = Vec::new();
    let mut grad_gamma1_layers = Vec::new();
    let mut grad_beta1_layers = Vec::new();
    let mut grad_gamma2_layers = Vec::new();
    let mut grad_beta2_layers = Vec::new();

    //各層のためにプレースホルダーを作成（後で上書き）
    for _ in 0..NUM_LAYERS {
        grad_w_q_heads_layers.push(Vec::new());
        grad_w_k_heads_layers.push(Vec::new());
        grad_w_v_heads_layers.push(Vec::new());
        grad_w_o_layers.push(Matrix::new(D_MODEL, D_MODEL));
        grad_w1_layers.push(Matrix::new(D_MODEL, D_FF));
        grad_w2_layers.push(Matrix::new(D_FF, D_MODEL));
        grad_gamma1_layers.push(vec![0.0; D_MODEL]);
        grad_beta1_layers.push(vec![0.0; D_MODEL]);
        grad_gamma2_layers.push(vec![0.0; D_MODEL]);
        grad_beta2_layers.push(vec![0.0; D_MODEL]);
    }

    // 現在の勾配（最終層の出力への勾配から開始）
    let mut current_grad = grad_output.clone();

    // 最終層から第1層まで逆向きにループ
    for layer in (0..NUM_LAYERS).rev() {
        // この層の入力を取得
        // Layer 0 の入力 = embedded
        // Layer i の入力 = Layer (i-1) の出力 =layer_outputs[i-1]
        let layer_input = if layer == 0 {
            embedded.clone()
        } else {
            layer_outputs[layer - 1].clone()
        };

        // この層の逆伝播を実行
        let (
            grad_input,
            grad_w_q_heads,
            grad_w_k_heads,
            grad_w_v_heads,
            grad_w_o,
            grad_w1,
            grad_w2,
            grad_gamma1,
            grad_beta1,
            grad_gamma2,
            grad_beta2,
        ) = backward_transformer_block(
            &current_grad,
            &layer_input,
            &tokens,
            &w_q_heads_layers[layer],
            &w_k_heads_layers[layer],
            &w_v_heads_layers[layer],
            &w_o_layers[layer],
            &w1_layers[layer],
            &w2_layers[layer],
            &gamma1_layers[layer],
            &beta1_layers[layer],
            &gamma2_layers[layer],
            &beta2_layers[layer],
        );

        // この層のパラメータ勾配を保存
        grad_w_q_heads_layers[layer] = grad_w_q_heads;
        grad_w_k_heads_layers[layer] = grad_w_k_heads;
        grad_w_v_heads_layers[layer] = grad_w_v_heads;
        grad_w_o_layers[layer] = grad_w_o;
        grad_w1_layers[layer] = grad_w1;
        grad_w2_layers[layer] = grad_w2;
        grad_gamma1_layers[layer] = grad_gamma1;
        grad_beta1_layers[layer] = grad_beta1;
        grad_gamma2_layers[layer] = grad_gamma2;
        grad_beta2_layers[layer] = grad_beta2;

        // この層の入力への勾配を、次の層（前の層）の出力への勾配として使用
        current_grad = grad_input;
    }

    // 最終的に current_grad には embeddedへの勾配が入っている
    let grad_embedded = current_grad;

    (
        grad_embedded,
        grad_w_q_heads_layers,
        grad_w_k_heads_layers,
        grad_w_v_heads_layers,
        grad_w_o_layers,
        grad_w1_layers,
        grad_w2_layers,
        grad_gamma1_layers,
        grad_beta1_layers,
        grad_gamma2_layers,
        grad_beta2_layers,
    )
}

// Feed-Forward層の逆伝播
// grad_output: Feed-Forward出力への勾配 [seq_len, d_model]
// input: 元の入力 [seq_len, d_model]
// w1: [d_model, d_ff]
// w2: [d_ff, d_model]
// 戻り値: (W1への勾配, W2への勾配, 入力への勾配)
pub fn backward_feedforward(
    grad_output: &Matrix,
    input: &Matrix,
    w1: &Matrix,
    w2: &Matrix,
) -> (Matrix, Matrix, Matrix) {
    // 順伝播を再計算（中間結果が必要）
    let hidden = input.multiply(w1); // [seq_len, d_ff]
    let activated = relu(&hidden); // [seq_len, d_ff]

    // === 逆伝播 ===

    // 1. W2への勾配: ∂L/∂W2 = activated^T @ grad_output
    let activated_t = activated.transpose(); // [d_ff,seq_len]
    let grad_w2 = activated_t.multiply(grad_output);
    // [d_ff, d_model]

    // 2. activated層への勾配: ∂L/∂activated = grad_output @ W2^T
    let w2_t = w2.transpose(); // [d_model, d_ff]
    let grad_activated = grad_output.multiply(&w2_t);
    // [seq_len, d_ff]

    // 3. ReLU逆伝播: 勾配マスクを適用（Hadamard積）
    let relu_mask = relu_gradient(&hidden); // [seq_len, d_ff]
    let grad_hidden = grad_activated.hadamard(&relu_mask); // [seq_len, d_ff]

    // 4. W1への勾配: ∂L/∂W1 = input^T @ grad_hidden
    let input_t = input.transpose(); // [d_model, seq_len]
    let grad_w1 = input_t.multiply(&grad_hidden); // [d_model, d_ff]

    // 5. 入力への勾配: ∂L/∂input = grad_hidden @ W1^T
    let w1_t = w1.transpose(); // [d_ff, d_model]
    let grad_input = grad_hidden.multiply(&w1_t); // [seq_len, d_model]

    (grad_w1, grad_w2, grad_input)
}

// 完全なTransformerブロックの順伝播（Pre-LN + 残差接続）
// embedded: 埋め込み層 [seq_len, d_model]
// w_q_heads, w_k_heads, w_v_heads:Multi-Head用QKV重み
// w_o: 出力プロジェクション [d_model, d_model]
// w1: Feed-Forward第1層 [d_model, d_ff]
// w2: Feed-Forward第2層 [d_ff, d_model]
// gamma1, beta1: Attention前のLayer Norm
// gamma2, beta2: Feed-Forward前のLayer Norm
// 戻り値: Transformerブロックの出力 [seq_len,d_model]
pub fn compute_transformer_block(
    embedded: &Matrix,
    tokens: &Vec<i32>,
    w_q_heads: &Vec<Matrix>,
    w_k_heads: &Vec<Matrix>,
    w_v_heads: &Vec<Matrix>,
    w_o: &Matrix,
    w1: &Matrix,
    w2: &Matrix,
    gamma1: &Vec<f64>,
    beta1: &Vec<f64>,
    gamma2: &Vec<f64>,
    beta2: &Vec<f64>,
) -> (
    Matrix, // 最終出力 (residual2)
    Matrix, // normalized1 (LayerNorm1の出力)
    Matrix, // attention_output
    Matrix, // residual1
    Matrix, // normalized2 (LayerNorm2の出力)
    Matrix, // ffn_output
) {
    // 1. Layer Norm → Multi-Head Attention → 残差接続
    let normalized1 = layer_normalization(embedded, gamma1, beta1);
    let attention_output =
        compute_multihead_attention(&normalized1, tokens, w_q_heads, w_k_heads, w_v_heads, w_o);
    let residual1 = embedded.add(&attention_output);
    // 残差接続

    // 2. Layer Norm → Feed-Forward → 残差接続
    let normalized2 = layer_normalization(&residual1, gamma2, beta2);
    let ffn_output = compute_feedforward(&normalized2, w1, w2);
    let residual2 = residual1.add(&ffn_output); //残差接続

    (
        residual2,
        normalized1,
        attention_output,
        residual1,
        normalized2,
        ffn_output,
    )
}

// ヘッドの出力を結合する補助関数
// 各ヘッドの出力 [seq_len, d_head] を横に並べて[seq_len, d_model] にする
fn concatenate_heads(head_outputs: &Vec<Matrix>) -> Matrix {
    let seq_len = head_outputs[0].rows;
    let mut data = Vec::new();

    for i in 0..seq_len {
        let mut row = Vec::new();
        // 各ヘッドの同じ行を横に結合
        for head_output in head_outputs {
            let head_row = head_output.get_row(i);
            row.extend(head_row);
        }
        data.push(row);
    }

    Matrix::from_vec(data)
}

// Multi-Head Attentionの順伝播
// embedded: 埋め込み層 [seq_len, d_model]
// w_q_heads, w_k_heads, w_v_heads: 各ヘッドのQKV重み行列群 (各 [d_model, d_head])
// w_o: 出力プロジェクション行列 [d_model, d_model]
// 戻り値: Multi-Head Attentionの出力 [seq_len,d_model]
pub fn compute_multihead_attention(
    embedded: &Matrix,
    tokens: &Vec<i32>,
    w_q_heads: &Vec<Matrix>,
    w_k_heads: &Vec<Matrix>,
    w_v_heads: &Vec<Matrix>,
    w_o: &Matrix,
) -> Matrix {
    let mut head_outputs = Vec::new();
    let mask = create_attention_mask(tokens);
    // 各ヘッドで独立にAttentionを計算
    for head in 0..NUM_HEADS {
        // 1. Q, K, Vを計算 (各 [seq_len, d_head])
        let q = embedded.multiply(&w_q_heads[head]);
        let k = embedded.multiply(&w_k_heads[head]);
        let v = embedded.multiply(&w_v_heads[head]);

        // 2. Attentionscoresを計算（D_HEADでスケーリング）
        let k_t = k.transpose();
        let scores = q.multiply(&k_t);
        let scale_factor = (D_HEAD as f64).sqrt();
        let scores_scaled = scores.scale(scale_factor);
        let scores_masked = scores_scaled.add(&mask);

        // 3. Attention weightsを計算（Softmax）
        let mut weights_data = Vec::new();
        for i in 0..scores_masked.rows {
            let row = scores_masked.get_row(i);
            let softmax_row = soft_max(&row);
            weights_data.push(softmax_row);
        }
        let weights = Matrix::from_vec(weights_data);

        // 4. Attention outputを計算
        let output = weights.multiply(&v);

        head_outputs.push(output);
    }

    // 5. すべてのヘッドの出力を結合 [seq_len, d_model]
    let concatenated = concatenate_heads(&head_outputs);

    // 6. 出力プロジェクション
    concatenated.multiply(w_o)
}

// split_heads関数: concatenate_headsの逆操作
// concatenate_headsの逆操作：[seq_len, d_model]を各ヘッド[seq_len, d_head]に分割
// grad_concat: 結合された勾配 [seq_len, d_model]
// 戻り値: 各ヘッドの勾配 Vec<Matrix>（各要素は[seq_len, d_head]）
fn split_heads(grad_concat: &Matrix) -> Vec<Matrix> {
    let seq_len = grad_concat.rows;
    let mut head_grads = Vec::new();

    // 各ヘッドの勾配を格納する行列を作成
    for _head in 0..NUM_HEADS {
        head_grads.push(Matrix::new(seq_len, D_HEAD));
    }

    // 各行を処理
    for i in 0..seq_len {
        let concat_row = grad_concat.get_row(i);

        // 各ヘッドにd_head個の要素を分配
        for head in 0..NUM_HEADS {
            for j in 0..D_HEAD {
                // concatenate_headsでは各ヘッドを順番に横に並べたので
                // head番目のヘッドの要素は [head * D_HEAD, (head + 1) * D_HEAD) の範囲にある
                let concat_col = head * D_HEAD + j;
                head_grads[head].set(i, j, concat_row[concat_col]);
            }
        }
    }

    head_grads
}

// Multi-Head Attentionの逆伝播
// すべての重み行列の勾配を計算
//
// embedded: 埋め込み層 [seq_len, d_model]
// grad_multihead_output: Multi-Head Attention出力への勾配 [seq_len, d_model]
// w_q_heads, w_k_heads, w_v_heads: 各ヘッドのQKV重み行列群
// w_o: 出力プロジェクション行列
//
// 戻り値: (埋め込み層への勾配, 各ヘッドのW_Q勾配群, W_K勾配群, W_V勾配群, W_O勾配)
pub fn backward_multihead_attention(
    embedded: &Matrix,
    grad_multihead_output: &Matrix,
    w_q_heads: &Vec<Matrix>,
    w_k_heads: &Vec<Matrix>,
    w_v_heads: &Vec<Matrix>,
    w_o: &Matrix,
) -> (Matrix, Vec<Matrix>, Vec<Matrix>, Vec<Matrix>, Matrix) {
    // 順伝播の再計算（各ヘッドのQ, K, V, Attention weightsが必要）
    let mut q_heads = Vec::new();
    let mut k_heads = Vec::new();
    let mut v_heads = Vec::new();
    let mut attention_weights_heads = Vec::new();
    let mut head_outputs = Vec::new();

    for head in 0..NUM_HEADS {
        let q = embedded.multiply(&w_q_heads[head]);
        let k = embedded.multiply(&w_k_heads[head]);
        let v = embedded.multiply(&w_v_heads[head]);

        let k_t = k.transpose();
        let scores = q.multiply(&k_t);
        let scale_factor = (D_HEAD as f64).sqrt();
        let scores_scaled = scores.scale(scale_factor);

        let mut weights_data = Vec::new();
        for i in 0..scores_scaled.rows {
            let row = scores_scaled.get_row(i);
            let softmax_row = soft_max(&row);
            weights_data.push(softmax_row);
        }
        let weights = Matrix::from_vec(weights_data);
        let output = weights.multiply(&v);

        q_heads.push(q);
        k_heads.push(k);
        v_heads.push(v);
        attention_weights_heads.push(weights);
        head_outputs.push(output);
    }

    let concatenated = concatenate_heads(&head_outputs);

    // === 逆伝播 ===

    // 1. W_Oの勾配
    let grad_w_o = concatenated.transpose().multiply(grad_multihead_output);

    // 2. 結合層への勾配
    let w_o_t = w_o.transpose();
    let grad_concatenated = grad_multihead_output.multiply(&w_o_t);

    // 3. 各ヘッドへの勾配に分割
    let grad_head_outputs = split_heads(&grad_concatenated);

    // 4. 各ヘッドで独立に逆伝播
    let mut grad_w_q_heads = Vec::new();
    let mut grad_w_k_heads = Vec::new();
    let mut grad_w_v_heads = Vec::new();

    // 埋め込み層への勾配を累積（全ヘッドから流れてくる）
    let seq_len = embedded.rows;
    let mut grad_embedded = Matrix::new(seq_len, D_MODEL);

    for head in 0..NUM_HEADS {
        let grad_head_output = &grad_head_outputs[head];
        let attention_weights = &attention_weights_heads[head];
        let v = &v_heads[head];
        let q = &q_heads[head];
        let k = &k_heads[head];

        // V行列への勾配: ∂L/∂V = attention_weights^T @ grad_attention_output
        let attention_weights_t = attention_weights.transpose();
        let grad_v = attention_weights_t.multiply(grad_head_output);

        // Attention重みへの勾配: ∂L/∂attention_weights = grad_attention_output @ V^T
        let v_t = v.transpose();
        let grad_attention_weights = grad_head_output.multiply(&v_t);

        // Softmaxの逆伝播
        let seq_len = attention_weights.rows;
        let mut grad_scores = Matrix::new(seq_len, seq_len);
        for i in 0..seq_len {
            let grad_output_row = grad_attention_weights.get_row(i);
            let softmax_row = attention_weights.get_row(i);
            let mut sum: f64 = 0.0;
            for j in 0..seq_len {
                sum += grad_output_row[j] * softmax_row[j];
            }
            for j in 0..seq_len {
                let grad = softmax_row[j] * (grad_output_row[j] - sum);
                grad_scores.set(i, j, grad);
            }
        }

        // Q, K行列への勾配
        let scale_factor = (D_HEAD as f64).sqrt();
        let grad_q_unscaled = grad_scores.multiply(k);
        let grad_q = grad_q_unscaled.scale(scale_factor);

        let grad_scores_t = grad_scores.transpose();
        let grad_k_unscaled = grad_scores_t.multiply(q);
        let grad_k = grad_k_unscaled.scale(scale_factor);

        // QKV重み行列への勾配
        let embedded_t = embedded.transpose();
        let grad_w_q = embedded_t.multiply(&grad_q);
        let grad_w_k = embedded_t.multiply(&grad_k);
        let grad_w_v = embedded_t.multiply(&grad_v);

        grad_w_q_heads.push(grad_w_q);
        grad_w_k_heads.push(grad_w_k);
        grad_w_v_heads.push(grad_w_v);

        // 埋め込み層への勾配を累積
        let w_q_t = w_q_heads[head].transpose();
        let w_k_t = w_k_heads[head].transpose();
        let w_v_t = w_v_heads[head].transpose();

        let grad_from_q = grad_q.multiply(&w_q_t);
        let grad_from_k = grad_k.multiply(&w_k_t);
        let grad_from_v = grad_v.multiply(&w_v_t);

        grad_embedded = grad_embedded.add(&grad_from_q);
        grad_embedded = grad_embedded.add(&grad_from_k);
        grad_embedded = grad_embedded.add(&grad_from_v);
    }

    (
        grad_embedded,
        grad_w_q_heads,
        grad_w_k_heads,
        grad_w_v_heads,
        grad_w_o,
    )
}

// === Phase 12: 複数層の訓練ループ ===

// 複数層のTransformerを使った訓練ループ
// すべてのパラメータを同時学習（埋め込み +　各層のTransformerブロック + 出力層）
pub fn train_multilayer_transformer(
    training_data: &Vec<(Vec<i32>, i32)>,
    mut embedding_matrix: Matrix,
    mut w_q_heads_layers: Vec<Vec<Matrix>>,
    mut w_k_heads_layers: Vec<Vec<Matrix>>,
    mut w_v_heads_layers: Vec<Vec<Matrix>>,
    mut w_o_layers: Vec<Matrix>,
    mut w1_layers: Vec<Matrix>,
    mut w2_layers: Vec<Matrix>,
    mut w_out: Matrix,
    mut gamma1_layers: Vec<Vec<f64>>,
    mut beta1_layers: Vec<Vec<f64>>,
    mut gamma2_layers: Vec<Vec<f64>>,
    mut beta2_layers: Vec<Vec<f64>>,
    learning_rate: f64,
    epochs: usize,
) -> (
    Matrix,           // embedding_matrix
    Vec<Vec<Matrix>>, // w_q_heads_layers
    Vec<Vec<Matrix>>, // w_k_heads_layers
    Vec<Vec<Matrix>>, // w_v_heads_layers
    Vec<Matrix>,      // w_o_layers
    Vec<Matrix>,      // w1_layers
    Vec<Matrix>,      // w2_layers
    Matrix,           // w_out
    Vec<Vec<f64>>,    // gamma1_layers
    Vec<Vec<f64>>,    // beta1_layers
    Vec<Vec<f64>>,    // gamma2_layers
    Vec<Vec<f64>>,    // beta2_layers
    Vec<f64>,         // loss_history
) {
    let mut loss_history = Vec::new();

    for _epoch in 0..epochs {
        let mut total_loss = 0.0;

        // 各訓練例で学習
        for (input, label) in training_data {
            // === 順伝播 ===
            let embedded = embed_tokens_with_matrix(input, &embedding_matrix);

            // 複数層のTransformerブロック
            let (final_output, layer_outputs) = compute_multilayer_transformer(
                &embedded,
                input,
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

            // 出力層
            let last_index = final_output.rows - 1;
            let last_output = final_output.get_row(last_index);
            let predictions = predict_next_token(&final_output, &w_out);

            // 損失計算（Cross-Entropy Loss）
            let mut loss = 0.0;
            for (token_id, prob) in &predictions {
                if *token_id == *label {
                    loss = -prob.ln();
                    break;
                }
            }
            total_loss += loss;

            // === 逆伝播 ===

            // 1. 出力層の勾配（Softmax +Cross-Entropy）
            let mut grad_output = Vec::new();
            for (token_id, prob) in &predictions {
                if *token_id == *label {
                    grad_output.push(prob - 1.0);
                } else {
                    grad_output.push(*prob);
                }
            }

            // 出力層の重みへの勾配
            let mut grad_w_out = Matrix::new(D_MODEL, VOCAB_SIZE);
            for i in 0..D_MODEL {
                for j in 0..VOCAB_SIZE {
                    grad_w_out.set(i, j, last_output[i] * grad_output[j]);
                }
            }

            // 2. Transformer最終層への勾配
            let mut grad_last_output = vec![0.0; D_MODEL];
            for i in 0..D_MODEL {
                let mut sum = 0.0;
                for j in 0..VOCAB_SIZE {
                    sum += w_out.get(i, j) * grad_output[j];
                }
                grad_last_output[i] = sum;
            }

            let seq_len = final_output.rows;
            let mut grad_final_output = Matrix::new(seq_len, D_MODEL);
            for j in 0..D_MODEL {
                grad_final_output.set(last_index, j, grad_last_output[j]);
            }

            // 3. 複数層のTransformerブロックの逆伝播
            let (
                grad_embedded,
                grad_w_q_heads_layers,
                grad_w_k_heads_layers,
                grad_w_v_heads_layers,
                grad_w_o_layers,
                grad_w1_layers,
                grad_w2_layers,
                grad_gamma1_layers,
                grad_beta1_layers,
                grad_gamma2_layers,
                grad_beta2_layers,
            ) = backward_multilayer_transformer(
                &grad_final_output,
                &embedded,
                input,
                &layer_outputs,
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

            // 4. 埋め込み層の勾配
            let mut grad_embedding_matrix = Matrix::new(VOCAB_SIZE, D_MODEL);
            for (pos, &token_id) in input.iter().enumerate() {
                let token_idx = token_id as usize;
                let grad_row = grad_embedded.get_row(pos);
                for j in 0..D_MODEL {
                    let current_grad = grad_embedding_matrix.get(token_idx, j);

                    grad_embedding_matrix.set(token_idx, j, current_grad + grad_row[j]);
                }
            }

            // === 重み更新 ===

            // 埋め込み行列
            for i in 0..VOCAB_SIZE {
                for j in 0..D_MODEL {
                    let old_weight = embedding_matrix.get(i, j);
                    let gradient = grad_embedding_matrix.get(i, j);
                    embedding_matrix.set(i, j, old_weight - learning_rate * gradient);
                }
            }

            // 各層のパラメータを更新
            for layer in 0..NUM_LAYERS {
                // Multi-Head QKV
                for head in 0..NUM_HEADS {
                    for i in 0..D_MODEL {
                        for j in 0..D_HEAD {
                            let old_wq = w_q_heads_layers[layer][head].get(i, j);
                            let grad_wq = grad_w_q_heads_layers[layer][head].get(i, j);

                            w_q_heads_layers[layer][head].set(
                                i,
                                j,
                                old_wq - learning_rate * grad_wq,
                            );

                            let old_wk = w_k_heads_layers[layer][head].get(i, j);
                            let grad_wk = grad_w_k_heads_layers[layer][head].get(i, j);

                            w_k_heads_layers[layer][head].set(
                                i,
                                j,
                                old_wk - learning_rate * grad_wk,
                            );

                            let old_wv = w_v_heads_layers[layer][head].get(i, j);
                            let grad_wv = grad_w_v_heads_layers[layer][head].get(i, j);

                            w_v_heads_layers[layer][head].set(
                                i,
                                j,
                                old_wv - learning_rate * grad_wv,
                            );
                        }
                    }
                }

                // W_O
                for i in 0..D_MODEL {
                    for j in 0..D_MODEL {
                        let old_weight = w_o_layers[layer].get(i, j);
                        let gradient = grad_w_o_layers[layer].get(i, j);
                        w_o_layers[layer].set(i, j, old_weight - learning_rate * gradient);
                    }
                }

                // Feed-Forward W1
                for i in 0..D_MODEL {
                    for j in 0..D_FF {
                        let old_weight = w1_layers[layer].get(i, j);
                        let gradient = grad_w1_layers[layer].get(i, j);
                        w1_layers[layer].set(i, j, old_weight - learning_rate * gradient);
                    }
                }

                // Feed-Forward W2
                for i in 0..D_FF {
                    for j in 0..D_MODEL {
                        let old_weight = w2_layers[layer].get(i, j);
                        let gradient = grad_w2_layers[layer].get(i, j);
                        w2_layers[layer].set(i, j, old_weight - learning_rate * gradient);
                    }
                }

                // Layer Normalizationパラメータ
                for i in 0..D_MODEL {
                    gamma1_layers[layer][i] -= learning_rate * grad_gamma1_layers[layer][i];
                    beta1_layers[layer][i] -= learning_rate * grad_beta1_layers[layer][i];
                    gamma2_layers[layer][i] -= learning_rate * grad_gamma2_layers[layer][i];
                    beta2_layers[layer][i] -= learning_rate * grad_beta2_layers[layer][i];
                }
            }

            // 出力層
            for i in 0..D_MODEL {
                for j in 0..VOCAB_SIZE {
                    let old_weight = w_out.get(i, j);
                    let gradient = grad_w_out.get(i, j);
                    w_out.set(i, j, old_weight - learning_rate * gradient);
                }
            }
        }

        // エポックごとの平均損失を記録
        let avg_loss = total_loss / training_data.len() as f64;
        loss_history.push(avg_loss);

        // 100エポックごとに表示
        if loss_history.len() % 100 == 0 || loss_history.len() == epochs {
            println!("Epoch {}: 平均損失 = {:.4}", loss_history.len(), avg_loss);
        }
    }

    (
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
        loss_history,
    )
}

// === Phase 12: 複数層のテキスト生成関数 ===

// 複数層のTransformerを使ったテキスト生成関数
// seed_text: 種となる文字列（最低SEQ_LEN文字必要）
// max_length: 生成する最大文字数
// vocab: Vocabulary
// 重み行列群: 学習済みの複数層Transformer重み
// 戻り値: 生成されたテキスト
pub fn generate_text_multilayer(
    seed_text: &str,
    max_length: usize,
    vocab: &Vocabulary,
    embedding_matrix: &Matrix,
    w_q_heads_layers: &Vec<Vec<Matrix>>,
    w_k_heads_layers: &Vec<Vec<Matrix>>,
    w_v_heads_layers: &Vec<Vec<Matrix>>,
    w_o_layers: &Vec<Matrix>,
    w1_layers: &Vec<Matrix>,
    w2_layers: &Vec<Matrix>,
    gamma1_layers: &Vec<Vec<f64>>,
    beta1_layers: &Vec<Vec<f64>>,
    gamma2_layers: &Vec<Vec<f64>>,
    beta2_layers: &Vec<Vec<f64>>,
    w_out: &Matrix,
) -> String {
    // 種文字列をトークンIDに変換
    let mut token_ids = vocab.encode(seed_text);

    // 最低SEQ_LEN文字必要
    if token_ids.len() < SEQ_LEN {
        return seed_text.to_string();
    }

    // max_lengthまで文字を生成
    for _ in 0..(max_length - seed_text.chars().count()) {
        // 最後のSEQ_LEN文字を取得
        let len = token_ids.len();
        let mut input = Vec::new();
        for j in 0..SEQ_LEN {
            input.push(token_ids[len - SEQ_LEN + j]);
        }

        // 複数層のTransformerで次の文字を予測
        let embedded = embed_tokens_with_matrix(&input, embedding_matrix);
        let (final_output, _layer_outputs) = compute_multilayer_transformer(
            &embedded,
            &input,
            w_q_heads_layers,
            w_k_heads_layers,
            w_v_heads_layers,
            w_o_layers,
            w1_layers,
            w2_layers,
            gamma1_layers,
            beta1_layers,
            gamma2_layers,
            beta2_layers,
        );
        let predictions = predict_next_token(&final_output, w_out);

        // 最も確率の高い文字を選択
        let next_token = predictions
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(id, _)| *id)
            .unwrap_or(0);

        // 文末トークンで停止
        if let Some(c) = vocab.id_to_char.get(next_token as usize) {
            if *c == '。' {
                token_ids.push(next_token);
                break; // 生成を停止
            }
        }

        // 生成した文字を追加
        token_ids.push(next_token);
    }

    // トークンIDを文字列に変換
    vocab.decode(&token_ids)
}

// === Phase 9: Vocabulary公開用関数 ===

pub fn create_vocabulary() -> Vocabulary {
    Vocabulary::new()
}

// トークン列を固定長にパディング
// tokens: 入力トークン列
// 戻り値: SEQ_LEN長にパディングされたトークン列
pub fn pad_sequence(tokens: &[i32]) -> Vec<i32> {
    let mut padded = <[_]>::to_vec(&tokens);
    while padded.len() < SEQ_LEN {
        padded.push(PAD_TOKEN);
    }
    padded.truncate(SEQ_LEN);
    padded
}

// Attention Mask行列を生成
// tokens: パディング済みトークン列（長さSEQ_LEN）
// 戻り値: マスク行列 [SEQ_LEN, SEQ_LEN]
//         PAD位置は-1e9、それ以外は0.0
fn create_attention_mask(tokens: &[i32]) -> Matrix {
    let mut mask = Matrix::zeros(SEQ_LEN, SEQ_LEN);
    for i in 0..SEQ_LEN {
        for j in 0..SEQ_LEN {
            if tokens[i] == PAD_TOKEN || tokens[j] == PAD_TOKEN {
                mask.set(i, j, -1e9);
            }
        }
    }
    mask
}
