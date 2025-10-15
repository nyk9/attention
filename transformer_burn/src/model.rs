use crate::config::{D_FF, D_HEAD, D_MODEL, NUM_HEADS, NUM_LAYERS, VOCAB_SIZE};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    w_1: Linear<B>,
    w_2: Linear<B>,
}
impl<B: Backend> FeedForward<B> {
    pub fn new(device: &B::Device) -> Self {
        let w1 = LinearConfig::new(D_MODEL, D_FF).init(device);
        let w2 = LinearConfig::new(D_FF, D_MODEL).init(device);

        Self { w_1: w1, w_2: w2 }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // x: [batch_size, seq_len, d_model]

        // 第1層: d_model → d_ff
        let hidden = self.w_1.forward(x);

        // ReLU活性化
        let activated = burn::tensor::activation::relu(hidden);

        // 第2層: d_ff → d_model
        self.w_2.forward(activated)
    }
}
impl<B: Backend> TransformerBlock<B> {
    pub fn new(device: &B::Device) -> Self {
        let attention = CustomMultiHeadAttention::new(device);
        let feed_forward = FeedForward::new(device);

        // Layer Normalization  (d_model次元で正規化)
        let layer_norm1 = LayerNormConfig::new(D_MODEL).init(device);
        let layer_norm2 = LayerNormConfig::new(D_MODEL).init(device);

        Self {
            attention,
            feed_forward,
            layer_norm1,
            layer_norm2,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        // Pre-LN方式: Layer Norm → Attention →   残差接続

        // 1. Layer Norm 1
        let normalized1 = self.layer_norm1.forward(x.clone());

        // 2. Multi-Head Attention
        let attention_output = self.attention.forward(normalized1, mask);

        // 3. 残差接続
        let residual1 = x + attention_output;

        // 4. Layer Norm 2
        let normalized2 = self.layer_norm2.forward(residual1.clone());

        // 5. Feed-Forward
        let ff_output = self.feed_forward.forward(normalized2);

        // 6. 残差接続
        let residual2 = residual1 + ff_output;

        residual2
    }
}
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attention: CustomMultiHeadAttention<B>,
    feed_forward: FeedForward<B>,
    layer_norm1: LayerNorm<B>,
    layer_norm2: LayerNorm<B>,
}

#[derive(Module, Debug)]
pub struct CustomMultiHeadAttention<B: Backend> {
    // 各ヘッドのQ, K, V用の線形変換（2ヘッド分）
    w_q: Vec<Linear<B>>,
    w_k: Vec<Linear<B>>,
    w_v: Vec<Linear<B>>,
    // 出力射影
    w_o: Linear<B>,
}

impl<B: Backend> CustomMultiHeadAttention<B> {
    pub fn new(device: &B::Device) -> Self {
        let mut w_q = Vec::new();
        let mut w_k = Vec::new();
        let mut w_v = Vec::new();

        // 各ヘッド用の重み行列を作成
        for _ in 0..NUM_HEADS {
            w_q.push(LinearConfig::new(D_MODEL, D_HEAD).init(device));
            w_k.push(LinearConfig::new(D_MODEL, D_HEAD).init(device));
            w_v.push(LinearConfig::new(D_MODEL, D_HEAD).init(device));
        }

        // 出力射影層 [d_model, d_model]
        let w_o = LinearConfig::new(D_MODEL, D_MODEL).init(device);

        Self { w_q, w_k, w_v, w_o }
    }

    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        // x: [batch_size, seq_len, d_model]
        let mut head_outputs = Vec::new();

        // 各ヘッドで処理
        for head_idx in 0..NUM_HEADS {
            let output = self.compute_head(x.clone(), head_idx, mask.clone());
            head_outputs.push(output);
        }

        // ヘッドの出力を結合 [batch_size, seq_len, d_model]
        let concat = self.concatenate_heads(head_outputs);

        // 出力射影
        self.w_o.forward(concat)
    }

    fn compute_head(
        &self,
        x: Tensor<B, 3>,
        head_idx: usize,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        // Q, K, V を計算 [batch_size, seq_len, d_head]
        let q = self.w_q[head_idx].forward(x.clone());
        let k = self.w_k[head_idx].forward(x.clone());
        let v = self.w_v[head_idx].forward(x.clone());

        // Attention Score: Q × K^T / sqrt(d_head)
        // [batch_size, seq_len, d_head] × [batch_size, d_head, seq_len]
        let k_t = k.transpose(); // 最後の2次元を転置
        let scores = q.matmul(k_t);
        let scale = (D_HEAD as f32).sqrt();
        let scores = scores / scale;

        // マスクを適用（オプション）
        let scores = if let Some(mask) = mask {
            // マスクが0の箇所に大きな負の値を設定
            let mask_value = Tensor::ones_like(&scores) * (-1e9);
            let mask = mask.unsqueeze::<3>(); // [batch, seq, 1]を追加
            scores.mask_where(mask.equal_elem(0), mask_value)
        } else {
            scores
        };

        // Softmax適用
        let attention_weights = burn::tensor::activation::softmax(scores, 2);

        // Attention × Value
        attention_weights.matmul(v)
    }

    fn concatenate_heads(&self, heads: Vec<Tensor<B, 3>>) -> Tensor<B, 3> {
        // 各ヘッドを最後の次元で結合
        Tensor::cat(heads, 2)
    }
}

#[derive(Module, Debug)]
pub struct TransformerModel<B: Backend> {
    embedding: Embedding<B>,

    // 後で追加: attention, feed_forward, etc.
    transformer_blocks: Vec<TransformerBlock<B>>,
    output_projection: Linear<B>,
}

impl<B: Backend> TransformerModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let embedding = EmbeddingConfig::new(VOCAB_SIZE, D_MODEL).init(device);

        // 4層のTransformerブロックを作成
        let mut transformer_blocks = Vec::new();
        for _ in 0..NUM_LAYERS {
            transformer_blocks.push(TransformerBlock::new(device));
        }

        // 出力射影層: [d_model] → [vocab_size]
        let output_projection = LinearConfig::new(D_MODEL, VOCAB_SIZE).init(device);

        Self {
            embedding,
            transformer_blocks,
            output_projection,
        }
    }

    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let embedded = self.embedding.forward(tokens);
        let mut x = self.add_positional_encoding(embedded);

        // 4層のTransformerブロックを順番に適用
        for block in &self.transformer_blocks {
            x = block.forward(x, None);
        }

        // 出力射影: [batch, seq_len, d_model] →  [batch, seq_len, vocab_size]
        let logits = self.output_projection.forward(x);

        logits
    }

    fn add_positional_encoding(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let shape = x.dims();
        let _batch_size = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];

        // Positional Encoding行列を生成 [seq_len,   d_model]
        let mut pos_encoding_data = Vec::new();

        for pos in 0..seq_len {
            for i in 0..d_model {
                let value = if i % 2 == 0 {
                    // sin(pos / 10000^(i / d_model))
                    let angle = pos as f32 / 10000_f32.powf(i as f32 / d_model as f32);
                    angle.sin()
                } else {
                    // cos(pos / 10000^((i-1) /   d_model))
                    let angle = pos as f32 / 10000_f32.powf((i - 1) as f32 / d_model as f32);
                    angle.cos()
                };
                pos_encoding_data.push(value);
            }
        }

        // Vec -> Tensor変換 [seq_len, d_model]
        let pos_encoding = Tensor::<B, 1>::from_floats(pos_encoding_data.as_slice(), &x.device())
            .reshape([seq_len, d_model]);

        // バッチ次元を追加して broadcast: [1,  seq_len, d_model]
        let pos_encoding = pos_encoding.unsqueeze::<3>();

        // Embedding + Positional Encoding
        x + pos_encoding
    }
}
