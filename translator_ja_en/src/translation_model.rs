use crate::config::{D_FF, D_HEAD, D_MODEL, NUM_DECODER_LAYERS, NUM_ENCODER_LAYERS, NUM_HEADS};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

// ===== 共通ヘルパー関数 =====

/// 位置エンコーディングを追加（Encoder/Decoder共通）
fn add_positional_encoding<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let shape = x.dims();
    let seq_len = shape[1];
    let d_model = shape[2];

    let mut pos_encoding_data = Vec::new();

    for pos in 0..seq_len {
        for i in 0..d_model {
            let value = if i % 2 == 0 {
                let angle = pos as f32 / 10000_f32.powf(i as f32 / d_model as f32);
                angle.sin()
            } else {
                let angle = pos as f32 / 10000_f32.powf((i - 1) as f32 / d_model as f32);
                angle.cos()
            };
            pos_encoding_data.push(value);
        }
    }

    let pos_encoding = Tensor::<B, 1>::from_floats(pos_encoding_data.as_slice(), &x.device())
        .reshape([seq_len, d_model]);

    let pos_encoding = pos_encoding.unsqueeze::<3>();

    x + pos_encoding
}

/// Attentionヘッドを連結（MultiHeadAttention/CrossAttention共通）
fn concatenate_heads<B: Backend>(heads: Vec<Tensor<B, 3>>) -> Tensor<B, 3> {
    Tensor::cat(heads, 2)
}

// ===== FeedForward =====

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
        let hidden = self.w_1.forward(x);
        let activated = burn::tensor::activation::relu(hidden);
        self.w_2.forward(activated)
    }
}

// ===== Multi-Head Self-Attention =====

#[derive(Module, Debug)]
pub struct CustomMultiHeadAttention<B: Backend> {
    w_q: Vec<Linear<B>>,
    w_k: Vec<Linear<B>>,
    w_v: Vec<Linear<B>>,
    w_o: Linear<B>,
}

impl<B: Backend> CustomMultiHeadAttention<B> {
    pub fn new(device: &B::Device) -> Self {
        let mut w_q = Vec::new();
        let mut w_k = Vec::new();
        let mut w_v = Vec::new();

        for _ in 0..NUM_HEADS {
            w_q.push(LinearConfig::new(D_MODEL, D_HEAD).init(device));
            w_k.push(LinearConfig::new(D_MODEL, D_HEAD).init(device));
            w_v.push(LinearConfig::new(D_MODEL, D_HEAD).init(device));
        }

        let w_o = LinearConfig::new(D_MODEL, D_MODEL).init(device);

        Self { w_q, w_k, w_v, w_o }
    }

    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        let mut head_outputs = Vec::new();

        for head_idx in 0..NUM_HEADS {
            let output = self.compute_head(x.clone(), head_idx, mask.clone());
            head_outputs.push(output);
        }

        let concat = self.concatenate_heads(head_outputs);
        self.w_o.forward(concat)
    }

    fn compute_head(
        &self,
        x: Tensor<B, 3>,
        head_idx: usize,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let q = self.w_q[head_idx].forward(x.clone());
        let k = self.w_k[head_idx].forward(x.clone());
        let v = self.w_v[head_idx].forward(x.clone());

        let k_t = k.transpose();
        let scores = q.matmul(k_t);
        let scale = (D_HEAD as f32).sqrt();
        let scores = scores / scale;

        let scores = if let Some(mask) = mask {
            let mask_value = Tensor::ones_like(&scores) * (-1e9);
            let mask = mask.unsqueeze::<3>();
            scores.mask_where(mask.equal_elem(0), mask_value)
        } else {
            scores
        };

        let attention_weights = burn::tensor::activation::softmax(scores, 2);
        attention_weights.matmul(v)
    }

    fn concatenate_heads(&self, heads: Vec<Tensor<B, 3>>) -> Tensor<B, 3> {
        concatenate_heads(heads)
    }
}

// ===== Transformer Block (Encoder用) =====

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attention: CustomMultiHeadAttention<B>,
    feed_forward: FeedForward<B>,
    layer_norm1: LayerNorm<B>,
    layer_norm2: LayerNorm<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(device: &B::Device) -> Self {
        let attention = CustomMultiHeadAttention::new(device);
        let feed_forward = FeedForward::new(device);

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
        // Pre-LN方式
        let normalized1 = self.layer_norm1.forward(x.clone());
        let attention_output = self.attention.forward(normalized1, mask);
        let residual1 = x + attention_output;

        let normalized2 = self.layer_norm2.forward(residual1.clone());
        let ff_output = self.feed_forward.forward(normalized2);
        let residual2 = residual1 + ff_output;

        residual2
    }
}

// ===== Cross-Attention =====

#[derive(Module, Debug)]
pub struct CustomCrossAttention<B: Backend> {
    w_q: Vec<Linear<B>>,
    w_k: Vec<Linear<B>>,
    w_v: Vec<Linear<B>>,
    w_o: Linear<B>,
}

impl<B: Backend> CustomCrossAttention<B> {
    pub fn new(device: &B::Device) -> Self {
        let mut w_q = Vec::new();
        let mut w_k = Vec::new();
        let mut w_v = Vec::new();

        for _ in 0..NUM_HEADS {
            w_q.push(LinearConfig::new(D_MODEL, D_HEAD).init(device));
            w_k.push(LinearConfig::new(D_MODEL, D_HEAD).init(device));
            w_v.push(LinearConfig::new(D_MODEL, D_HEAD).init(device));
        }

        let w_o = LinearConfig::new(D_MODEL, D_MODEL).init(device);

        Self { w_q, w_k, w_v, w_o }
    }

    pub fn forward(
        &self,
        query_input: Tensor<B, 3>,
        key_value_input: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let mut head_outputs = Vec::new();

        for head_idx in 0..NUM_HEADS {
            let output = self.compute_head(
                query_input.clone(),
                key_value_input.clone(),
                head_idx,
                mask.clone(),
            );
            head_outputs.push(output);
        }

        let concat = self.concatenate_heads(head_outputs);
        self.w_o.forward(concat)
    }

    fn compute_head(
        &self,
        query_input: Tensor<B, 3>,
        key_value_input: Tensor<B, 3>,
        head_idx: usize,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let q = self.w_q[head_idx].forward(query_input);
        let k = self.w_k[head_idx].forward(key_value_input.clone());
        let v = self.w_v[head_idx].forward(key_value_input);

        let k_t = k.transpose();
        let scores = q.matmul(k_t);
        let scale = (D_HEAD as f32).sqrt();
        let scores = scores / scale;

        let scores = if let Some(mask) = mask {
            let mask_value = Tensor::ones_like(&scores) * (-1e9);
            let mask = mask.unsqueeze::<3>();
            scores.mask_where(mask.equal_elem(0), mask_value)
        } else {
            scores
        };

        let attention_weights = burn::tensor::activation::softmax(scores, 2);
        attention_weights.matmul(v)
    }

    fn concatenate_heads(&self, heads: Vec<Tensor<B, 3>>) -> Tensor<B, 3> {
        concatenate_heads(heads)
    }
}

// ===== Decoder Block =====

#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    self_attention: CustomMultiHeadAttention<B>,
    cross_attention: CustomCrossAttention<B>,
    feed_forward: FeedForward<B>,
    layer_norm1: LayerNorm<B>,
    layer_norm2: LayerNorm<B>,
    layer_norm3: LayerNorm<B>,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn new(device: &B::Device) -> Self {
        let self_attention = CustomMultiHeadAttention::new(device);
        let cross_attention = CustomCrossAttention::new(device);
        let feed_forward = FeedForward::new(device);

        let layer_norm1 = LayerNormConfig::new(D_MODEL).init(device);
        let layer_norm2 = LayerNormConfig::new(D_MODEL).init(device);
        let layer_norm3 = LayerNormConfig::new(D_MODEL).init(device);

        Self {
            self_attention,
            cross_attention,
            feed_forward,
            layer_norm1,
            layer_norm2,
            layer_norm3,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        encoder_output: Tensor<B, 3>,
        tgt_mask: Option<Tensor<B, 2>>,
        src_mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        // Pre-LN方式

        // 1. Self-Attention（因果マスク付き）
        let normalized1 = self.layer_norm1.forward(x.clone());
        let self_attn_output = self.self_attention.forward(normalized1, tgt_mask);
        let residual1 = x + self_attn_output;

        // 2. Cross-Attention（Encoderの出力を参照）
        let normalized2 = self.layer_norm2.forward(residual1.clone());
        let cross_attn_output = self
            .cross_attention
            .forward(normalized2, encoder_output, src_mask);
        let residual2 = residual1 + cross_attn_output;

        // 3. Feed-Forward
        let normalized3 = self.layer_norm3.forward(residual2.clone());
        let ff_output = self.feed_forward.forward(normalized3);
        let residual3 = residual2 + ff_output;

        residual3
    }
}

// ===== Encoder =====

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    embedding: Embedding<B>,
    encoder_blocks: Vec<TransformerBlock<B>>,
}

impl<B: Backend> Encoder<B> {
    pub fn new(device: &B::Device, src_vocab_size: usize) -> Self {
        let embedding = EmbeddingConfig::new(src_vocab_size, D_MODEL).init(device);

        let mut encoder_blocks = Vec::new();
        for _ in 0..NUM_ENCODER_LAYERS {
            encoder_blocks.push(TransformerBlock::new(device));
        }

        Self {
            embedding,
            encoder_blocks,
        }
    }

    pub fn forward(
        &self,
        src_tokens: Tensor<B, 2, Int>,
        src_mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let embedded = self.embedding.forward(src_tokens);
        let mut x = self.add_positional_encoding(embedded);

        for block in &self.encoder_blocks {
            x = block.forward(x, src_mask.clone());
        }

        x
    }

    fn add_positional_encoding(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        add_positional_encoding(x)
    }
}

// ===== Decoder =====

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    embedding: Embedding<B>,
    decoder_blocks: Vec<DecoderBlock<B>>,
    output_projection: Linear<B>,
}

impl<B: Backend> Decoder<B> {
    pub fn new(device: &B::Device, tgt_vocab_size: usize) -> Self {
        let embedding = EmbeddingConfig::new(tgt_vocab_size, D_MODEL).init(device);

        let mut decoder_blocks = Vec::new();
        for _ in 0..NUM_DECODER_LAYERS {
            decoder_blocks.push(DecoderBlock::new(device));
        }

        let output_projection = LinearConfig::new(D_MODEL, tgt_vocab_size).init(device);

        Self {
            embedding,
            decoder_blocks,
            output_projection,
        }
    }

    pub fn forward(
        &self,
        tgt_tokens: Tensor<B, 2, Int>,
        encoder_output: Tensor<B, 3>,
        tgt_mask: Option<Tensor<B, 2>>,
        src_mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let embedded = self.embedding.forward(tgt_tokens);
        let mut x = self.add_positional_encoding(embedded);

        for block in &self.decoder_blocks {
            x = block.forward(
                x,
                encoder_output.clone(),
                tgt_mask.clone(),
                src_mask.clone(),
            );
        }

        let logits = self.output_projection.forward(x);

        logits
    }

    fn add_positional_encoding(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        add_positional_encoding(x)
    }
}

// ===== Seq2SeqModel（日英翻訳用） =====

#[derive(Module, Debug)]
pub struct Seq2SeqModel<B: Backend> {
    encoder: Encoder<B>,
    decoder: Decoder<B>,
}

impl<B: Backend> Seq2SeqModel<B> {
    pub fn new(device: &B::Device, src_vocab_size: usize, tgt_vocab_size: usize) -> Self {
        let encoder = Encoder::new(device, src_vocab_size);
        let decoder = Decoder::new(device, tgt_vocab_size);

        Self { encoder, decoder }
    }

    /// 訓練時のフォワードパス（Teacher Forcing）
    pub fn forward(
        &self,
        src_tokens: Tensor<B, 2, Int>,
        tgt_tokens: Tensor<B, 2, Int>,
        src_mask: Option<Tensor<B, 2>>,
        tgt_mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let encoder_output = self.encoder.forward(src_tokens, src_mask.clone());
        let logits = self
            .decoder
            .forward(tgt_tokens, encoder_output, tgt_mask, src_mask);

        logits
    }

    /// 推論時の自己回帰生成
    pub fn generate(
        &self,
        src_tokens: Tensor<B, 2, Int>,
        src_mask: Option<Tensor<B, 2>>,
        sos_id: usize,
        eos_id: usize,
        max_len: usize,
        tgt_vocab_size: usize,
    ) -> Tensor<B, 2, Int> {
        let encoder_output = self.encoder.forward(src_tokens.clone(), src_mask.clone());

        let batch_size = encoder_output.dims()[0];
        let device = encoder_output.device();

        let mut generated_ids = vec![vec![sos_id as i32]; batch_size];

        for _ in 0..max_len {
            let current_len = generated_ids[0].len();
            let flat_ids: Vec<i32> = generated_ids.iter().flatten().copied().collect();
            let tgt_tokens = Tensor::<B, 1, Int>::from_data(flat_ids.as_slice(), &device)
                .reshape([batch_size, current_len]);

            let logits =
                self.decoder
                    .forward(tgt_tokens, encoder_output.clone(), None, src_mask.clone());

            let last_logits = logits
                .clone()
                .slice([
                    0..batch_size,
                    current_len - 1..current_len,
                    0..tgt_vocab_size,
                ])
                .reshape([batch_size, tgt_vocab_size]);

            let predicted_ids = last_logits.argmax(1);

            let predicted_data: Vec<i32> = predicted_ids.to_data().to_vec().unwrap();
            let mut all_eos = true;
            for (i, &predicted_id) in predicted_data.iter().enumerate() {
                generated_ids[i].push(predicted_id);
                if predicted_id != eos_id as i32 {
                    all_eos = false;
                }
            }

            if all_eos {
                break;
            }
        }

        let max_generated_len = generated_ids.iter().map(|ids| ids.len()).max().unwrap();
        let mut padded_ids = Vec::new();
        for ids in generated_ids {
            let ids_len = ids.len();
            padded_ids.extend(ids);
            for _ in ids_len..max_generated_len {
                padded_ids.push(eos_id as i32);
            }
        }

        Tensor::<B, 1, Int>::from_data(padded_ids.as_slice(), &device)
            .reshape([batch_size, max_generated_len])
    }
}
