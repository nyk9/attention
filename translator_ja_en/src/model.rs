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

// ===== Encoder（Seq2Seq用） =====

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    embedding: Embedding<B>,
    encoder_blocks: Vec<TransformerBlock<B>>,
}

impl<B: Backend> Encoder<B> {
    pub fn new(device: &B::Device) -> Self {
        let embedding = EmbeddingConfig::new(VOCAB_SIZE, D_MODEL).init(device);

        // N層のEncoderBlock（TransformerBlockを再利用）
        let mut encoder_blocks = Vec::new();
        for _ in 0..NUM_LAYERS {
            encoder_blocks.push(TransformerBlock::new(device));
        }

        Self {
            embedding,
            encoder_blocks,
        }
    }

    /// Encoderのフォワードパス
    /// 入力: [batch, src_seq_len] のトークンID
    /// 出力: [batch, src_seq_len, d_model] のエンコード表現
    pub fn forward(&self, src_tokens: Tensor<B, 2, Int>, src_mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        // Embedding + Positional Encoding
        let embedded = self.embedding.forward(src_tokens);
        let mut x = self.add_positional_encoding(embedded);

        // N層のEncoderBlockを順番に適用
        for block in &self.encoder_blocks {
            x = block.forward(x, src_mask.clone());
        }

        x
    }

    fn add_positional_encoding(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let shape = x.dims();
        let _batch_size = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];

        // Positional Encoding行列を生成 [seq_len, d_model]
        let mut pos_encoding_data = Vec::new();

        for pos in 0..seq_len {
            for i in 0..d_model {
                let value = if i % 2 == 0 {
                    // sin(pos / 10000^(i / d_model))
                    let angle = pos as f32 / 10000_f32.powf(i as f32 / d_model as f32);
                    angle.sin()
                } else {
                    // cos(pos / 10000^((i-1) / d_model))
                    let angle = pos as f32 / 10000_f32.powf((i - 1) as f32 / d_model as f32);
                    angle.cos()
                };
                pos_encoding_data.push(value);
            }
        }

        // Vec -> Tensor変換 [seq_len, d_model]
        let pos_encoding = Tensor::<B, 1>::from_floats(pos_encoding_data.as_slice(), &x.device())
            .reshape([seq_len, d_model]);

        // バッチ次元を追加して broadcast: [1, seq_len, d_model]
        let pos_encoding = pos_encoding.unsqueeze::<3>();

        // Embedding + Positional Encoding
        x + pos_encoding
    }
}

// ===== Cross-Attention（Seq2Seq用） =====

#[derive(Module, Debug)]
pub struct CustomCrossAttention<B: Backend> {
    // Query用の線形変換（Decoder側の入力から）
    w_q: Vec<Linear<B>>,
    // Key/Value用の線形変換（Encoder側の出力から）
    w_k: Vec<Linear<B>>,
    w_v: Vec<Linear<B>>,
    // 出力射影
    w_o: Linear<B>,
}

impl<B: Backend> CustomCrossAttention<B> {
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

    /// Cross-Attentionのフォワードパス
    /// query_input: [batch, tgt_seq_len, d_model] (Decoder側の入力)
    /// key_value_input: [batch, src_seq_len, d_model] (Encoder側の出力)
    /// mask: オプションのマスク [batch, src_seq_len] (Encoderのパディングマスク)
    pub fn forward(
        &self,
        query_input: Tensor<B, 3>,
        key_value_input: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let mut head_outputs = Vec::new();

        // 各ヘッドで処理
        for head_idx in 0..NUM_HEADS {
            let output = self.compute_head(
                query_input.clone(),
                key_value_input.clone(),
                head_idx,
                mask.clone(),
            );
            head_outputs.push(output);
        }

        // ヘッドの出力を結合 [batch, tgt_seq_len, d_model]
        let concat = self.concatenate_heads(head_outputs);

        // 出力射影
        self.w_o.forward(concat)
    }

    fn compute_head(
        &self,
        query_input: Tensor<B, 3>,
        key_value_input: Tensor<B, 3>,
        head_idx: usize,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        // Q: Decoder側の入力から計算 [batch, tgt_seq_len, d_head]
        let q = self.w_q[head_idx].forward(query_input);

        // K, V: Encoder側の出力から計算 [batch, src_seq_len, d_head]
        let k = self.w_k[head_idx].forward(key_value_input.clone());
        let v = self.w_v[head_idx].forward(key_value_input);

        // Attention Score: Q × K^T / sqrt(d_head)
        // [batch, tgt_seq_len, d_head] × [batch, d_head, src_seq_len]
        let k_t = k.transpose(); // 最後の2次元を転置
        let scores = q.matmul(k_t);
        let scale = (D_HEAD as f32).sqrt();
        let scores = scores / scale;

        // マスクを適用（Encoderのパディングマスク）
        let scores = if let Some(mask) = mask {
            // マスクが0の箇所に大きな負の値を設定
            let mask_value = Tensor::ones_like(&scores) * (-1e9);
            let mask = mask.unsqueeze::<3>(); // [batch, src_seq_len, 1]を追加
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

// ===== DecoderBlock（Seq2Seq用） =====

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

        // Layer Normalization (d_model次元で正規化)
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

    /// DecoderBlockのフォワードパス
    /// x: [batch, tgt_seq_len, d_model] (Decoder側の入力)
    /// encoder_output: [batch, src_seq_len, d_model] (Encoderの出力)
    /// tgt_mask: オプションの因果マスク [batch, tgt_seq_len] (Decoderの因果マスク)
    /// src_mask: オプションのマスク [batch, src_seq_len] (Encoderのパディングマスク)
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        encoder_output: Tensor<B, 3>,
        tgt_mask: Option<Tensor<B, 2>>,
        src_mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        // Pre-LN方式: Layer Norm → Attention → 残差接続

        // 1. Self-Attention（因果マスク付き）
        let normalized1 = self.layer_norm1.forward(x.clone());
        let self_attn_output = self.self_attention.forward(normalized1, tgt_mask);
        let residual1 = x + self_attn_output;

        // 2. Cross-Attention（Encoderの出力を参照）
        let normalized2 = self.layer_norm2.forward(residual1.clone());
        let cross_attn_output = self.cross_attention.forward(normalized2, encoder_output, src_mask);
        let residual2 = residual1 + cross_attn_output;

        // 3. Feed-Forward
        let normalized3 = self.layer_norm3.forward(residual2.clone());
        let ff_output = self.feed_forward.forward(normalized3);
        let residual3 = residual2 + ff_output;

        residual3
    }
}

// ===== Decoder（Seq2Seq用） =====

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    embedding: Embedding<B>,
    decoder_blocks: Vec<DecoderBlock<B>>,
    output_projection: Linear<B>,
}

impl<B: Backend> Decoder<B> {
    pub fn new(device: &B::Device) -> Self {
        let embedding = EmbeddingConfig::new(VOCAB_SIZE, D_MODEL).init(device);

        // N層のDecoderBlock
        let mut decoder_blocks = Vec::new();
        for _ in 0..NUM_LAYERS {
            decoder_blocks.push(DecoderBlock::new(device));
        }

        // 出力射影層: [d_model] → [vocab_size]
        let output_projection = LinearConfig::new(D_MODEL, VOCAB_SIZE).init(device);

        Self {
            embedding,
            decoder_blocks,
            output_projection,
        }
    }

    /// Decoderのフォワードパス
    /// tgt_tokens: [batch, tgt_seq_len] のトークンID
    /// encoder_output: [batch, src_seq_len, d_model] (Encoderの出力)
    /// tgt_mask: オプションの因果マスク [batch, tgt_seq_len]
    /// src_mask: オプションのマスク [batch, src_seq_len] (Encoderのパディングマスク)
    /// 出力: [batch, tgt_seq_len, vocab_size]
    pub fn forward(
        &self,
        tgt_tokens: Tensor<B, 2, Int>,
        encoder_output: Tensor<B, 3>,
        tgt_mask: Option<Tensor<B, 2>>,
        src_mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        // Embedding + Positional Encoding
        let embedded = self.embedding.forward(tgt_tokens);
        let mut x = self.add_positional_encoding(embedded);

        // N層のDecoderBlockを順番に適用
        for block in &self.decoder_blocks {
            x = block.forward(x, encoder_output.clone(), tgt_mask.clone(), src_mask.clone());
        }

        // 出力射影: [batch, tgt_seq_len, d_model] → [batch, tgt_seq_len, vocab_size]
        let logits = self.output_projection.forward(x);

        logits
    }

    fn add_positional_encoding(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let shape = x.dims();
        let _batch_size = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];

        // Positional Encoding行列を生成 [seq_len, d_model]
        let mut pos_encoding_data = Vec::new();

        for pos in 0..seq_len {
            for i in 0..d_model {
                let value = if i % 2 == 0 {
                    // sin(pos / 10000^(i / d_model))
                    let angle = pos as f32 / 10000_f32.powf(i as f32 / d_model as f32);
                    angle.sin()
                } else {
                    // cos(pos / 10000^((i-1) / d_model))
                    let angle = pos as f32 / 10000_f32.powf((i - 1) as f32 / d_model as f32);
                    angle.cos()
                };
                pos_encoding_data.push(value);
            }
        }

        // Vec -> Tensor変換 [seq_len, d_model]
        let pos_encoding = Tensor::<B, 1>::from_floats(pos_encoding_data.as_slice(), &x.device())
            .reshape([seq_len, d_model]);

        // バッチ次元を追加して broadcast: [1, seq_len, d_model]
        let pos_encoding = pos_encoding.unsqueeze::<3>();

        // Embedding + Positional Encoding
        x + pos_encoding
    }
}

// ===== Seq2SeqModel（Encoder-Decoder統合） =====

#[derive(Module, Debug)]
pub struct Seq2SeqModel<B: Backend> {
    encoder: Encoder<B>,
    decoder: Decoder<B>,
}

impl<B: Backend> Seq2SeqModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let encoder = Encoder::new(device);
        let decoder = Decoder::new(device);

        Self { encoder, decoder }
    }

    /// 訓練時のフォワードパス（Teacher Forcing）
    /// src_tokens: [batch, src_seq_len] のトークンID（日本語入力）
    /// tgt_tokens: [batch, tgt_seq_len] のトークンID（手話タグ出力、SOSで開始）
    /// src_mask: オプションのマスク [batch, src_seq_len] (Encoderのパディングマスク)
    /// tgt_mask: オプションの因果マスク [batch, tgt_seq_len] (Decoderの因果マスク)
    /// 出力: [batch, tgt_seq_len, vocab_size]
    pub fn forward(
        &self,
        src_tokens: Tensor<B, 2, Int>,
        tgt_tokens: Tensor<B, 2, Int>,
        src_mask: Option<Tensor<B, 2>>,
        tgt_mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        // Encoderでソース文をエンコード
        let encoder_output = self.encoder.forward(src_tokens, src_mask.clone());

        // Decoderでターゲット文を生成
        let logits = self.decoder.forward(tgt_tokens, encoder_output, tgt_mask, src_mask);

        logits
    }

    /// 推論時の自己回帰生成
    /// src_tokens: [batch, src_seq_len] のトークンID（日本語入力）
    /// sos_id: Start of Sequence トークンID
    /// eos_id: End of Sequence トークンID
    /// max_len: 最大生成長
    /// 出力: [batch, generated_seq_len] の生成されたトークンID列
    pub fn generate(
        &self,
        src_tokens: Tensor<B, 2, Int>,
        src_mask: Option<Tensor<B, 2>>,
        sos_id: usize,
        eos_id: usize,
        max_len: usize,
    ) -> Tensor<B, 2, Int> {
        // Encoderでソース文をエンコード
        let encoder_output = self.encoder.forward(src_tokens.clone(), src_mask.clone());

        let batch_size = encoder_output.dims()[0];
        let device = encoder_output.device();

        // 最初のトークンはSOS
        let mut generated_ids = vec![vec![sos_id as i32]; batch_size];

        // 自己回帰生成ループ
        for _ in 0..max_len {
            // 現在までの生成トークンをTensorに変換
            let current_len = generated_ids[0].len();
            let flat_ids: Vec<i32> = generated_ids.iter().flatten().copied().collect();
            let tgt_tokens = Tensor::<B, 1, Int>::from_data(flat_ids.as_slice(), &device)
                .reshape([batch_size, current_len]);

            // Decoderで次のトークンを予測
            let logits = self.decoder.forward(
                tgt_tokens,
                encoder_output.clone(),
                None, // 因果マスクは不要（生成済み部分のみ参照）
                src_mask.clone(),
            );

            // 最後の位置のlogitsを取得: [batch, vocab_size]
            let last_logits = logits
                .clone()
                .slice([0..batch_size, current_len - 1..current_len, 0..VOCAB_SIZE])
                .reshape([batch_size, VOCAB_SIZE]);

            // 各バッチで最も確率の高いトークンを選択
            let predicted_ids = last_logits.argmax(1);

            // 生成されたトークンを追加
            let predicted_data: Vec<i32> = predicted_ids.to_data().to_vec().unwrap();
            let mut all_eos = true;
            for (i, &predicted_id) in predicted_data.iter().enumerate() {
                generated_ids[i].push(predicted_id);
                if predicted_id != eos_id as i32 {
                    all_eos = false;
                }
            }

            // 全バッチがEOSに到達したら終了
            if all_eos {
                break;
            }
        }

        // 生成されたトークンIDをTensorに変換
        let max_generated_len = generated_ids.iter().map(|ids| ids.len()).max().unwrap();
        let mut padded_ids = Vec::new();
        for ids in generated_ids {
            let ids_len = ids.len();
            padded_ids.extend(ids);
            // 長さを揃える（短い場合はEOSで埋める）
            for _ in ids_len..max_generated_len {
                padded_ids.push(eos_id as i32);
            }
        }

        Tensor::<B, 1, Int>::from_data(padded_ids.as_slice(), &device)
            .reshape([batch_size, max_generated_len])
    }
}

