//! 認識モデル: 手ポーズ列(126次元×Tフレーム) → 手話タグ
//!
//! Seq2SeqModel(日本語→タグの足場)との違いは入力側だけ:
//! - 足場: トークンID列 → Embedding → Transformer
//! - 認識: 手ポーズ列(連続値ベクトル) → Linear射影 → Transformer
//!
//! Decoder 側は足場と同じ自己回帰タグ生成だが、以下が異なる:
//! - 語彙が動的(TagVocabulary、データから構築)
//! - Self-Attention に因果マスクを内蔵(学習時に未来のトークンを
//!   覗く「カンニング」を防ぐ。足場の Decoder は None を渡していて
//!   この防御がない)

use crate::config::{RecognitionModelConfig, D_HEAD, D_MODEL, NUM_HEADS, POSE_FEATURE_DIM};
use crate::model::{CustomCrossAttention, FeedForward, TransformerBlock};
use crate::tag_vocabulary::TagVocabulary;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use std::path::Path;

/// sin/cos の Positional Encoding を加算する(model.rs の各実装と同じ式)
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
        .reshape([seq_len, d_model])
        .unsqueeze::<3>(); // [1, seq_len, d_model]

    x + pos_encoding
}

// ===== 因果マスク付き Self-Attention(Decoder用) =====

/// 因果マスク付き Multi-Head Self-Attention。
/// 位置 i は位置 j <= i だけを参照できる(未来を見ない)。
/// 自己回帰生成では推論時に未来が存在しないので、学習時にも
/// 同じ条件にしないと「未来を覗いて当てる」学習をしてしまう。
#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    w_q: Vec<Linear<B>>,
    w_k: Vec<Linear<B>>,
    w_v: Vec<Linear<B>>,
    w_o: Linear<B>,
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn new(device: &B::Device) -> Self {
        Self::new_with_dims(D_MODEL, NUM_HEADS, D_HEAD, device)
    }

    /// 寸法を指定して構築する版。認識モデルのサイズ縮小向けの口
    pub fn new_with_dims(d_model: usize, num_heads: usize, d_head: usize, device: &B::Device) -> Self {
        let mut w_q = Vec::new();
        let mut w_k = Vec::new();
        let mut w_v = Vec::new();
        for _ in 0..num_heads {
            w_q.push(LinearConfig::new(d_model, d_head).init(device));
            w_k.push(LinearConfig::new(d_model, d_head).init(device));
            w_v.push(LinearConfig::new(d_model, d_head).init(device));
        }
        let w_o = LinearConfig::new(d_model, d_model).init(device);
        Self { w_q, w_k, w_v, w_o }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let seq_len = x.dims()[1];
        let device = x.device();

        // 下三角行列(行=query位置, 列=key位置): j <= i の位置だけ 1.0
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = 1.0;
            }
        }
        let causal_mask = Tensor::<B, 1>::from_floats(mask_data.as_slice(), &device)
            .reshape([seq_len, seq_len])
            .unsqueeze::<3>(); // [1, seq, seq] バッチ次元へ broadcast

        let mut head_outputs = Vec::new();
        // ヘッド数はフィールド長から。寸法可変化に伴い定数参照をやめた
        for head_idx in 0..self.w_q.len() {
            let q = self.w_q[head_idx].forward(x.clone());
            let k = self.w_k[head_idx].forward(x.clone());
            let v = self.w_v[head_idx].forward(x.clone());

            // d_head はテンソルの最後の次元(q.dims()[2])から取る。定数参照だと寸法可変化に耐えないため。
            // q は matmul に値渡しで消費されるので、先に dims() を読んでおく
            let d_head = q.dims()[2];
            let scores = q.matmul(k.transpose()) / (d_head as f32).sqrt();

            // 未来(マスクが0)の位置に大きな負の値 → softmax 後ほぼ 0
            let neg = Tensor::ones_like(&scores) * (-1e9);
            let scores = scores.mask_where(causal_mask.clone().equal_elem(0), neg);

            let attention_weights = burn::tensor::activation::softmax(scores, 2);
            head_outputs.push(attention_weights.matmul(v));
        }

        let concat = Tensor::cat(head_outputs, 2);
        self.w_o.forward(concat)
    }
}

// ===== PoseEncoder =====

/// 手ポーズ列のエンコーダ。
/// Embedding の代わりに Linear で 126次元 → d_model に射影する。
/// (Embedding は「離散IDの表引き」、Linear は「連続値ベクトルの線形変換」。
///  ポーズは連続値なので Linear が対応物になる)
#[derive(Module, Debug)]
pub struct PoseEncoder<B: Backend> {
    input_projection: Linear<B>,
    encoder_blocks: Vec<TransformerBlock<B>>,
}

impl<B: Backend> PoseEncoder<B> {
    /// 設定駆動の構築。認識モデルのサイズ縮小はここで寸法を渡すことで実現する
    pub fn new(config: &RecognitionModelConfig, device: &B::Device) -> Self {
        let input_projection = LinearConfig::new(POSE_FEATURE_DIM, config.d_model).init(device);
        let mut encoder_blocks = Vec::new();
        for _ in 0..config.num_layers {
            encoder_blocks.push(TransformerBlock::new_with_dims(
                config.d_model,
                config.num_heads,
                config.d_head,
                config.d_ff,
                device,
            ));
        }
        Self {
            input_projection,
            encoder_blocks,
        }
    }

    /// pose: [batch, frames, POSE_FEATURE_DIM] → [batch, frames, d_model]
    pub fn forward(&self, pose: Tensor<B, 3>) -> Tensor<B, 3> {
        let projected = self.input_projection.forward(pose);
        let mut x = add_positional_encoding(projected);
        for block in &self.encoder_blocks {
            // フレーム列は全サンプル同じ長さ(build-dict --frames)なのでマスク不要。
            // 可変長対応(パディングマスク)は将来の課題
            x = block.forward(x, None);
        }
        x
    }
}

// ===== RecognitionDecoder =====

/// タグ列を自己回帰生成する Decoder(動的語彙 + 因果マスク付き)
#[derive(Module, Debug)]
pub struct RecognitionDecoder<B: Backend> {
    embedding: Embedding<B>,
    self_attentions: Vec<CausalSelfAttention<B>>,
    cross_attentions: Vec<CustomCrossAttention<B>>,
    feed_forwards: Vec<FeedForward<B>>,
    norms1: Vec<LayerNorm<B>>,
    norms2: Vec<LayerNorm<B>>,
    norms3: Vec<LayerNorm<B>>,
    output_projection: Linear<B>,
}

impl<B: Backend> RecognitionDecoder<B> {
    /// 設定駆動の構築。認識モデルのサイズ縮小はここで寸法を渡すことで実現する
    pub fn new(vocab_size: usize, config: &RecognitionModelConfig, device: &B::Device) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, config.d_model).init(device);
        let mut self_attentions = Vec::new();
        let mut cross_attentions = Vec::new();
        let mut feed_forwards = Vec::new();
        let mut norms1 = Vec::new();
        let mut norms2 = Vec::new();
        let mut norms3 = Vec::new();
        for _ in 0..config.num_layers {
            self_attentions.push(CausalSelfAttention::new_with_dims(
                config.d_model,
                config.num_heads,
                config.d_head,
                device,
            ));
            cross_attentions.push(CustomCrossAttention::new_with_dims(
                config.d_model,
                config.num_heads,
                config.d_head,
                device,
            ));
            feed_forwards.push(FeedForward::new_with_dims(config.d_model, config.d_ff, device));
            norms1.push(LayerNormConfig::new(config.d_model).init(device));
            norms2.push(LayerNormConfig::new(config.d_model).init(device));
            norms3.push(LayerNormConfig::new(config.d_model).init(device));
        }
        let output_projection = LinearConfig::new(config.d_model, vocab_size).init(device);
        Self {
            embedding,
            self_attentions,
            cross_attentions,
            feed_forwards,
            norms1,
            norms2,
            norms3,
            output_projection,
        }
    }

    /// tgt_tokens: [batch, tgt_len] → logits [batch, tgt_len, vocab_size]
    pub fn forward(
        &self,
        tgt_tokens: Tensor<B, 2, Int>,
        encoder_output: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let embedded = self.embedding.forward(tgt_tokens);
        let mut x = add_positional_encoding(embedded);

        // Pre-LN: LayerNorm → Attention/FF → 残差接続(DecoderBlock と同じ構成)。
        // 層数はフィールド長から。寸法可変化に伴い定数参照をやめた
        for i in 0..self.self_attentions.len() {
            // 1. 因果マスク付き Self-Attention
            let normalized1 = self.norms1[i].forward(x.clone());
            let self_attn = self.self_attentions[i].forward(normalized1);
            let residual1 = x + self_attn;

            // 2. Cross-Attention(エンコードされたポーズ列を参照)
            let normalized2 = self.norms2[i].forward(residual1.clone());
            let cross_attn =
                self.cross_attentions[i].forward(normalized2, encoder_output.clone(), None);
            let residual2 = residual1 + cross_attn;

            // 3. Feed-Forward
            let normalized3 = self.norms3[i].forward(residual2.clone());
            let ff = self.feed_forwards[i].forward(normalized3);
            x = residual2 + ff;
        }

        self.output_projection.forward(x)
    }
}

// ===== RecognitionModel(統合) =====

#[derive(Module, Debug)]
pub struct RecognitionModel<B: Backend> {
    encoder: PoseEncoder<B>,
    decoder: RecognitionDecoder<B>,
}

impl<B: Backend> RecognitionModel<B> {
    /// 既定サイズ(現行 const 相当)で構築する薄いラッパ。
    /// サイズ系 CLI フラグを一切指定しない既存の呼び出し元(pose_extractor 経由の推論など)の
    /// 挙動を変えないために残している
    pub fn new(vocab_size: usize, device: &B::Device) -> Self {
        Self::new_with_config(vocab_size, &RecognitionModelConfig::default(), device)
    }

    /// 設定駆動の構築。認識モデルのサイズ縮小はここが入口になる
    pub fn new_with_config(
        vocab_size: usize,
        config: &RecognitionModelConfig,
        device: &B::Device,
    ) -> Self {
        Self {
            encoder: PoseEncoder::new(config, device),
            decoder: RecognitionDecoder::new(vocab_size, config, device),
        }
    }

    /// 訓練時のフォワードパス(Teacher Forcing)
    /// pose: [batch, frames, POSE_FEATURE_DIM]
    /// tgt_tokens: [batch, tgt_len](SOS で始まるタグID列)
    /// 出力: [batch, tgt_len, vocab_size]
    pub fn forward(
        &self,
        pose: Tensor<B, 3>,
        tgt_tokens: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let encoder_output = self.encoder.forward(pose);
        self.decoder.forward(tgt_tokens, encoder_output)
    }

    /// 1 サンプルの手ポーズ列から、確率上位 k 個のタグを返す。
    /// Phase 0a の終了基準「10語で top-5 に正解」の判定に使う。
    /// features: [frames * POSE_FEATURE_DIM] のフラット列
    pub fn predict_topk(
        &self,
        features: &[f32],
        frames: usize,
        vocab: &TagVocabulary,
        k: usize,
        device: &B::Device,
    ) -> Vec<(String, f32)> {
        let pose = Tensor::<B, 1>::from_floats(features, device).reshape([
            1,
            frames,
            POSE_FEATURE_DIM,
        ]);
        let encoder_output = self.encoder.forward(pose);

        // SOS だけを入力し、最初のタグ位置の分布を見る(単語認識なのでこれで十分)
        let sos = [vocab.sos_id() as i32];
        let tgt = Tensor::<B, 1, Int>::from_data(sos.as_slice(), device).reshape([1, 1]);
        let logits = self.decoder.forward(tgt, encoder_output); // [1, 1, vocab]

        let vocab_size = vocab.vocab_size();
        let probs =
            burn::tensor::activation::softmax(logits.reshape([vocab_size]), 0);
        let probs_data: Vec<f32> = probs.to_data().to_vec().unwrap();

        // タグのみ(SOS/EOS/PAD を除外)を確率の高い順に並べる
        let mut ranked: Vec<(usize, f32)> = probs_data
            .iter()
            .copied()
            .enumerate()
            .filter(|(id, _)| *id < vocab.tags.len())
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k);

        ranked
            .into_iter()
            .map(|(id, p)| (vocab.id_to_tag(id).unwrap_or("?").to_string(), p))
            .collect()
    }
}

// ===== 保存 / 読み込み =====

/// 認識モデルと語彙をセットで保存する。
/// 語彙はタグ集合で ID が変わるため、必ずモデルと一緒に保存する。
/// あわせて `model_config.json` にモデルの寸法設定を書き出す。これが無いと、
/// 縮小して保存したモデルを読み込むときに元の構造(層数・d_model など)が分からなくなるため
pub fn save_recognition_model<B: Backend>(
    model: &RecognitionModel<B>,
    vocab: &TagVocabulary,
    config: &RecognitionModelConfig,
    save_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(save_dir)?;
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(save_dir.join("model"), &recorder)
        .map_err(|e| format!("認識モデル保存エラー: {:?}", e))?;
    vocab.save(save_dir)?;

    let config_path = save_dir.join("model_config.json");
    let config_file = std::fs::File::create(&config_path)?;
    serde_json::to_writer_pretty(config_file, config)?;

    println!("認識モデルを保存: {}", save_dir.display());
    Ok(())
}

/// 保存済みの認識モデルと語彙を読み込む。
/// 語彙を先に読まないとモデルの出力次元(vocab_size)が決まらない点に注意。
///
/// `model_config.json` があればそれで構造を復元し、無ければ `Default`(現行 const 相当)に
/// フォールバックする。これにより `model_config.json` を持たない既存モデル
/// (`models/rec_full` 等)もそのまま読める(後方互換)
pub fn load_recognition_model<B: Backend>(
    load_dir: &Path,
    device: &B::Device,
) -> Result<(RecognitionModel<B>, TagVocabulary), Box<dyn std::error::Error>> {
    let vocab = TagVocabulary::load(load_dir)?;

    let config_path = load_dir.join("model_config.json");
    let config = if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)?;
        serde_json::from_str::<RecognitionModelConfig>(&content)?
    } else {
        println!(
            "model_config.json が見つからないため既定サイズにフォールバックします: {}",
            load_dir.display()
        );
        RecognitionModelConfig::default()
    };
    // 実際に器を組むのに使う構成を必ず表示する。CLI 側で表示している構成は
    // 学習経路のものなので、読み込み時はこちらが真実
    println!(
        "モデル構成: d_model={} / num_heads={} / d_head={} / d_ff={} / num_layers={}",
        config.d_model, config.num_heads, config.d_head, config.d_ff, config.num_layers
    );

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model = RecognitionModel::<B>::new_with_config(vocab.vocab_size(), &config, device)
        .load_file(load_dir.join("model"), &recorder, device)
        .map_err(|e| {
            format!(
                "認識モデル読み込みエラー(保存時の構造と一致しません。model_config.json の内容と \
                 --model-size 系フラグの指定を確認してください): {:?}",
                e
            )
        })?;
    println!("認識モデルを読み込み: {}", load_dir.display());
    Ok((model, vocab))
}

// ===== ワンショット推論(CPU)=====

/// 推論専用バックエンド。学習は Wgpu(+Autodiff)だが、動画1本の判定は
/// CPU(NdArray)で十分速く、GPU 初期化も要らない。ここでバックエンドを
/// 固定して、呼び出し側(pose_extractor)が Burn の型を意識しなくて済むようにする。
pub type InferenceBackend = burn::backend::NdArray<f32>;

/// 保存済み認識モデルを CPU(NdArray)で読み込み、手ポーズ列(フラット)から
/// 上位 k タグを返す。`動画→タグ直結パス`(pose_extractor 側)の推論入口。
///
/// - `load_dir`: model.bin + tag_vocab.json を含むディレクトリ
/// - `features`: [frames * POSE_FEATURE_DIM] のフラット列
/// - `frames`: ダウンサンプル後のフレーム数(学習時の SEQ_LEN と揃えること)
/// - `k`: 返す上位件数
pub fn predict_from_features(
    load_dir: &Path,
    features: &[f32],
    frames: usize,
    k: usize,
) -> Result<Vec<(String, f32)>, Box<dyn std::error::Error>> {
    let device = <InferenceBackend as Backend>::Device::default();
    let (model, vocab) = load_recognition_model::<InferenceBackend>(load_dir, &device)?;
    Ok(model.predict_topk(features, frames, &vocab, k, &device))
}

// ===== テスト =====
//
// 学習経路の Autodiff<NdArray> は burn-ndarray 0.18.0 の mask_where が現モデル形状で
// パニックする既知バグがあるため使わない(CLAUDE.md 参照)。ここでは NdArray バックエンドで
// forward のみを検証し、「サイズ縮小しても save/load 往復で同じ構造が復元されること」と
// 「model_config.json が無い既存モデルでも Default にフォールバックして読めること」(後方互換)
// を確認する
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    type TestBackend = InferenceBackend;

    fn test_dir(name: &str) -> PathBuf {
        let dir = PathBuf::from(format!("tests/temp_recognition_{}", name));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).ok();
        }
        dir
    }

    fn cleanup(dir: &PathBuf) {
        if dir.exists() {
            std::fs::remove_dir_all(dir).ok();
        }
    }

    #[test]
    fn tiny_config_save_load_roundtrip_predicts() {
        let device = <TestBackend as Backend>::Device::default();
        let dir = test_dir("tiny_roundtrip");

        let vocab = TagVocabulary::from_labels(vec!["a".to_string(), "b".to_string()]);
        let config = RecognitionModelConfig::preset("tiny").unwrap();
        let model =
            RecognitionModel::<TestBackend>::new_with_config(vocab.vocab_size(), &config, &device);

        save_recognition_model(&model, &vocab, &config, &dir).expect("保存に失敗");
        assert!(
            dir.join("model_config.json").exists(),
            "model_config.json が書き出されていません"
        );

        let (loaded_model, loaded_vocab) =
            load_recognition_model::<TestBackend>(&dir, &device).expect("読み込みに失敗");
        assert_eq!(loaded_vocab.tags, vocab.tags);

        // tiny 構成(d_model=16 など)で復元された器に対して forward が通ることを確認
        let frames = 10;
        let features = vec![0.0f32; frames * POSE_FEATURE_DIM];
        let preds = loaded_model.predict_topk(&features, frames, &loaded_vocab, 2, &device);
        assert_eq!(preds.len(), 2);

        cleanup(&dir);
    }

    #[test]
    fn missing_model_config_falls_back_to_default() {
        let device = <TestBackend as Backend>::Device::default();
        let dir = test_dir("fallback_default");

        // Default(= 現行 const 相当)で保存したあと model_config.json を意図的に消し、
        // 「構造情報を持たない既存モデル」(models/rec_full 等)を模擬する
        let vocab = TagVocabulary::from_labels(vec!["x".to_string()]);
        let config = RecognitionModelConfig::default();
        let model =
            RecognitionModel::<TestBackend>::new_with_config(vocab.vocab_size(), &config, &device);
        save_recognition_model(&model, &vocab, &config, &dir).expect("保存に失敗");
        std::fs::remove_file(dir.join("model_config.json")).expect("model_config.json の削除に失敗");

        let (loaded_model, loaded_vocab) =
            load_recognition_model::<TestBackend>(&dir, &device).expect("読み込みに失敗");

        let frames = 10;
        let features = vec![0.0f32; frames * POSE_FEATURE_DIM];
        let preds = loaded_model.predict_topk(&features, frames, &loaded_vocab, 1, &device);
        assert_eq!(preds.len(), 1);

        cleanup(&dir);
    }
}
