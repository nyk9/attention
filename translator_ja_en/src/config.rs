// モデルハイパーパラメーター
pub const D_MODEL: usize = 64; // 埋め込み次元
pub const NUM_HEADS: usize = 4; // Multi-head Attentionのヘッド数
pub const D_HEAD: usize = D_MODEL / NUM_HEADS; // 各ヘッドの次元数
pub const D_FF: usize = D_MODEL * 4; // Feed-forward中間層の次元数
pub const SRC_SEQ_LEN: usize = 30; // ソース（日本語）シーケンス長
pub const TGT_SEQ_LEN: usize = 30; // ターゲット（英語）シーケンス長
pub const NUM_ENCODER_LAYERS: usize = 4; // Encoderレイヤー数
pub const NUM_DECODER_LAYERS: usize = 4; // Decoderレイヤー数

// 訓練設定
pub const LEARNING_RATE: f64 = 0.00005; // 学習率（Phase 2: モデル大型化で低減）
pub const EPOCHS: usize = 150; // エポック数（633サンプル、過学習防止）
pub const BATCH_SIZE: usize = 128; // バッチサイズ（GPU メモリ許容範囲）
