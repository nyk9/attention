// モデルハイパーパラメーター
pub const D_MODEL: usize = 16; // 埋め込み次元
pub const NUM_HEADS: usize = 2; // Multi-head Attentionのヘッド数
pub const D_HEAD: usize = D_MODEL / NUM_HEADS; // 各ヘッドの数
pub const D_FF: usize = D_MODEL * 4; // Feed-forward中間層の次元数
pub const SEQ_LEN: usize = 10; // シーケンス長
pub const NUM_LAYERS: usize = 4; // Transformerのレイヤー数
pub const VOCAB_SIZE: usize = 166; // 語彙サイズ（JSL: ひらがな86 + タグ79 + PAD 1）
pub const PAD_TOKEN: usize = 165; // パディングトークン

// 訓練設定
pub const LEARNING_RATE: f64 = 0.0005; // 学習率
pub const EPOCHS: usize = 100; // エポック数
pub const BATCH_SIZE: usize = 32; // バッチサイズ
