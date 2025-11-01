# 日英翻訳 Seq2Seq モデル訓練結果

## モデル設定

- **d_model**: 128
- **ヘッド数**: 4
- **Encoderレイヤ数**: 4
- **Decoderレイヤ数**: 4
- **d_ff**: 512
- **日本語語彙サイズ**: 168
- **英語語彙サイズ**: 798
- **日本語シーケンス長**: 30
- **英語シーケンス長**: 30

## 訓練設定

- **エポック数**: 150
- **学習率**: 0.0005
- **バッチサイズ**: 128
- **オプティマイザ**: Adam
- **最終Loss**: 0.162247

## 訓練情報

- **訓練日時**: 2025-11-01T13:11:12.202656+09:00
- **Burn バージョン**: 0.1.0

## 使用方法

### 推論（WGPU）

```bash
cargo run --release -- \
  --load models/test \
  --backend wgpu \
  --predict "こんにちは"
```

### 推論（NdArray / CPU）

```bash
cargo run --release -- \
  --load models/test \
  --backend ndarray \
  --predict "こんにちは"
```

### 推論（自動選択）

```bash
cargo run --release -- \
  --load models/test \
  --backend auto \
  --predict "こんにちは"
```

## ファイル構成

- `model.bin`: モデル重み（Burnバイナリ形式）
- `config.json`: モデル設定とハイパーパラメータ
- `metrics.json`: 訓練統計と損失履歴
- `README.md`: このファイル
