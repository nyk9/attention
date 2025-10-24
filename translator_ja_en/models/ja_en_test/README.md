# 日英翻訳 Seq2Seq モデル訓練結果

## モデル設定

- **d_model**: 16
- **ヘッド数**: 2
- **Encoderレイヤ数**: 4
- **Decoderレイヤ数**: 4
- **d_ff**: 64
- **日本語語彙サイズ**: 200
- **英語語彙サイズ**: 500
- **日本語シーケンス長**: 20
- **英語シーケンス長**: 20

## 訓練設定

- **エポック数**: 50
- **学習率**: 0.0005
- **バッチサイズ**: 32
- **オプティマイザ**: Adam
- **最終Loss**: 11.136560

## 訓練情報

- **訓練日時**: 2025-10-24T12:32:19.378598+09:00
- **Burn バージョン**: 0.1.0

## 使用方法

### 推論（WGPU）

```bash
cargo run --release -- \
  --load models/ja_en_test \
  --backend wgpu \
  --predict "こんにちは"
```

### 推論（NdArray / CPU）

```bash
cargo run --release -- \
  --load models/ja_en_test \
  --backend ndarray \
  --predict "こんにちは"
```

### 推論（自動選択）

```bash
cargo run --release -- \
  --load models/ja_en_test \
  --backend auto \
  --predict "こんにちは"
```

## ファイル構成

- `model.bin`: モデル重み（Burnバイナリ形式）
- `config.json`: モデル設定とハイパーパラメータ
- `metrics.json`: 訓練統計と損失履歴
- `README.md`: このファイル
