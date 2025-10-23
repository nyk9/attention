# JSL Seq2Seq モデル訓練結果

## モデル設定

- **d_model**: 16
- **ヘッド数**: 2
- **レイヤ数**: 4
- **d_ff**: 64
- **語彙サイズ**: 168
- **最大シーケンス長**: 10

## 訓練設定

- **エポック数**: 10000
- **学習率**: 0.00001
- **バッチサイズ**: 128
- **オプティマイザ**: Adam
- **最終Loss**: 0.050275

## 訓練情報

- **訓練日時**: 2025-10-22T23:08:56.909864+09:00
- **Burn バージョン**: 0.1.0

## 使用方法

### 推論（WGPU）

```bash
cargo run --release -p transformer_burn --features "wgpu,autodiff,ndarray" -- \
  --load models/test \
  --backend wgpu \
  --predict "ありがとう"
```

### 推論（NdArray / CPU）

```bash
cargo run --release -p transformer_burn --features "wgpu,autodiff,ndarray" -- \
  --load models/test \
  --backend ndarray \
  --predict "ありがとう"
```

### 推論（自動選択）

```bash
cargo run --release -p transformer_burn --features "wgpu,autodiff,ndarray" -- \
  --load models/test \
  --backend auto \
  --predict "ありがとう"
```

## ファイル構成

- `model.bin`: モデル重み（Burnバイナリ形式）
- `config.json`: モデル設定とハイパーパラメータ
- `metrics.json`: 訓練統計と損失履歴
- `README.md`: このファイル
