# Transformer/Attention 手動実装プロジェクト

## プロジェクト概要

日本語手話通訳アプリの開発に向けた、TransformerとAttentionメカニズムの学習プロジェクト。

**現在**: Phase 15b完了、次のフェーズ検討中
**プロジェクト**: `transformer_burn`（Burn 0.18.0使用）

## 技術スタック

- **言語**: Rust
- **フレームワーク**: Burn 0.18.0
- **バックエンド**: Wgpu（GPU）、Autodiff（自動微分）
- **最適化**: Adam

## 現在のモデル設定

- **語彙サイズ**: 168（日本語86 + 手話タグ80 + PAD + SOS/EOS）
- **シーケンス長**: 10
- **モデル次元**: d_model=16、2ヘッド（d_head=8）、d_ff=32
- **層数**: 4層Transformerブロック
- **アーキテクチャ**: Pre-LN方式、残差接続

---

## Phase 15b: Seq2Seq翻訳モデル（完了）

**目標**: 日本語→手話タグへの可変長翻訳（Encoder-Decoder）

**実装内容**:

- ✓ Encoder層（Self-Attention + FF + Positional Encoding）
- ✓ Cross-Attention層（Query: Decoder、Key/Value: Encoder）
- ✓ DecoderBlock（Self-Attention → Cross-Attention → FF）
- ✓ Seq2SeqModel（Encoder-Decoder統合）
- ✓ 可変長出力（SOS/EOSトークン、自己回帰生成）
- ✓ データ形式変更（固定5タグ → 可変長シーケンス）
- ✓ Teacher Forcing訓練
- ✓ 自己回帰推論

### 実装結果

**アーキテクチャ**:

- Encoder: 4層（Self-Attention + FF + LayerNorm）
- Decoder: 4層（Self-Attention + Cross-Attention + FF + LayerNorm）
- Pre-LN方式、残差接続
- マルチヘッドAttention（2ヘッド）

**動作確認**:

- 訓練: 100エポック、Loss 15.3 → 1.3
- 推論例:
  - 「わたしはたべます」→ `<私> <歩く>`（ほぼ正解）
  - 「ありがとう」→ `<ありがとう>`（正解）
  - 「おはようございます」→ `<今> <暑い>`（訓練データ外）

**制約**:

- 小規模モデル（d_model=16）
- 少量データ（47サンプル）
- 短いシーケンス（10トークン）

### 実装進捗

- [x] より深いモデル（3-4層）への拡張
- [x] 可変長シーケンス（パディングとマスキング）
- [x] モデル次元の拡大（d_model=16）
- [x] Burn移行（バッチ処理で8.6倍、GPU最適化で15.5倍高速化達成）
- [x] **Phase 15a**: 日本語→手話タグ翻訳（デコーダーのみ、固定5タグ出力）
- [x] **Phase 15b**: Seq2Seq翻訳モデル（エンコーダー・デコーダー、可変長出力）
- [ ] より長いシーケンス（20-50文字）への拡張
- [ ] 大規模モデル（d_model=64-256）への拡張

---

## 次のフェーズ候補

### Phase 16a: より長いシーケンス（20-50文字）

- 長文対応
- 位置エンコーディングの拡張
- メモリ効率の最適化

### Phase 16b: 大規模モデル（d_model=64-256）

- モデル次元の拡大
- ヘッド数の増加（4-8ヘッド）
- 層数の増加（6-12層）
- 訓練データの拡充

---

## ユーザー背景

- Next.js、TypeScript、Tailwind CSS、Supabaseでの開発経験
- **Rust初心者**: 基本文法は理解しているが、細かい仕様は学習中
- UI/UX設計、社会的価値、ケア・ステークホルダー視点を重視

---

## コミュニケーション指針（重要）

### YOU MUST follow these communication rules:

1. **言語**: 必ず日本語で回答すること

2. **スタイル**:
   - 論理的で簡潔、かつ実践的で実行可能
   - 段階的な説明を重視
   - 形式的だが過度に堅苦しくない
   - 絵文字を使わないこと

3. **コード指示の出し方**（CRITICAL）:
   - **必ず行数を明示する**: 「XXX行目に追加」「YYY行目を修正」と具体的に指定
   - **Rust初心者を考慮**: タイプ、所有権、トレイトなど、分かりにくい部分は丁寧に説明
   - **修正前後を明示**: 変更箇所が明確に分かるように提示
   - **複数ファイルの場合**: ファイル名と行数を両方明記

4. **不確実性**: 分からないことや限界は率直に認める

5. **絵文字**: 使用しない

### 指示の例

```
src/attention.rs の 215行目の後に追加:

修正前（103行目）:
pub fn create_output_weight() -> Matrix {

修正後:
fn create_output_weight() -> Matrix {
```

### 対話方針

- アルゴリズムの考え方、実装方針、デバッグのアプローチを説明
- 具体的なコードではなく、疑似コードや概念図で説明
- 行き詰まった際の問題解決をサポート
- 建設的で論理的なフィードバックを提供
- Rust特有の概念（借用、ライフタイム、トレイト）は必要に応じて補足説明

---

## Phase 16: モデル重みの保存・出力とクロスプラットフォーム推論

### 目的（What/Why）

- WindowsデスクトップPCのGPUで効率よく訓練し、MacBook Air M2で即座に推論・検証できる体制を整える。
- 学習済みモデルの重み（パラメータ行列）を保存・共有し、推論時に再現可能な形で読み込めるようにする。
- 学習・推論時の「重み/注意（Attention）行列」を必要に応じてエクスポートし、学習の可視化・分析を可能にする。

### 重み管理の方針

#### 保存形式の使い分け

| 用途                          | 形式                  | 理由                 |
| ----------------------------- | --------------------- | -------------------- |
| **モデル本体の保存/読み込み** | Burnバイナリ (`.bin`) | 軽量・高速・精度保持 |
| **設定・メタデータ**          | JSON                  | 人間可読・互換性     |
| **デバッグ・分析**            | CSV（オプション）     | 可視化・統計解析用   |
| **人間可読メモ**              | Markdown              | 訓練記録・再現手順   |

**CSV形式の位置づけ**:

- モデル全体の保存には**使用しない**（ファイルサイズ・速度・精度の問題）
- 小規模な分析・可視化用途に限定（特定層の重み、Attention行列など）
- デフォルトでは出力しない（フラグ有効時のみ）

**業界標準との比較**:

- Hugging Face: `safetensors` (安全・高速・言語間互換)
- PyTorch: `.bin` / `.pth` (pickle形式)
- llama.cpp: `GGUF` (量子化対応)
- Burn: `BinBytesRecorder` (バイナリ)、`PrettyJsonRecorder` (JSON)

### スコープ（Scope）

1. モデル重みの保存/読み込み
   - Burn 0.18の `BinBytesRecorder` を使用してモデル全体を保存/読み込み。
   - オプティマイザ（Adam）の状態は任意保存（継続訓練時のみ）。
   - 出力ディレクトリ構成:
     ```
     models/jsl-seq2seq/<timestamp>/
     ├── model.bin              # モデル本体（バイナリ）
     ├── config.json            # ハイパーパラメータ、語彙情報
     ├── metrics.json           # 訓練統計、重み統計
     ├── README.md              # 訓練メモ（自動生成）
     └── exports/               # オプション：分析用CSV
         ├── attn_*.csv         # Attention重み
         └── weights_*.csv      # 特定層の重み
     ```

2. バックエンド切替とクロスプラットフォーム推論
   - 実行時に `auto|wgpu|ndarray` を選択できるCLIフラグを追加。
   - `auto`: WGPUが利用可能ならWGPU、不可ならNdarrayを選択（MacではMetal/WGPUが期待、fallbackはNdarray）。
   - 訓練（Windows）: `wgpu+autodiff`。 推論（Mac）: `auto`（優先: WGPU→Ndarray）。

3. 重み・行列の分析用エクスポート（オプション機能）
   - **重み統計の自動出力**:
     - 全レイヤ（Embedding, Linear, LayerNorm）の統計値（mean/std/min/max）を `metrics.json` に保存。
     - 訓練後に自動実行、推論時は不要。
   - **CSV エクスポート**（フラグ有効時のみ）:
     - 特定層の重み: `--export-weights <layer_name>` で指定層をCSV出力。
     - Attention行列: `--export-attn` で推論時の注意分布をCSV出力（seq_len < 20のみ推奨）。
   - **用途**: ヒートマップ可視化、異常値検出、学習過程の分析。

4. 再現性・メタデータ
   - `config.json` に以下を保存:
     - ハイパーパラメータ（d_model, n_heads, n_layers, learning_rate等）
     - 語彙情報（vocab_size, vocab_hash）
     - 環境情報（Burnバージョン、乱数シード、訓練日時）
   - `README.md` の自動生成:
     - 訓練設定、最終Loss、推論例、実行コマンドを記録。
     - Git commit hashも記録（再現性確保）。

### 具体的成果物（Deliverables）

#### CLIインターフェース

- `--save <DIR>`: 訓練後に `<DIR>` へモデル・設定・統計を保存。
- `--load <DIR>`: 既存モデルを読み込んで推論/再訓練。
- `--backend <auto|wgpu|ndarray>`: 実行時バックエンド切替。
- `--export-weights <layer_name>`: 指定層の重みをCSVエクスポート（オプション）。
- `--export-attn`: 推論時Attention行列をCSVエクスポート（オプション、短尺入力のみ）。

#### 出力ファイル構成例

```
models/jsl-seq2seq/20251022_120000/
├── model.bin                          # モデル本体（Burnバイナリ形式）
├── config.json                        # 設定・メタデータ
├── metrics.json                       # 訓練曲線・重み統計
├── README.md                          # 訓練メモ（自動生成）
└── exports/                           # オプション：分析用CSV
    ├── attn_layer1_head0.csv          # Self-Attention（Layer 1, Head 0）
    ├── attn_cross_layer2_head1.csv    # Cross-Attention（Layer 2, Head 1）
    └── decoder_output_weight.csv      # 出力射影層の重み
```

#### config.json 例

```json
{
  "model": {
    "d_model": 16,
    "n_heads": 2,
    "n_layers": 4,
    "d_ff": 32,
    "vocab_size": 168,
    "max_seq_len": 10
  },
  "training": {
    "learning_rate": 0.001,
    "epochs": 200,
    "batch_size": 8,
    "optimizer": "Adam"
  },
  "metadata": {
    "burn_version": "0.18.0",
    "vocab_hash": "abc123...",
    "seed": 42,
    "trained_at": "2025-10-22T12:00:00Z",
    "git_commit": "a1b2c3d"
  }
}
```

### 実装方針（Design）

#### 1. シリアライズ（Burn Recorder API）

```rust
use burn::record::{FullPrecisionSettings, BinBytesRecorder, Recorder};

// 保存
let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
model
    .save_file("models/jsl-seq2seq/20251022_120000/model", &recorder)
    .expect("保存失敗");

// 読み込み
let model = Seq2SeqModel::load_file(
    "models/jsl-seq2seq/20251022_120000/model",
    &recorder,
    &device,
)
.expect("読み込み失敗");
```

- オプティマイザ状態は継続訓練時のみ保存（初回推論では不要）。
- `FullPrecisionSettings` でf32精度を保持。

#### 2. バックエンド抽象化

- 訓練: `type TrainingBackend = Autodiff<Wgpu>` 固定。
- 推論: `B: Backend` トレイト境界で汎用化。
- 実行時選択:
  ```rust
  match backend_flag {
      "wgpu" => run_inference::<Wgpu>(...),
      "ndarray" => run_inference::<NdArray>(...),
      "auto" => {
          if wgpu_available() {
              run_inference::<Wgpu>(...)
          } else {
              run_inference::<NdArray>(...)
          }
      }
  }
  ```

#### 3. Attention行列の捕捉（オプション機能）

- `CustomMultiHeadAttention` / `CustomCrossAttention` に `AttnHook` トレイトを導入。
- フラグ有効時のみ、Softmax後のテンソルをCPUへ転送。
- `seq_len > 20` の場合は警告を出して出力をスキップ（I/O負荷軽減）。
- CSV形式: `row=query_pos, col=key_pos, value=attention_weight`

#### 4. 重み統計の自動計算

- 訓練後に全レイヤを走査し、各パラメータテンソルの統計値を計算:
  ```rust
  fn compute_param_stats<B: Backend>(tensor: &Tensor<B, 2>) -> ParamStats {
      ParamStats {
          mean: tensor.mean().into_scalar(),
          std: tensor.std().into_scalar(),
          min: tensor.min().into_scalar(),
          max: tensor.max().into_scalar(),
      }
  }
  ```
- `metrics.json` に以下の構造で保存:
  ```json
  {
    "layers": {
      "encoder.embed.weight": {"mean": 0.02, "std": 0.15, ...},
      "decoder.self_attn.q_proj.weight": {...}
    }
  }
  ```

### CLI使用例

#### Windows（訓練＋保存）

```bash
# 基本訓練（モデル・設定・統計を保存）
cargo run --release -p transformer_burn --features "wgpu,autodiff,ndarray" -- \
  --train --epochs 200 --backend wgpu \
  --save models/jsl-seq2seq/$(date +%Y%m%d_%H%M%S)

# 特定層の重みもCSVエクスポート
cargo run --release -p transformer_burn --features "wgpu,autodiff,ndarray" -- \
  --train --epochs 200 --backend wgpu \
  --save models/jsl-seq2seq/$(date +%Y%m%d_%H%M%S) \
  --export-weights decoder.output_projection
```

#### Mac（推論）

```bash
# 基本推論（バックエンド自動選択）
cargo run --release -p transformer_burn --features "wgpu,autodiff,ndarray" -- \
  --load models/jsl-seq2seq/20251022_120000 \
  --backend auto \
  --predict "ありがとう"

# Attention行列も出力（短尺入力のみ推奨）
cargo run --release -p transformer_burn --features "wgpu,autodiff,ndarray" -- \
  --load models/jsl-seq2seq/20251022_120000 \
  --backend auto \
  --predict "ありがとう" \
  --export-attn
```

#### 継続訓練（Mac/Windows共通）

```bash
# 既存モデルから再開
cargo run --release -p transformer_burn --features "wgpu,autodiff,ndarray" -- \
  --load models/jsl-seq2seq/20251022_120000 \
  --train --epochs 100 \
  --save models/jsl-seq2seq/$(date +%Y%m%d_%H%M%S)
```

### テスト計画

| テスト項目                 | 検証内容                            | 合格基準                              |
| -------------------------- | ----------------------------------- | ------------------------------------- |
| **往復一致性**             | 保存→読み込み→同一入力で出力一致    | logitsの相対誤差 < 1e-5               |
| **バックエンド一致性**     | Wgpu vs Ndarray で推論結果比較      | 相対誤差 < 1e-4（浮動小数点誤差許容） |
| **CSV出力妥当性**          | Attention行列の形状・値域           | shape=(seq_len, seq_len), sum≈1.0     |
| **JSON妥当性**             | config.json / metrics.json のパース | エラーなくデシリアライズ可能          |
| **クロスプラットフォーム** | Windows訓練→Mac推論                 | 推論成功、結果一致                    |

#### テストコード例

```rust
#[test]
fn test_checkpoint_roundtrip() {
    let model = create_model();
    let input = create_test_input();

    // 保存前の出力
    let output_before = model.forward(input.clone());

    // 保存→読み込み
    save_model(&model, "test_checkpoint");
    let loaded_model = load_model("test_checkpoint");

    // 読み込み後の出力
    let output_after = loaded_model.forward(input);

    // 近似一致を検証
    assert_tensors_close(&output_before, &output_after, 1e-5);
}
```

### リスクと対策

| リスク                   | 影響                  | 対策                                  |
| ------------------------ | --------------------- | ------------------------------------- |
| **GPU/CPU数値誤差**      | 推論結果の微小な差異  | 許容閾値設定（1e-4）、Softmax安定化   |
| **大規模CSV出力**        | ディスク容量・I/O遅延 | seq_len制限、デフォルトオフ、警告表示 |
| **Burnバージョン不一致** | モデル読み込み失敗    | config.jsonにバージョン記録、CI検証   |
| **メタデータ欠損**       | 再現性喪失            | 保存時に全メタデータ強制出力          |
| **ファイル破損**         | 読み込みエラー        | チェックサム検証（将来拡張）          |

### マイルストーン

| Phase    | タスク             | 所要時間 | 成果物                               |
| -------- | ------------------ | -------- | ------------------------------------ |
| **16.1** | Recorder API統合   | 0.5日    | `--save/--load` 実装                 |
| **16.2** | バックエンド切替   | 0.5日    | `--backend` フラグ、推論経路分離     |
| **16.3** | 重み統計出力       | 0.5日    | `metrics.json` 自動生成              |
| **16.4** | CSV エクスポート   | 1日      | `--export-weights/--export-attn`     |
| **16.5** | メタデータ・README | 0.5日    | `config.json`, `README.md` 自動生成  |
| **16.6** | テスト・検証       | 1日      | 往復試験、クロスプラットフォーム検証 |

**合計見積もり**: 4日

### 実装チェックリスト

- [ ] `BinBytesRecorder` でモデル保存/読み込み
- [ ] CLIフラグ（`--save`, `--load`, `--backend`, `--export-*`）
- [ ] バックエンド自動選択ロジック（`auto`）
- [ ] 重み統計計算（mean/std/min/max）
- [ ] Attention行列の捕捉とCSV出力
- [ ] `config.json` / `metrics.json` 生成
- [ ] `README.md` 自動生成
- [ ] 往復一致性テスト
- [ ] バックエンド一致性テスト
- [ ] Windows→Mac クロスプラットフォーム検証

---

**Note**: このファイルはリビングドキュメントとして、実装の進捗に応じて更新する。
**最終更新**: Phase 16詳細計画更新（2025年10月22日）
