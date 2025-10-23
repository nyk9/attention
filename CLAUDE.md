# Transformer/Attention プロジェクト

## プロジェクト構成

### 1. transformer_burn - 手話翻訳AI
- 日本語→手話タグへの可変長翻訳（Seq2Seq）
- **ステータス**: Phase 15b完了、Phase 16（モデル保存・推論基盤）90%完了
- **語彙**: 168（日本語86 + 手話タグ80 + SOS/EOS/PAD）
- **データ**: 47サンプル

### 2. translator_ja_en - 日英翻訳AI
- 日本語→英語の機械翻訳（手話データ準備期間中の学習継続）
- **ステータス**: 初期セットアップ完了
- **語彙**: 未定（日本語 + 英語 + 特殊トークン）
- **データ**: 準備中（50-100サンプル予定）

---

## 技術スタック

- **言語**: Rust
- **フレームワーク**: Burn 0.18.0
- **バックエンド**: Wgpu（GPU）、Autodiff（自動微分）、NdArray（CPU）
- **アーキテクチャ**: Seq2Seq Transformer（Encoder-Decoder、Pre-LN方式）
- **最適化**: Adam

### 現在のモデル設定
- **モデル次元**: d_model=16、2ヘッド（d_head=8）、d_ff=64
- **層数**: Encoder 4層、Decoder 4層
- **シーケンス長**: 10トークン（初期）
- **訓練設定**: 学習率0.0005、バッチサイズ128

---

## コミュニケーション指針（重要）

### 言語とスタイル
- 必ず日本語で回答
- 論理的で簡潔、実践的で実行可能
- 絵文字を使わない

### コード指示
- **行数を明示**: 「XXX行目に追加」「YYY行目を修正」と具体的に指定
- **Rust初心者を考慮**: 所有権、トレイト、ライフタイムなど分かりにくい部分は丁寧に説明
- **修正前後を明示**: 変更箇所を明確に提示
- **複数ファイル**: ファイル名と行数を両方明記

### ユーザー背景
- Next.js、TypeScript、Supabaseでの開発経験あり
- **Rust初心者**: 基本文法は理解、細かい仕様は学習中
- UI/UX設計、社会的価値を重視

---

## Phase 15b: Seq2Seq翻訳モデル（完了）

### 実装内容
- ✓ Encoder-Decoder統合（各4層）
- ✓ Self-Attention、Cross-Attention
- ✓ 可変長出力（SOS/EOSトークン、自己回帰生成）
- ✓ Teacher Forcing訓練

### 実装結果
- 訓練: 100エポック、Loss 15.3 → 1.3
- 推論例: 「ありがとう」→ `<ありがとう>`（正解）
- 制約: 小規模モデル（d_model=16）、少量データ（47サンプル）

---

## Phase 16: モデル保存・推論基盤（90%完了）

### 実装済み機能
- ✅ モデル保存/読み込み（BinFileRecorder、バイナリ形式）
- ✅ クロスプラットフォーム推論（WGPU/NdArray切り替え、autoモード）
- ✅ CLIフラグ（--train, --save, --load, --predict, --backend, --export-attn）
- ✅ メタデータ管理（config.json, metrics.json, README.md自動生成）
- ✅ CSVエクスポート機能（Attention行列、テンソル出力）
- ✅ テストコード（往復一致性、クロスプラットフォーム）

### 出力ディレクトリ構成
```
models/<timestamp>/
├── model.bin       # モデル本体（Burnバイナリ）
├── config.json     # ハイパーパラメータ、語彙情報
├── metrics.json    # 訓練統計、損失履歴
├── README.md       # 訓練メモ（自動生成）
└── exports/        # オプション：分析用CSV
```

### 未実装
- Attention行列の捕捉（モデル変更が必要）
- オプティマイザ状態の保存（継続訓練用）

---

## CLI使用例

### 基本コマンド
```bash
# 訓練
cargo run --release -- --train --save models/run001

# 推論
cargo run --release -- --load models/run001 --predict "こんにちは"

# バックエンド選択（auto: WGPU→NdArrayフォールバック）
cargo run --release -- --load models/run001 --backend auto --predict "ありがとう"

# Attention行列CSV出力
cargo run --release -- --load models/run001 --predict "おはよう" --export-attn
```

---

## transformer_burn - 次のステップ

- [ ] テストコード実行確認
- [ ] 実際に訓練を実行してモデル保存を検証
- [ ] Phase 16a（長いシーケンス対応）またはPhase 16b（モデルスケールアップ）へ進む

---

## translator_ja_en - 次のステップ

### Phase 1（基盤構築）
- [ ] 日英対訳データセット準備（50-100サンプル、TSV形式）
- [ ] 語彙モジュール実装（translation_vocabulary.rs）
  - 日本語トークナイザー（1文字単位）
  - 英語トークナイザー（単語単位、小文字化、句読点処理）
  - 分離語彙空間（src_vocab, tgt_vocab）
- [ ] データローダー実装（translation_data.rs）
- [ ] 設定ファイル更新（config.rs）
- [ ] 初回訓練実行（動作確認）

### データセット例（data/translation_data_ja_en.txt）
```
こんにちは	Hello
ありがとう	Thank you
おはようございます	Good morning
私は学生です	I am a student
```

### Phase 2以降
- Phase 2: シーケンス長拡張（20-50トークン）
- Phase 3: モデルスケールアップ（d_model=64-128、4-8ヘッド）
- Phase 4: 実用化（BLEU評価、ビーム探索）

---

**最終更新**: 2025年10月23日
