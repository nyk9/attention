# Transformer/Attention プロジェクト

## プロジェクト構成

### 1. transformer_burn - 手話翻訳AI
- 日本語→手話タグへの可変長翻訳(Seq2Seq)
- **ステータス**: Phase 15b完了、Phase 16(モデル保存・推論基盤)90%完了
- **語彙**: 168(日本語86 + 手話タグ80 + SOS/EOS/PAD)
- **データ**: 47サンプル

### 2. translator_ja_en - 日英翻訳AI
- 日本語→英語の機械翻訳(手話データ準備期間中の学習継続)
- **ステータス**: 初期セットアップ完了
- **語彙**: 未定(日本語 + 英語 + 特殊トークン)
- **データ**: 準備中(50-100サンプル予定)

### 3. elan_parser - ELAN .eaf アノテーション抽出CLI
- ELAN(手話研究のデファクトアノテーションツール)出力の `.eaf` XML から、gloss 列を TSV/JSON 形式で抽出
- **ステータス**: v1完了(`ALIGNABLE_ANNOTATION` 対応、`REF_ANNOTATION` は v2)
- **使用クレート**: quick-xml 0.36, clap 4, serde, anyhow
- **バイナリ**: `elan-eaf-parse`
- **目的**: ろう者協力者から受け取るELANアノテーションを将来のMLパイプラインに繋ぐ前処理
- **Skill連携**: `~/.claude/skills/elan-eaf-parse/SKILL.md` から自動発見

### 4. pose_extractor - 動画→姿勢ランドマーク抽出CLI
- BlazePose ONNX(MediaPipe Pose 変換版)を `ort` クレートで呼び、動画フレームから39 keypoints(33+6補助)を抽出
- **ステータス**: Phase 0a-pipe-v1 の S1-S6(機能側)完了。対話ウィザード CLI 化 + バッチ処理対応済み。**精度上の課題が S6 検証で発見(下記参照)**
- **使用クレート**: ort 2.0.0-rc.12, ndarray 0.16, image 0.25, clap 4, dialoguer 0.11, serde, anyhow
- **外部依存**: ffmpeg 8.1.1(`brew install ffmpeg`、CLI subprocess 経由で動画読み込み)
- **モデル**: `models/blazepose_full.onnx`(5.3MB、HF `opencv/pose_estimation_mediapipe`)。git管理外
- **バイナリ**: `pose-extract`(引数なし起動でウィザード。`videos/` から動画選択、単発/バッチ両対応。dev サブコマンド `inspect <model>` / `test-infer <model>`)
- **使い方**: `pose_extractor/CLI.md` 参照(videos/ ディレクトリに動画を置く運用)
- **入力仕様**: NHWC (1, 256, 256, 3) float32 [0, 1] RGB(中央クロップ + 256×256 リサイズ、S6 検証後に stretch から変更)
- **出力仕様**: landmarks (1, 195) = 39 × [x, y, z, visibility, presence] + conf (1, 1) + その他3個
- **TSV出力**: wide 形式(1 行 = 1 フレーム、197 列: frame_idx, confidence, x0..pres38)

#### S6 検証で発見した問題

1920×1080 の手話動画 6 本(各 4-9 秒)をバッチ実行した結果:

- **stats 上は良好**: 全 6 本で confidence mean = 0.82-0.90
- **しかしオーバーレイで確認すると、39 ランドマークが顔と襟元の極狭い領域に集中** していて、手・肩・腰・膝・足の本来のランドマークが全く見えない(手のジェスチャーをしているフレームでも手の位置にドットが一切ない)

原因(推定): BlazePose の **Pose Landmark モデルは「人物がクロップされた正方形画像」を前提**。MediaPipe 本来のパイプラインは Pose Detection で bbox を取って人物クロップしてから Landmark に渡す 2 段構え。現状は後半のみを使い、1920×1080 をいきなり 256×256 に stretch しているので、モデルから見ると「全身が崩れた変な人物」になる。顔だけは確実に検出できるので confidence 数値だけ高く出る。

#### 改善案と採用方針(2026-05-23 決定)

| 案 | 工数 | 手話用途への適合 | 採用 |
|---|---|---|---|
| A. 中央クロップ(1080×1080 → 256×256) | 小 | 上半身は改善、脚は不可視のまま | **採用(第1段)** |
| B. Pose Detection モデル追加(2段パイプライン) | 中 | MediaPipe 本来のフロー | 不採用(手指が取れない) |
| C. MediaPipe Hands ONNX 併用(手指 21点×2) | 中-大 | 手話に最重要な手指が取れる | **採用(第2段)** |
| D. MediaPipe Holistic(Pose + Hands + Face) | 大 | 表情・手指・上半身姿勢を統合 | 将来採用予定 |

- **採用方針**: 段階的に A → C → 将来 D
  - 第1段: A の中央クロップで BlazePose の上半身ランドマークを正しく取れるようにする
  - 第2段: C で MediaPipe Hands ONNX を追加し、手指 21点×2手を併用
  - 将来: A+C が安定したら D の Holistic 統合(表情・微妙な手指・上半身姿勢を一体化)へ
- **進捗(2026-06-04 時点)**:
  - A: 実装完了。confidence は向上したが身体ランドマークは破綻したまま(人物が正方形内で小さい)
  - C-1(Palm Detection): 完了。SSD anchors 2016 を埋め込み、両手検出可能
  - C-2(Hand Landmark): 完了。21 keypoints/手を取得。回転アラインメントは未実装のため指関節の精度は MediaPipe 純正よりやや劣る
  - S7(transformer_burn 連携・生成補強方向): `build-dict` サブコマンド実装完了。`<タグ名>.mp4` 群からタグ→ハンドポーズ辞書(JSON)を構築。1 フレーム=126 次元(左右の手 21点×xyz、手のみ。身体ランドマークは破綻のため不採用)、`--frames` でダウンサンプル(既定 10=SEQ_LEN)。座標は width/height で正規化。既存動画 1 本でスモークテスト済み(coverage 約 62%)
- **S7-8 の方向性(2026-06-04 決定)**: 認識(動画→タグ)ではなく**生成補強**を採用。現行の日本語→手話タグ翻訳(transformer_burn)の出力タグに、`build-dict` で作る参照ポーズ列を紐付ける。モデル構造変更は不要(辞書引き)
- **次のステップ**:
  - タグ名動画の用意(`<タグ名>.mp4`、1動画=1タグ、ELAN不要)→ 辞書構築。数タグから段階的に
  - S8: transformer_burn 推論パスに `tag_pose_dict.json` 読み込みを追加し、出力タグ列に `pose_sequence` を付与
  - 任意: Hand Landmark の回転アラインメント実装で指関節精度向上

---

## 技術スタック

- **言語**: Rust
- **フレームワーク**: Burn 0.18.0
- **バックエンド**: Wgpu(GPU)、Autodiff(自動微分)、NdArray(CPU)
- **アーキテクチャ**: Seq2Seq Transformer(Encoder-Decoder、Pre-LN方式)
- **最適化**: Adam

### 現在のモデル設定
- **モデル次元**: d_model=16、2ヘッド(d_head=8)、d_ff=64
- **層数**: Encoder 4層、Decoder 4層
- **シーケンス長**: 10トークン(初期)
- **訓練設定**: 学習率0.0005、バッチサイズ128

---

## コミュニケーション指針(重要)

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

## Phase 15b: Seq2Seq翻訳モデル(完了)

### 実装内容
- ✓ Encoder-Decoder統合(各4層)
- ✓ Self-Attention、Cross-Attention
- ✓ 可変長出力(SOS/EOSトークン、自己回帰生成)
- ✓ Teacher Forcing訓練

### 実装結果
- 訓練: 100エポック、Loss 15.3 → 1.3
- 推論例: 「ありがとう」→ `<ありがとう>`(正解)
- 制約: 小規模モデル(d_model=16)、少量データ(47サンプル)

---

## Phase 16: モデル保存・推論基盤(90%完了)

### 実装済み機能
- ✅ モデル保存/読み込み(BinFileRecorder、バイナリ形式)
- ✅ クロスプラットフォーム推論(WGPU/NdArray切り替え、autoモード)
- ✅ CLIフラグ(--train, --save, --load, --predict, --backend, --export-attn)
- ✅ メタデータ管理(config.json, metrics.json, README.md自動生成)
- ✅ CSVエクスポート機能(Attention行列、テンソル出力)
- ✅ テストコード(往復一致性、クロスプラットフォーム)

### 出力ディレクトリ構成
```
models/<timestamp>/
├── model.bin       # モデル本体(Burnバイナリ)
├── config.json     # ハイパーパラメータ、語彙情報
├── metrics.json    # 訓練統計、損失履歴
├── README.md       # 訓練メモ(自動生成)
└── exports/        # オプション:分析用CSV
```

### 未実装
- Attention行列の捕捉(モデル変更が必要)
- オプティマイザ状態の保存(継続訓練用)

---

## CLI使用例

### 基本コマンド
```bash
# 訓練
cargo run --release -- --train --save models/run001

# 推論
cargo run --release -- --load models/run001 --predict "こんにちは"

# バックエンド選択(auto: WGPU→NdArrayフォールバック)
cargo run --release -- --load models/run001 --backend auto --predict "ありがとう"

# Attention行列CSV出力
cargo run --release -- --load models/run001 --predict "おはよう" --export-attn
```

### elan_parser
```bash
cd elan_parser
cargo build --release
./target/release/elan-eaf-parse tests/fixtures/sample.eaf
./target/release/elan-eaf-parse <input.eaf> --tier <tier_id> --format json
```

---

## transformer_burn - 次のステップ

- [ ] テストコード実行確認
- [ ] 実際に訓練を実行してモデル保存を検証
- [ ] Phase 16a(長いシーケンス対応)またはPhase 16b(モデルスケールアップ)へ進む

---

## translator_ja_en - 次のステップ

### Phase 1(基盤構築)
- [ ] 日英対訳データセット準備(50-100サンプル、TSV形式)
- [ ] 語彙モジュール実装(translation_vocabulary.rs)
  - 日本語トークナイザー(1文字単位)
  - 英語トークナイザー(単語単位、小文字化、句読点処理)
  - 分離語彙空間(src_vocab, tgt_vocab)
- [ ] データローダー実装(translation_data.rs)
- [ ] 設定ファイル更新(config.rs)
- [ ] 初回訓練実行(動作確認)

### データセット例(data/translation_data_ja_en.txt)
```
こんにちは	Hello
ありがとう	Thank you
おはようございます	Good morning
私は学生です	I am a student
```

### Phase 2以降
- Phase 2: シーケンス長拡張(20-50トークン)
- Phase 3: モデルスケールアップ(d_model=64-128、4-8ヘッド)
- Phase 4: 実用化(BLEU評価、ビーム探索)

---

## elan_parser - 次のステップ

- [ ] `REF_ANNOTATION`(階層 tier)対応
- [ ] Stereotype 処理
- [ ] 実データ(NHK STRL/NINJAL コーパス or 自前アノテーション)での検証
- [ ] `transformer_burn` 前処理パイプラインへの組み込み

---

**最終更新**: 2026年5月22日
