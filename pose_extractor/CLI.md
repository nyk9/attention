# pose-extract CLI ガイド

BlazePose ONNX を使った動画→姿勢ランドマーク抽出ツールの使い方。

## 前提

- macOS / Linux のターミナルから起動
- `ffmpeg` / `ffprobe` がインストール済み(`brew install ffmpeg`)
- BlazePose ONNX モデルが `pose_extractor/models/blazepose_full.onnx` に配置されている
  (Hugging Face `opencv/pose_estimation_mediapipe` から取得、5.3MB、git管理外)
- ビルド済み: `cd pose_extractor && cargo build --release`

## 基本ワークフロー

### 1. 動画を `videos/` に置く

```
pose_extractor/
└── videos/                  ← .gitignore 済み(中身は git に乗らない)
    ├── sign_sample_01.mov
    └── walk_test.mp4
```

`.mp4` `.mov` `.mkv` `.avi` `.webm` `.m4v` が認識されます。
iPhone で撮った `.MOV` も AirDrop でこのディレクトリに入れれば OK。

### 2. ウィザードを起動

```bash
./target/release/pose-extract
```

引数なしで起動するとウィザードに入ります。

### 3. 質問に答える

```
動画を選択
  > sign_sample_01.mov
    walk_test.mp4
    [パスを直接入力]

出力形式
  > JSON
    TSV (1行=1フレーム)

追加機能 (space で選択, enter で確定)
  [x] sigmoid を visibility/presence に適用
  [ ] ランドマーク オーバーレイ PNG 保存
  [ ] フレーム数制限 (動作確認用)

出力ファイル (空欄=stdout)
  > /tmp/result.json
```

- **動画を選択**: `videos/` の中身が出る。直接入力に切り替えれば任意パスも可
- **出力形式**:
  - **JSON**: 構造体まるごと(`video`/`model`/`sigmoid_applied`/`frames[]`)
  - **TSV**: 1 行 = 1 フレーム、197 列(`frame_idx`, `confidence`, `x0..pres38`)
- **追加機能(複数選択可)**:
  - **sigmoid**: 生のロジット値を [0,1] に正規化(visibility/presence のみ。confidence はモデル側で正規化済み)
  - **オーバーレイ PNG**: 等間隔のフレームにランドマークを描画し PNG 保存
    - 選択時に「保存先ディレクトリ」「枚数(既定 3)」を追加質問
    - 緑=visibility > 0.5、赤=visibility ≤ 0.5(BlazePose)
    - シアン枠=palm bbox、黄色ドット=palm 7 keypoints(MediaPipe Palm Detection)
    - マゼンタ/青ドット=hand 21 landmarks(handedness で色分け、MediaPipe Hand Landmark)
  - **フレーム数制限**: 動作確認用に先頭 N フレームのみ処理
    - 選択時に「最大フレーム数(既定 10)」を追加質問
  - **MediaPipe Hands (Palm + Landmark) を並走**: 手指検出を並列実行
    - `models/palm_detection_mediapipe.onnx` で手の bbox を抽出
    - `models/handpose_estimation_mediapipe.onnx` で 21 keypoints/手 を抽出
    - JSON 出力に `palm_frames` と `hand_frames` フィールドが追加される(TSV は pose のみ)
    - 注:回転アラインメントは未実装。`HAND_CROP_ENLARGE=3.0` の簡易クロップのため、指関節の位置精度は MediaPipe 純正よりやや劣る
- **出力ファイル**: 空欄で stdout、パス指定でそのファイルに書き出し

## 出力スキーマ

### JSON

```json
{
  "video": {
    "width": 320, "height": 240, "fps": 30.0,
    "frame_count": 60, "duration": 2.0
  },
  "model": "models/blazepose_full.onnx",
  "sigmoid_applied": true,
  "frames": [
    {
      "frame_idx": 0,
      "confidence": 0.0004,
      "landmarks": [
        { "x": 118.95, "y": 37.59, "z": -36.19,
          "visibility": 0.669, "presence": 0.180 },
        ... (39 個)
      ]
    },
    ...
  ]
}
```

座標系: x, y は **256x256 のリサイズ後画像空間(中央クロップ後)**。
元フレーム座標に戻すには:

```
side = min(width, height)
crop_x = (width  - side) / 2
crop_y = (height - side) / 2
x_orig = x / 256 * side + crop_x
y_orig = y / 256 * side + crop_y
```

z は相対的な奥行き。

### TSV(wide)

```
frame_idx  confidence  x0  y0  z0  vis0  pres0  x1 ... pres38
0          0.0004      ...
1          0.0009      ...
```

- 1 行目: ヘッダ
- 以降: 1 フレーム = 1 行、197 列
- pandas / DuckDB から `pd.read_csv(..., sep='\t')` 等で直接読める

## stats 出力(stderr)

実行ごとに以下を stderr に出力:

```
stats: confidence mean=0.066 max=0.579 min=0.000 | visibility(sigmoid) mean=0.276
```

- `confidence`: フレームごとの pose 検出 confidence
- `visibility(sigmoid)`: 全ランドマーク visibility の平均(sigmoid 統一済み)

人物が映った動画なら confidence mean が 0.7〜0.95 程度になるのが目安。
0.1 を切るようなら人物検出に失敗している可能性が高い。

Hands を並走させた場合、以下も追加で出力:

```
palm detection: 172 frames with palms, 342 palms total
hand landmark: 170 frames with hands, 335 hands total
```

- 通常 palm 数 ≥ hand 数(hand landmark の conf<0.5 が弾かれるため)

## build-dict サブコマンド(S7: transformer_burn 連携)

タグ名を付けた動画群から「タグ→ポーズ列」辞書(JSON)を構築する。
日本語→手話タグ翻訳(transformer_burn)の出力タグに、参照ポーズ列を
紐付けるための前処理。ウィザードと違い TTY 不要なのでスクリプトから呼べる。

```bash
# videos/ 内の <タグ名>.mp4 から辞書を作る
./target/release/pose-extract build-dict \
    --input-dir videos \
    --output tag_pose_dict.json \
    --frames 10
```

- **入力**: `--input-dir` 内の `<タグ名>.mp4`(ファイル名の拡張子を除いた部分がタグ名)
  - 例: `ありがとう.mp4` → タグ `ありがとう`
  - 1 動画 = 1 タグ。ELAN アノテーション不要(動画全体がそのサインを表す前提)
- **`--frames`**: 各クリップを何フレームにダウンサンプルするか(既定 10、transformer_burn の `SEQ_LEN` と揃える)
- **特徴量**: 1 フレーム = 126 次元 = 左手 21 点 + 右手 21 点、各 `[x, y, z]`
  - **手のみ**を使う(BlazePose の身体ランドマークは S6 で破綻が判明したため不採用)
  - 座標は正規化済み: `x/=width`, `y/=height`, `z/=width`
  - handedness で左右スロットに振り分け。検出されなかった手はゼロ埋め
- **出力 JSON**:

```json
{
  "metadata": {
    "frames": 10,
    "feature_dim": 126,
    "feature_layout": "left_hand[21*xyz] then right_hand[21*xyz]; missing hand = zeros",
    "normalization": "x/=width, y/=height, z/=width (z is relative depth)",
    "tag_count": 2
  },
  "tags": {
    "ありがとう": {
      "sequence": [[...126 floats...], ...10 frames...],
      "left_hand_coverage": 0.62,
      "right_hand_coverage": 0.61,
      "source": "ありがとう.mp4"
    }
  }
}
```

- **coverage**: そのタグで左右の手が検出できたフレームの割合。低い(< 0.3 等)なら
  動画の撮り方(手が画角外/小さすぎ)を見直す目安。

## 開発用サブコマンド

通常使わないが、ONNX モデルの調査用に残してある:

```bash
# モデルの入出力名を確認
./target/release/pose-extract inspect models/blazepose_full.onnx

# ダミー入力で推論し、出力テンソルの shape を確認
./target/release/pose-extract test-infer models/blazepose_full.onnx
```

## 入力前処理について

**中央クロップ + 256x256 リサイズ** を行っている(S6 検証後に stretch から変更)。
短辺基準で中央正方形を切り出してから 256x256 に縮小するため、アスペクト比は保持される。

- 1920×1080 → 1080×1080 中央クロップ → 256×256
- 横長フレームは左右が、縦長フレームは上下が落ちる
- 全身を映す動画では脚部分が落ちることがあるが、手話用途では脚は不要なので問題なし

BlazePose の Pose Landmark モデルは「正方形にクロップされた人物画像」を前提に
作られているため、stretch リサイズだとモデルから見て「歪んだ人物」になり、
顔以外のランドマークが破綻する(S6 検証で確認済み)。

## 自動化したくなった時

ウィザード(引数なし起動)は TTY が必要なため、CI やスクリプトから呼ぶことはできない。
辞書構築の自動化には上記 `build-dict` サブコマンドを使う(TTY 不要)。
個別動画の JSON/TSV 抽出をスクリプト化したい場合は、`RunConfig` を直接組み立てる
サブコマンド(例: `pose-extract batch --input ...`)を別途追加する想定。

## トラブルシューティング

### `Error: IO error: not a terminal`
ウィザードを TTY なし(パイプ・cron 等)で起動した。ターミナルから直接実行する。

### `ffprobe failed` / `failed to spawn ffmpeg`
`ffmpeg` がインストールされていない。`brew install ffmpeg`。

### `failed to load model: ...`
`models/blazepose_full.onnx` がない。Hugging Face から取得して配置。

### confidence が全フレーム 0.1 以下
- 人物が画面に小さくしか映っていない
- 人物が中央クロップの範囲外にいる(画面端に立っている等)
- 動画が暗すぎる / 解像度が低すぎる
