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
    - 緑=visibility > 0.5、赤=visibility ≤ 0.5
  - **フレーム数制限**: 動作確認用に先頭 N フレームのみ処理
    - 選択時に「最大フレーム数(既定 10)」を追加質問
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

座標系: x, y は **256x256 のリサイズ後画像空間**。元解像度に戻すには
`x * width / 256.0`、`y * height / 256.0` を掛ける。z は相対的な奥行き。

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

## 開発用サブコマンド

通常使わないが、ONNX モデルの調査用に残してある:

```bash
# モデルの入出力名を確認
./target/release/pose-extract inspect models/blazepose_full.onnx

# ダミー入力で推論し、出力テンソルの shape を確認
./target/release/pose-extract test-infer models/blazepose_full.onnx
```

## 入力前処理について

現状は **256x256 への伸ばしリサイズ(stretch)** のみ。アスペクト比は保持
していない。S6 検証で confidence が低いケースが多発するようなら、letterbox
(余白埋めでアスペクト比保持)を追加する予定。

## 自動化したくなった時

ウィザードは TTY が必要なため、CI やスクリプトから呼ぶことはできない。
S7-8(transformer_burn 連携)などで自動化が必要になったら、
`RunConfig` を直接組み立てるサブコマンド(例: `pose-extract batch --input ...`)を
別途追加する想定。現在は対話のみ。

## トラブルシューティング

### `Error: IO error: not a terminal`
ウィザードを TTY なし(パイプ・cron 等)で起動した。ターミナルから直接実行する。

### `ffprobe failed` / `failed to spawn ffmpeg`
`ffmpeg` がインストールされていない。`brew install ffmpeg`。

### `failed to load model: ...`
`models/blazepose_full.onnx` がない。Hugging Face から取得して配置。

### confidence が全フレーム 0.1 以下
- 人物が画面に小さくしか映っていない
- アスペクト比が極端で stretch リサイズの歪みが大きい(letterbox 検討)
- 動画が暗すぎる / 解像度が低すぎる
