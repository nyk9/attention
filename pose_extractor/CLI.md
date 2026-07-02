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

### 2. 起動してメニューから選ぶ

```bash
./target/release/pose-extract
```

このツールは **フラグ(`--xxx`)を使わず、すべて質問形式** で操作します。
起動すると最初にトップレベルメニューが出ます。

```
やることを選択
  > 動画から姿勢/手を抽出
    撮影セッション(録画フォルダを監視して自動取り込み)
    撮影進捗を確認
    撮影テイクを build-dict 用にエクスポート
    タグ→ポーズ辞書を構築(build-dict)
    動画からタグを認識(推論)
    [dev] ONNXモデルの入出力を調査(inspect)
    [dev] ダミー推論で出力shapeを確認(test-infer)
    終了
```

選んだ機能ごとに、必要な設定を順に質問されます(デフォルト値が出るので Enter で進めます)。
以降は「動画から姿勢/手を抽出」を選んだ場合の流れです。

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
  [ ] ランドマーク オーバーレイ保存(動画 mp4 / PNG)
  [ ] フレーム数制限 (動作確認用)
  [ ] MediaPipe Hands (Palm + Landmark) を並走

出力ファイル (空欄=stdout)
  > /tmp/result.json
```

- **動画を選択**: `videos/` の中身が出る。直接入力に切り替えれば任意パスも可
- **出力形式**:
  - **JSON**: 構造体まるごと(`video`/`model`/`sigmoid_applied`/`frames[]`)
  - **TSV**: 1 行 = 1 フレーム、197 列(`frame_idx`, `confidence`, `x0..pres38`)
- **追加機能(複数選択可)**:
  - **sigmoid**: 生のロジット値を [0,1] に正規化(visibility/presence のみ。confidence はモデル側で正規化済み)
  - **オーバーレイ保存**: フレームにランドマークを描画する。選択時に「保存先ディレクトリ」と
    出力形式を質問:
    - **動画 mp4(全フレーム・手の骨格つき/品質チェック向き)**: 全フレームを `frame_%05d.png` で
      吐き、ffmpeg で `overlay.mp4` に束ねる。手話のジェスチャーを通しで確認でき、回転アラインメント
      後に指が正しく曲がっているかの目視に向く(撮影品質チェックの主用途)
      - 中間 PNG(`frame_*.png`)は削除せず同じディレクトリに残す(フレーム単位で拡大確認できるため)
      - フレームが 0 本(空/破損動画・最大フレーム数 0)のときは動画化をスキップして警告のみ
    - **PNG 数枚(サンプル)**: 等間隔の N フレームのみ PNG 保存(選択時に「枚数(既定 3)」を追加質問)
    - 色の凡例(両形式共通):
      - 緑=visibility > 0.5、赤=visibility ≤ 0.5(BlazePose 39 点)
      - シアン枠=palm bbox、黄色ドット=palm 7 keypoints(MediaPipe Palm Detection)
      - マゼンタ/青=hand 21 landmarks(handedness で色分け)。**21 本の骨格線(ボーン)で関節を接続**し、
        指の曲がりを可視化(MediaPipe Hand Landmark)
    - 手の骨格線・並走 Hands を見るには「MediaPipe Hands を並走」も併せて選ぶ
  - **フレーム数制限**: 動作確認用に先頭 N フレームのみ処理
    - 選択時に「最大フレーム数(既定 10)」を追加質問
  - **MediaPipe Hands (Palm + Landmark) を並走**: 手指検出を並列実行
    - `models/palm_detection_mediapipe.onnx` で手の bbox を抽出
    - `models/handpose_estimation_mediapipe.onnx` で 21 keypoints/手 を抽出
    - JSON 出力に `palm_frames` と `hand_frames` フィールドが追加される(TSV は pose のみ)
    - 回転アラインメント実装済み(2026-06-20)。palm keypoint[0]手首→[2]中指MCP の向きが上向きに
      なる回転角でクロップを回転・双線形サンプリングしてから Hand Landmark へ渡す(MediaPipe 純正と同式)
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

## build-dict(メニュー:「タグ→ポーズ辞書を構築」)

タグ名を付けた動画群から「タグ→ポーズ列」辞書(JSON)を構築する。
認識モデル(transformer_burn の `--train-pose`)の学習データになる手ポーズ列を作る前処理。

メニューで「タグ→ポーズ辞書を構築(build-dict)」を選ぶと、以下を質問されます
(いずれも Enter で既定値):

```
入力ディレクトリ(<タグ名>.mp4 を置いた場所)  [videos]
出力 JSON パス                                [tag_pose_dict.json]
ダウンサンプルするフレーム数                  [10]
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

## 動画からタグを認識(メニュー:「動画からタグを認識(推論)」)

学習済みの認識モデル(手ポーズ列→タグ)を使って、**動画1本を入れたらタグが出る**直結パス。
build-dict→学習→判定の3手順のうち、判定だけをワンショットに短縮した近道です
(学習は引き続き transformer_burn の `--train-pose` で行う)。

メニューで選ぶと以下を質問されます:

1. **認識する動画**: `videos/` 内の動画一覧から選択(無ければパス直接入力)
2. **認識モデル(ディレクトリ)**: `../transformer_burn/models/` 配下で `model.bin` と
   `tag_vocab.json` の両方を持つもの(= 認識モデル)を一覧表示。足場の Seq2Seq モデルは
   `tag_vocab.json` を持たないので候補に出ません
3. **フレーム数**: 学習時の `SEQ_LEN`(既定 10)と揃える
4. **表示件数**: 上位何件のタグを出すか(既定 5)

動画から 126 次元の手ポーズ列を抽出し、モデルを **CPU(NdArray)で読み込んで**推論します
(動画1本の判定に GPU は不要)。出力は上位 k タグと確率、加えて手の検出率(coverage)。

```
=== 認識結果(上位 2) ===
1) 0205  1.000
2) 0225  0.000
```

注意:
- **coverage が低いときは結果を信用しない**。確率が高くても手が取れていなければ無意味
  (S6 の「confidence は高いのに身体が破綻」と同じ罠)。
- **学習に使った動画(in-sample)で当たるのは配線確認**であって精度の証拠ではない。
  本判定は未見テイクで行う。
- 内部実装は pose_extractor が transformer_burn を**ライブラリ依存**して 1 プロセスで完結する。
  代償として pose_extractor の初回ビルドは Burn を取り込むぶん時間がかかる。
- E2E スモーク(実モデル+実動画)は `#[ignore]` 付きテストで再現できる:
  `cargo test recognize_smoke_in_sample -- --ignored --nocapture`

## 撮影進捗ツール(progress / session)

50語×複数テイクの撮影で「次に何を撮るか・どこまで撮ったか」の管理を自動化する。
データのルートは既定で `../transformer_burn/data/raw_jsl`(Phase 0a の所定構成)。

### ディレクトリ構成

```
transformer_burn/data/raw_jsl/
├── words.tsv                  ← 語彙マスタ(初回に50語テンプレを自動生成。編集可)
├── index.tsv                  ← 撮影台帳(自動生成。手書き列は保持される)
├── 001_konnichiwa/
│   ├── 01.mp4                 ← <word_id>_<romaji>/<rep>.mp4
│   └── 02.mp4
└── 005_arigatou/
    └── 01.mp4
```

- **words.tsv**(人が編集): `word_id / romaji / label_ja / stage / target_takes`。
  `stage 1` = 先頭10語、`target_takes` = 各語の目標テイク数。撮影方針(段階・本数)を
  ここで変える。語彙はろう者協力者レビュー前の暫定。
- **index.tsv**(生成物): ファイルシステムを正としてスキャンで作る。ただし
  `quality_flag` / `notes` の2列だけは人が書く欄で、再スキャンでも消えない。

### 撮影進捗を確認(メニュー:「撮影進捗を確認」)

メニューで選ぶと、撮影データのルートと動作(更新 / 検証のみ)を質問されます:

```
撮影データのルート  [../transformer_burn/data/raw_jsl]
動作
  > index.tsv を更新して進捗表示
    検証のみ(index.tsv を書き換えない)
```

各語の「撮影済み/目標」と完了状況、stage 小計・全体%を表示する。
words.tsv に無い `word_id`(typo)、`<word_id>_<romaji>` 形式でないフォルダ、
プロトコル目安(3-4秒)から外れた長さ、実体が消えた台帳行などを警告する。

### 撮影セッション(メニュー:「撮影セッション」)

OBS / QuickTime の保存先を監視し、新しい録画を**自動で命名・振り分け・記帳・手検出
チェック**する。あなたの操作は「録画ボタンを押す→演じる→止める」だけ。

メニューで選ぶと以下を質問されます(録画フォルダは `~/Movies` を初期値として提示):

```
撮影データのルート(取り込み先)              [../transformer_burn/data/raw_jsl]
監視する録画フォルダ(OBS/QuickTime の保存先)  [~/Movies]
取り込み時の手検出チェック
  > 有効(手の検出が連続して続かなかったテイクを ng_hands+撮り直し提案)
    無効(取り込みが速い)
```

動作:

- 監視フォルダにファイルが書き終わると(サイズが約1.5秒変化しなくなったら)、
  stage 順→未達の最初の語へ自動で割り当てて取り込む
- `.mkv` 等は `ffmpeg -c copy` で **mp4 にコンテナ詰め替え**(再エンコードなし・高速)
- 取り込み後に Palm/Hand 検出をサンプルフレームに実行。**動画全体の平均ではなく、
  連続して手が検出できた最長区間**が1.0秒未満なら `ng_hands` フラグを付け、撮り直しを
  促す(NG テイクは目標数に数えない)。一人撮影だと録画の前後に手を下げた時間が
  できるため、全体平均だと正しく撮れたテイクまで弾いてしまう(2026-07-02 変更)
- セッション中のキー: `s`+Enter=現在の語をスキップ / `p`+Enter=進捗表示 / `q`+Enter=終了
- **開始時に監視フォルダにあった既存ファイルは取り込まない**(過去の録画の誤飲を防ぐ)

> 撮影は既製ツール(OBS/QuickTime)のまま。session はそれを置き換えず横で取り込みを担う
> (ツール内録画は Phase 0b 以降。プレビュー確保のため現状は既製ツールを使う方針)。

### build-dict 用にエクスポート(メニュー:「撮影テイクを build-dict 用にエクスポート」)

raw_jsl の `<word_id>_<romaji>/<rep>.mp4`(ネスト構成)から、build-dict が読める
`<romaji>-<rep>.mp4`(フラット構成)へテイクを並べ直す。stage 1 では手作業だった工程の自動化。

メニューで選ぶと以下を質問されます:

```
撮影データのルート            [../transformer_burn/data/raw_jsl]
対象 stage
  > すべての stage
    stage 1
    stage 2
エクスポート先ディレクトリ    [videos/dict_export_stage1 など(選んだ stage で変わる)]
続けて build-dict(ポーズ辞書構築)を実行しますか
  > はい
    いいえ(エクスポートのみ)
```

動作:

- index.tsv を実ファイルと同期した内容から、**`quality_flag` が `ng` で始まるテイクを除外**して
  エクスポートする(進捗表示の有効テイクと同じ規則。`ok`・空欄・手書きフラグは対象)
- ファイルはハードリンクで置く(容量を消費しない。別ボリューム等で失敗したらコピー)。
  **raw_jsl 側には一切書き込まない**
- **テイク番号は raw_jsl の番号をそのまま使う**(NG 除外で欠番になっても付け替えない。
  例: konnichiwa の ok テイクが 05〜07 なら `konnichiwa-05.mp4`〜`konnichiwa-07.mp4`)。
  番号を保持することで raw_jsl のどのテイクかを後から追跡できる
- 語ごとの本数と目標未達をサマリ表示する
- 「はい」を選ぶとそのまま build-dict に進み、出力 JSON(既定
  `../transformer_burn/data/pose_dict_stage<N>.json`)とフレーム数を聞いて辞書を構築する
- 実データでの回帰テスト: `cargo test export_stage1_smoke -- --ignored`
  (stage 1 のエクスポート結果が既存 pose_dict_stage1.json の30タグ名と一致することを確認)

## 開発用メニュー(inspect / test-infer)

通常使わないが、ONNX モデルの調査用にメニューに残してある。選ぶと `models/` 内の
`.onnx` 一覧から対象を選択する(直接パス入力も可):

- **inspect**: モデルの入出力名・shape を表示
- **test-infer**: ダミー入力で推論し、出力テンソルの shape を表示

## 入力前処理について

**中央クロップ + 256x256 リサイズ** を行っている(S6 検証後に stretch から変更)。
短辺基準で中央正方形を切り出してから 256x256 に縮小するため、アスペクト比は保持される。

- 1920×1080 → 1080×1080 中央クロップ → 256×256
- 横長フレームは左右が、縦長フレームは上下が落ちる
- 全身を映す動画では脚部分が落ちることがあるが、手話用途では脚は不要なので問題なし

BlazePose の Pose Landmark モデルは「正方形にクロップされた人物画像」を前提に
作られているため、stretch リサイズだとモデルから見て「歪んだ人物」になり、
顔以外のランドマークが破綻する(S6 検証で確認済み)。

## CLI の方針(フラグなし・質問形式)

このツールは **`--flag` を一切使わず、全機能を引数なし起動のメニュー+質問形式** で
操作する(UI/UX 重視・暗記不要の方針)。そのため現状は **TTY が必須**で、CI や
スクリプトからの非対話呼び出しはできない(以前あった `build-dict --input-dir ...` 等の
フラグは廃止した)。

将来、自動化のためにフラグ形式を併設し直す可能性はあるが、**その判断・タイミングは
開発者(ユーザー)が決める**。スクリプト連携が必要になったら、その時点で
`RunConfig` 等を組み立てる非対話サブコマンドの追加を検討する。

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
