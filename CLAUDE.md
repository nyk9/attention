# Transformer/Attention プロジェクト

## 翻訳方向(最重要・source of truth / 2026-06-04 確定)

- **本来の目標**: **手話 → 日本語(認識)** が第一。手話動画 → 姿勢/手ポーズ抽出 → タグ列 → (LLM整形) → 日本語、という認識パイプライン。最終ゴールは双方向だが、**先に作るのは認識方向**。
- **日本語 → 手話(生成)は後段**(ロードマップ上 v1.5 以降)。
- **注意(再発防止)**: 2026-06-04 に一時「生成補強(日本語→タグ + ポーズ辞書引き)」へ舵を切った経緯があるが、これは**認識がランドマーク破綻(下記 S6)で詰まったための暫定回避策**であり、**本来の目標ではない**。同日中に認識方向へ戻すと決定(reverted)。コードが「日本語→タグ」中心に見えても、それは Phase 15b の足場であって、目標方向ではない。
- **`transformer_burn` の現状の `日本語→手話タグ` モデルは"足場"**。認識では入力を「日本語トークン列」から「手ポーズ列(ランドマーク列)」に差し替えて `ポーズ列→タグ` モデルにする(Phase 0a 本来の計画)。
- **記録ルール**: 技術的ブロックによる迂回は必ず「暫定回避策(本来の目標は X、ブロック Y のため一時 Z)」と明示し、裸の「方向性決定」として書かない。目標(安定)と戦術(可変)を混同しない。

## プロジェクト構成

### 1. transformer_burn - 手話翻訳AI
- 現状: 日本語→手話タグの可変長翻訳(Seq2Seq)= **認識モデルの足場**(本来の目標は手話→日本語認識。上記「翻訳方向」参照)
- 認識化の計画: 入力を「日本語トークン列」→「手ポーズ列」に差し替え、`ポーズ列→タグ` モデルにする
- **ステータス**: Phase 15b完了、Phase 16(モデル保存・推論基盤)90%完了
- **語彙**: 168(日本語86 + 手話タグ80 + SOS/EOS/PAD)
- **データ**: 47サンプル

### 2. translator_ja_en - 日英翻訳AI
- 日本語→英語の機械翻訳(手話データ準備期間中の学習継続)
- **ステータス**: Phase 1〜3完了(データセット・語彙・モデルスケールアップ)。Phase 4(実用化)はビームサーチ推論まで実装済み、BLEU評価は未着手。**2026-07-02 訂正**: 本節は 2025-11-01 の学習後も約8ヶ月間「初期セットアップ完了」のまま未更新だった(pose_extractor 側の作業に集中していたため)。以下は実態に合わせた最新値
- **語彙**: 日本語168 + 英語798(特殊トークン込み)
- **データ**: 2879サンプル(`data/` 配下に複数データセットを統合: 基本文・挨拶拡張・文章拡張・英国英語表現など)
- **モデル**: d_model=128、4ヘッド、Encoder/Decoder各4層、d_ff=512、seq_len=30。`models/test/` に150エポック学習済みモデルあり(2025-11-01訓練、final loss 0.162)

### 3. elan_parser - ELAN .eaf アノテーション抽出CLI
- ELAN(手話研究のデファクトアノテーションツール)出力の `.eaf` XML から、gloss 列を TSV/JSON 形式で抽出
- **ステータス**: v1完了(`ALIGNABLE_ANNOTATION` 対応、`REF_ANNOTATION` は v2)
- **使用クレート**: quick-xml 0.36, clap 4, serde, anyhow
- **バイナリ**: `elan-eaf-parse`
- **目的**: ろう者協力者から受け取るELANアノテーションを将来のMLパイプラインに繋ぐ前処理
- **Skill連携**: `~/.claude/skills/elan-eaf-parse/SKILL.md` から自動発見

### 4. pose_extractor - 動画→姿勢ランドマーク抽出CLI
- BlazePose ONNX(MediaPipe Pose 変換版)を `ort` クレートで呼び、動画フレームから39 keypoints(33+6補助)を抽出
- **ステータス**: Phase 0a-pipe-v1 の機能側は **S1〜S8 + 動画→タグ直結パスまで実装完了**(PR #1 で main にマージ、2026-06-14)。対話ウィザード CLI・バッチ処理・撮影進捗ツール対応済み。**先頭10語の実撮影データで Phase 0a 終了基準を判定済み(2026-07-02、実質クリア。下記参照)**。次は stage 2(11-20語目)の撮影へ。**精度上の課題が S6 検証で発見済み(下記参照)**のため、認識は手ポーズ(C 系)中心で進行中
- **使用クレート**: ort 2.0.0-rc.12, ndarray 0.16, image 0.25, clap 4, dialoguer 0.11, serde, anyhow
- **外部依存**: ffmpeg 8.1.1(`brew install ffmpeg`、CLI subprocess 経由で動画読み込み)
- **モデル**: `models/blazepose_full.onnx`(5.3MB、HF `opencv/pose_estimation_mediapipe`)。git管理外
- **バイナリ**: `pose-extract`。**CLI はフラグを使わず、引数なし起動のトップレベルメニュー(質問形式)で全機能に入る**(2026-06-13 にフラグ廃止。`--flag` 暗記不要の UI/UX 方針 [[cli-interactive-wizard-preference]])。メニュー項目: 動画抽出 / 撮影セッション / 撮影進捗 / 辞書構築(build-dict)/ 動画→タグ認識(推論)/ dev:inspect / dev:test-infer。設定は dialoguer の Select/Input で対話的に尋ねる。**TTY 必須**(スクリプトからの非対話呼び出しは不可。将来フラグ併設の可能性はあるが判断はユーザーが行う)
- **撮影進捗ツール(2026-06-12 追加、2026-06-13 ウィザード化)**: メニュー「撮影進捗を確認」= `data/raw_jsl` をスキャンして `index.tsv` を自動生成・更新し進捗表示(words.tsv 照合・typo/長さ警告)。メニュー「撮影セッション」= OBS/QuickTime の保存先を監視し、新録画を自動で命名(`<word_id>_<romaji>/<rep>.mp4`)・mp4詰め替え・記帳・手検出チェック。撮影は既製ツールのまま横で取り込む方式(ツール内録画は Phase 0b 以降)。実装は `src/progress.rs`、words.tsv は初回に50語テンプレ自動生成(ろう者協力者レビュー前の暫定)
- **手検出チェックの判定方式を変更(2026-07-02)**: 実撮影(先頭10語「こんにちは」)で4テイク連続 `ng_hands` が発生。オーバーレイ用に抜いたフレームを目視した結果、クロップ/照明は問題なく、**一人撮影のため録画の前後に手を下げた時間が長く**(サイン自体は動画の半分弱の区間に収まる)、動画全体平均のカバレッジ(旧: 50%未満で`ng_hands`)が構造的に閾値を割ることが原因と判明。**「動画全体の平均」から「連続して手が検出できた最長区間(1.0秒未満で`ng_hands`)」に変更**(`progress.rs`: `HAND_COVERAGE_THRESHOLD` → `MIN_HAND_RUN_SECONDS`+`MAX_RUN_GAP_SAMPLES`、瞬断1サンプルまでは同一区間として橋渡し)。ユニットテスト5件追加。なお `build-dict` は各テイクから均等10フレームを抜くため、手ぶら時間が長いテイクは学習特徴としても質が落ちる点は変わらず、録画時は手ぶら時間を短くする方が望ましい
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
  - C-2(Hand Landmark): 完了。21 keypoints/手を取得。**回転アラインメント実装完了(2026-06-20)**: Palm keypoint[0]=手首・[2]=中指MCP を結ぶ向きが上向き(90°)になる回転角を求め(OpenCV Zoo `mp_handpose` / MediaPipe `DetectionsToRectsCalculator` と同式)、クロップを回転して双線形サンプリング→手ランドマークモデルへ。出力は逆アフィンで元画像へ戻す。回転角0のとき従来の軸並行クロップに一致する一般化(後方互換)。中心シフトも回転後フレームの y 軸方向へ適用(MediaPipe RectTransformation 準拠)
  - S7(動画→手ポーズ列 抽出基盤): `build-dict` サブコマンド実装完了。`<タグ名>.mp4` 群から 1 フレーム=126 次元(左右の手 21点×xyz、手のみ。身体ランドマークは破綻のため不採用)の手ポーズ列を抽出。`--frames` でダウンサンプル(既定 10=SEQ_LEN)、座標は width/height で正規化。既存動画 1 本でスモークテスト済み(coverage 約 62%)
- **S7-8 の方向性(2026-06-04: 生成補強を一度採用 → 同日 認識へ戻す reverted)**:
  - 一時、認識ではなく**生成補強**(日本語→タグ + `build-dict` のポーズ辞書引き)を採用したが、これは認識がランドマーク破綻で詰まったための**暫定回避策**。本来の目標(手話→日本語 認識)に戻すため**取り消し**。
  - **現方針 = 認識**: `build-dict` で得た「動画→手ポーズ列(126次元)」は、辞書ではなく**認識モデルの学習データ・入力特徴**として転用する(`(手ポーズ列, タグ)` ペア)。手のみ特徴なので身体ランドマーク破綻を回避できる。
- **S8(認識モデル: 手ポーズ列→タグ)進捗(2026-06-11)**:
  - **学習ループ実装完了・スモーク通過**。transformer_burn に認識モデルを追加(足場の Seq2Seq とは別モデルとして共存)
  - 新規モジュール: `tag_vocabulary.rs`(動的タグ語彙+SOS/EOS/PAD、モデルと一緒に保存)/ `pose_data.rs`(build-dict JSON 読み込み、stem 末尾の `-NN` テイク番号を剥がしてラベル化)/ `recognition.rs`(PoseEncoder = Linear 126→d_model + 既存 TransformerBlock、Decoder は**因果マスク付き** Self-Attention)/ `recognition_training.rs`(Teacher Forcing 学習+top-k 評価)
  - CLI: `--train-pose <dict.json> [--epochs N] [--save dir]` / `--load dir --predict-pose <dict.json>`
  - スモーク結果(既存6動画=2ラベル×3テイク、300エポック・約47秒・WGPU): Loss 4.50 → 0.00、in-sample top-1 6/6。save→load 往復も一致
  - 注意: 2クラス in-sample の 100% は配線確認であって精度の証拠ではない。本判定は撮影後の未見テイクで行う
- **動画→タグ直結パス 完了(2026-06-14、PR #1 で main にマージ)**:
  - pose-extract メニュー「動画からタグを認識(推論)」。pose_extractor が transformer_burn を lib 依存し、動画→手ポーズ列(126次元)→ 認識モデル(CPU/NdArray)→ top-k タグを **1 プロセスで完結**(S7 抽出 + S8 推論を結線)。学習は引き続き build-dict→`--train-pose`。
  - E2E スモーク通過(既存6動画+rec_smoke、0205-01.mp4 → top-1 "0205" 確率1.0)。回帰用に `#[ignore]` テスト `recognize_smoke_in_sample` 同梱(`cargo test recognize_smoke_in_sample -- --ignored`)。
  - 同 PR で checkpoint クロスバックエンドテストのバグ修正、レビュー指摘3件(overlay範囲 / フレームレート0除算 / TSVサニタイズ)も対応。大きい/private な生成物(`models/rec_*`・`data/pose_dict*.json`・`data/raw_jsl/`)は gitignore で非追跡。
  - 注意: in-sample の一致は配線確認であって精度の証拠ではない。本判定は撮影後の未見テイクで。
- **ビルド軽量化(feature ゲート化)完了(2026-06-19)**:
  - 直結パス導入時の代償だった「pose_extractor の初回ビルドが Burn(wgpu)取り込みで重い」を解消。transformer_burn 側で **wgpu(+autodiff)を `wgpu` feature でゲート**(`default = ["wgpu"]`、`ndarray` は CPU 推論で常用するため常時有効)。wgpu に触れるモジュール(checkpoint / inference / training / recognition_training)は `#[cfg(feature = "wgpu")]` で限定し、CPU 推論経路(recognition ほか)は常時公開。bin には `required-features = ["wgpu"]` を付与し、wgpu 無効時はビルド対象外にする。
  - pose_extractor は `transformer_burn = { path = "../transformer_burn", default-features = false }` で取り込み、wgpu バックエンド一式のコンパイルを回避。
  - 検証: 軽量 lib のみ `cargo check --no-default-features` 30秒(wgpu/naga/wgpu-hal 非コンパイル)/ pose_extractor `cargo check` 47秒(同上)/ transformer_burn `cargo test`(default=wgpu)全15件パス・回帰なし。学習・推論の機能と精度は不変(burn のバックエンド差し替えのみ)。
- **未見テイク評価ハーネス 完了(2026-06-15)**:
  - `transformer_burn --eval-holdout <dict.json> [--holdout-per-label N]`。pose dict を**テイク単位で train/held-out に分割**(各ラベルでテイク番号の後ろ N 件=既定1を未見に)→ train だけで学習 → **未見テイクで top-k を測る**。Phase 0a の合否判定を1コマンド化(従来は train/predict 用に dict を手で2つ作る必要があった)。
  - `pose_data.rs` に `take_from_stem` / `load_holdout_split`(+ユニットテスト)、`main.rs` に `--eval-holdout` 分岐を追加。学習・評価は既存 `train_recognition` / `evaluate_topk` を再利用。
  - ドライラン(既存6動画、各語 take3 を未見): train=4 / 未見=2 に正しく分割・学習・評価。top-1 1/2・top-5 2/2(2クラス極小なので配線確認。ただし take3 は学習未使用=**従来の in-sample と違い真の out-of-sample**)。
  - テイクが N 以下のラベルは held-out を作れないため train のみに入れ評価対象外(警告)。
- **オーバーレイ可視化ツール 完了(2026-06-21、PR #6)**:
  - pose-extract「動画から姿勢/手を抽出」のオーバーレイを強化。手ランドマーク21点に**骨格線(MediaPipe 21本のボーン)**を描き、指の曲がりを目視できるように(回転アラインメント PR #5 の効果確認用)。S6 の教訓「数値でなく目で確かめる」の本丸で、撮影開始後の手トラッキング品質チェックの主導線。
  - 出力形式に**オーバーレイ動画 mp4(全フレーム)**を追加(従来はサンプル PNG 数枚のみ)。全フレームを `frame_%05d.png` で吐き ffmpeg で `overlay.mp4` に束ねる。描画を抽出ループ内へ移し生フレームの全バッファ(`Vec<Array3<u8>>`)を廃止=メモリ削減。ゼロフレーム時は動画化をスキップし警告に降格。
  - 実装は `pose_extractor/src/main.rs`(`HAND_CONNECTIONS` / `draw_line` / `encode_overlay_video`、`RunConfig.overlay_video`)。ユニットテスト(骨格トポロジ・線描画)+ 実モデルでの E2E スモーク `overlay_video_smoke`(`#[ignore]`)を同梱。CLI.md 更新。サブエージェントレビュー対応済み(ゼロフレーム致命化の是正・骨格トポロジテスト追加ほか)。
- **Phase 0a 終了基準を先頭10語の実撮影データで判定(2026-07-02、実質クリア)**:
  - 撮影セッションツールで先頭10語×3テイク(「ok」判定分、計30本)を撮影 → 手動で `<romaji>-<rep>.mp4` のフラット命名に整理(build-dict は `<タグ名>.mp4` のフラット構成が前提で、`raw_jsl` の `<word_id>_<romaji>/<rep>.mp4` というネスト構成とは形式が違うため)→ `build-dict` でポーズ辞書化(`transformer_burn/data/pose_dict_stage1.json`、30タグ×10フレーム×126次元)→ `--eval-holdout`(各語 train=2/held-out=1)で評価
  - **結果: top-1 3/10(30%)、top-5 8/10(80%)**。学習lossは0.000178まで低下(20サンプルをほぼ暗記した状態での汎化を見ている)。外れた2語は `anata`・`watashi`
  - **外れた2語には具体的な原因の見立てあり**: build-dict の左右手カバレッジを見ると、`anata` は学習2本とも右手優位(R41%/R35%)なのに held-out だけ左手優位(L36%/R0%)。126次元特徴が「左手枠/右手枠」で構成されているため、held-out テイクだけ学習時と違う枠にデータが来て別物に見えた可能性が高い(予測は `arigatou` 0.97、`onegaishimasu` 1.00 と、際どい誤りでなく高確信度の誤りである点も整合)。`konnichiwa` など両手を安定して使う語は正解しており、単一モデル能力の問題というより**撮影/検出時の左右手ゆれ**が主因と見ている
  - **判断(2026-07-02 ユーザー確認済み)**: 80%は実質クリア(2語の外れに具体的な原因があり、パイプライン自体は機能していると判断)として **Phase 0a は実質クリア**。ただし n=10 の極小サンプルなので数値の頑健性は低い。左右手ゆれの根本対応(特徴表現の見直し、または anata/watashi の再撮影)は**いったん保留**し、stage 2(11-20語目)の撮影を優先。同じ問題が再発するか様子を見てから対応を判断する
  - 学習済みモデルは `--save` 未指定のため未保存(評価専用の使い捨て実行)。実運用モデルが要る場合は保留分含む全データで `--train-pose` から再学習する必要がある
- **左右手ゆれの真因判明+診断ツール `--inspect-dict` 追加(2026-07-02)**:
  - transformer_burn に診断専用 CLI `--inspect-dict <dict.json>` を追加(学習なし・モデル不要)。pose dict をラベル別に集計し、テイクごとの左右手カバレッジと「使われた手」(L/R/LR/-、カバレッジ20%以上で使用扱い)を表示し、テイク間で使われた手が食い違うラベルを警告する。実装は `pose_data.rs`(`used_hands` / `inspect_dict`)+ `main.rs` の早期分岐。ユニットテスト2件追加
  - stage 1 の pose dict を点検した結果、**10語中7語で使用手がテイク間で食い違い**(anata, arigatou, ohayou, onegaishimasu, sayounara, sumimasen, watashi)。食い違いゼロの3語(konnichiwa・konbanwa・otsukaresama)はいずれも両手サイン。**top-1 30% はほぼ「使用手の train/holdout 不一致」で説明可能**で、上記「外れた2語」の見立てが実は7語に及んでいた
  - sayounara(テイク01=右手 / 03=左手)と watashi(01=左手 / 02=右手)の生動画フレームを目視検証し、**handedness 検出は正しく、撮影者がテイクごとに実際に手を持ち替えていた**ことが真因と確定。「検出時のゆれ」ではなく「撮影時のゆれ」であり、モデル・抽出パイプラインの品質問題ではない
  - **方針(2026-07-02 ユーザー決定)**: 実運用で利用者の使用手が揺れることも見据え、**使用手は固定せず、あえて一部左手データを含めて撮影を続ける**。stage 1 の混在7語は様子見(撮り直しなし)。撮影セッションへの使用手警告機能は追加しない。評価時は `--inspect-dict` で使用手分布を把握した上で数値を解釈する
  - あわせて words.tsv の採用語彙をユーザーが修正(手話で同一表現の語の統合=are-sore、別表現になる語の分割=kare/kanojo など。撮影済み 001〜010 は不変)
- **次のステップ**:
  - stage 2(words.tsv の11-20語目: kare〜ii)を撮影 → 同じ手順(build-dict 用にフラット整理 → `--eval-holdout`)で評価。評価前に `--inspect-dict` で使用手分布を確認し、使用手混在を意図的に含む方針を踏まえて数値を解釈する
  - タグ → 自然な日本語(部品C)は LLM 整形を後段で(①プロンプト→②LoRA→③自作の順で安く試す)
  - 前提: S6 のランドマーク品質問題への取り組み継続。ただし認識は手ポーズ(C 系)中心で進められる

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

### Phase 1〜3(完了)
- ✓ 日英対訳データセット準備(2879サンプル、`data/` 配下に複数ファイルへ分割: 基本文・挨拶拡張・文章拡張・英国英語表現など)
- ✓ 語彙モジュール実装(translation_vocabulary.rs、日本語168語 + 英語798語、分離語彙空間)
- ✓ データローダー実装(translation_data.rs)
- ✓ シーケンス長拡張(seq_len=30)
- ✓ モデルスケールアップ(d_model=128、4ヘッド、Encoder/Decoder各4層、d_ff=512)
- ✓ ビームサーチ推論実装(translation_model.rs::generate_with_beam_search、CLIの--beam-widthで貪欲探索/ビーム探索を切替)

### データセット例(data/translation_data_ja_en.txt)
```
こんにちは	Hello
ありがとう	Thank you
おはようございます	Good morning
私は学生です	I am a student
```

### Phase 4(実用化、残タスク)
- [ ] BLEU評価の実装
- [ ] 訓練済みモデル(`models/test/`、150エポック、final loss 0.162、2025-11-01訓練)の翻訳品質を定性チェック

---

## elan_parser - 次のステップ

- [ ] `REF_ANNOTATION`(階層 tier)対応
- [ ] Stereotype 処理
- [ ] 実データ(NHK STRL/NINJAL コーパス or 自前アノテーション)での検証
- [ ] `transformer_burn` 前処理パイプラインへの組み込み

---

## インフラ / 計算環境 移行メモ

- **移行方針**: 手話動画の撮影完了後、開発機を MacBook → デスクトップ Windows PC に移行予定。GPU は新規購入(NVIDIA **RTX 5070 Ti**、16GB GDDR7、Blackwell世代を想定。2026-06時点の価格・流通を考慮して選定)
- **クラウド**: データ増大で学習時間が膨らんだら、クラウド GPU(NVIDIA/CUDA/Linux 前提)をレンタルして学習する想定
- **当面のボトルネック認識**: 現行モデルは極小(d_model=16、47サンプル)のため VRAM 16GB は大幅に余る。GPU の当面の主目的は「映像前処理(pose/Hands ONNX 抽出)の高速化」と「将来スケールへの余裕」。学習そのものが遅くて困るのは、モデルを大きくしてデータ・実験回数が増えた後

### 移行前に確認するチェックリスト(忘れないこと)

- [ ] **ローカル Linux(Ubuntu)で Rust/Burn のビルドが通るか事前検証**。Windows ローカルとクラウド Linux で OS が変わる段差を、移行前に潰しておく(Ubuntu は既に導入済み)
- [ ] Burn の **CUDA バックエンド**を使う場合、Blackwell(sm_120系)に CUDA Toolkit / ドライバ / Burn のバージョンが対応しているか確認。未対応なら wgpu(Vulkan経由)バックエンドで回避

---

**最終更新**: 2026年7月2日(**左右手ゆれの真因判明**: 診断ツール `--inspect-dict` を transformer_burn に追加して stage 1 を点検した結果、10語中7語で使用手がテイク間で食い違い、生動画の目視で「検出誤りではなく撮影者が実際に手を持ち替えていた」ことを確定。方針は「使用手は固定せず、あえて一部左手データを含める」(ユーザー決定)。words.tsv の採用語彙も修正(are-sore 統合・kare/kanojo 分割など)。同日それ以前: **Phase 0a 終了基準を先頭10語の実撮影データで判定**: top-1 30%・top-5 80%で実質クリアと判断、次は stage 2(11-20語目)の撮影へ。あわせて撮影セッションの手検出チェックを「動画全体平均」から「連続して手が検出できた最長区間」ベースに変更(一人撮影だと前後の手ぶら時間で平均が構造的に閾値を割るため)。同日中の一連の変更: `translator_ja_en` 節のドキュメント訂正(実態より約8ヶ月古い記載だった`初期セットアップ完了`を、実際は2025-11-01時点でPhase 1〜3完了済みだった内容に修正)。直前の更新(2026-06-21)=**オーバーレイ可視化ツール**を pose-extract に追加(PR #6)。手ランドマーク21点に骨格線を描きオーバーレイを全フレームmp4で出力可能に。さらに前=Hand Landmark の回転アラインメント(PR #5)、ビルド軽量化(PR #4)、未見テイク評価ハーネス `--eval-holdout`(PR #3)、認識方向パイプライン S1〜S8 + 動画→タグ直結パス(PR #1)を main にマージ済み)
