# クイズモード(desktop/)設計と実装の記録

作成: 2026-10-16。マネージャー経由でユーザーが承認した設計(2026-10-16)と、その実装の記録。決定の記録ではない(決定は CLAUDE.md / development-plan.md)。ブランチ `feature/desktop-quiz`。

## 目的

ライブ認識を使ったクイズで、ライブデモを楽しくわかりやすくする。

## 設計

- エンジン(pose_extractor)と `protocol.rs` は変更しない。`desktop/` の画面側だけで作る。
- `AppMode::Quiz`(`state.rs`)。エンジンはライブ認識と同じ設定(`mode` はライブ、区切りは自動。ライブ側で手動を選んでいても、クイズは自動で起動する)。`result` イベントをクイズの判定に流す。`apply` は Result を Live 以外で無視していたが、Quiz のときだけ `Quiz::on_event` へ流す(撮影ガイド・撮影では従来どおり無視)。ライブの `current` / `history` には入れない。
- 純粋ロジックは `desktop/src/quiz.rs`(GPUI に触れない)。画面は `desktop/src/quiz_view.rs`。
- 出題は `ready.vocab`(モデルの語彙)から、除外語を引いた候補から重複なしのランダム。候補が 5 に満たなければ候補の数だけ出題し、候補が 0 なら始めない(ログに警告)。

### 流れ

開始前(Ready)→ スタート(ボタン / スペース)→ お題表示・待機(Asking)→ 結果受信 / パス / 30 秒で時間切れ → 判定表示(Judged)→ 次へ(スペース / 3 秒で自動)→ … → 総括(Summary)→ もう一度。

- **古い結果の扱い**: 待機(Asking)のときだけ結果を受け付ける。開始前・判定表示中・総括に届いた結果は無視する。エンジンが止まったらゲームは中断(Ready に戻る)。
- スペースキー: 開始前・総括では開始、判定表示中は次へ、出題中は何もしない(うっかりパスにならないように)。

### 判定と点数(定数は `quiz.rs` の先頭。変えるときはここだけ)

| 判定 | 条件 | 点 | 表示 |
|---|---|---|---|
| 正解 | top-1 がお題と一致 | `POINTS_CORRECT` = 3 | ○ 正解 |
| おしい | top-3 に入る(top-1 は不一致) | `POINTS_CLOSE` = 1 | △ おしい |
| ちがう | 上記以外 | 0 | × ちがう |
| 時間切れ | `ANSWER_TIMEOUT_SECS` = 30 秒 | 0 | × 時間切れ |
| パス | パス操作 | 0 | － パス |

- 1 ゲーム `QUESTIONS_PER_GAME` = 5 問、満点 15 点。判定表示から次へ自動で進むまで `AUTO_ADVANCE_SECS` = 3 秒。「おしい」の順位 `CLOSE_RANK` = 3。
- 点数はユーザー要望「点数があったほうがわかりやすい」への提案値。数値の根拠となる測定はない。
- 判定は文字と形(○ △ × －)でも示し、色だけ・音だけに頼らない。音は使っていない。

### 人間判定「正解にする」

AI が読み違えたときの救済。判定表示中、「おしい」「ちがう」の問にだけボタンを出す(正解は不要、時間切れ・パスは AI の読みが無いので対象外)。押した問は 3 点(正解扱い)。AI の元の判定は記録に残し、総括の表では判定欄に「※人」の印を付ける(例「○ 正解※人」)。

### 出題から除く語(設定として持つ)

初期値: `atsui`, `are-sore`, `kore`, `juu`, `ichi`, `anata`。理由: 2026-10-12 の `rotate` 評価(`docs/experiments/full51-t10-roi.md`)で外れやすかった語(atsui・are-sore↔kore・juu↔ichi・anata)。デモで AI が外すと盛り上がりにくいための**仮の初期値**で、決定ではない。変更方法: `QuizConfig::excluded_tags`、または環境変数 `ATTENTION_QUIZ_EXCLUDE=a,b,c`(カンマ区切り、空文字なら除外なし)。画面からの変更 UI は無い。

### 画面

左のカメラ映像・骨格重ね描きは既存のまま。右側(クイズ用カラム)に、進捗(3 / 5 問目)・合計点、お題(大きく)、残り時間バー(10 秒以下で色が変わる。残り秒数も文字で表示)、判定バナー、AI の上位 3 件(既存の棒グラフ `top_bar_rows` を認識結果と共有)、総括(合計点と、問ごとの「お題 / AI の答え / 判定 / 点」の表)。モード切替に「クイズ」を追加(切替は待機中のみ = 既存の `mode_selectable`)。

### データの境界

ライブの映像・特徴量は保存しない。スコアと記録はメモリのみで、ファイルに書かない。ハイスコア保存はしない。クイズモードの start は通常のライブ認識と同じコマンドで、書き込みうるエンジン(撮影モード)は起動しない。

### 範囲外

手本動画の表示、「AI vs 観客」など別種のクイズ、スコア保存。

## 実装

| ファイル | 内容 |
|---|---|
| `desktop/src/quiz.rs`(新規) | 進行・判定・点数・出題選択(単体テスト 16 件) |
| `desktop/src/quiz_view.rs`(新規) | 操作(開始・パス・正解にする・次へ・スペース)と右カラムの描画、診断ログ |
| `desktop/src/state.rs` | `AppMode::Quiz`、`AppModel::quiz`、Result の振り分け、start コマンド(自動区切り)、`quiz_running` |
| `desktop/src/main.rs` | モード切替ボタン、右カラム、時計(`poll` で `tick`)、スペースキー、環境変数、`top_bar_rows` の共通化 |
| `desktop/src/engine.rs` | 動画入力のスモーク `quiz_scores_a_video_result_against_the_target`(`#[ignore]`) |

開発用の環境変数: `ATTENTION_APP_MODE=quiz`、`ATTENTION_QUIZ_EXCLUDE`、`ATTENTION_APP_CAPTURE_SCRIPT` に `quiz_start` / `quiz_pass` / `quiz_correct` / `quiz_next` を追加(既存の自動操作と同じ書式)、`ATTENTION_APP_DEBUG` で `[quiz]` の状態遷移ログ。

## 検証(2026-10-16、ワーカーによる)

- `cd desktop && cargo test`: 114 件パス・0 件失敗(ignore は実測 8 件。動画スモーク `quiz_scores_a_video_result_against_the_target` を含む。当初「7 件」と書いたのは、この ignored テストを足す前の数字)。変更前のベースラインは 95 件パス。追加 19 件 = quiz.rs 17 + state.rs 2(当初 113 件・18 件追加。レビュー対応で、おしい → 人間判定のテストを 1 件追加。あわせて state.rs の `toggle_allowed` の断言を、何も固定していなかった形からクイズモデルで false を確かめる形に直した)。`cargo build --release` は警告なしで通る。
- 動画スモーク(実エンジン、`raw_jsl/001_konnichiwa/05.mp4`、自動区切り): `result` は top-1 konnichiwa(p=0.996)。**このスモークが使ったモデルは `rec_t10_full` ではない**。`pose_extractor/target/release/pose-extract` が古く(2026-10-13 ビルド。`rec_t10_full` を既定にしたコミット 6891039 を含まないソースでビルドされたとみられる)、旧既定の `rec_demo_center_signspan`(`preprocess.json` なし、前処理は center-square + サイン区間の既定)で動いていた(`ready` の `model_dir` で確認)。そのため、判定ロジック(クイズの流れ・点数)の確認には使えるが、`rec_t10_full` の精度の根拠にはならない。お題を konnichiwa に絞ると「正解」。ohayou に絞ると「正解」にならない(`cargo test --release quiz_scores -- --ignored`。テストが断言しているのは `assert_ne!(… , Correct)` = 正解にならない、までで、「ちがう」かどうかは断言していない)。アプリ全体の run のログでは、お題 ohayou が「ちがう 0 点」になっていた。
- アプリ全体(`ATTENTION_APP_VIDEO` で動画入力。同じ古いバイナリ = `rec_demo_center_signspan`): ready → スタート → お題 konnichiwa → 結果受信 → 「正解 3 点」→ 次へ → 総括、の遷移をログで確認。お題 ohayou の run では「ちがう 0 点」→「正解にする」で「正解※人 3 点」。

### rec_t10_full での再実測と release バイナリの再ビルド(2026-10-16)

- 別ディレクトリ(`cargo build --release --target-dir <スクラッチパッド>`、クリーンビルド 2分23秒)に作った pose-extract で、model_dir なし・環境変数なしの `ready` は `model_dir=rec_t10_full`、`palm_roi=full-frame`、`frame_select=uniform`、`preprocess=preprocess.json`、警告なし。同じ動画(`001_konnichiwa/05.mp4`)の `result` は top-1 konnichiwa **p=0.984**(2位 konbanwa 0.016、3位 iu 0.0001)。クイズの動画スモーク(`quiz_scores_a_video_result_against_the_target`)は、この `rec_t10_full` でも「正解」になった(ohayou に絞ると正解にならない)。
- その後、ユーザーの承認(2026-10-16、ワーカー側でユーザー本人に確認)を得て、`pose_extractor/target/release/pose-extract` を `cd pose_extractor && cargo build --release`(増分 18.9 秒)で**置き換えた**。置き換え前のバイナリ(2026-10-13 16:53 ビルド、sha256 `5b494012…59d7b`)は作業用のスクラッチパッドに退避した(リポジトリ外・一時置き場。恒久の保管ではない)。置き換え後の sha256 は `f6271749…bb0`。
- 置き換え後の既定動作: model_dir なし・環境変数なしの `start` で、`ready` が `model_dir=rec_t10_full`、`preprocess=preprocess.json`、警告なし。`result` は konnichiwa p=0.9838。
- desktop も `cargo build --release` で最新にし、通常起動(`ATTENTION_ROOT` なし、`ATTENTION_APP_MODE=quiz`、動画入力)で、お題 konnichiwa が「○ 正解(3 点)」(`result` p=0.984)、「preprocess.json が無い」メッセージは出なかった。アプリの診断ログは `ready` の `model_dir` を出さないため、アプリ経由で `rec_t10_full` を読んだことは p=0.984(デモモデルは 0.996)と警告の不在からの推定で、`model_dir` の直接確認ではない。
- 上記は 1 本の動画での確認で、`rec_t10_full` の精度の根拠にはならない。

## 既知の表示のずれ

- 出題候補が `questions_per_game`(5)に満たないとき、開始前(Ready)の画面は「5問・満点 15 点」と出す(`quiz_view.rs` の Ready の表示。実際の問題数は `start` で候補の数に縮む)。開始後の「N / 総数」「合計 x / 満点」は実際の問題数で出る。現語彙(51 語)から除外 6 語を引いても候補は 45 語あるので、現状は起きない。

## 今後の候補(設計の変更になるため、今回は実装していない)

古い result が新しい問に混ざる件への案。いずれも挙動を変える。
- 問の切替後 N 秒は結果を捨てる(単純。N の値は未測定)。
- result の区切り区間(`span` / `duration_secs`)と問を出した時刻を突き合わせ、問を出す前に始まったサインの結果は捨てる(エンジンのイベントに時刻の基準が要るかもしれず、未調査)。
- 捨てた result をログに出す(Judged 中・Ready 中)。まず頻度を測るために有効。
- Ready 画面の問題数・満点を、実際の出題数(候補数との小さい方)で出す。

## 未確認

- **実機カメラでの動作は未確認**(ユーザーが確認する)。
- **画面の見た目は未確認**。スクリーンショットが撮れなかった(`screencapture` が `could not create image from display`)ため、レイアウト(右カラム 420px 内の総括の表の列幅、長い語のはみ出し、残り時間バーの見え方)は目視していない。確認したのは状態遷移(ログ)のみ。
- **自動区切りで結果が出るまでの待ちと 30 秒の関係**: ライブ条件(実カメラ・auto 区切り)で、サインを終えてから `result` が来るまでの遅れ、手を出し入れしたときの誤区切りは未測定。
- **古い result の扱い(既知の挙動と未測定)**:
  - 判定表示中(Judged)に届いた result は、無条件に捨てる。ログにも出ない(捨てたことが見えない)。
  - 逆に、古い result が**次の問の答えとして判定される**可能性がある。条件の候補(いずれも頻度は未測定):
    (a) 判定表示中に観客がやり直しのサインをし、その区切りの確定が 3 秒を超えて新しい問が始まった後になる。
    (b) パスを押した時点でサインの途中だった(パスで新しい問へ進み、その後に区切りが確定する)。
    (c) 手を出したままで 6 秒の強制終了(自動区切りの上限)に当たり、確定が新しい問の後になる。
- **お題を出す直前のサインの結果が新しい問に混ざる可能性**: 待機状態のときだけ受け付ける規則は入れたが、次の問に移った直後にエンジンが前のサインの区切りを確定して `result` を返した場合は、新しい問の答えとして判定される。起きる頻度は未測定。
- **人間判定の押す余裕**: 判定表示は 3 秒で自動で次へ進む。「正解にする」を押せるのは判定表示の間だけで、3 秒で足りるかは未確認。足りなければ自動送りを止める、総括からも押せるようにする、などが考えられる(未実装)。
- 点数(3 / 1 / 0)・除外語・5 問・30 秒・3 秒は提案値・仮の値で、デモでの体感は未確認。
- 語彙は `ready.vocab`(51 語。`rec_t10_full` でも、スモークで使った旧既定の `rec_demo_center_signspan` でも 51 語)。stage 3 の語(例: kazoku)はモデルに無いので出題されない。
- エンジン起動時に「モデルに preprocess.json が無いため、既定の前処理を使います」というメッセージが出た。原因は上記の古い release バイナリ(`preprocess.json` の読み込み失敗ではない)。同じバイナリに `start` の `model_dir` で `rec_t10_full` を指定すると `preprocess.json` を読み、警告は出ない。なお、この古いバイナリは環境変数 `ATTENTION_MODEL_DIR` を無視する。ソースの現在の既定(`pose_extractor/src/live.rs`)は `rec_t10_full`。再ビルド後の実測は未記録。
