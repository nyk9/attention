# 実験: 51語×10テイクで `rotate` 評価し、ROI を center-square と full-frame で比較する

> **結論(先に)**: 51語×10テイクの `rotate`(10 fold)で、**top-1 は約 86%、top-5 は約 98%**。
> pipe-v2 の基準値(top-1 70% / top-5 90%)を上回った。**ROI は center-square と full-frame で差の証拠がない**
> (`rotate` top-1 の差 +0.25 件、Welch t=0.21、各 20 run)。ただし**同一人物・同一環境の測定**で、
> 別人への汎化は測っておらず、別日への汎化は §6 の限られた条件(学習 3 テイク)のみ。

- **実施日**: 2026-10-12
- **ブランチ**: `main`(コード変更なし。評価コマンドを実行しただけ)
- **前提**: `take-count-10words.md`(10語の予備測定)/ `holdout-split.md`(分割方法)
- **決定との対応**: 「pipe-v2 のクリア判定は `rotate` で行う」「ROI は 51語が 10 テイクそろった時点で両方で測って比較」(いずれも 2026-10-12 ユーザー決定)に沿った測定
- **実行**: Sonnet のサブエージェント 2 本(辞書作成と評価)。私が全ログから top-1・top-5 の平均と差を再集計し、報告値と一致を確認した

> **この文書の位置づけ**: 測定結果の報告であり、決定事項ではない。
> **測定後のユーザー決定(2026-10-12)**: pipe-v2 は**クリア**(同一セッション条件での判定)、ROI の既定は **full-frame のまま変えない**。
> 決定の記録は `CLAUDE.md` / `development-plan.md` にある。

---

## 1. 条件

- **データ**: `index.tsv` の `quality_flag=ok` 510 テイク(51 語 × 各 10 テイク、`ng` 除外、テイク番号は raw_jsl のまま)
- **辞書**: `pose_dict_t10_center.json`(`palm_roi=center-square`)と `pose_dict_t10_full.json`(`palm_roi=full-frame`)。
  どちらも `frame_select=uniform`、510 タグ、51 ラベル各 10 件、10 フレーム × 126 次元。作成は各約 17 分(1.97 秒/本)
- **設定**: `--input-features handshape --model-size base --standardize`、300 エポック、ミラー拡張なし
- **`rotate` の fold**: **10 fold**(`--holdout-fold 0〜9`、各回で各語の 1 テイクを未見)。fold 数は暫定の選択で、
  決定ではない(`holdout-split.md` §5 の「10 テイク時の fold 設計」は未確定のまま)。fold ごとに seed 1・2 の 2 回、計 20 run
- **`last`**: テイク 10 を未見にし、seed 1〜5 の 5 run
- `--seed` は結果を固定しないので、run は独立な標本として扱った

## 2. 結果

各セルは 51 語中の件数の平均。

| ROI | 分割 | run 数 | top-1 平均 | top-1 レンジ | top-5 平均 | top-5 レンジ |
|---|---|---|---|---|---|---|
| center-square | rotate(10 fold) | 20 | 44.15(**86.6%**) | 37〜49 | 49.90(97.8%) | 48〜51 |
| full-frame | rotate(10 fold) | 20 | 43.90(**86.1%**) | 34〜49 | 49.85(97.7%) | 47〜51 |
| center-square | last | 5 | 48.4(94.9%) | 46〜50 | 50.8(99.6%) | 50〜51 |
| full-frame | last | 5 | 46.4(91.0%) | 45〜49 | 50.2(98.4%) | 49〜51 |

| 比較(center − full) | 差 | Welch t | 備考 |
|---|---|---|---|
| rotate top-1 | +0.25 件 | 0.21 | 差の証拠なし |
| rotate top-5 | +0.05 件 | 0.15 | 差の証拠なし |
| last top-1 | +2.0 件 | 1.81 | 各 5 run。弱い証拠にとどまる |
| last top-5 | +0.6 件 | 1.41 | 同上 |

- **標準偏差(rotate、20 run)**: center 3.83 件 / full 3.73 件。run 単位のレンジは広く(full の最小は 34 件 = 67%)、単発 run は数 pt 動く
- **fold 別 top-1 平均(2 seed)**: center は fold 0 の 38.0 から fold 6・9 の 48.0 まで、full は fold 2 の 35.0 から fold 8 の 47.5 まで。fold 2 は full で最低(35.0)、center では fold 0 が最低(38.0)で fold 2 は 37.5
- **手の不一致**(未見テイクの使用手が学習側に無い語数、/51): fold 2 だけ 34 と突出(どちらの ROI も)。ほかは 0〜4。
  `last` では center が 1、full が 0
- 3 テイク時代(2026-08-22、51 語)の `rotate` 平均は top-1 72.0% / top-5 90.6%(`holdout-split.md`、center-square、3 fold)。
  テイク数が 3→10 になった点が違うが、fold 数・辞書の再作成など条件も違うので、直接の差としては扱わない

## 3. 外れる語

`rotate` 全 20 run で top-1 を外した正解語(多い順。括弧内は最頻の誤り先):

| 順位 | center-square | full-frame |
|---|---|---|
| 1 | atsui 14(dekinai-muzukashii) | are-sore 11(kore) |
| 2 | are-sore 10(kore) | atsui 9(dekinai-muzukashii) |
| 3 | juu 10(ichi) | anata 9(chigau) |
| 4 | anata 8(juu) | juu 8(watashi) |
| 5 | ichi 6(juu) | sumimasen 8(arigatou) |

- 共通して外れやすい語: **atsui(→dekinai-muzukashii)、are-sore(↔kore)、juu(↔ichi)、anata**。
  are-sore ↔ kore、ichi ↔ juu、chiisai → ookii は、動作が近い語どうしの混同に見えるが、原因は切り分けていない
- 1 回でも外れた語は full-frame で 51 語中 37 語
- **CenterSquare の切れ**(手の 3.7% がクロップ外、危険群は方向・指示語。iku / kuru / are-sore など)との関係: iku は両 ROI で外れ上位に出た
  (center で 5 回、full でも 5 回)が、center だけで悪化した形跡は今回の数値からは読み取れない

## 4. 言えないこと

- **別人・別日の汎化**: 全テイクが同一人物・同一環境。10 テイクの大半は同じ日の連続撮影で、`rotate` の未見テイクは
  学習テイクと時間的に近い。この数値は「同じ人が同じ条件でもう一度打ったら」に近い
- **ROI の優劣**: 差の証拠がないことは、同等と言い切ることとは違う。語彙拡張(方向・指示語の増加)で切れの影響が
  変わるかは測っていない。tiles3 や person など他の ROI は今回測っていない
- **fold 数の影響**: 10 fold は暫定。k-fold(N 件ずつ)にした場合の値は測っていない
- 外れた語ペアの原因(動作の類似か、データか、左右手か)は切り分けていない
- 実行時間は並行実行のため 1 run あたり 312〜532 秒で、単独実行の目安(51 語で 1〜2 分)より長かった

## 5. 測定後の決定と、残っている未確定

- **決定済み(2026-10-12 ユーザー)**: pipe-v2 はクリア。ROI の既定は full-frame のまま変えない
- **作成済み**: 510 テイクで再学習した `models/rec_t10_full`(`rec_full` は未更新のまま残置)
- 未確定: 74 語(stage 3 の 23 語を撮影した後)での再測定の要否 / `rotate` の fold の切り方(10 fold は暫定)

## 6. 日をまたぐ測定(追加、同日実施)

古い撮影日(2026-07-02 / 08-01)の各語 3 テイク(153 件)だけで学習し、新しい日(2026-10-12)の各語 7 テイク(357 件)を未見にした。
`--holdout-split last --holdout-per-label 7`。学習が各語 3 テイクしかない点に注意(`rotate` は各語 9 テイクで学習)。seed 1〜5。

| ROI | top-1 平均 | top-1 レンジ | top-5 平均 | top-5 レンジ | 手の不一致 |
|---|---|---|---|---|---|
| center-square | 206.0/357(**57.7%**) | 194〜218 | 292.8(82.0%) | 285〜299 | 12/357 |
| full-frame | 203.6/357(**57.0%**) | 201〜211 | 294.8(82.6%) | 288〜306 | 5/357 |

- ROI の差: top-1 +2.4 件(Welch t=0.52)、top-5 −2.0 件(t=−0.50)。差の証拠なし
- **基準値(70%/90%)に届いていない。** ただし**学習が 3 テイクしかない条件**なので、10 テイク学習の日またぎ精度は測れていない。
  参考として、同じ 3 テイク学習の同日評価(51 語、2026-08-22 の `last`)は top-1 66〜69% で、日をまたぐと **約 10pt 低い** ようにみえる
  (未見テイクも辞書も違うので、厳密な対比ではない)
- 外れやすい語: are-sore(35/35 件が外れ、→kore)、kore(→kiku)、chiisai(→kaku)、anata、watashi(→juu)、ichi(→juu)。
  full-frame だけ konbanwa→konnichiwa が 29 件(35 件中)あった。5 run 合計の件数で、語ごとの未見は 7 テイク×5 run = 35 件
- **解釈の限界**: 日をまたぐ劣化の原因(立ち位置・カメラ位置・照明のずれか、学習テイク数の少なさか)は切り分けていない。
  学習 10 テイク(うち複数日)で新しい日を測るには、**別の日に撮った新しいテイク**が要る

## 7. 再現手順

```bash
# 辞書(ok テイクをフラットに並べた $TMPDIR/palm_roi_dict_takes/ が必要。無ければ run_export が作る。
# 古い同名ディレクトリが残っていると再利用されて古いデータで作られるので注意)
cd pose_extractor
DICT_ROI_MODE=center DICT_OUT=../transformer_burn/data/pose_dict_t10_center.json cargo test build_dict_with_palm_roi --release -- --ignored --nocapture
DICT_ROI_MODE=full   DICT_OUT=../transformer_burn/data/pose_dict_t10_full.json   cargo test build_dict_with_palm_roi --release -- --ignored --nocapture
# 評価(fold 0〜9、seed 1・2 を回す)
cd ../transformer_burn
D=data/pose_dict_t10_center.json
./target/release/transformer_burn --eval-holdout $D --input-features handshape --model-size base --standardize --holdout-split rotate --holdout-fold 0 --seed 1
```
