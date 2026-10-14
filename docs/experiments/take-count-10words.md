# 実験: 10語×10テイクで未見テイクの精度はどう変わるか(テイク数 3→10 の効果の予備測定)

> **結論(先に)**: 同じ10語で、学習が 3 テイク前後のとき top-1 は約 66%、10 テイクあるとき 92〜96%。
> top-5 はどちらも約 99〜100%。**ただし10語・同一人物・同一環境・同日撮影の測定であり、51語での効果や
> 別人・別日への汎化は測っていない。** pipe-v2 基準(51語での top-1 70% / top-5 90%)のクリア判定には使えない。

- **実施日**: 2026-10-12
- **ブランチ**: `main`(コード変更なし。評価コマンドを実行しただけ)
- **実行**: Sonnet のサブエージェント 3 本に評価コマンドを振り、ログを集計させた。top-1 の合計はログから検算済み(59 / 96 / 138 / 185)。top-5 は検算していない

> **この文書の位置づけ**: 測定結果の報告であり、**決定事項ではない**。`CLAUDE.md` は書き換えていない。

---

## 1. 条件

- **語**: word_id 001〜010 の 10 語(konnichiwa, ohayou, konbanwa, sayounara, arigatou, sumimasen, otsukaresama, onegaishimasu, watashi, anata)。11 語目の kare も 10 テイクあるが、辞書に入っていないので対象外
- **辞書**: `data/pose_dict_v2_center.json`(Palm ROI = center-square、frame_select = uniform、223 タグ。最終撮影 06:23 の後の 06:36 に作成)から上記 10 語だけを抜き出した
  - `w10_t10`: 100 テイク(10 語 × 10 テイク)
  - `w10_t3`: テイク番号 1〜3 のみ。ng 除外でテイク番号が欠けるため **24 テイク**(30 ではない)
- **設定**: `--input-features handshape --model-size base --standardize`、300 エポック、ミラー拡張なし
- `--seed` は結果を固定しないので、run ごとに独立な標本として扱った

## 2. 結果

| 条件 | 学習 / 未見 | run 数 | top-1 平均 | top-1 レンジ | top-5 平均 |
|---|---|---|---|---|---|
| A: 3 テイク、`last` | 15 / 9 | 10 | 65.6%(5.9/9) | 5〜7 | 98.9% |
| B1: 10 テイク、テイク10 を未見(`last`) | 90 / 10 | 10 | **96%**(9.6/10) | 9〜10 | 100% |
| B2: 10 テイク、テイク8〜10 を未見(`last`、`--holdout-per-label 3`) | 70 / 30 | 5 | **92%**(27.6/30) | 26〜29 | 99.3% |
| C: 10 テイク、`rotate` fold 0〜9 × seed 2 通り | 90 / 10 | 20 | **92.5%**(185/200) | run 単位 8〜10 | 100% |

- C の fold 別 top-1 平均は 85%(fold 0, 1, 3)〜100%(fold 4, 5, 7)
- A の学習は語によって 1〜2 テイクしかない(arigatou, ohayou, sumimasen は学習 1 テイク)

## 3. 外れ方

- **A**: arigatou→onegaishimasu と ohayou→otsukaresama が 10 run すべてで外れた(どちらも学習 1 テイクの語)。otsukaresama も 9/10 run で外れた
- **B1**: 外れは sumimasen→onegaishimasu のみ(10 run 中 4 run、onegaishimasu 0.59〜0.94)
- **B2**: anata が 15 件中 8 件外れ(→ ohayou / watashi)、sumimasen と konbanwa が各 2 件。B1・C ではほぼ外れない語なので、**分割によって変わる**
- **C**: 外れ 15 件の内訳は sumimasen→onegaishimasu 3、arigatou→onegaishimasu 3、otsukaresama→sumimasen 2、onegaishimasu→arigatou 2、ほか 5 件(各 1)。anata, konbanwa, konnichiwa は 20 run で 1 度も外れなかった。誤りの予測先は onegaishimasu が最多(6 件)
- **手の不一致**(未見テイクの使用手が学習側に無い件数): A 5/9、B1 1/10、B2 1/30、C は fold により 0〜4/10。C では不一致の数と top-1 の低さが対応しなかった(fold 2 は 4/10 だが top-1 9〜10/10)

## 4. 言えないこと

- **51語・74語での効果**: 10 語は語彙が少なく易しい。51 語の 3 テイク(top-1 68.8%、2026-08-22)と 10 語の 3 テイク(65.6%)が近い値だったことは、語数による難しさの差が小さい可能性を示すが、確認はしていない
- **A と B・C の厳密な対比**: A は未見が 9 件・学習が約 15 件と少なく、分割も違う。テイク数の効果と分母・分割の効果が混ざっている
- **別人・別日への汎化**: 全テイクが同一人物・同一環境(100 テイクのうち 70 は同日の 2026-10-12 に撮影。残り 30 は 2026-07-02)
- 外れた語ペアの原因(動作の類似か、データか)は切り分けていない

## 5. 未確定(ユーザー判断待ち)

- 51 語・74 語がそろった時点で `rotate` による評価を取り、pipe-v2 の判定材料にするか
- pipe-v2 基準のクリア判定の方法(`last` 単独か、`rotate` 平均か)

## 6. 再現手順

```bash
cd transformer_burn
C="--input-features handshape --model-size base --standardize"
D=<10語に絞った辞書>.json
./target/release/transformer_burn --eval-holdout $D $C --holdout-split last --seed 1
./target/release/transformer_burn --eval-holdout $D $C --holdout-split last --holdout-per-label 3 --seed 1
./target/release/transformer_burn --eval-holdout $D $C --holdout-split rotate --holdout-fold 0 --seed 1
```

1 run は単独で約 80 秒、3 本並列では 115〜155 秒だった。10 語への絞り込みは、辞書の `tags` をラベル(キー末尾の `-NN` を除く)で
フィルタした JSON を作るだけでよい(`metadata.tag_count` も合わせて更新する)。
