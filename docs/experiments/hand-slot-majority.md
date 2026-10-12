# 実験: 片手テイクのフレーム別スロットを「テイク内の多数決スロット」に揃えると、未見テイクの精度が変わるか

> **結論(先に)**: 後処理した辞書(510 テイク中 42 テイクを変更)で `rotate` 10 fold × 2 seed = 20 run を回し、
> 元辞書の再測定(同条件 20 run)と比べた。**top-1 は 44.35 vs 44.05 件(/51、差 +0.30、Welch t=0.24、df=33.9、p≈0.81)、
> top-5 は 50.10 vs 49.55 件(差 +0.55、t=1.21、df=35.0、p≈0.23)で、どちらも差の証拠なし**。
> 同等と言い切れる結果ではない(各 20 run の sd は 3〜5 件で、検出できるのは数件以上の差)。
> 「手の不一致」(未見テイクの使用手が学習側に無い語数)は平均 4.10 → 4.00 でほぼ変わらず、
> 不一致の大半(fold 2 の 33〜34 語)は今回の後処理で触れないテイク間の使用手の違いだった。
> **同一人物・同一環境の測定**で、後処理は辞書に対して行ったもの(`frame_hand_feature` 自体は変更していない)。

- **実施日**: 2026-10-12(評価は日付をまたいで 2026-10-13 未明まで)
- **ブランチ**: `docs/v01-acceptance-criteria`(コード変更なし。後処理スクリプトは作業用で、リポジトリには置いていない。全文は §7)
- **前提**: `full51-t10-roi.md`(51語×10テイクの `rotate` 評価)/ `holdout-split.md`(分割方法)

> **この文書の位置づけ**: 測定結果の報告であり、決定事項や推奨ではない。

---

## 1. 背景と仮説

手ポーズ特徴(126 次元 = 左手スロット[0..63] + 右手スロット[63..126])は、`pose_extractor/src/main.rs` の
`frame_hand_feature`(1544 行付近)で、フレームごとにモデルの handedness スコアを 0.5 で区切って左右スロットへ割り当てている。
時間方向の平滑化はない。片手のサインでも 1〜2 フレームだけ反対側のスロットに入る「揺れ」が起き得る。

`pose_dict_t10_full.json`(full-frame、51語×10テイク = 510 テイク、10 フレーム × 126 次元)を調べると、
59/510 テイクで左右両スロットが使われ、53 テイクで隣接サンプル間の L<->R 切替があった(wakaru 9/10、sumimasen 6、arigatou 5 など)。
フレームのスロット判定は「その 63 次元のどれかが非ゼロなら埋まっている」。

**仮説**: 片手のテイクでフレームごとのスロットをテイク内の多数決スロットに揃えると、未見テイクの精度が上がる。

## 2. 後処理(`pose_dict_t10_full_majslot.json`)

元辞書は変更していない。変種辞書は metadata を元からコピーし、`slot_postprocess: "majority-single-hand"` を追加した。

- 各フレームを L(左のみ)/ R(右のみ)/ B(両方)/ .(なし)に分類
- **B フレームを 1 つでも含むテイクは変更しない**(両手サインを壊さないため)
- B が無く L と R の両方があるテイクだけ処理: 件数の多いスロットを多数派とし、少数派スロットのフレーム(63 次元)を
  多数派スロットの位置へ**移動**して元の位置をゼロにする(x 座標の反転はしない)
- L と R が同数のテイクは変更しない
- 処理したテイクは `left_hand_coverage` / `right_hand_coverage` を、辞書の定義(10 フレーム中でそのスロットが非ゼロのフレームの割合)で再計算した。
  元辞書の 510 テイク全てでこの定義と保存値が一致することを確認してから適用した。
  これをしないと評価ログの「手の不一致」が後処理前の値を読み続ける(使用手の判定は `pose_data.rs` の `used_hands` が coverage を読む)

### 処理結果

| 区分 | テイク数 |
|---|---|
| 処理した(B なし、L と R の両方あり、件数差あり) | **42** |
| 変更しなかった: B フレームあり | 141 |
| 変更しなかった: L と R が同数 | 1(ohayou-02: L 1 / R 1) |
| 変更しなかった: 片側のみ、または手なし | 326 |
| 合計 | 510 |

移動したフレームは合計 **47**。処理した 42 テイクの語別の内訳:

| 語 | 処理 | 同数(未変更) | B あり(未変更) | その他(未変更) |
|---|---|---|---|---|
| wakaru | 9 | 0 | 0 | 1 |
| sumimasen | 4 | 0 | 3 | 3 |
| dame / ii / sayounara | 各 3 | 0 | 0 | 各 7 |
| iu | 2 | 0 | 3 | 5 |
| wakaranai | 2 | 0 | 1 | 7 |
| juu | 2 | 0 | 0 | 8 |
| tatoeba | 1 | 0 | 8 | 1 |
| arigatou | 1 | 0 | 6 | 3 |
| ohayou | 1 | 1 | 0 | 8 |
| miru | 1 | 0 | 1 | 8 |
| are-sore / dekiru / go / ichi / kirai / kuru / nana / ni / san / watashi | 各 1 | 0 | 0 | 各 9 |

(タスク記述にあった「両スロット使用 59 テイク」は coverage 基準の数と思われ、今回のフレーム単位分類の区分とは単位が違う。両者を突き合わせてはいない。
tatoeba・arigatou は B ありのテイクが多く、語全体の揺れの多くは今回の対象外だった。)

## 3. 条件

- **評価**: `--eval-holdout <辞書> --input-features handshape --model-size base --standardize --holdout-split rotate --holdout-fold K --seed S`
  (300 エポック、ミラー拡張なし)。K=0〜9、S=1・2 の 20 run を、変種(maj)と元辞書(base)の両方で実施した(計 40 run)
- **base は今回の再測定**: `full51-t10-roi.md` の既測定値(top-1 43.90、sd 3.73、top-5 49.85)とは別の 20 run。同時期に変種と交互に回した
- 同時実行は最大 3 本(wgpu、非決定的)。1 run あたり約 530 秒、40 run の所要は約 115 分(6880 秒)
- `--seed` は結果を固定しないので、run は独立な標本として扱い、Welch の独立 2 標本検定で比較した。p は t 分布の両側確率(数値積分)
- 集計は各 run のログから正規表現(`top-1: (\d+)/51 ... top-5: (\d+)/51`、`手の不一致: (\d+)/51`、`正解: X 予測: Y(`)で自分で行った

## 4. 結果

### 4.1 全体(各 20 run、51 語中の件数)

| 辞書 | top-1 平均 | sd | レンジ | top-5 平均 | sd | レンジ | 手の不一致 平均 |
|---|---|---|---|---|---|---|---|
| base(元辞書、今回の再測定) | 44.05(86.4%) | 4.66 | 33〜48 | 49.55(97.2%) | 1.64 | 45〜51 | 4.10 |
| **maj(後処理)** | 44.35(87.0%) | 3.25 | 36〜49 | 50.10(98.2%) | 1.21 | 47〜51 | 4.00 |
| 参考: 既測定値(`full51-t10-roi.md`) | 43.90(86.1%) | 3.73 | 34〜49 | 49.85(97.7%) | — | 47〜51 | — |

| 比較 | 差(maj − 対象) | Welch t | df | p の目安 | 判定 |
|---|---|---|---|---|---|
| top-1: maj vs base(今回) | +0.30 件 | 0.24 | 33.9 | 0.81 | 差の証拠なし |
| top-5: maj vs base(今回) | +0.55 件 | 1.21 | 35.0 | 0.23 | 差の証拠なし |
| top-1: maj vs 既測定値 | +0.45 件 | 0.41 | 37.3 | 0.68 | 差の証拠なし(既測定値は平均と sd のみ使用) |

- 今回の base の top-1 平均 44.05 は既測定値 43.90 と近いが、sd は 4.66 と既測定値(3.73)より大きかった(同じ条件の別標本でも sd が変わる)
- seed 別の top-1 平均: base は seed1 44.5 / seed2 43.6、maj は seed1 44.0 / seed2 44.7

### 4.2 fold 別(各 2 run = seed 1・2 の平均)

| fold | base top-1 | maj top-1 | base 手の不一致 | maj 手の不一致 |
|---|---|---|---|---|
| 0 | 43.5 | 43.0 | 3 | 3 |
| 1 | 38.5 | 41.5 | 3 | 3 |
| **2** | **33.0**(33, 33) | **37.5**(36, 39) | **34** | **33** |
| 3 | 45.5 | 42.5 | 0 | 0 |
| 4 | 46.5 | 46.0 | 0 | 0 |
| 5 | 46.5 | 45.5 | 0 | 0 |
| 6 | 47.0 | 45.5 | 0 | 0 |
| 7 | 46.5 | 46.0 | 0 | 0 |
| 8 | 47.5 | 48.5 | 1 | 1 |
| 9 | 46.0 | 47.5 | 0 | 0 |

- 手の不一致が突出する fold 2(33〜34 語)は後処理でほとんど変わらなかった(34 → 33)。
  fold 2 は `full51-t10-roi.md` でも手の不一致が突出していた fold で、`last` 単独の問題ではなく(撮影癖との噛み合わせは [[stage2-eval-holdout-pitfall]] 参照)、
  未見テイクが学習側と別の手を使っていることが原因とみられる(fold 2 の未見テイクを個別には確認していない)。そうした違いはテイク**間**の違いなので、
  テイク内の多数決では動かない。fold 2 の top-1 は base 33.0 → maj 37.5 と数値上は上がっているが、各 2 run のみで検定していない
- fold 1 は +3.0、fold 3 は −3.0 など、fold ごとの差は両方向に出ており、2 run ずつなので fold 単位の差は読み取れない

## 5. 外れ方(top-1 を外した語、各 20 run 合計。全 51 語分の外れ合計は base 139 / maj 133)

後処理の対象になった 22 語(§2 の表)の top-1 外れ合計は base 58 → maj 52。

注目していた語(sumimasen / wakaru / arigatou / tatoeba / dame / ii / iu / sayounara)の外れ回数:

| 語 | base | maj |
|---|---|---|
| sumimasen | 6 | 5 |
| wakaru | 1 | 0 |
| arigatou | 4 | 4 |
| tatoeba | 2 | 2 |
| dame | 1 | 1 |
| ii | 1 | 1 |
| iu | 3 | 4 |
| sayounara | 0 | 0 |

- 差が 3 回以上あった語(各 20 run 合計): dekinai-muzukashii 6 → 1、dekiru 6 → 3、juu 8 → 5、watashi 5 → 2(いずれも減少)、
  **nomu 3 → 8(増加)**。このうち後処理の対象になったのは dekiru・juu・watashi のみ(各 1〜2 テイク)。
  dekinai-muzukashii と nomu は後処理で変更したテイクが無い語なので、この差は後処理による変化ではなく run 間のばらつきの可能性がある
- 外れ上位は両方とも atsui(11 / 12)、are-sore(8 / 9)、juu(8 / 5)、ok(7 / 6)、iku(6 / 6)、sumimasen(6 / 5)で、`full51-t10-roi.md` §3 と同じ顔ぶれ
- 語別の差は 20 run の合計回数で、語ごとの有意性は検定していない

## 6. 言えないこと

- **同一人物・同一環境の測定**: 全テイクが同一人物・同一環境。別人・別日への汎化は測っていない
- **揺れと実際の持ち替えを区別していない**: 後処理は「テイク内で少数派のスロットに入ったフレームは検出の揺れだ」と見なして移動する。
  実際に撮影者がテイクの途中で手を持ち替えた場合も同じ扱いになる(stage 1 の調査では持ち替えが真因のテイクがあった。[[stage1-hand-switching-finding]])。
  処理した 42 テイクを個別に動画で確認してはいない
- **ライブ側には未実装で未測定**: ライブ認識(区切った録画の等間隔 10 枚)への多数決の適用は実装しておらず、ライブでの効果は測っていない
- **後処理は辞書に対してであり、`frame_hand_feature` 自体の変更ではない**: 抽出段で handedness を平滑化・多数決する実装とは別物。
  抽出段で行えば、手の検出結果や 10 フレームの選択に影響が及び得るが、それは測っていない。また、辞書は選択された 10 フレームだけを見ているので、
  動画全体の多数決とは違う結果になり得る
- **多数決で寄せた結果が学習・評価の両側に入る**: 変種辞書では、学習テイクも未見テイクも同じ後処理を受けている。
  未見テイクに対して正解ラベルを使った処理はしていないが、評価側のデータも加工後のものなので、実運用(未見の入力を加工なしで受ける)との差は残る
- **差の証拠がないことは、同等と言い切ることとは違う**。20 run の sd は 3〜5 件で、1〜2 件程度の差は検出できない。
  変更したテイクは 510 中 42(8%)、移動したフレームは約 5,100 中 47(約 0.9%)と、変更が小さい点も効果が出にくい理由として考えられる(切り分けていない)
- 手の不一致の主因(テイク全体で使用手が違う、fold 2)は、このテイク内多数決の対象外だった
- baseline は今回追加で 20 run を取り直した。既測定値との差(sd 3.73 vs 4.66)は、同条件でも標本によって動くことを示す

## 7. 再現手順

```bash
# 後処理(スクラッチパッドのスクリプト。全文は下)
python3 -I majslot.py data/pose_dict_t10_full.json data/pose_dict_t10_full_majslot.json
# 評価(fold 0〜9、seed 1・2。maj は majslot、base は pose_dict_t10_full.json を渡す)
cd transformer_burn
D=data/pose_dict_t10_full_majslot.json
./target/release/transformer_burn --eval-holdout $D --input-features handshape --model-size base --standardize --holdout-split rotate --holdout-fold 0 --seed 1
```

辞書 `pose_dict*.json` は `.gitignore` で非追跡(`transformer_burn/data/pose_dict_t10_full_majslot.json` もローカルのみ)。

後処理スクリプト(`majslot.py`):

```python
import json, sys
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
meta = dict(d["metadata"]); meta["slot_postprocess"] = "majority-single-hand"
out = {}
for k, v in d["tags"].items():
    seq = v["sequence"]
    cls = []
    for f in seq:
        l = any(x != 0 for x in f[:63]); r = any(x != 0 for x in f[63:])
        cls.append("B" if l and r else "L" if l else "R" if r else ".")
    nL, nR, nB = cls.count("L"), cls.count("R"), cls.count("B")
    nv = dict(v)
    if nB == 0 and nL and nR and nL != nR:
        maj = "L" if nL > nR else "R"
        newseq = []
        for f, c in zip(seq, cls):
            f = list(f)
            if c == "L" and maj == "R":
                f[63:] = f[:63]; f[:63] = [0.0] * 63
            elif c == "R" and maj == "L":
                f[:63] = f[63:]; f[63:] = [0.0] * 63
            newseq.append(f)
        nv["sequence"] = newseq
        nv["left_hand_coverage"] = sum(any(x != 0 for x in f[:63]) for f in newseq) / len(newseq)
        nv["right_hand_coverage"] = sum(any(x != 0 for x in f[63:]) for f in newseq) / len(newseq)
    out[k] = nv
json.dump({"metadata": meta, "tags": out}, open(dst, "w"), ensure_ascii=False)
```
