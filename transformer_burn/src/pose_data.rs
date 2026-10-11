use crate::config::POSE_FEATURE_DIM;
use crate::feature_stats::FeatureStats;
use crate::handshape_features::{handshape_sequence_with_parts, DescriptorParts, InputFeatures};
use crate::tag_vocabulary::TagVocabulary;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

/// pose_extractor `build-dict` が出力する JSON のスキーマ(読み込みに必要な部分のみ)。
/// 出力側: pose_extractor/src/main.rs の PoseDict / PoseDictMeta / TagEntry
#[derive(Debug, Deserialize)]
struct PoseDictFile {
    metadata: PoseDictMeta,
    tags: BTreeMap<String, PoseDictEntry>,
}

#[derive(Debug, Deserialize)]
struct PoseDictMeta {
    /// 各エントリのフレーム数 T(全エントリ共通)
    frames: usize,
    /// 1 フレームの特徴量次元(= 126 のはず)
    feature_dim: usize,
}

#[derive(Debug, Deserialize)]
struct PoseDictEntry {
    /// [T フレーム][feature_dim] の正規化済みハンドランドマーク列
    sequence: Vec<Vec<f32>>,
    left_hand_coverage: f32,
    right_hand_coverage: f32,
}

/// 認識モデルの学習サンプル 1 件 = (手ポーズ列, タグラベル)
#[derive(Debug, Clone)]
pub struct PoseSample {
    /// [T * feature_dim] にフラット化済みの手ポーズ列
    pub features: Vec<f32>,
    /// タグラベル(テイク番号を剥がした後の名前。例: "0205-01" → "0205")
    pub label: String,
}

/// 認識モデル用の学習データ一式
pub struct PoseTrainingData {
    pub samples: Vec<PoseSample>,
    /// 1 サンプルのフレーム数 T(エンコーダのシーケンス長)
    pub frames: usize,
    /// 1 フレームの特徴量次元(= POSE_FEATURE_DIM)
    pub feature_dim: usize,
}

/// ファイル名 stem からテイク番号サフィックスを剥がしてラベルにする。
/// 例: "0205-01" → "0205"、"ありがとう_2" → "ありがとう"、"挨拶" → "挨拶"
/// 区切りは '-' または '_' で、その後ろが全部数字のときだけ剥がす。
// [TEST向き] 純粋な文字列処理。境界(数字なし・全部数字・多段サフィックス)
pub fn label_from_stem(stem: &str) -> String {
    if let Some(pos) = stem.rfind(['-', '_']) {
        let suffix = &stem[pos + 1..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            return stem[..pos].to_string();
        }
    }
    stem.to_string()
}

/// stem 末尾のテイク番号を取り出す。`label_from_stem` と同じ区切り規則。
/// 例: "0205-01" → Some(1)、"ありがとう_2" → Some(2)、"挨拶" → None、"tag-" → None
// [TEST向き] 純粋な数値抽出。境界(数字なし・先頭ゼロ・空サフィックス・大きい番号)
pub fn take_from_stem(stem: &str) -> Option<u32> {
    let pos = stem.rfind(['-', '_'])?;
    let suffix = &stem[pos + 1..];
    if suffix.is_empty() || !suffix.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    suffix.parse::<u32>().ok()
}

/// 手が「使われている」とみなす最小カバレッジ。
/// build-dict は各テイクから均等 T フレーム(既定10)をサンプルするので、0.2 ≒ 2/10 フレーム
const USED_HAND_THRESHOLD: f32 = 0.2;

/// テイクの左右手カバレッジから「使われた手」を分類する
// [TEST向き] 純粋な分類。境界(閾値ちょうど・両手・両手なし)
fn used_hands(left_cov: f32, right_cov: f32) -> &'static str {
    match (
        left_cov >= USED_HAND_THRESHOLD,
        right_cov >= USED_HAND_THRESHOLD,
    ) {
        (true, true) => "LR",
        (true, false) => "L",
        (false, true) => "R",
        (false, false) => "-",
    }
}

/// pose dict をラベル別に集計し、各テイクの左右手カバレッジ表を表示。
/// テイク間で「使われた手」が食い違うラベル(anata で起きた左右手ゆれ)を警告し、
/// 食い違ったラベル名の一覧を返す(テスト用)。学習は行わない診断専用
pub fn inspect_dict(path: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let (entries, frames, feature_dim) = read_raw_entries(path)?;
    println!(
        "pose dict: {} (frames={} feature_dim={})",
        path.display(),
        frames,
        feature_dim
    );

    let mut by_label: BTreeMap<String, Vec<RawEntry>> = BTreeMap::new();
    for e in entries {
        by_label.entry(e.label.clone()).or_default().push(e);
    }

    let label_count = by_label.len();
    let mut flagged = Vec::new();
    for (label, mut group) in by_label {
        // テイク番号昇順(番号なしは後ろに stem 順)。holdout 分割と同じ「後ろが未見」感覚で読めるように
        group.sort_by(|a, b| match (a.take, b.take) {
            (Some(x), Some(y)) => x.cmp(&y),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.stem.cmp(&b.stem),
        });

        println!("\nラベル '{}' ({}テイク):", label, group.len());
        let mut kinds = std::collections::BTreeSet::new();
        for e in &group {
            let hands = used_hands(e.left_cov, e.right_cov);
            kinds.insert(hands);
            println!(
                "  {:<24} L={:>3.0}% R={:>3.0}%  [{}]",
                e.stem,
                e.left_cov * 100.0,
                e.right_cov * 100.0,
                hands
            );
        }
        if kinds.len() > 1 {
            let kinds_str: Vec<&str> = kinds.into_iter().collect();
            println!(
                "  警告: テイク間で使われた手が食い違っています ({})。126次元特徴の左右枠が\n        テイクごとに変わるため、未見テイクの誤認識(anata 問題)の原因になります",
                kinds_str.join(" / ")
            );
            flagged.push(label);
        }
    }

    println!("\n--- 集計 ---");
    if flagged.is_empty() {
        println!("{}ラベル中、使われた手の食い違い: なし", label_count);
    } else {
        println!(
            "{}ラベル中 {}ラベルで使われた手が食い違い: {}",
            label_count,
            flagged.len(),
            flagged.join(", ")
        );
    }
    Ok(flagged)
}

/// 検証済みの 1 エントリ(分割前の中間表現)
struct RawEntry {
    stem: String,
    label: String,
    take: Option<u32>,
    features: Vec<f32>,
    left_cov: f32,
    right_cov: f32,
}

/// build-dict JSON を読み込み・検証して中間エントリ列にする(load と分割で共用)
fn read_raw_entries(
    path: &Path,
) -> Result<(Vec<RawEntry>, usize, usize), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("pose dict が読み込めません {}: {}", path.display(), e))?;
    let dict: PoseDictFile = serde_json::from_str(&content)?;

    if dict.metadata.feature_dim != POSE_FEATURE_DIM {
        return Err(format!(
            "feature_dim 不一致: dict={} 期待={}",
            dict.metadata.feature_dim, POSE_FEATURE_DIM
        )
        .into());
    }

    let mut entries = Vec::new();
    for (stem, entry) in &dict.tags {
        // 形状の検証: フレーム数と次元が metadata 通りか
        if entry.sequence.len() != dict.metadata.frames {
            return Err(format!(
                "'{}' のフレーム数が不正: {} (期待 {})",
                stem,
                entry.sequence.len(),
                dict.metadata.frames
            )
            .into());
        }
        for (i, frame) in entry.sequence.iter().enumerate() {
            if frame.len() != dict.metadata.feature_dim {
                return Err(format!(
                    "'{}' フレーム{} の次元が不正: {} (期待 {})",
                    stem,
                    i,
                    frame.len(),
                    dict.metadata.feature_dim
                )
                .into());
            }
        }

        entries.push(RawEntry {
            stem: stem.clone(),
            label: label_from_stem(stem),
            take: take_from_stem(stem),
            features: entry.sequence.iter().flatten().copied().collect(),
            left_cov: entry.left_hand_coverage,
            right_cov: entry.right_hand_coverage,
        });
    }

    if entries.is_empty() {
        return Err("pose dict にエントリがありません".into());
    }
    Ok((entries, dict.metadata.frames, dict.metadata.feature_dim))
}

/// 1 枠(片手分)の次元数(21 点 × [x, y, z])
const HAND_SLOT_DIM: usize = POSE_FEATURE_DIM / 2;

/// 手ポーズ特徴量を左右反転する(ミラー拡張)。
///
/// レイアウト(pose_extractor `frame_hand_feature` 側): 1 フレーム = 左手枠[0..63] +
/// 右手枠[63..126]、各枠は 21 点 × [x, y, z]、x は width で正規化済み(概ね [0, 1])。
/// 反転規則: 枠ごと入れ替え(利き手が反転したとみなす)+ 各枠内の x 座標を `1.0 - x` に反転。
/// y, z は水平反転で値が変わらないためそのまま。検出なし(全ゼロ)の枠は反転してもゼロのまま
/// (ゼロを 1.0 反転して偽の座標を作らないための安全策)。
///
/// 撮影時の左右手ゆれ([[stage1-hand-switching-finding]])に対してモデルを
/// 汎化させるための学習データ拡張(同じサインを逆の手で行った場合の疑似データを作る)
// [TEST向き] 純粋関数。境界(片手のみ・両手・両手ゼロ・複数フレーム・反転の対合性)
pub fn mirror_features(features: &[f32]) -> Vec<f32> {
    assert_eq!(
        features.len() % POSE_FEATURE_DIM,
        0,
        "features の長さが feature_dim({}) の倍数ではありません: {}",
        POSE_FEATURE_DIM,
        features.len()
    );
    let frames = features.len() / POSE_FEATURE_DIM;
    let mut out = vec![0.0f32; features.len()];
    for t in 0..frames {
        let frame = &features[t * POSE_FEATURE_DIM..(t + 1) * POSE_FEATURE_DIM];
        let (left, right) = frame.split_at(HAND_SLOT_DIM);
        let out_frame = &mut out[t * POSE_FEATURE_DIM..(t + 1) * POSE_FEATURE_DIM];
        let (out_left, out_right) = out_frame.split_at_mut(HAND_SLOT_DIM);
        mirror_hand_slot(right, out_left);
        mirror_hand_slot(left, out_right);
    }
    out
}

/// 片手枠(63次元)を反対側の枠へ x 反転してコピーする。全ゼロ(未検出)ならゼロのまま
fn mirror_hand_slot(src: &[f32], dst: &mut [f32]) {
    if src.iter().all(|&v| v == 0.0) {
        return;
    }
    for k in 0..HAND_SLOT_DIM / 3 {
        dst[k * 3] = 1.0 - src[k * 3]; // x
        dst[k * 3 + 1] = src[k * 3 + 1]; // y
        dst[k * 3 + 2] = src[k * 3 + 2]; // z
    }
}

/// テイク単位の train/held-out 分割の結果
pub struct HoldoutSplit {
    /// 学習に使うデータ(held-out を除いたテイク)
    pub train: PoseTrainingData,
    /// 評価に使う未見テイク
    pub test: PoseTrainingData,
    /// (ラベル, train件数, held-out件数)
    pub per_label: Vec<(String, usize, usize)>,
    /// テイク不足で held-out を作れず train のみに入れたラベル
    pub train_only: Vec<String>,
    /// 手の不一致: held-out テイクのうち、同ラベルの train テイクに「同じ使用手(L/R/LR)」が
    /// 1 つも無いもの。(held-out の stem, その使用手)。学習=右手のみ/未見=左手のみのような
    /// 系統的ズレで評価が歪んでいる度合いの直接指標
    pub hand_mismatch: Vec<(String, &'static str)>,
}

/// held-out テイクの選び方
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoldoutSplitMode {
    /// ラベルごとにテイク番号昇順の後ろ N 件(従来の既定)
    Last,
    /// ラベルごとに擬似乱数で N 件(`seed` と ラベル名から決定的に決まる)
    Random { seed: u64 },
    /// ラベルごとにテイク番号昇順の `fold` 番目から連続 N 件(循環)。
    /// fold を 0..max_takes で回せば、各テイクがちょうど 1 回 held-out になる(全ラベルが同数テイクのとき)
    Rotate { fold: usize },
}

impl HoldoutSplitMode {
    /// CLI 文字列から解決する。`seed` は random、`fold` は rotate で使う
    pub fn parse(name: &str, seed: u64, fold: usize) -> Result<Self, String> {
        match name {
            "last" => Ok(Self::Last),
            "random" => Ok(Self::Random { seed }),
            "rotate" => Ok(Self::Rotate { fold }),
            _ => Err(format!(
                "未知の分割方法です: \"{}\"(last / random / rotate から選んでください)",
                name
            )),
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::Last => "last".to_string(),
            Self::Random { seed } => format!("random(seed={})", seed),
            Self::Rotate { fold } => format!("rotate(fold={})", fold),
        }
    }
}

/// xorshift64*。依存を増やさず分割用の決定的な擬似乱数を得るだけの用途
struct SplitRng(u64);

impl SplitRng {
    /// seed とラベル名(FNV-1a)から初期状態を作る。0 は xorshift の不動点なので避ける
    fn new(seed: u64, label: &str) -> Self {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in label.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let mut x = h ^ seed.wrapping_mul(0x9E3779B97F4A7C15);
        if x == 0 {
            x = 0x9E3779B97F4A7C15;
        }
        let mut r = Self(x);
        // 初期の相関を捨てる
        for _ in 0..4 {
            r.next();
        }
        r
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

/// 番号付きテイク `count` 個(昇順)のうち held-out にする位置の集合を返す
// [TEST向き] 純粋関数。last の不変・random の決定性と件数・rotate の循環
fn held_out_positions(
    mode: HoldoutSplitMode,
    label: &str,
    count: usize,
    n: usize,
) -> std::collections::BTreeSet<usize> {
    let n = n.min(count);
    match mode {
        HoldoutSplitMode::Last => (count - n..count).collect(),
        HoldoutSplitMode::Rotate { fold } => (0..n).map(|j| (fold + j) % count).collect(),
        HoldoutSplitMode::Random { seed } => {
            let mut rng = SplitRng::new(seed, label);
            let mut idx: Vec<usize> = (0..count).collect();
            // 先頭 n 個だけ部分 Fisher-Yates
            for i in 0..n {
                let j = i + (rng.next() % (count - i) as u64) as usize;
                idx.swap(i, j);
            }
            idx[..n].iter().copied().collect()
        }
    }
}

impl PoseTrainingData {
    /// build-dict の JSON を読み込んで学習データにする。
    /// 同じラベルのエントリ(複数テイク)はそれぞれ独立したサンプルになる。
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let (entries, frames, feature_dim) = read_raw_entries(path)?;
        let mut samples = Vec::new();
        for e in entries {
            println!(
                "  {} → ラベル '{}' (手カバレッジ L={:.0}% R={:.0}%)",
                e.stem,
                e.label,
                e.left_cov * 100.0,
                e.right_cov * 100.0
            );
            samples.push(PoseSample {
                features: e.features,
                label: e.label,
            });
        }
        Ok(Self {
            samples,
            frames,
            feature_dim,
        })
    }

    /// build-dict JSON をテイク単位で train/held-out に分割して読み込む。
    /// 各ラベルでテイク番号昇順の後ろ `held_out_per_label` 件を未見テイク(test)にし、
    /// 残りを train にする。テイクが足りないラベルは全部 train に入れ test には出さない
    /// (= 未見評価できないが学習からは外さない)。
    pub fn load_holdout_split(
        path: &Path,
        held_out_per_label: usize,
    ) -> Result<HoldoutSplit, Box<dyn std::error::Error>> {
        Self::load_holdout_split_with(path, held_out_per_label, HoldoutSplitMode::Last)
    }

    /// `load_holdout_split` の分割方法指定版。`Last` は従来と完全に同一
    pub fn load_holdout_split_with(
        path: &Path,
        held_out_per_label: usize,
        mode: HoldoutSplitMode,
    ) -> Result<HoldoutSplit, Box<dyn std::error::Error>> {
        if held_out_per_label == 0 {
            return Err("holdout_per_label は 1 以上が必要です".into());
        }
        let (entries, frames, feature_dim) = read_raw_entries(path)?;

        // ラベルごとにまとめる(BTreeMap でラベル順は決定的)
        let mut by_label: BTreeMap<String, Vec<RawEntry>> = BTreeMap::new();
        for e in entries {
            by_label.entry(e.label.clone()).or_default().push(e);
        }

        let mut train_samples = Vec::new();
        let mut test_samples = Vec::new();
        let mut per_label = Vec::new();
        let mut train_only = Vec::new();
        let mut hand_mismatch = Vec::new();

        for (label, group) in by_label {
            // held-out 候補はテイク番号付きのみ。番号なし(「最後のテイク」が定義できない)は
            // 常に train に入れ、評価対象にしない。これで held-out が stem 順に左右されない
            let (mut numbered, unnumbered): (Vec<RawEntry>, Vec<RawEntry>) =
                group.into_iter().partition(|e| e.take.is_some());
            numbered.sort_by_key(|e| e.take.expect("partition で Some のみが numbered に入る"));
            let numbered_count = numbered.len();
            let unnumbered_count = unnumbered.len();

            // 番号なしは無条件で train
            let mut train_hands: Vec<&'static str> = Vec::new();
            for e in unnumbered {
                train_hands.push(used_hands(e.left_cov, e.right_cov));
                train_samples.push(PoseSample {
                    features: e.features,
                    label: label.clone(),
                });
            }

            if numbered_count <= held_out_per_label {
                // 番号付きテイクが足りず held-out を作れない → 残りも全部 train
                train_only.push(label.clone());
                for e in numbered {
                    train_samples.push(PoseSample {
                        features: e.features,
                        label: label.clone(),
                    });
                }
                per_label.push((label, numbered_count + unnumbered_count, 0));
                continue;
            }

            // 分割方法に従って held-out の位置を決める
            let held = held_out_positions(mode, &label, numbered_count, held_out_per_label);
            let mut held_entries: Vec<(String, &'static str)> = Vec::new();
            for (i, e) in numbered.into_iter().enumerate() {
                let hands = used_hands(e.left_cov, e.right_cov);
                let sample = PoseSample {
                    features: e.features,
                    label: label.clone(),
                };
                if held.contains(&i) {
                    held_entries.push((e.stem, hands));
                    test_samples.push(sample);
                } else {
                    train_hands.push(hands);
                    train_samples.push(sample);
                }
            }
            for (stem, hands) in held_entries {
                if !train_hands.contains(&hands) {
                    hand_mismatch.push((stem, hands));
                }
            }
            per_label.push((label, numbered_count - held.len() + unnumbered_count, held.len()));
        }

        if test_samples.is_empty() {
            return Err(format!(
                "held-out 用の未見テイクがありません。各ラベルに最低 {} テイク必要です(holdout_per_label={})",
                held_out_per_label + 1,
                held_out_per_label
            )
            .into());
        }

        Ok(HoldoutSplit {
            train: PoseTrainingData {
                samples: train_samples,
                frames,
                feature_dim,
            },
            test: PoseTrainingData {
                samples: test_samples,
                frames,
                feature_dim,
            },
            per_label,
            train_only,
            hand_mismatch,
        })
    }

    /// 各サンプルの左右反転コピーを追加した学習データを返す(件数は2倍)。
    /// ラベルは元のまま(左右反転しても同じサインという想定)。
    /// held-out(未見テイク評価用)には使わないこと: 評価は実撮影データの分布で行うべきで、
    /// 拡張データを混ぜると「解けたことにする」測定になってしまう
    pub fn with_mirror_augmentation(&self) -> PoseTrainingData {
        let mut samples = self.samples.clone();
        samples.extend(self.samples.iter().map(|s| PoseSample {
            features: mirror_features(&s.features),
            label: s.label.clone(),
        }));
        PoseTrainingData {
            samples,
            frames: self.frames,
            feature_dim: self.feature_dim,
        }
    }

    /// 生の手ポーズ列(126次元)を手形記述子(66次元)に変換した学習データを返す。
    /// ラベル・サンプル件数・フレーム数は変わらず、1 フレームの次元だけが変わる。
    /// `InputFeatures::Raw` を渡した場合は変換せずクローンを返す(呼び出し側で分岐しなくて済む)
    pub fn to_input_features(&self, mode: InputFeatures) -> PoseTrainingData {
        self.to_input_features_with_parts(mode, DescriptorParts::all())
    }

    /// `to_input_features` の成分選択版(アブレーション用)
    pub fn to_input_features_with_parts(
        &self,
        mode: InputFeatures,
        parts: DescriptorParts,
    ) -> PoseTrainingData {
        if mode.is_raw() {
            return PoseTrainingData {
                samples: self.samples.clone(),
                frames: self.frames,
                feature_dim: self.feature_dim,
            };
        }
        let samples = self
            .samples
            .iter()
            .map(|s| PoseSample {
                features: handshape_sequence_with_parts(&s.features, self.frames, mode, parts),
                label: s.label.clone(),
            })
            .collect();
        PoseTrainingData {
            samples,
            frames: self.frames,
            feature_dim: parts.feature_dim(),
        }
    }

    /// このデータから標準化の統計量を求める。
    /// **train 分割に対してのみ呼ぶこと**(held-out に対して呼ぶと評価が汚染される)
    pub fn fit_standardizer(&self) -> FeatureStats {
        FeatureStats::fit(
            self.samples.iter().map(|s| s.features.as_slice()),
            self.feature_dim,
        )
    }

    /// 与えられた統計量で標準化したデータを返す。統計量は train 由来のものを使う
    pub fn standardized(&self, stats: &FeatureStats) -> PoseTrainingData {
        assert_eq!(
            stats.feature_dim(),
            self.feature_dim,
            "標準化の統計量の次元({})がデータの次元({})と一致しません",
            stats.feature_dim(),
            self.feature_dim
        );
        PoseTrainingData {
            samples: self
                .samples
                .iter()
                .map(|s| PoseSample {
                    features: stats.apply(&s.features),
                    label: s.label.clone(),
                })
                .collect(),
            frames: self.frames,
            feature_dim: self.feature_dim,
        }
    }

    /// データに現れたラベルからタグ語彙を構築する
    pub fn build_vocabulary(&self) -> TagVocabulary {
        TagVocabulary::from_labels(self.samples.iter().map(|s| s.label.clone()))
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// バッチ化。各バッチは (入力特徴のフラット列, ターゲットID列) のペア。
    /// 入力: [batch * T * feature_dim] のフラット Vec(呼び出し側で reshape)
    /// ターゲット: 各サンプル [SOS, タグID, EOS](単語認識なので長さ3固定)
    pub fn batches(
        &self,
        batch_size: usize,
        vocab: &TagVocabulary,
    ) -> Vec<(Vec<f32>, Vec<Vec<i32>>, usize)> {
        let mut batches = Vec::new();

        for chunk in self.samples.chunks(batch_size) {
            let mut batch_features = Vec::new();
            let mut batch_targets = Vec::new();

            for sample in chunk {
                batch_features.extend_from_slice(&sample.features);

                let tag_id = vocab
                    .tag_to_id(&sample.label)
                    .expect("語彙はデータから構築しているので必ず存在する")
                    as i32;
                batch_targets.push(vec![vocab.sos_id() as i32, tag_id, vocab.eos_id() as i32]);
            }

            let actual_batch = chunk.len();
            batches.push((batch_features, batch_targets, actual_batch));
        }

        batches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // [TEST向き] テイク番号剥がしの境界ケース
    #[test]
    fn label_from_stem_cases() {
        // 期待値の根拠: '-'+数字 はテイク番号とみなして剥がす
        assert_eq!(label_from_stem("0205-01"), "0205");
        assert_eq!(label_from_stem("ありがとう_2"), "ありがとう");
        // サフィックスに数字以外が混ざる場合は剥がさない
        assert_eq!(label_from_stem("hello-world"), "hello-world");
        // 区切りが無ければそのまま
        assert_eq!(label_from_stem("挨拶"), "挨拶");
        // 区切りで終わる(空サフィックス)場合は剥がさない
        assert_eq!(label_from_stem("tag-"), "tag-");
    }

    // [TEST向き] テイク番号抽出の境界ケース
    #[test]
    fn take_from_stem_cases() {
        assert_eq!(take_from_stem("0205-01"), Some(1));
        assert_eq!(take_from_stem("ありがとう_2"), Some(2));
        assert_eq!(take_from_stem("word-10"), Some(10));
        // 数字以外が混ざる/区切りなし/空サフィックスは None
        assert_eq!(take_from_stem("挨拶"), None);
        assert_eq!(take_from_stem("hello-world"), None);
        assert_eq!(take_from_stem("tag-"), None);
    }

    // [TEST向き] 使われた手の分類。境界(閾値ちょうど 0.2 は「使用」側)
    #[test]
    fn used_hands_cases() {
        // 期待値の根拠: USED_HAND_THRESHOLD = 0.2、以上で「使用」
        assert_eq!(used_hands(0.5, 0.5), "LR");
        assert_eq!(used_hands(0.2, 0.19), "L"); // 閾値ちょうどは使用扱い
        assert_eq!(used_hands(0.0, 0.35), "R");
        assert_eq!(used_hands(0.1, 0.1), "-");
    }

    #[test]
    fn inspect_dict_flags_hand_mismatch() {
        use std::io::Write;
        let row = vec![0.0f32; POSE_FEATURE_DIM];
        // "anata": take1-2 は右手のみ、take3 は左手のみ → 食い違いフラグ
        // "konnichiwa": 全テイク両手 → フラグなし
        let entry = |l: f32, r: f32| {
            serde_json::json!({
                "sequence": [row.clone()],
                "left_hand_coverage": l,
                "right_hand_coverage": r,
                "source": "x"
            })
        };
        let dict = serde_json::json!({
            "metadata": {
                "frames": 1,
                "feature_dim": POSE_FEATURE_DIM,
                "feature_layout": "",
                "normalization": "",
                "tag_count": 6
            },
            "tags": {
                "anata-1": entry(0.0, 0.41),
                "anata-2": entry(0.0, 0.35),
                "anata-3": entry(0.36, 0.0),
                "konnichiwa-1": entry(0.9, 0.9),
                "konnichiwa-2": entry(0.8, 0.7),
                "konnichiwa-3": entry(0.9, 0.8)
            }
        });

        let dir = std::env::temp_dir().join(format!("pose_inspect_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dict.json");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(dict.to_string().as_bytes())
            .unwrap();

        let flagged = inspect_dict(&path).unwrap();
        assert_eq!(flagged, vec!["anata".to_string()]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn holdout_split_holds_out_last_take_per_label() {
        use std::io::Write;
        // frames=1, feature_dim=126 の最小 dict を一時ファイルに書く
        let row = vec![0.0f32; POSE_FEATURE_DIM];
        let entry = serde_json::json!({
            "sequence": [row],
            "left_hand_coverage": 1.0,
            "right_hand_coverage": 1.0,
            "source": "x"
        });
        let dict = serde_json::json!({
            "metadata": {
                "frames": 1,
                "feature_dim": POSE_FEATURE_DIM,
                "feature_layout": "",
                "normalization": "",
                "tag_count": 4
            },
            "tags": {
                "0205-01": entry.clone(),
                "0205-02": entry.clone(),
                "0205-03": entry.clone(),
                "0225-01": entry
            }
        });

        let dir = std::env::temp_dir().join(format!("pose_holdout_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dict.json");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(dict.to_string().as_bytes())
            .unwrap();

        let split = PoseTrainingData::load_holdout_split(&path, 1).unwrap();
        // 0205: 3テイク → train2 / held-out1、0225: 1テイク → train_only(test には出さない)
        assert_eq!(split.train.len(), 3, "train = 0205×2 + 0225×1");
        assert_eq!(split.test.len(), 1, "held-out = 0205 の take3 のみ");
        assert!(split.test.samples.iter().all(|s| s.label == "0205"));
        assert!(split.train_only.contains(&"0225".to_string()));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn holdout_split_holds_out_only_numbered_takes() {
        use std::io::Write;
        let row = vec![0.0f32; POSE_FEATURE_DIM];
        let entry = serde_json::json!({
            "sequence": [row],
            "left_hand_coverage": 1.0,
            "right_hand_coverage": 1.0,
            "source": "x"
        });
        // ラベル "word": 番号付き 2 + 番号なし 1。ラベル "solo": 番号なし 1 のみ
        let dict = serde_json::json!({
            "metadata": {
                "frames": 1,
                "feature_dim": POSE_FEATURE_DIM,
                "feature_layout": "",
                "normalization": "",
                "tag_count": 4
            },
            "tags": {
                "word-01": entry.clone(),
                "word-02": entry.clone(),
                "word":    entry.clone(),
                "solo":    entry
            }
        });

        let dir =
            std::env::temp_dir().join(format!("pose_holdout_num_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dict.json");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(dict.to_string().as_bytes())
            .unwrap();

        let split = PoseTrainingData::load_holdout_split(&path, 1).unwrap();
        // held-out は番号付きの最後(word-02)のみ。番号なし "word" と "solo" は train。
        assert_eq!(split.test.len(), 1, "held-out = word の take2 のみ");
        assert!(split.test.samples.iter().all(|s| s.label == "word"));
        // train = word-01 + word(番号なし) + solo
        assert_eq!(split.train.len(), 3);
        // solo は番号付きテイクが無いので train_only、word は held-out できたので含まれない
        assert!(split.train_only.contains(&"solo".to_string()));
        assert!(!split.train_only.contains(&"word".to_string()));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// (stem, 左カバレッジ, 右カバレッジ) から最小 dict(frames=1)を一時ファイルに書いて返す
    fn write_hand_dict(tag: &str, rows: &[(&str, f32, f32)]) -> (std::path::PathBuf, std::path::PathBuf) {
        use std::io::Write;
        let row = vec![0.0f32; POSE_FEATURE_DIM];
        let mut tags = serde_json::Map::new();
        for (stem, l, r) in rows {
            tags.insert(
                stem.to_string(),
                serde_json::json!({
                    "sequence": [row.clone()],
                    "left_hand_coverage": l,
                    "right_hand_coverage": r,
                    "source": "x"
                }),
            );
        }
        let dict = serde_json::json!({
            "metadata": {
                "frames": 1,
                "feature_dim": POSE_FEATURE_DIM,
                "feature_layout": "",
                "normalization": "",
                "tag_count": rows.len()
            },
            "tags": tags
        });
        let dir = std::env::temp_dir().join(format!("pose_split_{}_{}", tag, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dict.json");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(dict.to_string().as_bytes())
            .unwrap();
        (dir, path)
    }

    /// 3 ラベル × 3 テイク(撮影癖の再現: テイク1・2 = 右手、テイク3 = 左手)
    fn habit_rows() -> Vec<(&'static str, f32, f32)> {
        vec![
            ("a-01", 0.0, 1.0), ("a-02", 0.0, 1.0), ("a-03", 1.0, 0.0),
            ("b-01", 0.0, 1.0), ("b-02", 0.0, 1.0), ("b-03", 1.0, 0.0),
            ("c-01", 1.0, 1.0), ("c-02", 1.0, 1.0), ("c-03", 1.0, 1.0),
        ]
    }

    #[test]
    fn split_last_via_with_equals_legacy() {
        let (dir, path) = write_hand_dict("last", &habit_rows());
        let legacy = PoseTrainingData::load_holdout_split(&path, 1).unwrap();
        let with = PoseTrainingData::load_holdout_split_with(&path, 1, HoldoutSplitMode::Last)
            .unwrap();
        assert_eq!(legacy.per_label, with.per_label);
        assert_eq!(legacy.train.len(), with.train.len());
        assert_eq!(legacy.test.len(), with.test.len());
        // last は各ラベルのテイク3 が未見 = a, b は左手のみで train(右手)と食い違う
        assert_eq!(with.test.len(), 3);
        let stems: Vec<&str> = with.hand_mismatch.iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(stems, vec!["a-03", "b-03"], "c は LR 同士で一致");
        assert!(with.hand_mismatch.iter().all(|(_, h)| *h == "L"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn held_out_positions_random_is_deterministic_and_sized() {
        let m = HoldoutSplitMode::Random { seed: 7 };
        let a = held_out_positions(m, "x", 10, 3);
        let b = held_out_positions(m, "x", 10, 3);
        assert_eq!(a, b, "同 seed・同ラベルは同結果");
        assert_eq!(a.len(), 3, "重複なしで N 件");
        assert!(a.iter().all(|&i| i < 10));
        // seed またはラベルが違えば(ほぼ確実に)違う選び方になる。全 seed で同一なら乱数が効いていない
        let distinct: std::collections::BTreeSet<Vec<usize>> = (0..20u64)
            .map(|sd| {
                held_out_positions(HoldoutSplitMode::Random { seed: sd }, "x", 10, 1)
                    .into_iter()
                    .collect()
            })
            .collect();
        assert!(distinct.len() > 3, "seed を変えると選ばれる位置が変わる");
    }

    #[test]
    fn split_random_holds_out_n_per_label_without_overlap() {
        let (dir, path) = write_hand_dict("random", &habit_rows());
        for seed in 0..10u64 {
            let sp = PoseTrainingData::load_holdout_split_with(
                &path,
                1,
                HoldoutSplitMode::Random { seed },
            )
            .unwrap();
            assert_eq!(sp.test.len(), 3, "3 ラベル × 1 件");
            assert_eq!(sp.train.len(), 6);
            for (_, tr, te) in &sp.per_label {
                assert_eq!((*tr, *te), (2, 1));
            }
        }
        // 同 seed なら held-out の stem(= hand_mismatch の出方)も再現する
        let m = HoldoutSplitMode::Random { seed: 3 };
        let x = PoseTrainingData::load_holdout_split_with(&path, 1, m).unwrap();
        let y = PoseTrainingData::load_holdout_split_with(&path, 1, m).unwrap();
        assert_eq!(x.hand_mismatch, y.hand_mismatch);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rotate_holds_out_each_take_exactly_once_over_all_folds() {
        for n in [1usize, 2] {
            let mut count = [0usize; 3];
            for fold in 0..3 {
                for p in held_out_positions(HoldoutSplitMode::Rotate { fold }, "x", 3, n) {
                    count[p] += 1;
                }
            }
            assert_eq!(count, [n; 3], "N={} で全 fold を回すと各テイクがちょうど N 回", n);
        }
        // 循環: fold=2, N=2, 3テイク → 位置 2 と 0
        let p: Vec<usize> = held_out_positions(HoldoutSplitMode::Rotate { fold: 2 }, "x", 3, 2)
            .into_iter()
            .collect();
        assert_eq!(p, vec![0, 2]);
    }

    #[test]
    fn rotate_split_hand_mismatch_oracle() {
        // a, b: 右,右,左 / c: LR×3。fold 別の手計算:
        //  fold0 = テイク1 を未見 → a-01,b-01 は R。train に R(02)がある → 不一致 0
        //  fold1 = テイク2 → 同様に 0
        //  fold2 = テイク3 → a-03,b-03 は L。train は R のみ → 不一致 2
        let (dir, path) = write_hand_dict("rotate", &habit_rows());
        let counts: Vec<usize> = (0..3)
            .map(|fold| {
                PoseTrainingData::load_holdout_split_with(
                    &path,
                    1,
                    HoldoutSplitMode::Rotate { fold },
                )
                .unwrap()
                .hand_mismatch
                .len()
            })
            .collect();
        assert_eq!(counts, vec![0, 0, 2]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn holdout_split_mode_parse() {
        assert_eq!(HoldoutSplitMode::parse("last", 1, 2).unwrap(), HoldoutSplitMode::Last);
        assert_eq!(
            HoldoutSplitMode::parse("random", 5, 2).unwrap(),
            HoldoutSplitMode::Random { seed: 5 }
        );
        assert_eq!(
            HoldoutSplitMode::parse("rotate", 5, 2).unwrap(),
            HoldoutSplitMode::Rotate { fold: 2 }
        );
        assert!(HoldoutSplitMode::parse("x", 0, 0).is_err());
    }

    // [TEST向き] ミラー反転の核。境界(両手・片手のみ・両手ゼロ・複数フレーム)
    #[test]
    fn mirror_features_swaps_slots_and_flips_x() {
        // 1フレーム: 左手枠は point0=(0.1,0.2,0.3)、他は0。右手枠は point0=(0.7,0.8,0.9)、他は0
        let mut frame = vec![0.0f32; POSE_FEATURE_DIM];
        frame[0] = 0.1;
        frame[1] = 0.2;
        frame[2] = 0.3;
        frame[HAND_SLOT_DIM] = 0.7;
        frame[HAND_SLOT_DIM + 1] = 0.8;
        frame[HAND_SLOT_DIM + 2] = 0.9;

        let mirrored = mirror_features(&frame);

        // 元・右手枠(0.7,0.8,0.9) → 反転して左手枠へ、x = 1.0 - 0.7 = 0.3、y,zはそのまま
        assert!((mirrored[0] - 0.3).abs() < 1e-6);
        assert!((mirrored[1] - 0.8).abs() < 1e-6);
        assert!((mirrored[2] - 0.9).abs() < 1e-6);
        // 元・左手枠(0.1,0.2,0.3) → 反転して右手枠へ、x = 1.0 - 0.1 = 0.9
        assert!((mirrored[HAND_SLOT_DIM] - 0.9).abs() < 1e-6);
        assert!((mirrored[HAND_SLOT_DIM + 1] - 0.2).abs() < 1e-6);
        assert!((mirrored[HAND_SLOT_DIM + 2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn mirror_features_leaves_undetected_hand_zero() {
        // 左手枠のみ検出(右手枠は全ゼロ = 未検出)
        let mut frame = vec![0.0f32; POSE_FEATURE_DIM];
        frame[0] = 0.4;
        frame[1] = 0.5;
        frame[2] = 0.6;

        let mirrored = mirror_features(&frame);

        // 右手枠(未検出)から反転コピーされた新しい左手枠は、ゼロ入力なのでゼロのまま
        assert!(mirrored[0..HAND_SLOT_DIM].iter().all(|&v| v == 0.0));
        // 左手枠(検出あり)から反転コピーされた新しい右手枠は x が反転
        assert!((mirrored[HAND_SLOT_DIM] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn mirror_features_is_involution_over_multiple_frames() {
        // 2フレーム分のランダムでない適当な非ゼロ値。2回反転すれば元に戻るはず
        let mut features = vec![0.0f32; POSE_FEATURE_DIM * 2];
        for (i, v) in features.iter_mut().enumerate() {
            *v = (i % 7) as f32 * 0.1 + 0.05; // 全て非ゼロ(両手とも「検出あり」扱いにする)
        }

        let once = mirror_features(&features);
        let twice = mirror_features(&once);

        for (a, b) in features.iter().zip(twice.iter()) {
            assert!((a - b).abs() < 1e-6, "2回反転すれば元に戻るはず: {} vs {}", a, b);
        }
    }

    #[test]
    fn with_mirror_augmentation_doubles_samples_and_keeps_labels() {
        let data = PoseTrainingData {
            samples: vec![
                PoseSample {
                    features: {
                        let mut f = vec![0.0f32; POSE_FEATURE_DIM];
                        f[0] = 0.2;
                        f
                    },
                    label: "word".to_string(),
                },
                PoseSample {
                    features: vec![0.0f32; POSE_FEATURE_DIM],
                    label: "other".to_string(),
                },
            ],
            frames: 1,
            feature_dim: POSE_FEATURE_DIM,
        };

        let augmented = data.with_mirror_augmentation();
        assert_eq!(augmented.len(), 4, "元2件 + ミラー2件 = 4件");
        let labels: Vec<&str> = augmented.samples.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(
            labels.iter().filter(|&&l| l == "word").count(),
            2,
            "'word' は元+ミラーの2件になるはず"
        );
        // ミラー後のサンプルは元の特徴とは異なる(左右反転が実際に効いている)
        let mirrored_word = &augmented.samples[2];
        assert_eq!(mirrored_word.label, "word");
        assert_ne!(mirrored_word.features, augmented.samples[0].features);
    }
}
