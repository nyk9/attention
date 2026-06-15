use crate::config::POSE_FEATURE_DIM;
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

        for (label, group) in by_label {
            // held-out 候補はテイク番号付きのみ。番号なし(「最後のテイク」が定義できない)は
            // 常に train に入れ、評価対象にしない。これで held-out が stem 順に左右されない
            let (mut numbered, unnumbered): (Vec<RawEntry>, Vec<RawEntry>) =
                group.into_iter().partition(|e| e.take.is_some());
            numbered.sort_by_key(|e| e.take.expect("partition で Some のみが numbered に入る"));
            let numbered_count = numbered.len();
            let unnumbered_count = unnumbered.len();

            // 番号なしは無条件で train
            for e in unnumbered {
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

            // テイク番号昇順の後ろ held_out_per_label 件を held-out にする
            let split_at = numbered_count - held_out_per_label;
            for (i, e) in numbered.into_iter().enumerate() {
                let sample = PoseSample {
                    features: e.features,
                    label: label.clone(),
                };
                if i < split_at {
                    train_samples.push(sample);
                } else {
                    test_samples.push(sample);
                }
            }
            per_label.push((label, split_at + unnumbered_count, held_out_per_label));
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
        })
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
}
