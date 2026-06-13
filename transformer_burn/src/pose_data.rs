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

impl PoseTrainingData {
    /// build-dict の JSON を読み込んで学習データにする。
    /// 同じラベルのエントリ(複数テイク)はそれぞれ独立したサンプルになる。
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
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

        let mut samples = Vec::new();
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

            let label = label_from_stem(stem);
            let features: Vec<f32> = entry.sequence.iter().flatten().copied().collect();
            println!(
                "  {} → ラベル '{}' (手カバレッジ L={:.0}% R={:.0}%)",
                stem,
                label,
                entry.left_hand_coverage * 100.0,
                entry.right_hand_coverage * 100.0
            );
            samples.push(PoseSample { features, label });
        }

        if samples.is_empty() {
            return Err("pose dict にエントリがありません".into());
        }

        Ok(Self {
            samples,
            frames: dict.metadata.frames,
            feature_dim: dict.metadata.feature_dim,
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
}
