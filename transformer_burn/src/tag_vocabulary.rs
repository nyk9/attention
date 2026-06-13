use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

/// 認識モデル(手ポーズ列→タグ)用のタグ語彙。
///
/// JslVocabulary(日本語86文字+タグ80の固定168語)と違い、
/// 学習データ(pose dict)に現れたタグ名から動的に構築する。
/// 語彙の並びは [タグ0, タグ1, ..., SOS, EOS, PAD] で、
/// 末尾3つが特殊トークン。タグ集合が変わると ID も変わるため、
/// モデル保存時に必ず一緒に保存し、読み込み時に復元する。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagVocabulary {
    /// タグ名の一覧(ID 順、特殊トークンは含まない)
    pub tags: Vec<String>,
    /// タグ名 → ID の逆引き(JSON には保存せず復元時に作り直す)
    #[serde(skip)]
    tag_to_id: HashMap<String, usize>,
}

impl TagVocabulary {
    /// ラベル集合から語彙を構築する。
    /// BTreeSet 経由なので順序は常にソート済み = 再現性がある。
    pub fn from_labels<I: IntoIterator<Item = String>>(labels: I) -> Self {
        let unique: BTreeSet<String> = labels.into_iter().collect();
        let tags: Vec<String> = unique.into_iter().collect();
        let mut vocab = Self {
            tags,
            tag_to_id: HashMap::new(),
        };
        vocab.rebuild_index();
        vocab
    }

    /// tag_to_id を tags から作り直す(デシリアライズ後にも呼ぶ)
    fn rebuild_index(&mut self) {
        self.tag_to_id = self
            .tags
            .iter()
            .enumerate()
            .map(|(id, tag)| (tag.clone(), id))
            .collect();
    }

    /// 特殊トークン込みの語彙サイズ(= Decoder の出力次元)
    pub fn vocab_size(&self) -> usize {
        self.tags.len() + 3 // + SOS, EOS, PAD
    }

    pub fn sos_id(&self) -> usize {
        self.tags.len()
    }

    pub fn eos_id(&self) -> usize {
        self.tags.len() + 1
    }

    pub fn pad_id(&self) -> usize {
        self.tags.len() + 2
    }

    pub fn tag_to_id(&self, tag: &str) -> Option<usize> {
        self.tag_to_id.get(tag).copied()
    }

    pub fn id_to_tag(&self, id: usize) -> Option<&str> {
        self.tags.get(id).map(|s| s.as_str())
    }

    /// 語彙を JSON で保存(モデルと同じディレクトリに置く)
    pub fn save(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("tag_vocab.json");
        let file = std::fs::File::create(&path)?;
        serde_json::to_writer_pretty(file, self)?;
        println!("タグ語彙を保存: {} ({}タグ)", path.display(), self.tags.len());
        Ok(())
    }

    /// 保存済みの語彙 JSON を読み込む
    pub fn load(dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let path = dir.join("tag_vocab.json");
        let content = std::fs::read_to_string(&path)
            .map_err(|e| format!("タグ語彙が読み込めません {}: {}", path.display(), e))?;
        let mut vocab: TagVocabulary = serde_json::from_str(&content)?;
        vocab.rebuild_index();
        println!("タグ語彙を読み込み: {} ({}タグ)", path.display(), vocab.tags.len());
        Ok(vocab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // [TEST向き] 語彙構築の不変条件: 重複除去・ソート順・特殊トークンIDの位置
    #[test]
    fn from_labels_dedupes_and_sorts() {
        let vocab = TagVocabulary::from_labels(vec![
            "b".to_string(),
            "a".to_string(),
            "b".to_string(), // 重複は1つになる
        ]);
        // 期待値の根拠: BTreeSet なので "a" < "b" の辞書順、重複は消える
        assert_eq!(vocab.tags, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(vocab.vocab_size(), 5); // 2タグ + SOS/EOS/PAD
        assert_eq!(vocab.sos_id(), 2);
        assert_eq!(vocab.eos_id(), 3);
        assert_eq!(vocab.pad_id(), 4);
    }

    // [TEST向き] round-trip: tag -> id -> tag が元に戻る
    #[test]
    fn tag_id_round_trip() {
        let vocab = TagVocabulary::from_labels(vec!["挨拶".to_string(), "感謝".to_string()]);
        for tag in &vocab.tags {
            let id = vocab.tag_to_id(tag).expect("既知タグはIDを持つ");
            assert_eq!(vocab.id_to_tag(id), Some(tag.as_str()));
        }
        assert_eq!(vocab.tag_to_id("未知"), None);
    }
}
