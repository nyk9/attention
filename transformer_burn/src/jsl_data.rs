use crate::config::SEQ_LEN;
use crate::jsl_vocabulary::JslVocabulary;
use std::fs;

pub struct JslTrainingData {
    pub samples: Vec<(Vec<i32>, [i32; 5])>, // (入力: 日本語文, 出力: 5タグ位置)
}

impl JslTrainingData {
    pub fn load(vocab: &JslVocabulary, file_path: &str) -> Self {
        let content = fs::read_to_string(file_path).expect("JSL訓練データファイルが読み込めません");

        let mut samples = Vec::new();

        for line in content.lines() {
            let line = line.trim();

            // 空行とコメント行をスキップ
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // タブ区切りで分割：日本語文\tタグ列
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 2 {
                eprintln!(
                    "Warning: Invalid line format (expected TAB-separated): {}",
                    line
                );
                continue;
            }

            let japanese_text = parts[0].trim();
            let tag_sequence = parts[1].trim();

            // 日本語文をエンコード（入力シーケンス）
            let japanese_tokens = vocab.encode(japanese_text);
            let padded_input = vocab.pad_sequence(&japanese_tokens, SEQ_LEN);

            // タグをスペース区切りで分割し、各タグの最初のIDを取得
            let tag_ids: Vec<i32> = tag_sequence
                .split(' ')
                .filter(|s| !s.is_empty()) // 空文字列を除外
                .filter_map(|tag| {
                    let encoded = vocab.encode(tag);
                    if encoded.is_empty() {
                        eprintln!("Warning: Empty encoding for tag '{}'", tag);
                        None
                    } else {
                        Some(encoded[0]) // 最初のトークンIDを取得
                    }
                })
                .collect();

            // 5個を超える場合は警告
            if tag_ids.len() > 5 {
                eprintln!(
                    "Warning: Tag count {} exceeds 5, truncating: {}",
                    tag_ids.len(),
                    tag_sequence
                );
            }

            // 固定5位置の配列を作成
            let pad_id = (vocab.vocab_size - 1) as i32; // PADトークンID
            let mut tag_output: [i32; 5] = [pad_id; 5]; // PADで初期化

            for (i, &id) in tag_ids.iter().take(5).enumerate() {
                tag_output[i] = id;
            }

            samples.push((padded_input, tag_output));
        }

        println!("JSL訓練サンプル数: {}", samples.len());

        Self { samples }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn batches(&self, batch_size: usize) -> Vec<(Vec<Vec<i32>>, Vec<[i32; 5]>)> {
        let mut batches = Vec::new();

        for chunk in self.samples.chunks(batch_size) {
            let mut batch_inputs = Vec::new();
            let mut batch_targets = Vec::new();

            for (input, target) in chunk {
                batch_inputs.push(input.clone());
                batch_targets.push(*target);
            }

            batches.push((batch_inputs, batch_targets));
        }

        batches
    }
}
