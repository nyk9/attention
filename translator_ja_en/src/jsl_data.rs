use crate::config::SEQ_LEN;
use crate::jsl_vocabulary::JslVocabulary;
use std::fs;

pub struct JslTrainingData {
    pub samples: Vec<(Vec<i32>, Vec<i32>)>, // (入力: 日本語文, 出力: 可変長タグシーケンス)
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

            // ターゲットシーケンス: [SOS, tag1, tag2, ..., EOS]
            let mut target_sequence = Vec::new();
            target_sequence.push(vocab.sos_id as i32); // SOS
            target_sequence.extend(&tag_ids); // タグ列
            target_sequence.push(vocab.eos_id as i32); // EOS

            samples.push((padded_input, target_sequence));
        }

        println!("JSL訓練サンプル数: {}", samples.len());

        Self { samples }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn batches(&self, batch_size: usize, pad_id: usize) -> Vec<(Vec<Vec<i32>>, Vec<Vec<i32>>)> {
        let mut batches = Vec::new();

        for chunk in self.samples.chunks(batch_size) {
            let mut batch_inputs = Vec::new();
            let mut batch_targets = Vec::new();

            // バッチ内の最大ターゲット長を取得
            let max_target_len = chunk.iter().map(|(_, target)| target.len()).max().unwrap_or(0);

            for (input, target) in chunk {
                batch_inputs.push(input.clone());

                // ターゲットシーケンスをパディング
                let mut padded_target = target.clone();
                while padded_target.len() < max_target_len {
                    padded_target.push(pad_id as i32);
                }
                batch_targets.push(padded_target);
            }

            batches.push((batch_inputs, batch_targets));
        }

        batches
    }
}
