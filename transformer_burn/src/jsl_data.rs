use crate::jsl_vocabulary::JslVocabulary;
use crate::config::SEQ_LEN;
use std::fs;

pub struct JslTrainingData {
    pub samples: Vec<(Vec<i32>, i32)>, // (入力トークン列, 次トークン)
}

impl JslTrainingData {
    pub fn load(vocab: &JslVocabulary, file_path: &str) -> Self {
        let content = fs::read_to_string(file_path)
            .expect("JSL訓練データファイルが読み込めません");

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
                eprintln!("Warning: Invalid line format (expected TAB-separated): {}", line);
                continue;
            }

            let japanese_text = parts[0].trim();
            let tag_sequence = parts[1].trim();

            // 日本語文とタグ列を結合
            // 例: "私は食べます" + "<私> <食べる>" -> "私は食べます<私><食べる>"
            let combined = format!("{}{}", japanese_text, tag_sequence.replace(" ", ""));

            // トークン化
            let tokens = vocab.encode(&combined);

            // 次トークン予測のサンプルを生成
            // 各位置について、前のトークン列から次のトークンを予測
            for i in 1..tokens.len() {
                let input = tokens[0..i].to_vec();
                let target = tokens[i];

                // パディング（config::SEQ_LENを使用）
                let padded_input = vocab.pad_sequence(&input, SEQ_LEN);

                samples.push((padded_input, target));
            }
        }

        println!("JSL訓練サンプル数: {}", samples.len());

        Self { samples }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn batches(&self, batch_size: usize) -> Vec<(Vec<Vec<i32>>, Vec<i32>)> {
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
