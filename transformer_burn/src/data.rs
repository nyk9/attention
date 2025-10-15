use crate::vocabulary::Vocabulary;
use std::fs;

pub struct TrainingData {
    pub samples: Vec<(Vec<i32>, i32)>, // (入力トークン列, 次トークン)
}

impl TrainingData {
    pub fn load(vocab: &Vocabulary, file_path: &str) -> Self {
        let content = fs::read_to_string(file_path).expect("訓練データファイルが読み込めません");

        let mut samples = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // 文字列をトークン化
            let tokens = vocab.encode(line);

            // 各位置について、前の文字列から次の文字を予測するペアを作成
            for i in 1..tokens.len() {
                let input = tokens[0..i].to_vec();
                let target = tokens[i];

                // パディング
                let padded_input = vocab.pad_sequence(&input);

                samples.push((padded_input, target));
            }
        }

        println!("訓練サンプル数: {}", samples.len());

        Self { samples }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }
}
