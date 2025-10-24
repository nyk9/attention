use crate::translation_vocabulary::{SourceVocabulary, TargetVocabulary};
use std::fs;

pub struct TranslationData {
    pub samples: Vec<(Vec<i32>, Vec<i32>)>, // (入力: 日本語, 出力: 英語シーケンス)
}

impl TranslationData {
    /// TSV形式のファイルから日英対訳データを読み込む
    /// 形式: 日本語文[TAB]英語文
    pub fn load(
        src_vocab: &SourceVocabulary,
        tgt_vocab: &TargetVocabulary,
        file_path: &str,
        src_seq_len: usize,
    ) -> Self {
        let content =
            fs::read_to_string(file_path).expect("日英対訳データファイルが読み込めません");

        let mut samples = Vec::new();

        for line in content.lines() {
            let line = line.trim();

            // 空行とコメント行をスキップ
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // タブ区切りで分割
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 2 {
                eprintln!(
                    "Warning: 無効な行形式 (TAB区切りではありません): {}",
                    line
                );
                continue;
            }

            let japanese_text = parts[0].trim();
            let english_text = parts[1].trim();

            // 日本語文をエンコード（入力シーケンス）
            let japanese_tokens = src_vocab.encode(japanese_text);
            if japanese_tokens.is_empty() {
                eprintln!("Warning: 日本語文のエンコードが空です: {}", japanese_text);
                continue;
            }
            let padded_input = src_vocab.pad_sequence(&japanese_tokens, src_seq_len);

            // 英語文をエンコード
            let english_tokens = tgt_vocab.encode(english_text);
            if english_tokens.is_empty() {
                eprintln!("Warning: 英語文のエンコードが空です: {}", english_text);
                continue;
            }

            // ターゲットシーケンス: [SOS, word1, word2, ..., EOS]
            let mut target_sequence = Vec::new();
            target_sequence.push(tgt_vocab.sos_id as i32); // SOS
            target_sequence.extend(&english_tokens); // 英語単語列
            target_sequence.push(tgt_vocab.eos_id as i32); // EOS

            samples.push((padded_input, target_sequence));
        }

        println!("翻訳訓練サンプル数: {}", samples.len());

        Self { samples }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// バッチを生成（可変長ターゲットに対応）
    pub fn batches(
        &self,
        batch_size: usize,
        pad_id: usize,
    ) -> Vec<(Vec<Vec<i32>>, Vec<Vec<i32>>)> {
        let mut batches = Vec::new();

        for chunk in self.samples.chunks(batch_size) {
            let mut batch_inputs = Vec::new();
            let mut batch_targets = Vec::new();

            // バッチ内の最大ターゲット長を取得
            let max_target_len = chunk
                .iter()
                .map(|(_, target)| target.len())
                .max()
                .unwrap_or(0);

            for (input, target) in chunk {
                batch_inputs.push(input.clone());

                // ターゲットをパディング
                let mut padded_target = target.clone();
                padded_target.resize(max_target_len, pad_id as i32);
                batch_targets.push(padded_target);
            }

            batches.push((batch_inputs, batch_targets));
        }

        batches
    }
}
