use crate::config::{PAD_TOKEN, SEQ_LEN};
use std::collections::HashMap;

pub struct Vocabulary {
    pub char_to_id: HashMap<char, usize>,
    pub id_to_char: Vec<char>,
    pub vocab_size: usize,
}

impl Vocabulary {
    pub fn new() -> Self {
        let mut chars = Vec::new();

        // ひらがな（あ〜ん）
        let hiragana = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん";
        for c in hiragana.chars() {
            chars.push(c);
        }
        // 濁音
        let dakuon = "がぎぐげござじずぜぞだぢづでどばびぶべぼ";
        for c in dakuon.chars() {
            chars.push(c);
        }

        // 半濁音
        let handakuon = "ぱぴぷぺぽ";
        for c in handakuon.chars() {
            chars.push(c);
        }

        // 小文字（拗音用）
        let kogaki = "ぁぃぅぇぉゃゅょっ";
        for c in kogaki.chars() {
            chars.push(c);
        }

        // 記号
        chars.push('。');
        chars.push('、');
        chars.push('！');
        chars.push('？');
        chars.push(' ');
        chars.push('ー');
        chars.push('['); // [PAD] = 0
        chars.push(']'); // [UNK] = 1

        // char_to_id の構築
        let mut char_to_id = HashMap::new();
        for (id, &c) in chars.iter().enumerate() {
            char_to_id.insert(c, id);
        }

        let vocab_size = chars.len();

        Vocabulary {
            char_to_id,
            id_to_char: chars,
            vocab_size,
        }
    }

    // 文字列をトークンIDのベクトルに変換
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let mut token_ids = Vec::new();

        for c in text.chars() {
            if let Some(&id) = self.char_to_id.get(&c) {
                token_ids.push(id as i32);
            } else {
                // 未知の文字は [UNK] (id=1) に変換
                token_ids.push(1);
            }
        }

        token_ids
    }

    pub fn pad_sequence(&self, tokens: &[i32]) -> Vec<i32> {
        let mut padded = tokens.to_vec();

        //  SEQ_LENに満たない場合はPAD_TOKENで埋める
        while padded.len() < SEQ_LEN {
            padded.push(PAD_TOKEN as i32);
        }

        // SEQ_LENを超える場合は切り詰める
        if padded.len() > SEQ_LEN {
            padded.truncate(SEQ_LEN);
        }

        padded
    }

    // トークンIDのベクトルを文字列に変換
    pub fn decode(&self, token_ids: &Vec<i32>) -> String {
        let mut text = String::new();

        for &id in token_ids {
            if id >= 0 && (id as usize) < self.vocab_size {
                text.push(self.id_to_char[id as usize]);
            } else {
                // 範囲外のIDは無視
                text.push('?');
            }
        }

        text
    }

    // 次文字予測の確率分布を文字と確率のペアに変換
    pub fn decode_predictions(&self, predictions: &Vec<(i32, f64)>) -> Vec<(char, f64)> {
        predictions
            .iter()
            .map(|(id, prob)| {
                let c = if *id >= 0 && (*id as usize) < self.vocab_size {
                    self.id_to_char[*id as usize]
                } else {
                    '?'
                };
                (c, *prob)
            })
            .collect()
    }
}
