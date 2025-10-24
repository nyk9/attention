use std::collections::HashMap;

/// 日本語語彙（ソース言語）
pub struct SourceVocabulary {
    pub char_to_id: HashMap<char, usize>,
    pub id_to_char: Vec<char>,
    pub vocab_size: usize,
}

impl SourceVocabulary {
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

        // カタカナ追加（外来語対応）
        let katakana = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン";
        for c in katakana.chars() {
            chars.push(c);
        }

        // カタカナ濁音・半濁音
        let katakana_dakuon = "ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポ";
        for c in katakana_dakuon.chars() {
            chars.push(c);
        }

        // カタカナ小文字
        let katakana_kogaki = "ァィゥェォヵヶャュョッ";
        for c in katakana_kogaki.chars() {
            chars.push(c);
        }

        // char_to_id の構築
        let mut char_to_id = HashMap::new();
        for (id, &c) in chars.iter().enumerate() {
            char_to_id.insert(c, id);
        }

        let vocab_size = chars.len();

        SourceVocabulary {
            char_to_id,
            id_to_char: chars,
            vocab_size,
        }
    }

    /// 日本語文字列をトークンIDに変換（1文字単位）
    pub fn encode(&self, text: &str) -> Vec<i32> {
        text.chars()
            .filter_map(|c| self.char_to_id.get(&c).map(|&id| id as i32))
            .collect()
    }

    /// トークンIDを日本語文字列に変換
    pub fn decode(&self, token_ids: &[i32]) -> String {
        token_ids
            .iter()
            .filter_map(|&id| {
                if id >= 0 && (id as usize) < self.vocab_size {
                    Some(self.id_to_char[id as usize])
                } else {
                    None
                }
            })
            .collect()
    }

    /// シーケンスを指定長でパディング
    pub fn pad_sequence(&self, tokens: &[i32], seq_len: usize) -> Vec<i32> {
        let mut padded = tokens.to_vec();
        padded.resize(seq_len, 0); // PAD_IDは別途定義
        padded.truncate(seq_len);
        padded
    }
}

/// 英語語彙（ターゲット言語）
pub struct TargetVocabulary {
    pub word_to_id: HashMap<String, usize>,
    pub id_to_word: Vec<String>,
    pub vocab_size: usize,
    pub sos_id: usize, // Start of Sequence
    pub eos_id: usize, // End of Sequence
    pub pad_id: usize, // Padding
    pub unk_id: usize, // Unknown
}

impl TargetVocabulary {
    /// データセットから語彙を構築
    pub fn from_dataset(sentences: &[&str]) -> Self {
        let mut words = Vec::new();

        // 特殊トークン
        words.push("<PAD>".to_string()); // ID 0
        words.push("<UNK>".to_string()); // ID 1
        words.push("<SOS>".to_string()); // ID 2
        words.push("<EOS>".to_string()); // ID 3

        // データセットから単語を収集
        let mut word_set = std::collections::HashSet::new();
        for sentence in sentences {
            for word in Self::tokenize(sentence) {
                word_set.insert(word);
            }
        }

        // アルファベット順にソート（再現性のため）
        let mut sorted_words: Vec<String> = word_set.into_iter().collect();
        sorted_words.sort();
        words.extend(sorted_words);

        // word_to_id の構築
        let mut word_to_id = HashMap::new();
        for (id, word) in words.iter().enumerate() {
            word_to_id.insert(word.clone(), id);
        }

        let vocab_size = words.len();

        TargetVocabulary {
            word_to_id,
            id_to_word: words,
            vocab_size,
            pad_id: 0,
            unk_id: 1,
            sos_id: 2,
            eos_id: 3,
        }
    }

    /// 英語文を単語単位でトークン化（小文字化、句読点処理）
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .replace(",", " ,")
            .replace(".", " .")
            .replace("!", " !")
            .replace("?", " ?")
            .split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }

    /// 英語文をトークンIDに変換
    pub fn encode(&self, text: &str) -> Vec<i32> {
        Self::tokenize(text)
            .iter()
            .map(|word| {
                self.word_to_id
                    .get(word)
                    .copied()
                    .unwrap_or(self.unk_id) as i32
            })
            .collect()
    }

    /// トークンIDを英語文に変換
    pub fn decode(&self, token_ids: &[i32]) -> String {
        token_ids
            .iter()
            .filter_map(|&id| {
                if id >= 0 && (id as usize) < self.vocab_size {
                    let word = &self.id_to_word[id as usize];
                    // 特殊トークンは出力しない
                    if word == "<PAD>" || word == "<SOS>" || word == "<EOS>" {
                        None
                    } else {
                        Some(word.clone())
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<String>>()
            .join(" ")
    }

    /// シーケンスを指定長でパディング
    pub fn pad_sequence(&self, tokens: &[i32], seq_len: usize) -> Vec<i32> {
        let mut padded = tokens.to_vec();
        padded.resize(seq_len, self.pad_id as i32);
        padded.truncate(seq_len);
        padded
    }
}
