use std::collections::HashMap;

pub struct JslVocabulary {
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: Vec<String>,
    pub vocab_size: usize,
    pub tag_start_id: usize, // タグの開始ID（日本語文字とタグを区別するため）
}

impl JslVocabulary {
    pub fn new() -> Self {
        let mut tokens = Vec::new();

        // === 日本語文字（vocabulary.rsと同様） ===

        // ひらがな（あ〜ん）
        let hiragana = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん";
        for c in hiragana.chars() {
            tokens.push(c.to_string());
        }

        // 濁音
        let dakuon = "がぎぐげござじずぜぞだぢづでどばびぶべぼ";
        for c in dakuon.chars() {
            tokens.push(c.to_string());
        }

        // 半濁音
        let handakuon = "ぱぴぷぺぽ";
        for c in handakuon.chars() {
            tokens.push(c.to_string());
        }

        // 小文字（拗音用）
        let kogaki = "ぁぃぅぇぉゃゅょっ";
        for c in kogaki.chars() {
            tokens.push(c.to_string());
        }

        // 記号
        tokens.push("。".to_string());
        tokens.push("、".to_string());
        tokens.push("！".to_string());
        tokens.push("？".to_string());
        tokens.push(" ".to_string());
        tokens.push("ー".to_string());

        // タグの開始位置を記録
        let tag_start_id = tokens.len();

        // === 手話タグ ===

        // 1. 時間・挨拶関連
        tokens.push("朝".to_string());
        tokens.push("昼".to_string());
        tokens.push("夜".to_string());
        tokens.push("ありがとう".to_string());
        tokens.push("お願い".to_string());
        tokens.push("挨拶".to_string());

        // 2. 動作関連
        tokens.push("食べる".to_string());
        tokens.push("飲む".to_string());
        tokens.push("行く".to_string());
        tokens.push("来る".to_string());
        tokens.push("見る".to_string());
        tokens.push("聞く".to_string());
        tokens.push("言う".to_string());
        tokens.push("寝る".to_string());
        tokens.push("起きる".to_string());
        tokens.push("歩く".to_string());
        tokens.push("座る".to_string());
        tokens.push("立つ".to_string());
        tokens.push("読む".to_string());
        tokens.push("書く".to_string());
        tokens.push("やる".to_string());
        tokens.push("いる".to_string());

        // 3. 場所関連
        tokens.push("家".to_string());
        tokens.push("学校".to_string());
        tokens.push("会社".to_string());
        tokens.push("医".to_string());
        tokens.push("駅".to_string());
        tokens.push("公".to_string());
        tokens.push("建物".to_string());

        // 4. 人称・指示関連
        tokens.push("私".to_string());
        tokens.push("あなた".to_string());
        tokens.push("彼".to_string());
        tokens.push("彼女".to_string());
        tokens.push("これ".to_string());
        tokens.push("それ".to_string());

        // 5. 感情・状態関連
        tokens.push("悲しい".to_string());
        tokens.push("辛い".to_string());
        tokens.push("寒い".to_string());
        tokens.push("暑い".to_string());
        tokens.push("楽しい".to_string());
        tokens.push("怖い".to_string());
        tokens.push("面白い".to_string());
        tokens.push("厳しい".to_string());

        // 6. その他
        tokens.push("はい".to_string());
        tokens.push("いいえ".to_string());
        tokens.push("OK".to_string());
        tokens.push("ダメ".to_string());
        tokens.push("分かる".to_string());
        tokens.push("分からない".to_string());
        tokens.push("何".to_string());
        tokens.push("なぜ".to_string());
        tokens.push("原因".to_string());
        tokens.push("どこ".to_string());
        tokens.push("場所".to_string());
        tokens.push("誰".to_string());
        tokens.push("何時".to_string());
        tokens.push("いくら".to_string());

        // 7. 数
        tokens.push("1".to_string());
        tokens.push("2".to_string());
        tokens.push("3".to_string());
        tokens.push("4".to_string());
        tokens.push("5".to_string());
        tokens.push("6".to_string());
        tokens.push("7".to_string());
        tokens.push("8".to_string());
        tokens.push("9".to_string());
        tokens.push("10".to_string());
        tokens.push("千".to_string());
        tokens.push("万".to_string());
        tokens.push("億".to_string());

        // 8. 日付・日時
        tokens.push("今".to_string());
        tokens.push("明日".to_string());
        tokens.push("昨日".to_string());
        tokens.push("来週".to_string());
        tokens.push("先週".to_string());
        tokens.push("月".to_string());
        tokens.push("日".to_string());
        tokens.push("週".to_string());
        tokens.push("年".to_string());

        // PADトークン（最後に追加）
        tokens.push("PAD".to_string());

        // token_to_id の構築
        let mut token_to_id = HashMap::new();
        for (id, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), id);
        }

        let vocab_size = tokens.len();

        JslVocabulary {
            token_to_id,
            id_to_token: tokens,
            vocab_size,
            tag_start_id,
        }
    }

    /// 文字列（日本語文字 + タグ）をトークンIDのベクトルに変換
    /// 例: "私は食べます<私><食べる>" -> [35, 36, 37, ...]
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let mut token_ids = Vec::new();
        let mut i = 0;
        let chars: Vec<char> = text.chars().collect();

        while i < chars.len() {
            // タグの開始を検出（'<'）
            if chars[i] == '<' {
                // タグの終了位置を探す
                let mut j = i + 1;
                while j < chars.len() && chars[j] != '>' {
                    j += 1;
                }

                if j < chars.len() {
                    // タグ名を抽出 (例: "<私>" -> "私")
                    let tag_name: String = chars[i + 1..j].iter().collect();

                    if let Some(&id) = self.token_to_id.get(&tag_name) {
                        token_ids.push(id as i32);
                    } else {
                        eprintln!("Warning: Unknown tag '{}' in text", tag_name);
                    }

                    i = j + 1; // '>' の次へ
                } else {
                    // 対応する '>' が見つからない場合、'<' を通常の文字として処理
                    if let Some(&id) = self.token_to_id.get(&chars[i].to_string()) {
                        token_ids.push(id as i32);
                    }
                    i += 1;
                }
            } else {
                // 通常の文字として処理
                if let Some(&id) = self.token_to_id.get(&chars[i].to_string()) {
                    token_ids.push(id as i32);
                } else {
                    // 未知の文字は警告（スキップ）
                    eprintln!("Warning: Unknown character '{}' in text", chars[i]);
                }
                i += 1;
            }
        }

        token_ids
    }

    /// トークンIDのベクトルをPADで埋めてmax_lengthに調整
    pub fn pad_sequence(&self, tokens: &[i32], max_length: usize) -> Vec<i32> {
        let mut padded = tokens.to_vec();
        let pad_id = self.vocab_size - 1; // PADは最後のID

        // max_lengthに満たない場合はPADで埋める
        while padded.len() < max_length {
            padded.push(pad_id as i32);
        }

        // max_lengthを超える場合は切り詰める
        if padded.len() > max_length {
            padded.truncate(max_length);
        }

        padded
    }

    /// トークンIDのベクトルを文字列に変換
    /// 例: [35, 36, 37, 100, 105] -> "私は食<私><食べる>"
    pub fn decode(&self, token_ids: &[i32]) -> String {
        let mut result = String::new();
        let pad_id = self.vocab_size - 1;

        for &id in token_ids {
            // PADトークンはスキップ
            if id == pad_id as i32 {
                continue;
            }

            if id >= 0 && (id as usize) < self.vocab_size {
                let token = &self.id_to_token[id as usize];

                // タグかどうかを判定（tag_start_id以降）
                if (id as usize) >= self.tag_start_id {
                    result.push_str(&format!("<{}>", token));
                } else {
                    // 日本語文字はそのまま追加
                    result.push_str(token);
                }
            } else {
                result.push('?');
            }
        }

        result
    }

    /// タグのみを抽出してデコード（推論結果の表示用）
    /// 例: [35, 36, 37, 100, 105] -> "<私> <食べる>"
    pub fn decode_tags_only(&self, token_ids: &[i32]) -> String {
        let mut tags = Vec::new();
        let pad_id = self.vocab_size - 1;

        for &id in token_ids {
            // PADトークンはスキップ
            if id == pad_id as i32 {
                continue;
            }

            if id >= 0 && (id as usize) < self.vocab_size {
                // タグのみ抽出
                if (id as usize) >= self.tag_start_id {
                    let tag = &self.id_to_token[id as usize];
                    tags.push(format!("<{}>", tag));
                }
            }
        }

        tags.join(" ")
    }
}
