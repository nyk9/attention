mod config;
mod jsl_vocabulary;
mod jsl_data;

fn main() {
    let vocab = jsl_vocabulary::JslVocabulary::new();
    let data = jsl_data::JslTrainingData::load(&vocab, "data/training_data_jsl.txt");
    
    println!("最初の5サンプルを表示:");
    for (i, (input, target)) in data.samples.iter().take(5).enumerate() {
        println!("\nサンプル{}:", i + 1);
        println!("  入力: {:?}", input);
        println!("  入力デコード: {}", vocab.decode(input));
        println!("  ターゲット: {:?}", target);
        println!("  ターゲットデコード: {}", vocab.decode(target));
        
        // 各タグ位置の詳細
        for (j, &tag_id) in target.iter().enumerate() {
            if tag_id < vocab.vocab_size as i32 {
                let token = &vocab.id_to_token[tag_id as usize];
                let is_tag = tag_id as usize >= vocab.tag_start_id;
                println!("    位置{}: ID={}, トークン='{}', タグ={}", j, tag_id, token, is_tag);
            }
        }
    }
}
