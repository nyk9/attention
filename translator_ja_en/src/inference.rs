use crate::checkpoint::{load_model_generic, TrainingBackend};
use crate::config::SEQ_LEN;
use crate::jsl_vocabulary::JslVocabulary;
use crate::model::Seq2SeqModel;
use burn::backend::ndarray::NdArray;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::prelude::*;
use burn::tensor::Int;
use std::path::PathBuf;

/// 推論（TrainingBackend用）
pub fn predict_tags(
    model: &Seq2SeqModel<TrainingBackend>,
    vocab: &JslVocabulary,
    input_text: &str,
    device: &<TrainingBackend as Backend>::Device,
) -> String {
    predict_tags_generic(model, vocab, input_text, device)
}

/// 推論（ジェネリックBackend）
pub fn predict_tags_generic<B: Backend>(
    model: &Seq2SeqModel<B>,
    vocab: &JslVocabulary,
    input_text: &str,
    device: &B::Device,
) -> String {
    // トークン化とパディング
    let tokens = vocab.encode(input_text);
    let tokens = vocab.pad_sequence(&tokens, SEQ_LEN);

    // Tensorに変換
    let src_tokens = Tensor::<B, 1, Int>::from_data(tokens.as_slice(), device)
        .reshape([1, SEQ_LEN]);

    // 自己回帰生成
    let generated_ids = model.generate(src_tokens, None, vocab.sos_id, vocab.eos_id, 10);

    // トークンIDをデコード
    let generated_data: Vec<i32> = generated_ids.to_data().to_vec().unwrap();

    // タグのみ抽出
    let mut predicted_tags = Vec::new();
    for &id in &generated_data {
        let id_usize = id as usize;

        // SOS、EOS、PADをスキップ
        if id_usize == vocab.sos_id || id_usize == vocab.eos_id || id_usize == vocab.pad_id {
            continue;
        }

        // タグのみ抽出
        if id_usize >= vocab.tag_start_id && id_usize < vocab.vocab_size {
            predicted_tags.push(format!("<{}>", vocab.id_to_token[id_usize]));
        }
    }

    predicted_tags.join(" ")
}

/// バックエンドを選択して推論実行
pub fn run_inference(
    backend_name: &str,
    load_dir: &PathBuf,
    predict_text: &str,
    vocab: &JslVocabulary,
) -> Result<String, Box<dyn std::error::Error>> {
    match backend_name {
        "wgpu" => {
            let device = WgpuDevice::default();
            let model = load_model_generic::<Wgpu>(load_dir, &device)?;
            Ok(predict_tags_generic(&model, vocab, predict_text, &device))
        }
        "ndarray" => {
            let device = Default::default();
            let model = load_model_generic::<NdArray>(load_dir, &device)?;
            Ok(predict_tags_generic(&model, vocab, predict_text, &device))
        }
        "auto" => {
            // autoの場合はWGPUを試し、失敗したらNdArrayにフォールバック
            println!("バックエンド: 自動選択中...");
            let wgpu_result = std::panic::catch_unwind(|| {
                let device = WgpuDevice::default();
                load_model_generic::<Wgpu>(load_dir, &device)
            });

            match wgpu_result {
                Ok(Ok(model)) => {
                    println!("バックエンド: WGPU（自動選択）");
                    let device = WgpuDevice::default();
                    Ok(predict_tags_generic(&model, vocab, predict_text, &device))
                }
                _ => {
                    println!("バックエンド: NdArray（WGPU利用不可のためフォールバック）");
                    let device = Default::default();
                    let model = load_model_generic::<NdArray>(load_dir, &device)?;
                    Ok(predict_tags_generic(&model, vocab, predict_text, &device))
                }
            }
        }
        _ => Err(format!("未対応のバックエンド: {}", backend_name).into()),
    }
}
