use crate::config::SRC_SEQ_LEN;
use crate::translation_model::Seq2SeqModel;
use crate::translation_training::TrainingBackend;
use crate::translation_vocabulary::{SourceVocabulary, TargetVocabulary};
use burn::backend::ndarray::NdArray;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::Int;
use std::path::PathBuf;

/// 推論（TrainingBackend用）
pub fn predict_translation(
    model: &Seq2SeqModel<TrainingBackend>,
    src_vocab: &SourceVocabulary,
    tgt_vocab: &TargetVocabulary,
    input_text: &str,
    device: &<TrainingBackend as Backend>::Device,
) -> String {
    predict_translation_generic(model, src_vocab, tgt_vocab, input_text, device)
}

/// 推論（ジェネリックBackend）
pub fn predict_translation_generic<B: Backend>(
    model: &Seq2SeqModel<B>,
    src_vocab: &SourceVocabulary,
    tgt_vocab: &TargetVocabulary,
    input_text: &str,
    device: &B::Device,
) -> String {
    // 日本語トークン化とパディング
    let tokens = src_vocab.encode(input_text);
    let tokens = src_vocab.pad_sequence(&tokens, SRC_SEQ_LEN);

    // Tensorに変換
    let src_tokens =
        Tensor::<B, 1, Int>::from_data(tokens.as_slice(), device).reshape([1, SRC_SEQ_LEN]);

    // 自己回帰生成（最大20トークン）
    let generated_ids = model.generate(
        src_tokens,
        None,
        tgt_vocab.sos_id,
        tgt_vocab.eos_id,
        20,
        tgt_vocab.vocab_size,
    );

    // トークンIDをデコード
    let generated_data: Vec<i32> = generated_ids.to_data().to_vec().unwrap();

    // 英語文にデコード
    tgt_vocab.decode(&generated_data)
}

/// モデルを読み込み（ジェネリック）
pub fn load_model_generic<B: Backend>(
    load_dir: &PathBuf,
    device: &B::Device,
    src_vocab_size: usize,
    tgt_vocab_size: usize,
) -> Result<Seq2SeqModel<B>, Box<dyn std::error::Error>> {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model_path = load_dir.join("model");

    let model = Seq2SeqModel::<B>::new(device, src_vocab_size, tgt_vocab_size);

    let record = recorder
        .load(model_path, device)
        .map_err(|e| format!("モデル読み込みエラー: {:?}", e))?;

    Ok(model.load_record(record))
}

/// バックエンドを選択して推論実行
pub fn run_translation_inference(
    backend_name: &str,
    load_dir: &PathBuf,
    predict_text: &str,
    src_vocab: &SourceVocabulary,
    tgt_vocab: &TargetVocabulary,
) -> Result<String, Box<dyn std::error::Error>> {
    let src_vocab_size = src_vocab.vocab_size;
    let tgt_vocab_size = tgt_vocab.vocab_size;

    match backend_name {
        "wgpu" => {
            let device = WgpuDevice::default();
            let model = load_model_generic::<Wgpu>(load_dir, &device, src_vocab_size, tgt_vocab_size)?;
            Ok(predict_translation_generic(
                &model,
                src_vocab,
                tgt_vocab,
                predict_text,
                &device,
            ))
        }
        "ndarray" => {
            let device = Default::default();
            let model = load_model_generic::<NdArray>(load_dir, &device, src_vocab_size, tgt_vocab_size)?;
            Ok(predict_translation_generic(
                &model,
                src_vocab,
                tgt_vocab,
                predict_text,
                &device,
            ))
        }
        "auto" => {
            // autoの場合はWGPUを試し、失敗したらNdArrayにフォールバック
            println!("バックエンド: 自動選択中...");
            let wgpu_result = std::panic::catch_unwind(|| {
                let device = WgpuDevice::default();
                load_model_generic::<Wgpu>(load_dir, &device, src_vocab_size, tgt_vocab_size)
            });

            match wgpu_result {
                Ok(Ok(model)) => {
                    println!("バックエンド: WGPU（自動選択）");
                    let device = WgpuDevice::default();
                    Ok(predict_translation_generic(
                        &model,
                        src_vocab,
                        tgt_vocab,
                        predict_text,
                        &device,
                    ))
                }
                _ => {
                    println!("バックエンド: NdArray（WGPU利用不可のためフォールバック）");
                    let device = Default::default();
                    let model = load_model_generic::<NdArray>(load_dir, &device, src_vocab_size, tgt_vocab_size)?;
                    Ok(predict_translation_generic(
                        &model,
                        src_vocab,
                        tgt_vocab,
                        predict_text,
                        &device,
                    ))
                }
            }
        }
        _ => Err(format!("未対応のバックエンド: {}", backend_name).into()),
    }
}
