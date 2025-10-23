use crate::model::Seq2SeqModel;
use burn::backend::ndarray::NdArray;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use std::path::PathBuf;

pub type TrainingBackend = Autodiff<Wgpu>;

/// モデルを保存
pub fn save_model(
    model: &Seq2SeqModel<TrainingBackend>,
    save_dir: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;

    fs::create_dir_all(save_dir)?;

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model_path = save_dir.join("model");

    model
        .clone()
        .save_file(model_path, &recorder)
        .map_err(|e| format!("モデル保存エラー: {:?}", e))?;

    println!("モデルを保存: {}", save_dir.display());
    Ok(())
}

/// モデルを読み込み（TrainingBackend用）
pub fn load_model(
    load_dir: &PathBuf,
    device: &<TrainingBackend as Backend>::Device,
) -> Result<Seq2SeqModel<TrainingBackend>, Box<dyn std::error::Error>> {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model_path = load_dir.join("model");

    let model = Seq2SeqModel::<TrainingBackend>::new(device)
        .load_file(model_path, &recorder, device)
        .map_err(|e| format!("モデル読み込みエラー: {:?}", e))?;

    println!("モデルを読み込み: {}", load_dir.display());
    Ok(model)
}

/// モデルを読み込み（ジェネリックなBackend用）
pub fn load_model_generic<B: Backend>(
    load_dir: &PathBuf,
    device: &B::Device,
) -> Result<Seq2SeqModel<B>, Box<dyn std::error::Error>> {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model_path = load_dir.join("model");

    let model = Seq2SeqModel::<B>::new(device)
        .load_file(model_path, &recorder, device)
        .map_err(|e| format!("モデル読み込みエラー: {:?}", e))?;

    println!(
        "モデルを読み込み（{}バックエンド）: {}",
        std::any::type_name::<B>(),
        load_dir.display()
    );
    Ok(model)
}
