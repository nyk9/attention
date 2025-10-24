use crate::translation_model::Seq2SeqModel;
use crate::translation_training::TrainingBackend;
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use std::path::PathBuf;

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
