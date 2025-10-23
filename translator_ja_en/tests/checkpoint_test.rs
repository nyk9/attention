use burn::backend::ndarray::NdArray;
use burn::backend::wgpu::WgpuDevice;
use burn::prelude::*;
use burn::tensor::Int;
use std::fs;
use std::path::PathBuf;

// テスト用のモジュールをインポート
use transformer_burn::checkpoint::{load_model_generic, save_model, TrainingBackend};
use transformer_burn::config::SEQ_LEN;
use transformer_burn::model::Seq2SeqModel;

/// テスト用の一時ディレクトリを作成
fn create_test_dir() -> PathBuf {
    let test_dir = PathBuf::from("tests/temp_checkpoint_test");
    if test_dir.exists() {
        fs::remove_dir_all(&test_dir).ok();
    }
    fs::create_dir_all(&test_dir).unwrap();
    test_dir
}

/// テスト用の一時ディレクトリを削除
fn cleanup_test_dir(test_dir: &PathBuf) {
    if test_dir.exists() {
        fs::remove_dir_all(test_dir).ok();
    }
}

/// テンソル間の近似一致を検証
fn assert_tensors_close<B: Backend>(
    a: &Tensor<B, 3>,
    b: &Tensor<B, 3>,
    tolerance: f32,
) -> bool {
    let diff = (a.clone() - b.clone()).abs();
    let diff_data: Vec<f32> = diff.to_data().to_vec().unwrap();
    let max_diff = diff_data.iter().copied().fold(0.0_f32, f32::max);
    max_diff < tolerance
}

#[test]
fn test_checkpoint_roundtrip_wgpu() {
    println!("=== テスト: モデル保存/読み込み（WGPU） ===");

    // デバイス初期化
    let device = WgpuDevice::default();

    // テスト用ディレクトリ作成
    let test_dir = create_test_dir();

    // モデル作成
    let model = Seq2SeqModel::<TrainingBackend>::new(&device);

    // テスト入力作成（ダミーデータ）
    let src_tokens = Tensor::<TrainingBackend, 1, Int>::from_data(
        vec![1, 2, 3, 4, 5, 0, 0, 0, 0, 0].as_slice(),
        &device,
    )
    .reshape([1, SEQ_LEN]);

    let tgt_tokens = Tensor::<TrainingBackend, 1, Int>::from_data(
        vec![1, 2, 3, 0, 0, 0, 0, 0, 0, 0].as_slice(),
        &device,
    )
    .reshape([1, SEQ_LEN]);

    // 保存前の出力
    let output_before = model.forward(src_tokens.clone(), tgt_tokens.clone(), None, None);

    // モデル保存
    save_model(&model, &test_dir).expect("モデル保存失敗");

    // モデル読み込み
    let loaded_model =
        load_model_generic::<TrainingBackend>(&test_dir, &device).expect("モデル読み込み失敗");

    // 読み込み後の出力
    let output_after = loaded_model.forward(src_tokens, tgt_tokens, None, None);

    // 近似一致を検証
    let is_close = assert_tensors_close(&output_before, &output_after, 1e-5);
    assert!(
        is_close,
        "保存前後の出力が一致しません（許容誤差: 1e-5）"
    );

    println!("✓ 保存前後の出力が一致しました");

    // クリーンアップ
    cleanup_test_dir(&test_dir);

    println!("=== テスト完了 ===");
}

#[test]
fn test_checkpoint_roundtrip_ndarray() {
    println!("=== テスト: モデル保存/読み込み（NdArray） ===");

    // デバイス初期化
    let train_device = WgpuDevice::default();
    let infer_device: <NdArray as Backend>::Device = Default::default();

    // テスト用ディレクトリ作成
    let test_dir = PathBuf::from("tests/temp_checkpoint_test_ndarray");
    if test_dir.exists() {
        fs::remove_dir_all(&test_dir).ok();
    }
    fs::create_dir_all(&test_dir).unwrap();

    // モデル作成（WGPU）
    let model = Seq2SeqModel::<TrainingBackend>::new(&train_device);

    // テスト入力作成（WGPU）
    let src_tokens = Tensor::<TrainingBackend, 1, Int>::from_data(
        vec![1, 2, 3, 4, 5, 0, 0, 0, 0, 0].as_slice(),
        &train_device,
    )
    .reshape([1, SEQ_LEN]);

    let tgt_tokens = Tensor::<TrainingBackend, 1, Int>::from_data(
        vec![1, 2, 3, 0, 0, 0, 0, 0, 0, 0].as_slice(),
        &train_device,
    )
    .reshape([1, SEQ_LEN]);

    // 保存前の出力（WGPU）
    let output_before = model.forward(src_tokens.clone(), tgt_tokens.clone(), None, None);

    // モデル保存
    save_model(&model, &test_dir).expect("モデル保存失敗");

    // モデル読み込み（NdArray）
    let loaded_model =
        load_model_generic::<NdArray>(&test_dir, &infer_device).expect("モデル読み込み失敗");

    // テスト入力作成（NdArray）
    let src_tokens_nd = Tensor::<NdArray, 1, Int>::from_data(
        vec![1, 2, 3, 4, 5, 0, 0, 0, 0, 0].as_slice(),
        &infer_device,
    )
    .reshape([1, SEQ_LEN]);

    let tgt_tokens_nd = Tensor::<NdArray, 1, Int>::from_data(
        vec![1, 2, 3, 0, 0, 0, 0, 0, 0, 0].as_slice(),
        &infer_device,
    )
    .reshape([1, SEQ_LEN]);

    // 読み込み後の出力（NdArray）
    let output_after = loaded_model.forward(src_tokens_nd, tgt_tokens_nd, None, None);

    // WGPU出力をホストに転送
    let output_before_data: Vec<f32> = output_before.to_data().to_vec().unwrap();
    let output_after_data: Vec<f32> = output_after.to_data().to_vec().unwrap();

    // 各要素の相対誤差を計算
    let mut max_rel_error = 0.0_f32;
    for (a, b) in output_before_data.iter().zip(output_after_data.iter()) {
        let rel_error = ((a - b) / a.max(1e-8)).abs();
        max_rel_error = max_rel_error.max(rel_error);
    }

    println!("最大相対誤差: {:.6}", max_rel_error);
    assert!(
        max_rel_error < 1e-4,
        "WGPU/NdArray間の出力が一致しません（最大相対誤差: {:.6}）",
        max_rel_error
    );

    println!("✓ WGPU/NdArray間の出力が一致しました（相対誤差: {:.6}）", max_rel_error);

    // クリーンアップ
    cleanup_test_dir(&test_dir);

    println!("=== テスト完了 ===");
}

#[test]
fn test_config_files_created() {
    println!("=== テスト: 設定ファイル生成 ===");

    // デバイス初期化
    let device = WgpuDevice::default();

    // テスト用ディレクトリ作成
    let test_dir = create_test_dir();

    // モデル作成・保存
    let model = Seq2SeqModel::<TrainingBackend>::new(&device);
    save_model(&model, &test_dir).expect("モデル保存失敗");

    // model.bin の存在確認
    assert!(
        test_dir.join("model.bin").exists(),
        "model.bin が作成されていません"
    );
    println!("✓ model.bin が作成されました");

    // クリーンアップ
    cleanup_test_dir(&test_dir);

    println!("=== テスト完了 ===");
}
