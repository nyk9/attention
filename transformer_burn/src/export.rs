use burn::prelude::*;
use std::fs;
use std::path::PathBuf;

/// Attention行列をCSVファイルにエクスポート
///
/// attn_weights: [batch, n_heads, seq_len, seq_len] のAttention重み
/// save_dir: 保存先ディレクトリ
/// layer_name: レイヤー名（例: "encoder_layer1", "decoder_cross_layer2"）
pub fn export_attention_to_csv<B: Backend>(
    attn_weights: &Tensor<B, 4>,
    save_dir: &PathBuf,
    layer_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let exports_dir = save_dir.join("exports");
    fs::create_dir_all(&exports_dir)?;

    let dims = attn_weights.dims();
    let batch_size = dims[0];
    let n_heads = dims[1];
    let seq_len = dims[2];

    // シーケンス長が長すぎる場合は警告
    if seq_len > 20 {
        println!(
            "警告: シーケンス長が{}と長いため、CSV出力をスキップします（推奨: 20以下）",
            seq_len
        );
        return Ok(());
    }

    // データをホストに転送
    let attn_data: Vec<f32> = attn_weights.to_data().to_vec().unwrap();

    // 各バッチ・各ヘッドごとにCSVファイルを作成
    for batch_idx in 0..batch_size {
        for head_idx in 0..n_heads {
            let filename = format!("{}_batch{}_head{}.csv", layer_name, batch_idx, head_idx);
            let filepath = exports_dir.join(filename);

            let mut csv_content = String::new();

            // ヘッダー行（列番号）
            csv_content.push_str("query\\key");
            for key_idx in 0..seq_len {
                csv_content.push_str(&format!(",{}", key_idx));
            }
            csv_content.push('\n');

            // データ行
            for query_idx in 0..seq_len {
                csv_content.push_str(&format!("{}", query_idx));
                for key_idx in 0..seq_len {
                    let index = batch_idx * (n_heads * seq_len * seq_len)
                        + head_idx * (seq_len * seq_len)
                        + query_idx * seq_len
                        + key_idx;
                    let value = attn_data[index];
                    csv_content.push_str(&format!(",{:.6}", value));
                }
                csv_content.push('\n');
            }

            fs::write(&filepath, csv_content)?;
            println!("  Attention行列を出力: {}", filepath.display());
        }
    }

    Ok(())
}

/// 2次元テンソルをCSVファイルにエクスポート
///
/// tensor: [rows, cols] の2次元テンソル
/// save_dir: 保存先ディレクトリ
/// filename: ファイル名（拡張子なし）
pub fn export_tensor_to_csv<B: Backend>(
    tensor: &Tensor<B, 2>,
    save_dir: &PathBuf,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let exports_dir = save_dir.join("exports");
    fs::create_dir_all(&exports_dir)?;

    let dims = tensor.dims();
    let rows = dims[0];
    let cols = dims[1];

    // データをホストに転送
    let tensor_data: Vec<f32> = tensor.to_data().to_vec().unwrap();

    let filepath = exports_dir.join(format!("{}.csv", filename));

    let mut csv_content = String::new();

    // ヘッダー行（列番号）
    csv_content.push_str("row\\col");
    for col_idx in 0..cols {
        csv_content.push_str(&format!(",{}", col_idx));
    }
    csv_content.push('\n');

    // データ行
    for row_idx in 0..rows {
        csv_content.push_str(&format!("{}", row_idx));
        for col_idx in 0..cols {
            let index = row_idx * cols + col_idx;
            let value = tensor_data[index];
            csv_content.push_str(&format!(",{:.6}", value));
        }
        csv_content.push('\n');
    }

    fs::write(&filepath, csv_content)?;
    println!("  テンソルを出力: {}", filepath.display());

    Ok(())
}
