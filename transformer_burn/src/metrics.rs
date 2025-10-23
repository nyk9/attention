use crate::config;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 訓練メトリクス
#[derive(Serialize, Deserialize, Debug)]
pub struct TrainingMetrics {
    /// 訓練曲線（エポックごとの損失）
    pub loss_history: Vec<f32>,
    /// 最終損失
    pub final_loss: f32,
    /// エポック数
    pub epochs: usize,
    /// 学習率
    pub learning_rate: f64,
    /// バッチサイズ
    pub batch_size: usize,
}

/// モデル設定
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
}

/// メタデータ
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Metadata {
    pub burn_version: String,
    pub trained_at: String,
}

/// 統合メトリクスファイル
#[derive(Serialize, Deserialize, Debug)]
pub struct MetricsFile {
    pub model_config: ModelConfig,
    pub training: TrainingMetrics,
    pub metadata: Metadata,
}

/// メトリクスを保存
pub fn save_metrics(
    save_dir: &PathBuf,
    training_metrics: &TrainingMetrics,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;

    fs::create_dir_all(save_dir)?;

    let model_config = ModelConfig {
        d_model: config::D_MODEL,
        n_heads: config::NUM_HEADS,
        n_layers: config::NUM_LAYERS,
        d_ff: config::D_FF,
        vocab_size: config::VOCAB_SIZE,
        max_seq_len: config::SEQ_LEN,
    };

    let metadata = Metadata {
        burn_version: env!("CARGO_PKG_VERSION").to_string(),
        trained_at: chrono::Local::now().to_rfc3339(),
    };

    // metrics.jsonを保存
    let metrics_file = MetricsFile {
        model_config: model_config.clone(),
        training: TrainingMetrics {
            loss_history: training_metrics.loss_history.clone(),
            final_loss: training_metrics.final_loss,
            epochs: training_metrics.epochs,
            learning_rate: training_metrics.learning_rate,
            batch_size: training_metrics.batch_size,
        },
        metadata: metadata.clone(),
    };

    let metrics_json = serde_json::to_string_pretty(&metrics_file)?;
    fs::write(save_dir.join("metrics.json"), metrics_json)?;
    println!("メトリクスを保存: {}", save_dir.join("metrics.json").display());

    // config.jsonを保存
    save_config(save_dir, &model_config, training_metrics, &metadata)?;

    // README.mdを自動生成
    save_readme(save_dir, &model_config, training_metrics, &metadata)?;

    Ok(())
}

/// config.jsonを保存
fn save_config(
    save_dir: &PathBuf,
    model_config: &ModelConfig,
    training_metrics: &TrainingMetrics,
    metadata: &Metadata,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;

    #[derive(Serialize)]
    struct ConfigFile {
        model: ModelConfig,
        training: TrainingConfig,
        metadata: Metadata,
    }

    #[derive(Serialize)]
    struct TrainingConfig {
        learning_rate: f64,
        epochs: usize,
        batch_size: usize,
        optimizer: String,
    }

    let config_file = ConfigFile {
        model: model_config.clone(),
        training: TrainingConfig {
            learning_rate: training_metrics.learning_rate,
            epochs: training_metrics.epochs,
            batch_size: training_metrics.batch_size,
            optimizer: "Adam".to_string(),
        },
        metadata: metadata.clone(),
    };

    let config_json = serde_json::to_string_pretty(&config_file)?;
    fs::write(save_dir.join("config.json"), config_json)?;
    println!("設定を保存: {}", save_dir.join("config.json").display());

    Ok(())
}

/// README.mdを自動生成
fn save_readme(
    save_dir: &PathBuf,
    model_config: &ModelConfig,
    training_metrics: &TrainingMetrics,
    metadata: &Metadata,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;

    let readme_content = format!(
        r#"# JSL Seq2Seq モデル訓練結果

## モデル設定

- **d_model**: {}
- **ヘッド数**: {}
- **レイヤ数**: {}
- **d_ff**: {}
- **語彙サイズ**: {}
- **最大シーケンス長**: {}

## 訓練設定

- **エポック数**: {}
- **学習率**: {}
- **バッチサイズ**: {}
- **オプティマイザ**: Adam
- **最終Loss**: {:.6}

## 訓練情報

- **訓練日時**: {}
- **Burn バージョン**: {}

## 使用方法

### 推論（WGPU）

```bash
cargo run --release -p transformer_burn --features "wgpu,autodiff,ndarray" -- \
  --load {} \
  --backend wgpu \
  --predict "ありがとう"
```

### 推論（NdArray / CPU）

```bash
cargo run --release -p transformer_burn --features "wgpu,autodiff,ndarray" -- \
  --load {} \
  --backend ndarray \
  --predict "ありがとう"
```

### 推論（自動選択）

```bash
cargo run --release -p transformer_burn --features "wgpu,autodiff,ndarray" -- \
  --load {} \
  --backend auto \
  --predict "ありがとう"
```

## ファイル構成

- `model.bin`: モデル重み（Burnバイナリ形式）
- `config.json`: モデル設定とハイパーパラメータ
- `metrics.json`: 訓練統計と損失履歴
- `README.md`: このファイル
"#,
        model_config.d_model,
        model_config.n_heads,
        model_config.n_layers,
        model_config.d_ff,
        model_config.vocab_size,
        model_config.max_seq_len,
        training_metrics.epochs,
        training_metrics.learning_rate,
        training_metrics.batch_size,
        training_metrics.final_loss,
        metadata.trained_at,
        metadata.burn_version,
        save_dir.display(),
        save_dir.display(),
        save_dir.display(),
    );

    fs::write(save_dir.join("README.md"), readme_content)?;
    println!("READMEを生成: {}", save_dir.join("README.md").display());

    Ok(())
}
