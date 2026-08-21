// モデルハイパーパラメーター
pub const D_MODEL: usize = 64; // 埋め込み次元
pub const NUM_HEADS: usize = 2; // Multi-head Attentionのヘッド数
pub const D_HEAD: usize = D_MODEL / NUM_HEADS; // 各ヘッドの数
pub const D_FF: usize = D_MODEL * 4; // Feed-forward中間層の次元数
pub const SEQ_LEN: usize = 10; // シーケンス長
pub const NUM_LAYERS: usize = 4; // Transformerのレイヤー数
pub const VOCAB_SIZE: usize = 168; // 語彙サイズ（JSL: ひらがな86 + タグ79 + SOS/EOS/PAD 3）
pub const PAD_TOKEN: usize = 167; // パディングトークン

// 訓練設定
pub const LEARNING_RATE: f64 = 0.00005; // 学習率
pub const EPOCHS: usize = 10000; // エポック数
pub const BATCH_SIZE: usize = 128; // バッチサイズ

// ===== 認識モデル(手ポーズ列→タグ)用 =====
// pose_extractor build-dict の出力(1フレーム = 左右の手 21点×xyz = 126次元)を入力とする
pub const POSE_FEATURE_DIM: usize = 126; // 手ポーズ特徴量の次元
pub const POSE_EPOCHS: usize = 300; // 認識モデルの既定エポック数(--epochs で上書き可)
pub const POSE_LEARNING_RATE: f64 = 0.0005; // 認識モデルの学習率

// ===== 認識モデルのサイズ設定(実行時に可変) =====
//
// 上の D_MODEL/NUM_HEADS/D_HEAD/D_FF/NUM_LAYERS はコンパイル時定数で、足場の Seq2Seq
// (TransformerModel/Encoder/Decoder/Seq2SeqModel)は今回も引き続きこれを直接使う。
//
// 一方、認識モデル(手ポーズ列→タグ)は51語×3テイク=153サンプルしかなく、
// 現行構成(約48万パラメータ)はデータ量に対して大きすぎるのではという仮説がある。
// これを安く検証するため、認識モデルだけ寸法を実行時に選べるようにする
// (CLI フラグ未指定なら下の Default = 現行 const と完全に同じ値になる)。
use serde::{Deserialize, Serialize};

/// 認識モデル(PoseEncoder + RecognitionDecoder)の寸法設定。
/// `--model-size` プリセット + 個別フラグ(`--d-model` など)で CLI から解決され、
/// 保存時に `model_config.json` として書き出すことで読み込み時にも同じ構造を復元できる
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RecognitionModelConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_head: usize, // d_model / num_heads を保存時点で確定させておく(毎回割り算しない)
    pub d_ff: usize,
    pub num_layers: usize,
}

impl Default for RecognitionModelConfig {
    /// 「何も指定しない」ときの値。現行 const と完全に一致させることで、
    /// サイズ系フラグ未指定時の挙動を変えない
    fn default() -> Self {
        Self {
            d_model: D_MODEL,
            num_heads: NUM_HEADS,
            d_head: D_HEAD,
            d_ff: D_FF,
            num_layers: NUM_LAYERS,
        }
    }
}

impl RecognitionModelConfig {
    /// 名前付きプリセット。`base` が既定(= 現行 const)で、`small`/`tiny`/`micro` の順に縮小する。
    /// パラメータ数の目安は 51語データでの概算(実装プラン参照)
    pub fn preset(name: &str) -> Option<Self> {
        let (d_model, num_heads, d_ff, num_layers) = match name {
            "base" => (D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS), // 約483K params相当
            "small" => (32, 2, 128, 2),                       // 約67K params相当
            "tiny" => (16, 2, 64, 2),                         // 約19K params相当
            "micro" => (16, 2, 32, 1),                        // 約9K params相当
            _ => return None,
        };
        if num_heads == 0 || d_model % num_heads != 0 {
            // プリセット自身が壊れていたら早期に気づけるようにしておく(呼び出し側の責任にしない)
            return None;
        }
        Some(Self {
            d_model,
            num_heads,
            d_head: d_model / num_heads,
            d_ff,
            num_layers,
        })
    }

    /// プリセット名 + 個別上書きフラグから設定を解決する。
    /// 優先順位: プリセット(未指定なら "base") → 個別フラグでの上書き → 検証。
    /// `d_model` を指定して `d_ff` を指定しない場合は `d_ff = d_model * 4`
    /// (D_FF = D_MODEL * 4 という既存の関係を保つ)。
    /// 学習を始める前に理由がわかるメッセージで落とすため、結果は Result で返す
    pub fn resolve(
        preset: Option<&str>,
        d_model: Option<usize>,
        num_heads: Option<usize>,
        d_ff: Option<usize>,
        num_layers: Option<usize>,
    ) -> Result<Self, String> {
        let preset_name = preset.unwrap_or("base");
        let base = Self::preset(preset_name)
            .ok_or_else(|| format!("未知のモデルサイズプリセットです: \"{}\"（base/small/tiny/micro から選んでください）", preset_name))?;

        let resolved_d_model = d_model.unwrap_or(base.d_model);
        let resolved_num_heads = num_heads.unwrap_or(base.num_heads);
        // --d-model のみ指定して --d-ff を指定しない場合は d_model*4 を自動導出する。
        // d_model 自体が未指定なら base.d_ff をそのまま使う
        let resolved_d_ff = d_ff.unwrap_or_else(|| {
            if d_model.is_some() {
                resolved_d_model * 4
            } else {
                base.d_ff
            }
        });
        let resolved_num_layers = num_layers.unwrap_or(base.num_layers);

        if resolved_d_model == 0 {
            return Err("--d-model には 1 以上を指定してください".to_string());
        }
        if resolved_num_heads == 0 {
            return Err("--num-heads には 1 以上を指定してください".to_string());
        }
        if resolved_d_ff == 0 {
            return Err("--d-ff には 1 以上を指定してください".to_string());
        }
        if resolved_num_layers == 0 {
            return Err("--num-layers には 1 以上を指定してください".to_string());
        }
        if resolved_d_model % resolved_num_heads != 0 {
            return Err(format!(
                "--d-model({}) は --num-heads({}) で割り切れる必要があります",
                resolved_d_model, resolved_num_heads
            ));
        }

        Ok(Self {
            d_model: resolved_d_model,
            num_heads: resolved_num_heads,
            d_head: resolved_d_model / resolved_num_heads,
            d_ff: resolved_d_ff,
            num_layers: resolved_num_layers,
        })
    }
}

#[cfg(test)]
mod recognition_model_config_tests {
    use super::*;

    #[test]
    fn default_matches_current_consts() {
        // 既定経路が現状から変わっていないことの回帰テスト
        let default = RecognitionModelConfig::default();
        assert_eq!(default.d_model, D_MODEL);
        assert_eq!(default.num_heads, NUM_HEADS);
        assert_eq!(default.d_head, D_HEAD);
        assert_eq!(default.d_ff, D_FF);
        assert_eq!(default.num_layers, NUM_LAYERS);
    }

    #[test]
    fn base_preset_matches_default() {
        let base = RecognitionModelConfig::preset("base").unwrap();
        assert_eq!(base, RecognitionModelConfig::default());
    }

    #[test]
    fn small_preset_values() {
        let small = RecognitionModelConfig::preset("small").unwrap();
        assert_eq!(small.d_model, 32);
        assert_eq!(small.num_heads, 2);
        assert_eq!(small.d_head, 16);
        assert_eq!(small.d_ff, 128);
        assert_eq!(small.num_layers, 2);
    }

    #[test]
    fn tiny_preset_values() {
        let tiny = RecognitionModelConfig::preset("tiny").unwrap();
        assert_eq!(tiny.d_model, 16);
        assert_eq!(tiny.num_heads, 2);
        assert_eq!(tiny.d_head, 8);
        assert_eq!(tiny.d_ff, 64);
        assert_eq!(tiny.num_layers, 2);
    }

    #[test]
    fn micro_preset_values() {
        let micro = RecognitionModelConfig::preset("micro").unwrap();
        assert_eq!(micro.d_model, 16);
        assert_eq!(micro.num_heads, 2);
        assert_eq!(micro.d_head, 8);
        assert_eq!(micro.d_ff, 32);
        assert_eq!(micro.num_layers, 1);
    }

    #[test]
    fn unknown_preset_is_none() {
        assert!(RecognitionModelConfig::preset("giant").is_none());
    }

    #[test]
    fn resolve_no_args_is_default() {
        let resolved = RecognitionModelConfig::resolve(None, None, None, None, None).unwrap();
        assert_eq!(resolved, RecognitionModelConfig::default());
    }

    #[test]
    fn resolve_preset_only() {
        let resolved =
            RecognitionModelConfig::resolve(Some("tiny"), None, None, None, None).unwrap();
        assert_eq!(resolved, RecognitionModelConfig::preset("tiny").unwrap());
    }

    #[test]
    fn resolve_with_individual_override() {
        // small をベースに num_layers だけ上書き
        let resolved =
            RecognitionModelConfig::resolve(Some("small"), None, None, None, Some(4)).unwrap();
        assert_eq!(resolved.d_model, 32);
        assert_eq!(resolved.num_heads, 2);
        assert_eq!(resolved.d_ff, 128);
        assert_eq!(resolved.num_layers, 4);
    }

    #[test]
    fn resolve_d_model_only_derives_d_ff() {
        // --d-model だけ指定した場合、d_ff = d_model * 4 が自動導出されること
        let resolved =
            RecognitionModelConfig::resolve(None, Some(48), None, None, None).unwrap();
        assert_eq!(resolved.d_model, 48);
        assert_eq!(resolved.num_heads, NUM_HEADS); // base 由来のまま
        assert_eq!(resolved.d_ff, 192); // 48 * 4
        assert_eq!(resolved.d_head, 24); // 48 / 2
    }

    #[test]
    fn resolve_rejects_indivisible_d_model_num_heads() {
        let err = RecognitionModelConfig::resolve(None, Some(30), Some(4), None, None)
            .unwrap_err();
        assert!(err.contains("30"));
        assert!(err.contains("4"));
    }

    #[test]
    fn resolve_rejects_zero_d_model() {
        assert!(RecognitionModelConfig::resolve(None, Some(0), None, None, None).is_err());
    }

    #[test]
    fn resolve_rejects_zero_num_heads() {
        assert!(RecognitionModelConfig::resolve(None, None, Some(0), None, None).is_err());
    }

    #[test]
    fn resolve_rejects_unknown_preset() {
        assert!(RecognitionModelConfig::resolve(Some("giant"), None, None, None, None).is_err());
    }
}
