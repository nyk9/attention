//! 入力特徴量の標準化(z-score)。
//!
//! # なぜ必要か
//!
//! 手形記述子(`handshape_features.rs`)は桁の違う量を 1 本のベクトルに混ぜている:
//! 関節角は 0〜π(約 0〜3.14)、正規化距離は 0〜2 程度、手首座標は 0〜1、検出フラグは 0/1。
//! Linear 射影の初期化は全次元を同じスケールとみなすため、桁の大きい次元が
//! 学習初期の勾配を支配しやすい。先行調査の最近傍探索も z-score を掛けた上で
//! 成績を出していた。
//!
//! # 汚染を避ける約束
//!
//! 平均・標準偏差は **train 分割からのみ** 算出する。held-out を含めて計算すると
//! 未見テイクの情報が前処理に漏れ、評価が甘くなる(転導的な情報漏洩)。
//! そのため `fit` は `PoseTrainingData` を 1 つしか受け取らない設計にしてある。
//!
//! # 保存と再現
//!
//! 統計量はモデルと同じディレクトリに `feature_stats.json` として保存する。
//! **このファイルが存在すること = そのモデルは標準化済みの入力で学習された** という
//! 取り決めにしている(設定フラグを別に持つと 2 か所が食い違いうるため)。
//! 推論側(`--predict-pose` / pose_extractor 経由)は必ず同じ変換を掛けてから
//! モデルへ渡す。ここが崩れると保存モデルは実用時に壊れる。

use serde::{Deserialize, Serialize};
use std::path::Path;

/// 標準偏差がこの値未満の次元は「定数次元」とみなし、割り算を 1.0 にする。
/// 0 割りを避けつつ、定数次元を巨大な値へ増幅しないための下限
const MIN_STD: f32 = 1e-6;

/// 特徴量 1 次元ごとの平均と標準偏差
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureStats {
    /// 次元ごとの平均(長さ = feature_dim)
    pub mean: Vec<f32>,
    /// 次元ごとの標準偏差(長さ = feature_dim、下限 MIN_STD 適用済み)
    pub std: Vec<f32>,
}

impl FeatureStats {
    /// フラット化された特徴量列の集合から統計量を求める。
    ///
    /// - `samples`: 各要素が `[frames * feature_dim]` のフラット列
    /// - 平均・標準偏差は「全サンプル × 全フレーム」をまとめた母集団に対して計算する
    ///   (1 フレームを 1 観測とみなす)
    ///
    /// 呼び出し側は **train 分割だけ** を渡すこと(モジュールドキュメント参照)
    pub fn fit<'a, I>(samples: I, feature_dim: usize) -> Self
    where
        I: IntoIterator<Item = &'a [f32]>,
    {
        assert!(feature_dim > 0, "feature_dim には 1 以上が必要です");
        let mut sum = vec![0.0f64; feature_dim];
        let mut sum_sq = vec![0.0f64; feature_dim];
        let mut count: usize = 0;

        for features in samples {
            assert_eq!(
                features.len() % feature_dim,
                0,
                "特徴量の長さ({})が feature_dim({})の倍数ではありません",
                features.len(),
                feature_dim
            );
            for frame in features.chunks(feature_dim) {
                for (j, &v) in frame.iter().enumerate() {
                    sum[j] += v as f64;
                    sum_sq[j] += (v as f64) * (v as f64);
                }
                count += 1;
            }
        }

        assert!(count > 0, "標準化の統計量を取る観測が 1 件もありません");
        let n = count as f64;
        let mut mean = vec![0.0f32; feature_dim];
        let mut std = vec![0.0f32; feature_dim];
        for j in 0..feature_dim {
            let m = sum[j] / n;
            // 母分散(train を母集団とみなす)。数値誤差で微小な負になりうるので下で clamp
            let var = (sum_sq[j] / n - m * m).max(0.0);
            mean[j] = m as f32;
            let s = var.sqrt() as f32;
            std[j] = if s < MIN_STD { 1.0 } else { s };
        }
        Self { mean, std }
    }

    /// 特徴次元(= mean の長さ)
    pub fn feature_dim(&self) -> usize {
        self.mean.len()
    }

    /// フラット列に標準化を適用する(フレーム単位に同じ平均・標準偏差を掛ける)
    pub fn apply(&self, features: &[f32]) -> Vec<f32> {
        let dim = self.feature_dim();
        assert_eq!(
            features.len() % dim,
            0,
            "特徴量の長さ({})が feature_dim({})の倍数ではありません",
            features.len(),
            dim
        );
        let mut out = Vec::with_capacity(features.len());
        for frame in features.chunks(dim) {
            for (j, &v) in frame.iter().enumerate() {
                out.push((v - self.mean[j]) / self.std[j]);
            }
        }
        out
    }

    /// 保存先ファイル名。モデルディレクトリ直下に置く
    pub fn file_name() -> &'static str {
        "feature_stats.json"
    }

    pub fn save(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(dir)?;
        let file = std::fs::File::create(dir.join(Self::file_name()))?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    /// モデルディレクトリから統計量を読む。
    /// ファイルが無ければ `None`(= 標準化なしで学習されたモデル)。
    /// 既存モデル(`models/rec_full` 等)はこの経路で従来通り動く
    pub fn load(dir: &Path) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let path = dir.join(Self::file_name());
        if !path.exists() {
            return Ok(None);
        }
        let content = std::fs::read_to_string(&path)?;
        let stats: Self = serde_json::from_str(&content)?;
        if stats.mean.len() != stats.std.len() {
            return Err(format!(
                "{} の mean({})と std({})の長さが一致しません",
                path.display(),
                stats.mean.len(),
                stats.std.len()
            )
            .into());
        }
        Ok(Some(stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // [TEST向き] 手計算で検証できる小さな例
    #[test]
    fn fit_computes_population_mean_and_std() {
        // feature_dim=2、2 サンプル × 2 フレーム = 4 観測
        // 次元0: 1,3,5,7 → 平均4、母標準偏差 sqrt(5)≈2.2360680
        // 次元1: 全部 2 → 平均2、標準偏差 0 → 下限で 1.0
        let a = vec![1.0f32, 2.0, 3.0, 2.0];
        let b = vec![5.0f32, 2.0, 7.0, 2.0];
        let stats = FeatureStats::fit(vec![a.as_slice(), b.as_slice()], 2);
        assert!((stats.mean[0] - 4.0).abs() < 1e-5);
        assert!((stats.std[0] - 5.0f32.sqrt()).abs() < 1e-4);
        assert!((stats.mean[1] - 2.0).abs() < 1e-5);
        assert_eq!(stats.std[1], 1.0, "定数次元の標準偏差は 1.0 に置き換える");
    }

    // [TEST向き] 適用後、train 自身の平均は 0・標準偏差は 1 になること
    #[test]
    fn apply_normalizes_training_data() {
        let a = vec![1.0f32, 2.0, 3.0, 2.0];
        let b = vec![5.0f32, 2.0, 7.0, 2.0];
        let stats = FeatureStats::fit(vec![a.as_slice(), b.as_slice()], 2);
        let za = stats.apply(&a);
        let zb = stats.apply(&b);
        let col0: Vec<f32> = vec![za[0], za[2], zb[0], zb[2]];
        let mean: f32 = col0.iter().sum::<f32>() / 4.0;
        let var: f32 = col0.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "平均が 0 になっていません: {}", mean);
        assert!((var.sqrt() - 1.0).abs() < 1e-4, "標準偏差が 1 になっていません");
        // 定数次元は平均を引くだけ(2 - 2 = 0)
        assert_eq!(za[1], 0.0);
        assert_eq!(zb[3], 0.0);
    }

    // [TEST向き] 保存 → 読み込みの往復。無ければ None(既存モデルの後方互換)
    #[test]
    fn save_load_roundtrip_and_missing_file() {
        let dir = std::path::PathBuf::from("tests/temp_feature_stats");
        if dir.exists() {
            std::fs::remove_dir_all(&dir).ok();
        }
        std::fs::create_dir_all(&dir).unwrap();

        assert!(
            FeatureStats::load(&dir).unwrap().is_none(),
            "ファイルが無いときは None を返すこと"
        );

        let stats = FeatureStats {
            mean: vec![1.0, 2.0, 3.0],
            std: vec![0.5, 1.5, 2.5],
        };
        stats.save(&dir).unwrap();
        let loaded = FeatureStats::load(&dir).unwrap().expect("読み込めるはず");
        assert_eq!(loaded, stats);

        std::fs::remove_dir_all(&dir).ok();
    }
}
