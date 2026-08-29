//! 手形記述子の実装検証: 学習を一切せず、最近傍探索だけでどこまで当たるかを測る。
//!
//! # なぜこのテストがあるか
//!
//! `handshape_features.rs` の記述子は「先行調査の Python 実装が top-1 約55% を出した」
//! ことを根拠に導入した。Rust へ書き起こす過程で式や添字を間違えても、
//! 学習モデル側の評価は非決定的でノイズが大きく、間違いに気づけない。
//!
//! そこでモデルを介さず、記述子そのものの分離性能を決定的に測る。
//! ここが 50% 前後を保っていれば記述子の書き起こしは正しい。
//!
//! 実データ(`data/pose_dict_full.json`)が要るので `#[ignore]` 付き。実行:
//! ```text
//! cargo test --test handshape_nn_test -- --ignored --nocapture
//! ```

use std::path::Path;
use transformer_burn::handshape_features::{InputFeatures, HANDSHAPE_FEATURE_DIM};
use transformer_burn::pose_data::{PoseTrainingData, PoseSample};

/// 各次元を train 側の平均・標準偏差で標準化する。
/// 統計量は train からのみ算出する(未見テイクを見て正規化すると転導的な
/// 情報漏洩になり、測定が甘くなるため)
fn zscore(train: &[Vec<f32>], test: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let dim = train[0].len();
    let n = train.len() as f32;
    let mut mean = vec![0.0f32; dim];
    for v in train {
        for j in 0..dim {
            mean[j] += v[j];
        }
    }
    for m in mean.iter_mut() {
        *m /= n;
    }
    let mut sd = vec![0.0f32; dim];
    for v in train {
        for j in 0..dim {
            sd[j] += (v[j] - mean[j]).powi(2);
        }
    }
    for s in sd.iter_mut() {
        *s = (*s / n).sqrt();
        if *s < 1e-9 {
            *s = 1.0; // 定数次元はそのまま(0 割り回避)
        }
    }
    let apply = |rows: &[Vec<f32>]| -> Vec<Vec<f32>> {
        rows.iter()
            .map(|v| (0..dim).map(|j| (v[j] - mean[j]) / sd[j]).collect())
            .collect()
    };
    (apply(train), apply(test))
}

fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// テイク平均モードでは全フレームが同じ値なので、先頭フレームだけを取り出す
fn first_frame(sample: &PoseSample) -> Vec<f32> {
    sample.features[..HANDSHAPE_FEATURE_DIM].to_vec()
}

#[test]
#[ignore = "実データ data/pose_dict_full.json が必要"]
fn handshape_nearest_neighbour_beats_chance_by_far() {
    let path = Path::new("data/pose_dict_full.json");
    if !path.exists() {
        eprintln!("pose dict が無いためスキップ: {}", path.display());
        return;
    }

    // 学習モデルの評価と同じ分割(各語のテイク番号の最後1件を未見)
    let split = PoseTrainingData::load_holdout_split(path, 1).expect("分割に失敗");
    let train = split.train.to_input_features(InputFeatures::HandshapeMean);
    let test = split.test.to_input_features(InputFeatures::HandshapeMean);

    let train_vecs: Vec<Vec<f32>> = train.samples.iter().map(first_frame).collect();
    let test_vecs: Vec<Vec<f32>> = test.samples.iter().map(first_frame).collect();
    let (train_z, test_z) = zscore(&train_vecs, &test_vecs);

    let mut top1 = 0;
    let mut top5 = 0;
    for (i, q) in test_z.iter().enumerate() {
        let truth = &test.samples[i].label;
        let mut ranked: Vec<(f32, &str)> = train_z
            .iter()
            .enumerate()
            .map(|(j, g)| (distance(q, g), train.samples[j].label.as_str()))
            .collect();
        ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // 同じ語が複数テイク並ぶので、ラベル単位に畳んでから上位5件を見る
        let mut seen: Vec<&str> = Vec::new();
        for (_, label) in &ranked {
            if !seen.contains(label) {
                seen.push(label);
            }
            if seen.len() >= 5 {
                break;
            }
        }
        if seen.first() == Some(&truth.as_str()) {
            top1 += 1;
        }
        if seen.iter().any(|l| l == truth) {
            top5 += 1;
        }
    }

    let n = test_z.len();
    println!(
        "手形記述子 + 最近傍(学習なし): gallery={} probe={} top-1 {}/{} ({:.1}%) top-5 {}/{} ({:.1}%)",
        train_z.len(),
        n,
        top1,
        n,
        top1 as f32 / n as f32 * 100.0,
        top5,
        n,
        top5 as f32 / n as f32 * 100.0
    );

    // 51語からの検索なので偶然は top-1 約2%・top-5 約10%。
    // 先行調査(Python)は top-1 約55%・top-5 約80%。書き起こしを間違えると
    // ここが偶然近くまで落ちるので、緩めの下限で回帰を検出する
    assert!(
        top1 as f32 / n as f32 > 0.40,
        "top-1 が想定(約55%)から大きく外れています: {}/{}",
        top1,
        n
    );
    assert!(
        top5 as f32 / n as f32 > 0.65,
        "top-5 が想定(約80%)から大きく外れています: {}/{}",
        top5,
        n
    );
}
