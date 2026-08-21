#![recursion_limit = "256"]
mod checkpoint;
mod config;
mod handshape_features;
mod inference;
mod jsl_data;
mod jsl_vocabulary;
mod metrics;
mod model;
mod pose_data;
mod recognition;
mod recognition_training;
mod tag_vocabulary;
mod training;

use checkpoint::{TrainingBackend, load_model, save_model};
use inference::{predict_tags, run_inference};
use metrics::save_metrics;
use training::train_jsl;

use burn::backend::wgpu::WgpuDevice;
use burn::prelude::{Backend, Module};
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

/// Seq2Seq日本語→手話タグ翻訳モデル
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// 訓練モード（訓練を実行する場合に指定）
    #[arg(long)]
    train: bool,

    /// モデルを保存するディレクトリ
    #[arg(long)]
    save: Option<PathBuf>,

    /// モデルを読み込むディレクトリ
    #[arg(long)]
    load: Option<PathBuf>,

    /// 推論するテキスト
    #[arg(long)]
    predict: Option<String>,

    /// バックエンドの選択（auto, wgpu, ndarray）
    #[arg(long, default_value = "wgpu")]
    backend: String,

    /// Attention行列をCSVエクスポート（推論時のみ）
    #[arg(long)]
    export_attn: bool,

    /// 認識モデル訓練: build-dict が出力した pose dict JSON を指定
    #[arg(long)]
    train_pose: Option<PathBuf>,

    /// 認識モデル推論: pose dict JSON の各エントリを top-5 評価（--load 必須）
    #[arg(long)]
    predict_pose: Option<PathBuf>,

    /// 認識モデルのエポック数
    #[arg(long, default_value_t = config::POSE_EPOCHS)]
    epochs: usize,

    /// 未見テイク評価: pose dict をテイク単位で train/held-out に分割し、学習後に未見テイクで top-k を測る
    #[arg(long)]
    eval_holdout: Option<PathBuf>,

    /// 診断: pose dict の左右手カバレッジをラベル別に点検し、テイク間で使われた手が食い違う語を警告(学習なし)
    #[arg(long)]
    inspect_dict: Option<PathBuf>,

    /// held-out にするテイク数(ラベルごと、テイク番号の後ろから)。--eval-holdout 用
    #[arg(long, default_value_t = 1)]
    holdout_per_label: usize,

    /// ミラー拡張: 学習データに左右反転コピーを追加する(--train-pose / --eval-holdout の train 側のみ。
    /// held-out の未見テイクには適用しない)
    #[arg(long)]
    mirror_augment: bool,

    /// 乱数シード(モデル重み初期化に使用)。--train-pose / --eval-holdout / --predict-pose で
    /// 同じ値を指定すれば結果が再現できる。バッチの並び順は現状シャッフルしていないため、
    /// 揺れの発生源は重み初期化のみ
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// 認識モデルのサイズプリセット(base/small/tiny/micro)。未指定なら base(= 現状の設定)。
    /// --d-model 等の個別フラグはこのプリセットを上書きする
    #[arg(long)]
    model_size: Option<String>,

    /// 認識モデルの埋め込み次元 d_model を個別指定(未指定ならプリセット値)
    #[arg(long)]
    d_model: Option<usize>,

    /// 認識モデルの Multi-head Attention のヘッド数を個別指定(未指定ならプリセット値)
    #[arg(long)]
    num_heads: Option<usize>,

    /// 認識モデルの Feed-forward 中間層の次元数を個別指定
    /// (未指定時: --d-model のみ指定していれば d_model*4 を自動導出、それ以外はプリセット値)
    #[arg(long)]
    d_ff: Option<usize>,

    /// 認識モデルの Transformer 層数を個別指定(未指定ならプリセット値)
    #[arg(long)]
    num_layers: Option<usize>,

    /// 認識モデルの入力特徴量(raw / handshape / handshape-mean)。
    /// raw(既定)= 従来通り生の 126 次元ハンドランドマーク。
    /// handshape = フレームごとの手形記述子 66 次元(関節角・指先間距離・手首座標)。
    /// handshape-mean = テイク平均の手形記述子を全フレームに複製したもの
    #[arg(long, default_value = "raw")]
    input_features: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    let args = Args::parse();

    // 診断モード: モデルもバックエンドも使わない JSON 分析なのでここで完結させる
    if let Some(dict_path) = &args.inspect_dict {
        pose_data::inspect_dict(dict_path)?;
        return Ok(());
    }

    // 認識モデル(手ポーズ列→タグ)のモード。足場の Seq2Seq とはモデルも語彙も別物なので
    // ここで分岐して早期 return する
    if args.train_pose.is_some() || args.predict_pose.is_some() || args.eval_holdout.is_some() {
        run_recognition(&args)?;
        let duration = start_time.elapsed();
        println!("\n実行時間: {:.2}秒", duration.as_secs_f64());
        return Ok(());
    }

    let jsl_vocab = jsl_vocabulary::JslVocabulary::new();
    let training_device = WgpuDevice::default();

    // モデルの初期化または読み込み
    let mut model = if let Some(load_dir) = &args.load {
        load_model(load_dir, &training_device)?
    } else {
        model::Seq2SeqModel::<TrainingBackend>::new(&training_device)
    };

    // 訓練モード
    if args.train {
        println!("\n===== 訓練開始 =====");
        let jsl_data = jsl_data::JslTrainingData::load(&jsl_vocab, "data/training_data_jsl.txt");
        println!("訓練サンプル数: {}サンプル", jsl_data.len());

        let (trained_model, training_metrics) =
            train_jsl(model, &jsl_data, &jsl_vocab, &training_device);
        model = trained_model;
        println!("訓練完了！");

        // モデルとメトリクスを保存
        if let Some(save_dir) = &args.save {
            save_model(&model, save_dir)?;
            save_metrics(save_dir, &training_metrics)?;
        }
    }

    // 推論モード
    if let Some(predict_text) = &args.predict {
        println!("\n===== 推論テスト =====");

        if args.load.is_some() {
            // モデルが読み込まれている場合は指定されたバックエンドで推論
            let predicted_tags = run_inference(
                &args.backend,
                args.load.as_ref().unwrap(),
                predict_text,
                &jsl_vocab,
            )?;
            println!("入力: {} → 予測タグ: {}", predict_text, predicted_tags);
        } else {
            // 訓練直後の場合はTrainingBackendで推論
            let predicted_tags = predict_tags(&model, &jsl_vocab, predict_text, &training_device);
            println!("入力: {} → 予測タグ: {}", predict_text, predicted_tags);
        }
    }

    // デモモード（引数なし）
    if !args.train && args.predict.is_none() && args.load.is_none() {
        println!("===== デモモード =====");
        println!("使用方法:");
        println!("  訓練: cargo run --release -- --train --save models/test");
        println!("  推論: cargo run --release -- --load models/test --predict \"ありがとう\"");
        println!(
            "  継続訓練: cargo run --release -- --load models/test --train --save models/test2"
        );
    }

    let duration = start_time.elapsed();
    println!("\n実行時間: {:.2}秒", duration.as_secs_f64());

    Ok(())
}

/// 認識モデル(手ポーズ列→タグ)の訓練・推論を実行する
///
/// 使用例:
///   訓練: cargo run --release -- --train-pose data/pose_dict_smoke.json --save models/rec001
///   推論: cargo run --release -- --load models/rec001 --predict-pose data/pose_dict_smoke.json
///   未見テイク評価: cargo run --release -- --eval-holdout data/pose_dict_smoke.json [--holdout-per-label 1]
fn run_recognition(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let device = WgpuDevice::default();
    // モデル重み初期化を再現可能にする(データのシャッフルは無いため揺れの発生源はここのみ)。
    // モデル構築より前に呼ぶ必要がある
    TrainingBackend::seed(args.seed);
    println!("seed: {}", args.seed);

    // 認識モデルのサイズ設定を解決する(サイズ系フラグ未指定なら base = 現行 const と同じ値)。
    // 学習を始める前に検証エラーで落としたいので、モデルを組む前・分岐の外側で一度だけ呼ぶ
    // 入力特徴量の種類も、学習を始める前に解決しておく(未知の値なら理由付きで落とす)
    let input_features = handshape_features::InputFeatures::parse(&args.input_features)?;
    let model_config = config::RecognitionModelConfig::resolve(
        args.model_size.as_deref(),
        args.d_model,
        args.num_heads,
        args.d_ff,
        args.num_layers,
    )?
    .with_input_features(input_features);

    // --predict-pose(--load 経由の推論)ではモデル構造は保存済みモデル側が真実。
    // サイズ系フラグを黙って無視すると誤用に気づけないため、何かを始める前にエラーで止める
    // (--eval-holdout × --load のエラー処理と同じ流儀)
    let size_flag_given = args.model_size.is_some()
        || args.d_model.is_some()
        || args.num_heads.is_some()
        || args.d_ff.is_some()
        || args.num_layers.is_some();
    if args.predict_pose.is_some() && !input_features.is_raw() {
        return Err(
            "--predict-pose(--load 経由の推論)では入力表現も保存済みモデル側が真実のため、\
             --input-features は指定できません(モデルの model_config.json から自動で決まります)"
                .into(),
        );
    }
    if args.predict_pose.is_some() && size_flag_given {
        return Err(
            "--predict-pose(--load 経由の推論)ではモデル構造は保存済みモデル側が真実のため、\
             サイズ系フラグ(--model-size/--d-model/--num-heads/--d-ff/--num-layers)は指定できません"
                .into(),
        );
    }

    // ここで解決した構成でモデルを組むのは学習経路だけ。推論経路では保存済みモデル側の
    // 構成が使われるため、ここで表示すると実際に読み込まれる構造と食い違って誤解を招く
    // (推論時の構成表示は load_recognition_model 側が行う)
    if args.predict_pose.is_none() {
        println!(
            "入力特徴量: {} ({}次元/フレーム)",
            input_features.name(),
            model_config.input_dim
        );
        println!(
            "モデル構成: d_model={} / num_heads={} / d_head={} / d_ff={} / num_layers={}",
            model_config.d_model,
            model_config.num_heads,
            model_config.d_head,
            model_config.d_ff,
            model_config.num_layers
        );
    }

    // --- 未見テイク評価モード(Phase 0a の合否判定) ---
    if let Some(dict_path) = &args.eval_holdout {
        // train から新規学習するモードなので、--load(既存モデル読み込み)は意味を持たない。
        // 黙って無視すると誤用に気づきにくいのでエラーにする
        if args.load.is_some() {
            return Err("--eval-holdout は train から新規学習するため --load とは併用できません".into());
        }
        println!("\n===== 未見テイク評価(手ポーズ列→タグ) =====");
        println!("pose dict: {}", dict_path.display());
        println!("holdout_per_label: {}", args.holdout_per_label);

        let split =
            pose_data::PoseTrainingData::load_holdout_split(dict_path, args.holdout_per_label)?;

        println!("\n--- テイク単位の分割 ---");
        for (label, tr, te) in &split.per_label {
            println!("  {:<12} train={} held-out={}", label, tr, te);
        }
        if !split.train_only.is_empty() {
            println!(
                "  注意: テイク不足で held-out を作れず train のみに入れたラベル: {:?}",
                split.train_only
            );
        }
        println!(
            "学習サンプル {} 件 / 未見テイク {} 件",
            split.train.len(),
            split.test.len()
        );

        // ミラー拡張は train 側のみに適用する。held-out(未見テイク)は実撮影データの
        // 分布のまま評価しないと「解けたことにする」測定になってしまうため対象外。
        //
        // 拡張は必ず「生の 126 次元のうちに」行う。手形記述子は左右反転に不変なので、
        // 記述子へ変換したあとで反転しても完全に同じベクトルが増えるだけで拡張にならない
        // (handshape_features::tests::mirror_invariance 参照)。そのため記述子を使うときは
        // ミラー拡張を無効化し、黙って効かないのではなく理由を表示する
        let mut train_data = split.train;
        if args.mirror_augment {
            if input_features.is_raw() {
                let before = train_data.len();
                train_data = train_data.with_mirror_augmentation();
                println!(
                    "ミラー拡張適用: 学習サンプル {} 件 → {} 件",
                    before,
                    train_data.len()
                );
            } else {
                println!(
                    "注意: --input-features {} は左右反転に不変な記述子のため、\
                     --mirror-augment は同一ベクトルの複製にしかならず適用しません",
                    input_features.name()
                );
            }
        }
        // 生ポーズ → 手形記述子への変換(raw ならクローンのみで内容は不変)
        let train_data = train_data.to_input_features(input_features);
        let test_data = split.test.to_input_features(input_features);

        // 語彙は学習データ(train)から構築。held-out のラベルは train の部分集合なので必ず含まれる
        let vocab = train_data.build_vocabulary();
        let model = recognition::RecognitionModel::<TrainingBackend>::new_with_config(
            vocab.vocab_size(),
            &model_config,
            &device,
        );
        println!("パラメータ数: {}", model.num_params());
        let (model, loss_history) = recognition_training::train_recognition(
            model,
            &train_data,
            &vocab,
            &device,
            args.epochs,
        );
        println!(
            "訓練完了: Loss {:.6} → {:.6}",
            loss_history.first().unwrap_or(&0.0),
            loss_history.last().unwrap_or(&0.0)
        );

        println!("\n===== 未見テイクでの評価(これが Phase 0a の合否判定) =====");
        recognition_training::evaluate_topk(&model, &test_data, &vocab, &device);

        if let Some(save_dir) = &args.save {
            recognition::save_recognition_model(&model, &vocab, &model_config, save_dir)?;
        }
        return Ok(());
    }

    // --- 訓練モード ---
    if let Some(dict_path) = &args.train_pose {
        println!("\n===== 認識モデル訓練(手ポーズ列→タグ) =====");
        println!("pose dict: {}", dict_path.display());

        if args.load.is_some() {
            // 継続訓練は語彙の一致確認が必要になるため未対応(将来課題)
            return Err("--train-pose と --load の併用(継続訓練)は未対応です".into());
        }

        let mut data = pose_data::PoseTrainingData::load(dict_path)?;
        if args.mirror_augment {
            if input_features.is_raw() {
                let before = data.len();
                data = data.with_mirror_augmentation();
                println!("ミラー拡張適用: {} 件 → {} 件", before, data.len());
            } else {
                // 理由は --eval-holdout 側のコメント参照(記述子は左右反転に不変)
                println!(
                    "注意: --input-features {} は左右反転に不変な記述子のため、\
                     --mirror-augment は同一ベクトルの複製にしかならず適用しません",
                    input_features.name()
                );
            }
        }
        // 生ポーズ → 手形記述子への変換(raw ならクローンのみで内容は不変)
        let data = data.to_input_features(input_features);
        let vocab = data.build_vocabulary();
        println!(
            "サンプル数: {} / タグ数: {} ({:?})",
            data.len(),
            vocab.tags.len(),
            vocab.tags
        );

        let model = recognition::RecognitionModel::<TrainingBackend>::new_with_config(
            vocab.vocab_size(),
            &model_config,
            &device,
        );
        println!("パラメータ数: {}", model.num_params());
        let (model, loss_history) =
            recognition_training::train_recognition(model, &data, &vocab, &device, args.epochs);

        println!(
            "訓練完了: Loss {:.6} → {:.6}",
            loss_history.first().unwrap_or(&0.0),
            loss_history.last().unwrap_or(&0.0)
        );

        // 学習データ自身での top-5 確認(ループが回っているかのスモーク)
        recognition_training::evaluate_topk(&model, &data, &vocab, &device);

        if let Some(save_dir) = &args.save {
            recognition::save_recognition_model(&model, &vocab, &model_config, save_dir)?;
        }
        return Ok(());
    }

    // --- 推論モード ---
    if let Some(dict_path) = &args.predict_pose {
        // サイズ系フラグとの併用チェックは run_recognition 冒頭で済ませている
        println!("\n===== 認識モデル推論(手ポーズ列→タグ) =====");
        let load_dir = args
            .load
            .as_ref()
            .ok_or("--predict-pose には --load <モデルディレクトリ> が必要です")?;

        // モデルを組む前に、そのモデルがどの入力表現で学習されたかを読む
        let saved_config = recognition::load_recognition_config(load_dir)?;
        let (model, vocab) =
            recognition::load_recognition_model::<TrainingBackend>(load_dir, &device)?;
        let data = pose_data::PoseTrainingData::load(dict_path)?
            .to_input_features(saved_config.input_features);
        if !saved_config.input_features.is_raw() {
            println!(
                "保存済みモデルの入力表現に合わせて pose dict を変換しました: {}",
                saved_config.input_features.name()
            );
        }
        recognition_training::evaluate_topk(&model, &data, &vocab, &device);
    }

    Ok(())
}
