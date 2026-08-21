#![recursion_limit = "256"]

// wgpu(+autodiff)に依存するのは「学習・足場 Seq2Seq・WGPU推論」の経路だけ。
// これらは wgpu feature 有効時のみコンパイルする。pose_extractor は
// default-features = false で取り込み、下記の常時公開モジュール(CPU推論経路)
// だけを使うため、wgpu バックエンドのビルドを丸ごと回避できる。
#[cfg(feature = "wgpu")]
pub mod checkpoint;
pub mod config;
pub mod export;
pub mod handshape_features;
#[cfg(feature = "wgpu")]
pub mod inference;
pub mod jsl_data;
pub mod jsl_vocabulary;
pub mod metrics;
pub mod model;
pub mod pose_data;
pub mod recognition;
#[cfg(feature = "wgpu")]
pub mod recognition_training;
pub mod tag_vocabulary;
#[cfg(feature = "wgpu")]
pub mod training;
