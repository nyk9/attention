use anyhow::Result;
use clap::{Parser, Subcommand};
use lan_handoff::{client, serve};

#[derive(Parser)]
#[command(
    name = "lan-handoff",
    about = "同一LAN内で学習データを Mac から Windows へ渡す一時配信サーバー / クライアント"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 配信サーバーを起動する(--yes を付けるまではドライラン)
    Serve(serve::ServeArgs),
    /// サーバーからファイルを取得して sha256 を検証する
    Fetch(client::FetchArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Serve(args) => serve::run(args, None).await,
        Cmd::Fetch(args) => client::fetch(args).await,
    }
}
