//! 受け取り側(Windows でも Mac でも動く)のクライアント。
//!
//! サーバーが表示した URL(`https://IP:PORT/TOKEN#証明書の指紋`)の指紋と一致する
//! サーバーとしか通信しない。manifest を取得し、各ファイルを並列にダウンロードして
//! sha256 を検証する。すでに同じ内容のファイルがあればスキップするので、途中で止まっても
//! 再実行で続きから取れる。
//!
//! 悪意のあるサーバー/manifest を想定した防御:
//! - 書き込み先は dest 配下に限る(.. や絶対パス、symlink 経由の脱出を拒否)
//! - ドットで始まる名前(.git/.cargo など)、Windows の予約名、実行物・ビルド設定になりうる拡張子を拒否
//! - 既存ファイルと内容が違う場合は --overwrite が無い限り上書きしない
//! - プロキシ(環境変数)とリダイレクトを無効化(トークンとデータが第三者へ流れない)
//! - manifest・ファイルのサイズに上限

use crate::manifest;
use anyhow::{anyhow, bail, Context, Result};
use clap::Args;
use futures_util::{stream, StreamExt};
use percent_encoding::{utf8_percent_encode, NON_ALPHANUMERIC};
use reqwest::Client;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{verify_tls12_signature, verify_tls13_signature, CryptoProvider};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;

const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
const MAX_MANIFEST_ENTRIES: usize = 100_000;
/// 書き込んでよい拡張子(データ・モデル・動画のみ)。実行物やビルド設定になりうるものは許可しない。
const ALLOWED_EXTENSIONS: &[&str] = &[
    "mp4", "mov", "m4v", "json", "tsv", "csv", "txt", "md", "onnx", "bin", "mpk", "lock", "png", "jpg", "jpeg",
];

#[derive(Args, Clone, Debug)]
pub struct FetchArgs {
    /// サーバーが表示した URL(https://IP:PORT/TOKEN#指紋)。`#` 以降も含めて指定する
    #[arg(long)]
    pub url: String,
    /// 保存先(配信側の root と同じ相対パス構造で書き出す。Windows ではリポジトリのルートを指定する)
    #[arg(long)]
    pub dest: PathBuf,
    /// 並列ダウンロード数(1〜32)
    #[arg(long, default_value_t = 4)]
    pub jobs: usize,
    /// 指定したプレフィックスで始まる相対パスだけ取得(複数指定可)
    #[arg(long = "only")]
    pub only: Vec<String>,
    /// 取得せず、取得予定の一覧だけを表示する(書き込みは一切しない)
    #[arg(long)]
    pub list: bool,
    /// 既存ファイルの内容が違うとき上書きする(付けないとエラーにする)
    #[arg(long)]
    pub overwrite: bool,
    /// 1ファイルあたりの上限サイズ(MiB)
    #[arg(long, default_value_t = 8192)]
    pub max_file_mib: u64,
    /// 完了してもサーバーへ完了通知を送らない(サーバーは時間切れまで残る)
    #[arg(long)]
    pub no_done: bool,
}

/// URL を (トークン入りのベース URL, 証明書の指紋) に分ける。https と指紋が無ければ拒否する。
pub fn split_url(url: &str) -> Result<(String, String)> {
    let (base, fp) = url
        .split_once('#')
        .context("URL に証明書の指紋(#以降)がありません。サーバーが表示した URL を丸ごと指定してください")?;
    if !base.starts_with("https://") {
        bail!("https:// の URL だけ使えます(平文 HTTP は使いません)");
    }
    let fp = fp.trim().to_ascii_lowercase();
    if fp.len() != 64 || !fp.chars().all(|c| c.is_ascii_hexdigit()) {
        bail!("証明書の指紋の形式が不正です(sha256 の 64 桁の16進数)");
    }
    Ok((base.trim_end_matches('/').to_string(), fp))
}

/// 証明書の sha256 指紋が一致するサーバーだけを信頼する検証器。
#[derive(Debug)]
struct PinVerifier {
    fingerprint: String,
    provider: Arc<CryptoProvider>,
}

impl ServerCertVerifier for PinVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        if hex::encode(Sha256::digest(end_entity.as_ref())) == self.fingerprint {
            Ok(ServerCertVerified::assertion())
        } else {
            Err(rustls::Error::General(
                "サーバー証明書の指紋が URL の指紋と一致しません(別のサーバー、または盗聴の可能性)".into(),
            ))
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(message, cert, dss, &self.provider.signature_verification_algorithms)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(message, cert, dss, &self.provider.signature_verification_algorithms)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider.signature_verification_algorithms.supported_schemes()
    }
}

/// 指紋ピン留め TLS・プロキシ無効・リダイレクト無効の HTTP クライアント。
pub fn pinned_client(fingerprint: &str) -> Result<Client> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let cfg = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(PinVerifier {
            fingerprint: fingerprint.to_string(),
            provider,
        }))
        .with_no_client_auth();
    Ok(Client::builder()
        .use_preconfigured_tls(cfg)
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(10))
        .read_timeout(Duration::from_secs(60))
        .build()?)
}

/// reqwest のエラーは URL(=トークン)を含むので、表示前に URL を落とす。
fn nourl(e: reqwest::Error) -> anyhow::Error {
    anyhow!(e.without_url())
}

fn local_path(dest: &Path, rel: &str) -> PathBuf {
    rel.split('/').fold(dest.to_path_buf(), |p, c| p.join(c))
}

fn encode_rel(rel: &str) -> String {
    rel.split('/')
        .map(|seg| utf8_percent_encode(seg, NON_ALPHANUMERIC).to_string())
        .collect::<Vec<_>>()
        .join("/")
}

/// 書き込んでよい名前か(manifest の検査に加えて、拡張子の許可リストと 8.3 短縮名の拒否を適用)。
pub fn client_safe(rel: &str) -> bool {
    if !manifest::safe_rel(rel) || !manifest::portable_name(rel) || rel.contains('~') {
        return false;
    }
    let name = rel.rsplit('/').next().unwrap_or(rel);
    match name.rsplit_once('.') {
        Some((_, ext)) => ALLOWED_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()),
        None => false,
    }
}

enum Outcome {
    Downloaded(u64),
    Skipped,
}

async fn already_ok(path: &Path, sha: &str) -> bool {
    if !path.is_file() {
        return false;
    }
    let p = path.to_path_buf();
    let want = sha.to_string();
    tokio::task::spawn_blocking(move || manifest::sha256_file(&p).map(|h| h == want).unwrap_or(false))
        .await
        .unwrap_or(false)
}

/// 親ディレクトリが dest の内側にあり、書き込み先自体が symlink でないことを確かめる。
async fn ensure_inside(dest_c: &Path, path: &Path) -> Result<()> {
    let parent = path.parent().context("親ディレクトリがありません")?;
    tokio::fs::create_dir_all(parent).await?;
    let parent_c = tokio::fs::canonicalize(parent).await?;
    if !parent_c.starts_with(dest_c) {
        bail!("保存先が dest の外を指しています(symlink?): {}", path.display());
    }
    if let Ok(md) = tokio::fs::symlink_metadata(path).await {
        if md.file_type().is_symlink() {
            bail!("保存先が symlink です: {}", path.display());
        }
    }
    Ok(())
}

async fn download_once(
    client: &Client,
    url: &str,
    sha: &str,
    path: &Path,
    dest_c: &Path,
    max_bytes: u64,
) -> Result<u64> {
    ensure_inside(dest_c, path).await?;
    let mut part = path.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    let _ = tokio::fs::remove_file(&part).await;

    let resp = client.get(url).send().await.map_err(nourl)?.error_for_status().map_err(nourl)?;
    if resp.content_length().map_or(false, |l| l > max_bytes) {
        bail!("ファイルが上限({} MiB)を超えています", max_bytes >> 20);
    }
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&part)
        .await?;
    let mut hasher = Sha256::new();
    let mut total = 0u64;
    let mut body = resp.bytes_stream();
    let result: Result<()> = async {
        while let Some(chunk) = body.next().await {
            let chunk = chunk.map_err(nourl)?;
            total += chunk.len() as u64;
            if total > max_bytes {
                bail!("ファイルが上限({} MiB)を超えました", max_bytes >> 20);
            }
            hasher.update(&chunk);
            file.write_all(&chunk).await?;
        }
        file.flush().await?;
        Ok(())
    }
    .await;
    drop(file);
    if let Err(e) = result {
        let _ = tokio::fs::remove_file(&part).await;
        return Err(e);
    }

    let got = hex::encode(hasher.finalize());
    if got != sha {
        let _ = tokio::fs::remove_file(&part).await;
        bail!("sha256 が一致しません(期待 {sha} / 実際 {got})");
    }
    tokio::fs::rename(&part, path).await?;
    Ok(total)
}

async fn fetch_one(
    client: &Client,
    base: &str,
    dest: &Path,
    dest_c: &Path,
    sha: &str,
    rel: &str,
    overwrite: bool,
    max_bytes: u64,
) -> Result<Outcome> {
    let path = local_path(dest, rel);
    if already_ok(&path, sha).await {
        return Ok(Outcome::Skipped);
    }
    if path.exists() && !overwrite {
        bail!("既存ファイルと内容が違います(上書きするなら --overwrite): {rel}");
    }
    let url = format!("{}/files/{}", base, encode_rel(rel));
    let mut last_err = None;
    for attempt in 1..=3 {
        match download_once(client, &url, sha, &path, dest_c, max_bytes).await {
            Ok(n) => return Ok(Outcome::Downloaded(n)),
            Err(e) => {
                last_err = Some(e);
                if attempt < 3 {
                    tokio::time::sleep(Duration::from_millis(500 * attempt)).await;
                }
            }
        }
    }
    Err(last_err.unwrap())
}

pub async fn fetch(args: FetchArgs) -> Result<()> {
    let (base, fingerprint) = split_url(&args.url)?;
    let client = pinned_client(&fingerprint)?;

    let resp = client
        .get(format!("{base}/manifest.sha256"))
        .send()
        .await
        .map_err(nourl)
        .context("サーバーに接続できません(URL・指紋・ネットワーク・ファイアウォールを確認)")?
        .error_for_status()
        .map_err(nourl)
        .context("manifest の取得に失敗(トークン不一致、または接続元IPが許可されていない可能性)")?;
    if resp.content_length().map_or(false, |l| l as usize > MAX_MANIFEST_BYTES) {
        bail!("manifest が大きすぎます");
    }
    // 受信しながら上限を確認する(Content-Length の無い chunked 転送でメモリを使い切られないように)
    let mut buf: Vec<u8> = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(nourl)?;
        if buf.len() + chunk.len() > MAX_MANIFEST_BYTES {
            bail!("manifest が大きすぎます");
        }
        buf.extend_from_slice(&chunk);
    }
    let text = String::from_utf8(buf).context("manifest が UTF-8 ではありません")?;
    let mut list = manifest::parse(&text)?;
    if list.len() > MAX_MANIFEST_ENTRIES {
        bail!("manifest のファイル数が多すぎます");
    }
    if !args.only.is_empty() {
        list.retain(|(_, rel)| args.only.iter().any(|p| rel.starts_with(p.as_str())));
    }
    if list.is_empty() {
        bail!("取得対象が 0 件です");
    }
    let rejected: Vec<&str> = list
        .iter()
        .map(|(_, rel)| rel.as_str())
        .filter(|rel| !client_safe(rel))
        .collect();
    if !rejected.is_empty() {
        for r in rejected.iter().take(20) {
            eprintln!("拒否: {r}");
        }
        bail!(
            "安全ではない名前のファイルが {} 件あります(隠しファイル・予約名・許可されていない拡張子など)。取得を中止しました",
            rejected.len()
        );
    }

    if args.list {
        for (sha, rel) in &list {
            println!("{}  {}", &sha[..12], rel);
        }
        println!("{} ファイル(--list のため何も書き込んでいません)", list.len());
        return Ok(());
    }

    tokio::fs::create_dir_all(&args.dest).await?;
    let dest_c = tokio::fs::canonicalize(&args.dest).await?;
    let max_bytes = args.max_file_mib.saturating_mul(1 << 20);
    println!("対象 {} ファイル、保存先 {}、並列 {}", list.len(), args.dest.display(), args.jobs);

    let t0 = Instant::now();
    let jobs = args.jobs.clamp(1, 32);
    let total = list.len();
    let mut done = 0usize;
    let (mut downloaded, mut skipped, mut bytes_total) = (0usize, 0usize, 0u64);
    let mut failures: Vec<(String, String)> = Vec::new();

    let mut results = stream::iter(list.into_iter().map(|(sha, rel)| {
        let client = client.clone();
        let base = base.clone();
        let dest = args.dest.clone();
        let dest_c = dest_c.clone();
        let overwrite = args.overwrite;
        async move {
            let r = fetch_one(&client, &base, &dest, &dest_c, &sha, &rel, overwrite, max_bytes).await;
            (rel, r)
        }
    }))
    .buffer_unordered(jobs);

    while let Some((rel, r)) = results.next().await {
        done += 1;
        match r {
            Ok(Outcome::Downloaded(n)) => {
                downloaded += 1;
                bytes_total += n;
            }
            Ok(Outcome::Skipped) => skipped += 1,
            Err(e) => {
                eprintln!("FAIL {rel}: {e:#}");
                failures.push((rel, format!("{e:#}")));
            }
        }
        if done % 50 == 0 || done == total {
            println!("  {done}/{total}");
        }
    }

    let secs = t0.elapsed().as_secs_f64().max(0.001);
    println!(
        "完了: 取得 {downloaded} / スキップ(検証済み) {skipped} / 失敗 {} ({:.1} 秒、{:.1} MB、{:.1} MB/s)",
        failures.len(),
        secs,
        bytes_total as f64 / 1e6,
        bytes_total as f64 / 1e6 / secs
    );

    if !failures.is_empty() {
        bail!("{} ファイルの取得に失敗しました。再実行すると失敗分だけ取り直します", failures.len());
    }
    if !args.no_done {
        match client.post(format!("{base}/done")).send().await {
            Ok(resp) if resp.status().is_success() => println!("サーバーへ完了通知を送りました。"),
            Ok(resp) => eprintln!("完了通知が拒否されました: {}", resp.status()),
            Err(e) => eprintln!("完了通知を送れませんでした: {}", e.without_url()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_requires_https_and_fingerprint() {
        let fp = "ab".repeat(32);
        let (base, got) = split_url(&format!("https://192.168.0.2:9000/tok#{fp}")).unwrap();
        assert_eq!(base, "https://192.168.0.2:9000/tok");
        assert_eq!(got, fp);
        assert!(split_url("https://192.168.0.2:9000/tok").is_err());
        assert!(split_url(&format!("http://192.168.0.2:9000/tok#{fp}")).is_err());
        assert!(split_url("https://192.168.0.2:9000/tok#zz").is_err());
    }

    #[test]
    fn client_refuses_code_and_hidden_names() {
        assert!(client_safe("transformer_burn/data/raw_jsl/001_x/01.mp4"));
        assert!(client_safe("transformer_burn/data/pose_dict_t10_full.json"));
        assert!(client_safe("transformer_burn/Cargo.lock"));
        assert!(client_safe("pose_extractor/models/palm_detection_mediapipe.onnx"));
        assert!(client_safe("transformer_burn/models/rec_t10_full/model.bin"));
        for bad in [
            ".cargo/config.toml", "lan_handoff/build.rs", "transformer_burn/Cargo.toml",
            ".git/hooks/post-checkout", "a/run.sh", "a/x.EXE", "a/x.ps1", "sub/CON", "a/b.dll",
            "transformer_burn/CARGO~1.TOM", "rust-toolchain", "a/x.vbs", "a/x.hta", "a/x.jar",
        ] {
            assert!(!client_safe(bad), "{bad} は拒否されるべき");
        }
    }
}
