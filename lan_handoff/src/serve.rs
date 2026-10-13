//! 配信サーバー。
//!
//! 守りの層(外側から順に):
//! 1. LAN の IPv4 アドレスにだけ bind する(0.0.0.0 や LAN 外のアドレスは拒否)
//! 2. accept 直後(TLS ハンドシェイクの前)にクライアント IP を判定する。
//!    既定は同一サブネット。顔動画を含むときは --allow-client(受け取り側 IP)を必須にする
//! 3. TLS 1.3(起動ごとの自己署名証明書)。証明書の sha256 指紋を URL の `#` 以降で受け渡し、
//!    クライアントはその指紋と一致するサーバーとしか通信しない(盗聴・なりすまし対策)
//! 4. 起動ごとのランダムトークンを URL パスに含める。不一致は 404(理由を区別しない)
//! 5. 最初にトークンを提示した受け取り側 IP に固定する(別端末は以後拒否)
//! 6. 許可リスト(manifest のキー)に完全一致するパスだけを返す。ドットファイルは配信しない。
//!    配信の直前にも「実体のパスが列挙時と同一(symlink に差し替えられていない)」ことを確認する
//! 7. 同時接続数の上限、TLS ハンドシェイク/ヘッダ受信のタイムアウト(Slowloris 対策)
//! 8. 一定時間で自動停止し、クライアントの完了通知(POST /<token>/done)でも停止する
//!
//! 実際の配信は tokio のマルチスレッドランタイム上で接続ごとに並列に処理される。

use crate::{manifest, net};
use anyhow::{bail, Context, Result};
use axum::{
    body::Body,
    http::{header, Method, Request, StatusCode},
    response::{IntoResponse, Response},
};
use clap::{Args, ValueEnum};
use hyper::{body::Incoming, server::conn::http1, service::service_fn};
use hyper_util::rt::{TokioIo, TokioTimer};
use percent_encoding::percent_decode_str;
use rand::RngCore;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::convert::Infallible;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{oneshot, Notify, Semaphore};
use tokio_rustls::TlsAcceptor;
use tower::ServiceExt;
use tower_http::services::ServeFile;

/// 同時に受け付ける接続数の上限。受け取り側の並列数(既定4、最大32)より余裕を持たせる。
const MAX_CONNECTIONS: usize = 32;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const HEADER_TIMEOUT: Duration = Duration::from_secs(10);

/// このリポジトリ用のよく使う配信セット(root からの相対パス)。
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Preset {
    /// 学習用の pose dict(51語×10テイク、full-frame)
    Dict,
    /// transformer_burn の Cargo.lock(Windows のビルド固定用)
    Lock,
    /// 手検出・手ランドマークの ONNX 2本
    Onnx,
    /// 撮影動画の全量(顔が写っている)
    RawJsl,
    /// 保存済み認識モデル rec_t10_full
    RecModel,
}

impl Preset {
    fn paths(self) -> &'static [&'static str] {
        match self {
            Preset::Dict => &["transformer_burn/data/pose_dict_t10_full.json"],
            Preset::Lock => &["transformer_burn/Cargo.lock"],
            Preset::Onnx => &[
                "pose_extractor/models/palm_detection_mediapipe.onnx",
                "pose_extractor/models/handpose_estimation_mediapipe.onnx",
            ],
            Preset::RawJsl => &["transformer_burn/data/raw_jsl"],
            Preset::RecModel => &["transformer_burn/models/rec_t10_full"],
        }
    }
}

#[derive(Args, Clone, Debug)]
pub struct ServeArgs {
    /// リポジトリのルート。省略時はカレントから上へたどって transformer_burn/ のある場所を探す
    #[arg(long)]
    pub root: Option<PathBuf>,
    /// 配信セット(複数指定可)
    #[arg(long = "preset", value_enum)]
    pub presets: Vec<Preset>,
    /// root からの相対パスを追加(ファイルまたはディレクトリ。複数指定可)
    #[arg(long = "path")]
    pub paths: Vec<String>,
    /// bind する IPv4。省略時は LAN のプライベート IPv4 を自動検出(候補が複数ならこの指定が必要)
    #[arg(long)]
    pub bind: Option<Ipv4Addr>,
    /// ポート。0 なら空きポートを自動選択
    #[arg(long, default_value_t = 0)]
    pub port: u16,
    /// 受け付けるクライアント IP(複数指定可)。指定すると同一サブネット許可は無効になる。
    /// 顔動画・画像を含むときは必須
    #[arg(long = "allow-client")]
    pub allow_clients: Vec<IpAddr>,
    /// 自動停止までの分数(1〜120)
    #[arg(long, default_value_t = 10)]
    pub timeout_min: u64,
    /// 実際に配信を始める。付けないと配信内容を表示するだけ(ドライラン)
    #[arg(long)]
    pub yes: bool,
}

struct AppState {
    token: String,
    manifest_text: String,
    files: HashMap<String, PathBuf>,
    root_c: PathBuf,
    bind_ip: Ipv4Addr,
    mask: Ipv4Addr,
    allow_clients: Vec<IpAddr>,
    /// 最初にトークンを提示した受け取り側 IP。以後はこの IP だけを受け付ける
    pinned: Mutex<Option<IpAddr>>,
    done: Notify,
    /// クライアント IP ごとの (ファイル応答数, 応答ヘッダ上のバイト数)
    stats: Mutex<BTreeMap<IpAddr, (u64, u64)>>,
}

impl AppState {
    fn client_allowed(&self, ip: IpAddr) -> bool {
        let ip = match ip {
            IpAddr::V6(v6) => v6.to_ipv4_mapped().map(IpAddr::V4).unwrap_or(ip),
            other => other,
        };
        // 同じ Mac 自身からの動作確認(接続元が bind した LAN IP になる)は常に許可する
        if ip == IpAddr::V4(self.bind_ip) {
            return true;
        }
        if !self.allow_clients.is_empty() {
            return self.allow_clients.contains(&ip);
        }
        match ip {
            IpAddr::V4(v4) => net::same_subnet(v4, self.bind_ip, self.mask),
            IpAddr::V6(_) => false,
        }
    }

    /// accept 直後の判定。受け取り側が固定された後は、固定された IP と自分自身以外を TLS 処理の前に落とす。
    fn accept_ok(&self, ip: IpAddr) -> bool {
        if !self.client_allowed(ip) {
            return false;
        }
        let ip4 = match ip {
            IpAddr::V6(v6) => v6.to_ipv4_mapped().map(IpAddr::V4).unwrap_or(ip),
            other => other,
        };
        if ip4 == IpAddr::V4(self.bind_ip) {
            return true;
        }
        match *self.pinned.lock().unwrap() {
            Some(p) => p == ip4,
            None => true,
        }
    }

    /// トークン提示後の受け取り側 IP 固定。このマシン自身(動作確認)は固定の対象にしない。
    fn check_pin(&self, ip: IpAddr) -> bool {
        if ip == IpAddr::V4(self.bind_ip) {
            return true;
        }
        let mut p = self.pinned.lock().unwrap();
        match *p {
            None => {
                *p = Some(ip);
                true
            }
            Some(x) => x == ip,
        }
    }
}

fn ct_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn not_found() -> Response<Body> {
    StatusCode::NOT_FOUND.into_response()
}

async fn handle(st: Arc<AppState>, peer: SocketAddr, req: Request<Body>) -> Response<Body> {
    let ip = peer.ip();
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    let trimmed = path.trim_start_matches('/');
    let (tok, rest) = trimmed.split_once('/').unwrap_or((trimmed, ""));
    if !ct_eq(tok, &st.token) {
        eprintln!("DENY  {ip} (トークン不一致)");
        return not_found();
    }
    if !st.check_pin(ip) {
        eprintln!("DENY  {ip} (別の受け取り側が先に接続済み)");
        return not_found();
    }

    match (&method, rest) {
        (&Method::GET, "manifest.sha256") | (&Method::HEAD, "manifest.sha256") => {
            eprintln!("GET   {ip} manifest.sha256");
            (
                [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                st.manifest_text.clone(),
            )
                .into_response()
        }
        (&Method::POST, "done") => {
            eprintln!("DONE  {ip} (完了通知。停止します)");
            st.done.notify_one();
            StatusCode::NO_CONTENT.into_response()
        }
        (&Method::GET, r) | (&Method::HEAD, r) if r.starts_with("files/") => {
            let Ok(rel) = percent_decode_str(&r["files/".len()..]).decode_utf8() else {
                return not_found();
            };
            let Some(abs) = st.files.get(rel.as_ref()) else {
                eprintln!("DENY  {ip} 許可リスト外: {rel}");
                return not_found();
            };
            // 列挙後にファイルが symlink などへ差し替えられていないことを配信の直前に確認する
            match tokio::fs::canonicalize(abs).await {
                Ok(c) if &c == abs && c.starts_with(&st.root_c) => {}
                _ => {
                    eprintln!("DENY  {ip} 実体が変わっています: {rel}");
                    return not_found();
                }
            }
            let resp = ServeFile::new(abs.clone())
                .oneshot(req)
                .await
                .unwrap_or_else(|never: Infallible| match never {});
            let status = resp.status();
            if status.is_success() && method == Method::GET {
                let len = resp
                    .headers()
                    .get(header::CONTENT_LENGTH)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(0);
                let mut stats = st.stats.lock().unwrap();
                let e = stats.entry(ip).or_insert((0, 0));
                e.0 += 1;
                e.1 += len;
            }
            eprintln!("{method} {ip} {rel} -> {}", status.as_u16());
            resp.map(Body::new)
        }
        _ => not_found(),
    }
}

fn resolve_root(arg: Option<PathBuf>) -> Result<PathBuf> {
    let root = match arg {
        Some(r) => r,
        None => {
            let cwd = std::env::current_dir()?;
            cwd.ancestors()
                .find(|d| d.join("transformer_burn").is_dir() && d.join("pose_extractor").is_dir())
                .map(|d| d.to_path_buf())
                .context("リポジトリのルートが見つかりません。--root で指定してください")?
        }
    };
    let canon = root
        .canonicalize()
        .with_context(|| format!("root が開けません: {}", root.display()))?;
    // 広すぎる root(ファイルシステムのルートやホームディレクトリ)は拒否する
    let home = ["HOME", "USERPROFILE"]
        .iter()
        .filter_map(|k| std::env::var_os(k))
        .filter_map(|h| PathBuf::from(h).canonicalize().ok())
        .any(|h| h == canon);
    if canon.parent().is_none() || home {
        bail!("root が広すぎます({})。リポジトリのルートを指定してください", canon.display());
    }
    Ok(canon)
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64)
}

/// 顔が写っている可能性があるファイル(動画・画像、または撮影データのディレクトリ)。
fn is_sensitive(rel: &str) -> bool {
    let lower = rel.to_ascii_lowercase();
    lower.contains("raw_jsl/")
        || [".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".png", ".jpg", ".jpeg"]
            .iter()
            .any(|e| lower.ends_with(e))
}

fn tls_config(bind_ip: Ipv4Addr) -> Result<(Arc<rustls::ServerConfig>, String)> {
    let ck = rcgen::generate_simple_self_signed(vec![bind_ip.to_string()])?;
    let cert_der = ck.cert.der().clone();
    let fingerprint = hex::encode(Sha256::digest(cert_der.as_ref()));
    let key = rustls::pki_types::PrivatePkcs8KeyDer::from(ck.key_pair.serialize_der());
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let cfg = rustls::ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], rustls::pki_types::PrivateKeyDer::Pkcs8(key))?;
    Ok((Arc::new(cfg), fingerprint))
}

/// 配信サーバーを実行する。`ready` が渡されたとき、起動直後に `https://ip:port/token#指紋` を送る(テスト用)。
pub async fn run(args: ServeArgs, ready: Option<oneshot::Sender<String>>) -> Result<()> {
    if !(1..=120).contains(&args.timeout_min) {
        bail!("--timeout-min は 1〜120 の範囲で指定してください");
    }
    let root = resolve_root(args.root.clone())?;

    let mut targets: Vec<String> = args
        .presets
        .iter()
        .flat_map(|p| p.paths().iter().map(|s| s.to_string()))
        .collect();
    targets.extend(args.paths.iter().cloned());
    targets.sort();
    targets.dedup();
    if targets.is_empty() {
        bail!("配信対象がありません。--preset か --path を指定してください");
    }

    let items = manifest::collect(&root, &targets)?;
    if items.is_empty() {
        bail!("配信対象のファイルが 0 件です");
    }
    let total: u64 = items.iter().map(|i| i.size).sum();
    let sensitive_count = items.iter().filter(|i| is_sensitive(&i.rel)).count();

    let candidates = net::lan_candidates();
    let bind_ip = match args.bind {
        Some(ip) => {
            if ip.is_unspecified()
                || !(ip.is_loopback() || candidates.iter().any(|c| c.ip == ip))
            {
                bail!("--bind は LAN のプライベート IPv4(またはテスト用の 127.0.0.1)だけ指定できます");
            }
            ip
        }
        None => match candidates.as_slice() {
            [] => bail!("LAN の IPv4 アドレスが見つかりません。--bind で指定してください"),
            [one] => one.ip,
            many => {
                let list: Vec<_> = many.iter().map(|c| format!("{}={}", c.name, c.ip)).collect();
                bail!("LAN の候補が複数あります({})。--bind で選んでください", list.join(", "));
            }
        },
    };
    let mask = net::mask_for(bind_ip);

    println!("== lan-handoff ==");
    println!("root:       {}", root.display());
    println!("files:      {}", items.len());
    println!("bytes:      {} ({})", total, gib(total));
    // ディレクトリ別の内訳(承認の判断材料)
    let mut groups: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for i in &items {
        let dir: Vec<&str> = i.rel.split('/').collect();
        let key = dir[..dir.len().saturating_sub(1).min(3)].join("/");
        let e = groups.entry(key).or_insert((0, 0));
        e.0 += 1;
        e.1 += i.size;
    }
    for (k, (n, b)) in groups.iter().take(20) {
        println!("  {:>5} files {:>10.1} MB  {}", n, *b as f64 / 1e6, if k.is_empty() { "(root)" } else { k });
    }
    if groups.len() > 20 {
        let (rn, rb) = groups.values().skip(20).fold((0usize, 0u64), |a, v| (a.0 + v.0, a.1 + v.1));
        println!("  ほか {} グループ: {} files {:.1} MB", groups.len() - 20, rn, rb as f64 / 1e6);
    }
    if sensitive_count > 0 {
        println!("SENSITIVE:  顔が写っている可能性のある動画・画像が {sensitive_count} 件含まれます");
    }
    println!("bind:       {bind_ip} (mask {mask})");
    if args.allow_clients.is_empty() {
        println!("clients:    同一サブネットのみ (+ このマシン自身)。LAN への配信では --allow-client が必須");
        if u32::from(mask).count_ones() < 24 {
            println!("WARNING:    サブネットが広いです({mask})。--allow-client の指定を強く勧めます");
        }
    } else {
        let list: Vec<_> = args.allow_clients.iter().map(|c| c.to_string()).collect();
        println!("clients:    {} のみ (+ このマシン自身)", list.join(", "));
    }
    println!("transport:  TLS 1.3(自己署名、指紋ピン留め)。最初に接続した受け取り側 IP に固定");
    println!("auto stop:  {} 分", args.timeout_min);

    // 内容の判定(拡張子)には漏れがありうるので、LAN に公開するときは内容に関係なく受け取り側IPの指定を必須にする
    let needs_allow = (sensitive_count > 0 || !bind_ip.is_loopback()) && args.allow_clients.is_empty();
    if needs_allow {
        println!("REQUIRED:   LAN へ配信するため --allow-client <受け取り側のIP> が必要です");
    }
    if !args.yes {
        println!();
        println!("ドライランです。配信は始めていません。始めるには --yes を付けて再実行してください。");
        return Ok(());
    }
    if needs_allow {
        bail!("--allow-client <受け取り側のIP> を指定してください(LAN への配信、または顔動画・画像を含む配信では必須)");
    }

    let t0 = std::time::Instant::now();
    let entries = tokio::task::spawn_blocking(move || manifest::hash_all(items)).await??;
    eprintln!("sha256 計算: {:.1} 秒", t0.elapsed().as_secs_f32());

    let mut token_bytes = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut token_bytes);
    let token = hex::encode(token_bytes);
    let (tls_cfg, fingerprint) = tls_config(bind_ip)?;
    let acceptor = TlsAcceptor::from(tls_cfg);

    let state = Arc::new(AppState {
        token: token.clone(),
        manifest_text: manifest::render(&entries),
        files: entries.iter().map(|e| (e.rel.clone(), e.abs.clone())).collect(),
        root_c: root,
        bind_ip,
        mask,
        allow_clients: args.allow_clients.clone(),
        pinned: Mutex::new(None),
        done: Notify::new(),
        stats: Mutex::new(BTreeMap::new()),
    });

    let listener = tokio::net::TcpListener::bind((bind_ip, args.port))
        .await
        .with_context(|| format!("{bind_ip}:{} に bind できません", args.port))?;
    let addr = listener.local_addr()?;
    let base = format!("https://{}:{}/{}#{}", bind_ip, addr.port(), token, fingerprint);

    println!();
    println!("LAN_HANDOFF_URL={base}");
    println!("LAN_HANDOFF_FILES={}", entries.len());
    println!("LAN_HANDOFF_BYTES={total}");
    println!("配信を開始しました。完了通知か {} 分で自動停止します。", args.timeout_min);
    if let Some(tx) = ready {
        let _ = tx.send(base.clone());
    }

    let (stop_tx, mut stop_rx) = oneshot::channel::<()>();
    let accept_state = state.clone();
    let conn_lifetime = Duration::from_secs(args.timeout_min * 60);
    let accept_loop = tokio::spawn(async move {
        let sem = Arc::new(Semaphore::new(MAX_CONNECTIONS));
        loop {
            let (sock, peer) = tokio::select! {
                biased;
                _ = &mut stop_rx => break,
                r = listener.accept() => match r {
                    Ok(x) => x,
                    Err(e) => {
                        eprintln!("accept エラー: {e}");
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        continue;
                    }
                },
            };
            // TLS ハンドシェイクの前に接続元を判定する(許可外には何も応答しない)
            if !accept_state.accept_ok(peer.ip()) {
                eprintln!("DENY  {} (許可されていない接続元)", peer.ip());
                continue;
            }
            let Ok(permit) = sem.clone().try_acquire_owned() else {
                eprintln!("DENY  {} (同時接続数の上限)", peer.ip());
                continue;
            };
            let st = accept_state.clone();
            let acceptor = acceptor.clone();
            tokio::spawn(async move {
                let _permit = permit;
                let tls = match tokio::time::timeout(HANDSHAKE_TIMEOUT, acceptor.accept(sock)).await {
                    Ok(Ok(s)) => s,
                    _ => return,
                };
                let svc = service_fn(move |req: Request<Incoming>| {
                    let st = st.clone();
                    async move { Ok::<_, Infallible>(handle(st, peer, req.map(Body::new)).await) }
                });
                // 受信を止めて接続枠を占有されないよう、1接続の寿命にも上限を設ける
                let serving = http1::Builder::new()
                    .timer(TokioTimer::new())
                    .header_read_timeout(HEADER_TIMEOUT)
                    .serve_connection(TokioIo::new(tls), svc);
                let _ = tokio::time::timeout(conn_lifetime, serving).await;
            });
        }
    });

    tokio::select! {
        _ = state.done.notified() => {
            // 完了通知への応答を返し切ってから止める
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        _ = tokio::time::sleep(Duration::from_secs(args.timeout_min * 60)) => {
            eprintln!("タイムアウト({} 分)。停止します。", args.timeout_min);
        }
        _ = tokio::signal::ctrl_c() => {
            eprintln!("中断されました。停止します。");
        }
    }
    let _ = stop_tx.send(());
    let _ = tokio::time::timeout(Duration::from_secs(5), accept_loop).await;

    eprintln!("停止しました。受け取り側の集計(応答ヘッダ基準。途中で切れた転送も含む):");
    for (ip, (files, bytes)) in state.stats.lock().unwrap().iter() {
        eprintln!("  {ip}: {files} ファイル / {:.1} MB", *bytes as f64 / 1e6);
    }
    if let Some(p) = *state.pinned.lock().unwrap() {
        eprintln!("  固定された受け取り側: {p}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(allow: Vec<IpAddr>) -> AppState {
        AppState {
            token: "t".repeat(32),
            manifest_text: String::new(),
            files: HashMap::new(),
            root_c: PathBuf::from("/"),
            bind_ip: Ipv4Addr::new(192, 168, 11, 3),
            mask: Ipv4Addr::new(255, 255, 255, 0),
            allow_clients: allow,
            pinned: Mutex::new(None),
            done: Notify::new(),
            stats: Mutex::new(BTreeMap::new()),
        }
    }

    #[test]
    fn client_policy() {
        let s = state(vec![]);
        assert!(s.client_allowed("192.168.11.40".parse().unwrap()));
        assert!(s.client_allowed("192.168.11.3".parse().unwrap())); // 自分自身
        assert!(!s.client_allowed("192.168.12.40".parse().unwrap()));
        assert!(!s.client_allowed("10.0.0.5".parse().unwrap()));
        assert!(!s.client_allowed("::1".parse().unwrap()));

        let s = state(vec!["192.168.11.40".parse().unwrap()]);
        assert!(s.client_allowed("192.168.11.40".parse().unwrap()));
        assert!(!s.client_allowed("192.168.11.41".parse().unwrap())); // 同一サブネットでも指定外は拒否
    }

    #[test]
    fn first_client_is_pinned_but_host_is_not() {
        let s = state(vec![]);
        let host: IpAddr = "192.168.11.3".parse().unwrap();
        let a: IpAddr = "192.168.11.40".parse().unwrap();
        let b: IpAddr = "192.168.11.41".parse().unwrap();
        assert!(s.check_pin(host)); // 動作確認は固定の対象外
        assert!(s.check_pin(a));
        assert!(s.check_pin(a));
        assert!(!s.check_pin(b)); // 後から来た別端末は拒否
        assert!(s.check_pin(host));
    }

    #[test]
    fn after_pinning_only_the_pinned_client_reaches_tls() {
        let s = state(vec![]);
        let a: IpAddr = "192.168.11.40".parse().unwrap();
        let b: IpAddr = "192.168.11.41".parse().unwrap();
        assert!(s.accept_ok(a) && s.accept_ok(b)); // 固定前は同一サブネットを受け付ける
        assert!(s.check_pin(a));
        assert!(s.accept_ok(a));
        assert!(!s.accept_ok(b)); // 固定後は TLS 処理の前に落とす
        assert!(s.accept_ok("192.168.11.3".parse().unwrap()));
    }

    #[test]
    fn sensitive_detection_is_by_file_not_by_preset() {
        assert!(is_sensitive("transformer_burn/data/raw_jsl/index.tsv"));
        assert!(is_sensitive("anything/clip.MP4"));
        assert!(is_sensitive("a/frame.png"));
        assert!(!is_sensitive("transformer_burn/data/pose_dict_t10_full.json"));
        assert!(!is_sensitive("pose_extractor/models/palm_detection_mediapipe.onnx"));
    }
}
