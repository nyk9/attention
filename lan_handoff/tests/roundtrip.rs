//! サーバーとクライアントを同一プロセスで起動し、配信→取得→検証→完了停止までと、
//! 外部から見た守り(平文拒否・指紋ピン留め・許可リスト・トークン・隠しファイル)を確認する。

use lan_handoff::{client, serve};
use std::net::Ipv4Addr;
use std::path::Path;
use std::time::Duration;
use tokio::sync::oneshot;

fn args(root: &Path, paths: &[&str], timeout_min: u64) -> serve::ServeArgs {
    serve::ServeArgs {
        root: Some(root.to_path_buf()),
        presets: vec![],
        paths: paths.iter().map(|s| s.to_string()).collect(),
        bind: Some(Ipv4Addr::LOCALHOST),
        port: 0,
        allow_clients: vec![],
        timeout_min,
        yes: true,
    }
}

fn make_repo(tmp: &Path) -> (std::path::PathBuf, Vec<u8>) {
    let root = tmp.join("repo");
    std::fs::create_dir_all(root.join("data/sub")).unwrap();
    std::fs::create_dir_all(root.join("private")).unwrap();
    let big: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
    std::fs::write(root.join("data/big.bin"), &big).unwrap();
    std::fs::write(root.join("data/sub/small 日本語.txt"), b"hello").unwrap();
    std::fs::write(root.join("data/.env"), b"SECRET=1").unwrap();
    std::fs::write(root.join("private/secret.txt"), b"secret").unwrap();
    (root, big)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn serve_fetch_roundtrip_and_isolation() {
    let tmp = tempfile::tempdir().unwrap();
    let (root, big) = make_repo(tmp.path());

    let (tx, rx) = oneshot::channel();
    let server = tokio::spawn(serve::run(args(&root, &["data"], 5), Some(tx)));
    let url = rx.await.unwrap();
    let (base, fp) = client::split_url(&url).unwrap();
    let http = client::pinned_client(&fp).unwrap();
    let status = |u: String| {
        let http = http.clone();
        async move { http.get(u).send().await.unwrap().status().as_u16() }
    };

    // 許可リスト外、隠しファイル、トラバーサル、トークン違い
    assert_eq!(status(format!("{base}/files/private/secret.txt")).await, 404);
    assert_eq!(status(format!("{base}/files/data/.env")).await, 404);
    assert_eq!(status(format!("{base}/files/..%2Fprivate%2Fsecret.txt")).await, 404);
    assert_eq!(status(format!("{base}/files/data%2F..%2Fprivate%2Fsecret.txt")).await, 404);
    let wrong = base.rsplit_once('/').unwrap().0.to_string() + "/deadbeef";
    assert_eq!(status(format!("{wrong}/manifest.sha256")).await, 404);
    assert_eq!(status(format!("{base}/manifest.sha256")).await, 200);

    // manifest に隠しファイルや許可リスト外が載らない
    let manifest = http
        .get(format!("{base}/manifest.sha256"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(manifest.contains("data/big.bin") && !manifest.contains(".env"));
    assert!(!manifest.contains("private"));

    // Range 取得(部分取得・再開に使える)
    let r = http
        .get(format!("{base}/files/data/big.bin"))
        .header("Range", "bytes=10-19")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status().as_u16(), 206);
    assert_eq!(r.bytes().await.unwrap().as_ref(), &big[10..20]);

    // done はトークンが無ければ効かない
    let r = http.post(format!("{wrong}/done")).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 404);

    // 指紋が違うクライアントは接続できない(なりすまし・盗聴対策)
    let bad = client::pinned_client(&"00".repeat(32)).unwrap();
    assert!(bad.get(format!("{base}/manifest.sha256")).send().await.is_err());

    // 平文 HTTP では何も返らない
    let plain = base.replacen("https://", "http://", 1);
    // 平文用のテスト用クライアントは rustls の既定プロバイダが必要(本番コードは使わない)
    let _ = rustls::crypto::ring::default_provider().install_default();
    let plain_client = reqwest::Client::builder().no_proxy().build().unwrap();
    let plain_res = plain_client
        .get(format!("{plain}/manifest.sha256"))
        .timeout(Duration::from_secs(5))
        .send()
        .await;
    assert!(plain_res.map_or(true, |r| !r.status().is_success()));

    // 取得: 内容が一致し、隠しファイルと許可リスト外は届かない
    let out = tmp.path().join("win");
    client::fetch(client::FetchArgs {
        url: url.clone(),
        dest: out.clone(),
        jobs: 4,
        only: vec![],
        list: false,
        overwrite: false,
        max_file_mib: 8192,
        no_done: false,
    })
    .await
    .unwrap();
    assert_eq!(std::fs::read(out.join("data/big.bin")).unwrap(), big);
    assert_eq!(std::fs::read(out.join("data/sub/small 日本語.txt")).unwrap(), b"hello");
    assert!(!out.join("private").exists());
    assert!(!out.join("data/.env").exists());

    // 完了通知でサーバーが止まる
    tokio::time::timeout(Duration::from_secs(10), server)
        .await
        .expect("サーバーが完了通知で停止しませんでした")
        .unwrap()
        .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fetch_list_writes_nothing_and_overwrite_needs_flag() {
    let tmp = tempfile::tempdir().unwrap();
    let (root, _) = make_repo(tmp.path());
    let (tx, rx) = oneshot::channel();
    let server = tokio::spawn(serve::run(args(&root, &["data/sub"], 5), Some(tx)));
    let url = rx.await.unwrap();

    let out = tmp.path().join("win");
    let base_args = client::FetchArgs {
        url: url.clone(),
        dest: out.clone(),
        jobs: 2,
        only: vec![],
        list: true,
        overwrite: false,
        max_file_mib: 8192,
        no_done: true,
    };
    client::fetch(base_args.clone()).await.unwrap();
    assert!(!out.exists(), "--list は何も書き込まない");

    // 既存ファイルの内容が違うと、--overwrite が無い限りエラー
    std::fs::create_dir_all(out.join("data/sub")).unwrap();
    std::fs::write(out.join("data/sub/small 日本語.txt"), b"DIFFERENT").unwrap();
    let mut a = base_args.clone();
    a.list = false;
    assert!(client::fetch(a.clone()).await.is_err());
    assert_eq!(std::fs::read(out.join("data/sub/small 日本語.txt")).unwrap(), b"DIFFERENT");
    a.overwrite = true;
    a.no_done = false;
    client::fetch(a).await.unwrap();
    assert_eq!(std::fs::read(out.join("data/sub/small 日本語.txt")).unwrap(), b"hello");

    tokio::time::timeout(Duration::from_secs(10), server).await.unwrap().unwrap().unwrap();
}

#[tokio::test]
async fn dry_run_does_not_listen_and_sensitive_needs_allow_client() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("repo");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("a.txt"), b"a").unwrap();
    std::fs::write(root.join("clip.mp4"), b"video").unwrap();

    let mut dry = args(&root, &["a.txt"], 1);
    dry.yes = false;
    tokio::time::timeout(Duration::from_secs(5), serve::run(dry, None))
        .await
        .unwrap()
        .unwrap();

    // 動画を含むのに --allow-client が無いと、--yes でも起動しない
    let err = serve::run(args(&root, &["clip.mp4"], 1), None).await.unwrap_err();
    assert!(format!("{err:#}").contains("--allow-client"));
}

#[tokio::test]
async fn bind_is_restricted_to_lan_or_loopback() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("repo");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("a.txt"), b"a").unwrap();
    for ip in [Ipv4Addr::UNSPECIFIED, Ipv4Addr::new(8, 8, 8, 8)] {
        let mut a = args(&root, &["a.txt"], 1);
        a.bind = Some(ip);
        a.yes = false;
        assert!(serve::run(a, None).await.is_err(), "{ip} への bind は拒否されるべき");
    }
    // 隠しファイルを --path で直接指定しても拒否される
    std::fs::write(root.join(".env"), b"x").unwrap();
    assert!(serve::run(args(&root, &[".env"], 1), None).await.is_err());
}
