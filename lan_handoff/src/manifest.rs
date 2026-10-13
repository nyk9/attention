//! 配信対象の列挙、sha256 計算、manifest.sha256 の生成・解析。
//!
//! 配信するのは「許可リストに載ったファイルの相対パス」だけで、サーバーはリクエストされた
//! パスをファイルシステムへそのまま渡さない(manifest のキーとの完全一致のみ)。
//! そのため `..` を使ったディレクトリ脱出は構造的に起こらない。

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// 配信対象に含めない OS 由来のファイル名。
const SKIP_NAMES: &[&str] = &[".DS_Store", "Thumbs.db"];

pub struct FileItem {
    pub rel: String,
    pub abs: PathBuf,
    pub size: u64,
}

pub struct Entry {
    pub rel: String,
    pub abs: PathBuf,
    pub size: u64,
    pub sha256: String,
}

/// manifest に載せてよい相対パスか。絶対パス・`..`・バックスラッシュ・`:` を拒否する。
/// Windows で別ドライブや ADS を指せないようにする意味もある。
pub fn safe_rel(rel: &str) -> bool {
    if rel.is_empty() || rel.starts_with('/') || rel.contains('\\') || rel.contains(':') {
        return false;
    }
    if rel.chars().any(|c| c.is_control()) {
        return false;
    }
    rel.split('/')
        .all(|c| !c.is_empty() && c != "." && c != "..")
}

/// Windows でも安全に扱える名前か。ドットで始まる要素(.git / .cargo / .env など)、
/// 予約名(CON, NUL, COM1 ...)、末尾のドット・空白、使えない文字を拒否する。
/// サーバー側(配信前)とクライアント側(書き込み前)の両方で使う。
pub fn portable_name(rel: &str) -> bool {
    rel.split('/').all(|c| {
        !c.starts_with('.')
            && !c.ends_with('.')
            && !c.ends_with(' ')
            && !c.chars().any(|ch| "<>\"|?*".contains(ch))
            && !win_reserved(c)
    })
}

fn win_reserved(c: &str) -> bool {
    let stem = c.split('.').next().unwrap_or("").trim_end().to_ascii_uppercase();
    matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$")
        || ((stem.starts_with("COM") || stem.starts_with("LPT"))
            && stem.chars().count() == 4
            && stem.chars().nth(3).map_or(false, |d| "123456789¹²³".contains(d)))
}

/// `root` 配下の `targets`(ファイルまたはディレクトリの相対パス)を列挙する。
/// シンボリックリンクは辿らず、root の外へ出るパスは拒否する。
pub fn collect(root: &Path, targets: &[String]) -> Result<Vec<FileItem>> {
    let root_c = root
        .canonicalize()
        .with_context(|| format!("root が開けません: {}", root.display()))?;
    let mut out: BTreeMap<String, FileItem> = BTreeMap::new();

    for target in targets {
        let t = target.trim_end_matches('/');
        if !safe_rel(t) || !portable_name(t) {
            bail!("安全でない/配信できないパスです(隠しファイル・予約名など): {target}");
        }
        let canon = root_c
            .join(t)
            .canonicalize()
            .with_context(|| format!("見つかりません: {t}"))?;
        if !canon.starts_with(&root_c) {
            bail!("root の外を指しています: {t}");
        }
        if canon.is_file() {
            add(&root_c, &canon, &mut out)?;
            continue;
        }
        for entry in WalkDir::new(&canon)
            .follow_links(false)
            .sort_by_file_name()
            .into_iter()
            // ドットで始まるファイル/ディレクトリ(.env, .git, ._* など)は配信しない
            .filter_entry(|e| e.depth() == 0 || !e.file_name().to_string_lossy().starts_with('.'))
        {
            let entry = entry?;
            if entry.file_type().is_symlink() {
                eprintln!("skip symlink: {}", entry.path().display());
                continue;
            }
            if !entry.file_type().is_file() {
                continue;
            }
            let name = entry.file_name().to_string_lossy();
            if SKIP_NAMES.contains(&name.as_ref()) {
                continue;
            }
            add(&root_c, entry.path(), &mut out)?;
        }
    }
    Ok(out.into_values().collect())
}

fn add(root_c: &Path, path: &Path, out: &mut BTreeMap<String, FileItem>) -> Result<()> {
    let rel = path.strip_prefix(root_c)?;
    let parts: Vec<&str> = rel
        .components()
        .map(|c| {
            c.as_os_str()
                .to_str()
                .with_context(|| format!("UTF-8 でないパス: {}", path.display()))
        })
        .collect::<Result<_>>()?;
    let rel_s = parts.join("/");
    if !safe_rel(&rel_s) {
        bail!("安全でないパスです: {rel_s}");
    }
    if !portable_name(&rel_s) {
        bail!("Windows で扱えない名前のため配信できません: {rel_s}");
    }
    let size = std::fs::metadata(path)?.len();
    out.insert(
        rel_s.clone(),
        FileItem {
            rel: rel_s,
            abs: path.to_path_buf(),
            size,
        },
    );
    Ok(())
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// 全ファイルの sha256 を並列に計算する(CPU コア数だけスレッドを使う)。
pub fn hash_all(items: Vec<FileItem>) -> Result<Vec<Entry>> {
    items
        .into_par_iter()
        .map(|i| {
            let sha256 = sha256_file(&i.abs)
                .with_context(|| format!("ハッシュ計算に失敗: {}", i.abs.display()))?;
            Ok(Entry {
                rel: i.rel,
                abs: i.abs,
                size: i.size,
                sha256,
            })
        })
        .collect()
}

/// `shasum -a 256` と同じ「sha256␣␣相対パス」形式。
pub fn render(entries: &[Entry]) -> String {
    entries
        .iter()
        .map(|e| format!("{}  {}\n", e.sha256, e.rel))
        .collect()
}

/// manifest を (sha256, 相対パス) の一覧へ解析する。不正な行があれば全体を拒否する。
pub fn parse(text: &str) -> Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let (sha, rel) = line
            .split_once("  ")
            .with_context(|| format!("manifest {} 行目の形式が不正です", i + 1))?;
        if sha.len() != 64 || !sha.chars().all(|c| c.is_ascii_hexdigit()) {
            bail!("manifest {} 行目の sha256 が不正です", i + 1);
        }
        if !safe_rel(rel) || !portable_name(rel) {
            bail!("manifest {} 行目のパスが安全ではありません: {rel}", i + 1);
        }
        out.push((sha.to_ascii_lowercase(), rel.to_string()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_rel_rejects_escapes() {
        assert!(safe_rel("a/b/c.mp4"));
        assert!(safe_rel("transformer_burn/data/raw_jsl/001_x/01.mp4"));
        for bad in ["", "/etc/passwd", "../x", "a/../b", "a//b", "a\\b", "C:/x", "a/./b", "a/"] {
            assert!(!safe_rel(bad), "{bad} は拒否されるべき");
        }
    }

    #[test]
    fn portable_name_rejects_windows_traps_and_hidden() {
        assert!(portable_name("transformer_burn/data/raw_jsl/001_x/01.mp4"));
        assert!(portable_name("a/Movie on 2026-08-01 at 21.06 #2.mov"));
        for bad in [
            ".git/hooks/post-checkout", "a/.cargo/config.toml", ".env", "sub/CON", "sub/con.txt",
            "NUL", "COM1", "lpt9.log", "COM¹", "trail.", "trail ", "a/b?.txt", "a/b*.txt", "a/b|c",
        ] {
            assert!(!portable_name(bad), "{bad} は拒否されるべき");
        }
        assert!(portable_name("COM10")); // COM10 は予約名ではない
    }

    #[test]
    fn parse_roundtrip_and_rejects_bad_lines() {
        let sha = "a".repeat(64);
        let text = format!("{sha}  dir/file.bin\n");
        let parsed = parse(&text).unwrap();
        assert_eq!(parsed, vec![(sha.clone(), "dir/file.bin".to_string())]);
        assert!(parse(&format!("{sha}  ../evil\n")).is_err());
        assert!(parse("zz  file\n").is_err());
        assert!(parse(&format!("{sha} file\n")).is_err());
    }

    #[test]
    fn collect_stays_inside_root_and_skips_junk() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("repo");
        std::fs::create_dir_all(root.join("d/sub")).unwrap();
        std::fs::write(root.join("d/a.txt"), b"a").unwrap();
        std::fs::write(root.join("d/sub/b.txt"), b"b").unwrap();
        std::fs::write(root.join("d/.DS_Store"), b"x").unwrap();
        std::fs::write(root.join("d/.env"), b"SECRET=1").unwrap();
        std::fs::create_dir_all(root.join("d/.git")).unwrap();
        std::fs::write(root.join("d/.git/config"), b"x").unwrap();
        std::fs::write(dir.path().join("outside.txt"), b"secret").unwrap();

        let got = collect(&root, &["d".to_string()]).unwrap();
        let rels: Vec<_> = got.iter().map(|i| i.rel.as_str()).collect();
        assert_eq!(rels, vec!["d/a.txt", "d/sub/b.txt"]);

        assert!(collect(&root, &["../outside.txt".to_string()]).is_err());
        assert!(collect(&root, &["missing".to_string()]).is_err());
    }
}
