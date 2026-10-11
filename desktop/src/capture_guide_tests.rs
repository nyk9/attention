//! 撮影ガイド(R0)が「何も書き込まない」ことを固定するテスト。
//!
//! このファイルは `capture_guide/` の外に置いてある。テストの準備で一時ディレクトリにファイルを作る
//! (= 書き込み API を使う)ので、`capture_guide/` の中に置くと、下のソース検査が自分自身を見つけてしまうため。
//!
//! 1. ソースの検査: `capture_guide/` の全ソースに、ファイルの書き込み・作成・削除・名前変更・権限変更の
//!    API の文字列が無いこと。`fs::` の使い方は読み取り系(read_to_string / read_dir / metadata)だけに限る
//! 2. 実際の動作: 一時ディレクトリを `ATTENTION_DATA_DIR` 相当にして進捗を読み込み、前後でディレクトリの中身
//!    (ファイル名・大きさ・更新時刻・内容)が変わらないこと。words.tsv が無い場合も、作らないこと

use crate::capture_guide::progress::{load_progress, next_word_text};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// 書き込み系の API(文字列そのものがこのファイルに現れても検査対象外なので、そのまま書いてよい)
const FORBIDDEN: &[&str] = &[
    "fs::write",
    "File::create",
    "File::options",
    "OpenOptions",
    "create_dir",
    "remove_file",
    "remove_dir",
    "fs::rename",
    "fs::copy",
    "hard_link",
    "symlink",
    "set_permissions",
    "set_len",
    "write_all",
    "Command::new",
    "process::Command",
];

/// `fs::` の後に続いてよい名前(読み取りだけ)
const ALLOWED_FS: &[&str] = &["read_to_string", "read_dir", "metadata"];

fn guide_sources() -> Vec<(PathBuf, String)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/capture_guide");
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            let text = std::fs::read_to_string(&p).unwrap();
            out.push((p, text));
        }
    }
    out.sort();
    out
}

#[test]
fn capture_guide_sources_have_no_write_apis() {
    let sources = guide_sources();
    // mod.rs / edge.rs / hand_run.rs / progress.rs / silhouette.rs
    assert!(sources.len() >= 5, "検査対象が少なすぎる: {:?}", sources.iter().map(|s| &s.0).collect::<Vec<_>>());
    for (path, text) in &sources {
        for bad in FORBIDDEN {
            assert!(!text.contains(bad), "{} に書き込み系の API `{bad}` がある", path.display());
        }
        // fs:: の後ろは読み取り系の名前だけ
        for (i, _) in text.match_indices("fs::") {
            let rest = &text[i + 4..];
            let name: String = rest.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
            assert!(
                ALLOWED_FS.contains(&name.as_str()),
                "{} に読み取り以外の fs:: の使い方がある: fs::{name}",
                path.display()
            );
        }
    }
}

/// ディレクトリの中身のスナップショット(相対パス → (大きさ, 更新時刻, 内容))
fn snapshot(root: &Path) -> BTreeMap<String, (u64, std::time::SystemTime, Vec<u8>)> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, (u64, std::time::SystemTime, Vec<u8>)>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let e = e.unwrap();
            let p = e.path();
            let rel = p.strip_prefix(root).unwrap().display().to_string();
            let md = std::fs::metadata(&p).unwrap();
            if md.is_dir() {
                out.insert(format!("{rel}/"), (0, md.modified().unwrap(), Vec::new()));
                walk(root, &p, out);
            } else {
                out.insert(rel, (md.len(), md.modified().unwrap(), std::fs::read(&p).unwrap()));
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("attn_guide_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn loading_progress_does_not_change_the_data_dir() {
    let d = temp_dir("readonly");
    std::fs::write(
        d.join("words.tsv"),
        "word_id\tromaji\tlabel_ja\tstage\ttarget_takes\n001\tkonnichiwa\tこんにちは\t1\t10\n002\tohayou\tおはよう\t1\t10\n",
    )
    .unwrap();
    std::fs::write(
        d.join("index.tsv"),
        "file_path\tword_id\tword_romaji\tword_label_ja\trep_idx\trecorded_at\tduration_ms\tquality_flag\tnotes\n\
         001_konnichiwa/01.mp4\t001\tkonnichiwa\tこんにちは\t1\t\t4000\tng_hands\t\n\
         001_konnichiwa/02.mp4\t001\tkonnichiwa\tこんにちは\t2\t\t4000\tok\t\n",
    )
    .unwrap();
    std::fs::create_dir_all(d.join("001_konnichiwa")).unwrap();
    for f in ["01.mp4", "02.mp4", "03.mp4"] {
        std::fs::write(d.join("001_konnichiwa").join(f), b"not a real video").unwrap();
    }
    let before = snapshot(&d);
    // 何度読んでも(画面は 3 秒ごとに読み直す)中身は変わらない
    let mut text = String::new();
    for _ in 0..3 {
        let v = load_progress(&d).unwrap();
        text = next_word_text(&v);
    }
    // 01 は ng なので数えない。02(ok)と 03(index に無い = 有効)で 2 本 → 次は 3 本目
    assert_eq!(text, "次の語: こんにちは(001)、3 本目 / 目標 10");
    assert_eq!(snapshot(&d), before, "読み込みでデータディレクトリが変わった");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn missing_words_tsv_is_not_created() {
    // progress.rs の load_words は words.tsv が無いとテンプレートを書き出すが、撮影ガイドは書かない
    let d = temp_dir("nowords");
    let before = snapshot(&d);
    let err = load_progress(&d).unwrap_err();
    assert!(err.contains("words.tsv を読めません"), "{err}");
    assert!(err.contains("読み取り専用"), "{err}");
    assert!(!d.join("words.tsv").exists(), "words.tsv が作られた");
    assert!(!d.join("index.tsv").exists(), "index.tsv が作られた");
    assert_eq!(snapshot(&d), before);
    let _ = std::fs::remove_dir_all(&d);
}

/// 目視確認用(手動実行): エンジンの出力 JSONL(ATTENTION_APP_PREVIEW_JSONL)の preview のうち、
/// ATTENTION_APP_PREVIEW_NS(カンマ区切り、既定 "25")番目に撮影ガイドの重ね描きをして、
/// ATTENTION_APP_PREVIEW_OUT_DIR に guide_<N>.png として保存する。実映像が写るので、出力先は
/// スクラッチパッドなどリポジトリ外にすること(撮影ガイド本体ではなく、テスト専用の書き出し)
#[test]
#[ignore]
fn visual_check_guide_preview() {
    let (Ok(src), Ok(out_dir)) =
        (std::env::var("ATTENTION_APP_PREVIEW_JSONL"), std::env::var("ATTENTION_APP_PREVIEW_OUT_DIR"))
    else {
        return eprintln!("skip: 環境変数が無い");
    };
    let ns: Vec<usize> = std::env::var("ATTENTION_APP_PREVIEW_NS")
        .unwrap_or_else(|_| "25".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let mut k = 0;
    for line in std::fs::read_to_string(src).unwrap().lines() {
        if let Ok(Some(crate::protocol::EngineEvent::Preview { jpeg_b64, hands, crop, .. })) =
            crate::protocol::parse_event(line)
        {
            k += 1;
            if ns.contains(&k) {
                let (img, edge) = crate::capture_guide::compose_guide_preview(&jpeg_b64, &hands, crop).unwrap();
                let out = Path::new(&out_dir).join(format!("guide_{k}.png"));
                img.save(&out).unwrap();
                eprintln!("saved {} edge={:?}", out.display(), edge.message());
            }
        }
    }
}

#[test]
fn empty_words_tsv_is_reported() {
    let d = temp_dir("emptywords");
    std::fs::write(d.join("words.tsv"), "word_id\tromaji\tlabel_ja\tstage\ttarget_takes\n").unwrap();
    let err = load_progress(&d).unwrap_err();
    assert!(err.contains("words.tsv に語がありません"), "{err}");
    let _ = std::fs::remove_dir_all(&d);
}
