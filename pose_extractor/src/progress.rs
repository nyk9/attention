//! 撮影進捗ツール(progress / session サブコマンド)
//!
//! - progress: data_dir をスキャンして words.tsv(語彙マスタ)と照合し、
//!   index.tsv を自動生成・更新して進捗表を表示する
//! - session: 録画フォルダ(OBS/QuickTime の保存先)を監視し、新しい録画を
//!   自動で命名・振り分け・記帳・手検出チェックまで行うガイド付きモード
//!
//! 設計方針: ファイルシステムが正、index.tsv は生成物。ただし
//! quality_flag / notes の2列だけは人が書く欄なので再スキャンでも保持する。

use anyhow::{Context, Result};
use ort::session::Session;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

/// words.tsv の1行 = 撮影対象の単語
#[derive(Debug, Clone)]
pub struct WordSpec {
    pub word_id: String, // "001" のようなゼロ埋め文字列
    pub romaji: String,
    pub label_ja: String,
    pub stage: u32,        // 撮影ステージ(1 = 先頭10語)
    pub target_takes: u32, // 目標テイク数
}

/// index.tsv の1行 = 撮影済みテイク1本
#[derive(Debug, Clone)]
pub struct IndexRow {
    pub file_path: String, // data_dir からの相対パス
    pub word_id: String,
    pub word_romaji: String,
    pub word_label_ja: String,
    pub rep_idx: u32,
    pub recorded_at: String,
    pub duration_ms: u64,
    pub quality_flag: String, // "" / "ok" / "ng_hands" / 手書きの任意値
    pub notes: String,        // 手書き専用
}

/// スキャンで見つかったテイクファイル
struct ScannedTake {
    rel_path: String,
    word_id: String,
    romaji: String,
    rep_idx: u32,
    abs: PathBuf,
}

const INDEX_FILE: &str = "index.tsv";
const WORDS_FILE: &str = "words.tsv";
const INDEX_HEADER: &str =
    "file_path\tword_id\tword_romaji\tword_label_ja\trep_idx\trecorded_at\tduration_ms\tquality_flag\tnotes";

/// 撮影プロトコルの目安(3-4秒/動画)から大きく外れたら警告する範囲
const DURATION_MIN_MS: u64 = 2000;
const DURATION_MAX_MS: u64 = 10000;

/// 手検出カバレッジがこの値未満なら ng_hands フラグ(撮り直し推奨)
const HAND_COVERAGE_THRESHOLD: f32 = 0.5;

/// Phase 0a の50語リスト(ろう者協力者レビュー前の暫定)。
/// stage 1 = 先頭10語(挨拶8 + わたし/あなた)、target_takes は編集可
const WORDS_TEMPLATE: &str = "\
word_id\tromaji\tlabel_ja\tstage\ttarget_takes
001\tkonnichiwa\tこんにちは\t1\t3
002\tohayou\tおはよう\t1\t3
003\tkonbanwa\tこんばんは\t1\t3
004\tsayounara\tさようなら\t1\t3
005\tarigatou\tありがとう\t1\t3
006\tsumimasen\tすみません\t1\t3
007\totsukaresama\tお疲れさま\t1\t3
008\tonegaishimasu\tお願いします\t1\t3
009\twatashi\tわたし\t1\t3
010\tanata\tあなた\t1\t3
011\tkare-kanojo\tかれ・かのじょ\t2\t3
012\tkore\tこれ\t2\t3
013\tsore\tそれ\t2\t3
014\tare\tあれ\t2\t3
015\thai\tはい\t2\t3
016\tiie\tいいえ\t2\t3
017\twakaru\tわかる\t2\t3
018\twakaranai\tわからない\t2\t3
019\tok\tOK\t2\t3
020\tii\tいい\t2\t3
021\tdame\tダメ\t2\t3
022\tmaamaa\tまあまあ\t2\t3
023\ttaberu\t食べる\t2\t3
024\tnomu\t飲む\t2\t3
025\tiku\t行く\t2\t3
026\tkuru\t来る\t2\t3
027\tmiru\t見る\t2\t3
028\tkiku\t聞く\t2\t3
029\thanasu\t話す\t2\t3
030\tkaku\t書く\t2\t3
031\tyomu\t読む\t2\t3
032\tsuki\t好き\t2\t3
033\tkirai\t嫌い\t2\t3
034\tdekiru\tできる\t2\t3
035\tookii\t大きい\t2\t3
036\tchiisai\t小さい\t2\t3
037\tatsui\t暑い\t2\t3
038\tsamui\t寒い\t2\t3
039\tatarashii\t新しい\t2\t3
040\tfurui\t古い\t2\t3
041\tichi\t1\t2\t3
042\tni\t2\t2\t3
043\tsan\t3\t2\t3
044\tyon\t4\t2\t3
045\tgo\t5\t2\t3
046\troku\t6\t2\t3
047\tnana\t7\t2\t3
048\thachi\t8\t2\t3
049\tkyuu\t9\t2\t3
050\tjuu\t10\t2\t3
";

// ===== エントリポイント =====

/// progress サブコマンド: スキャン → index.tsv 更新 → 進捗表示
pub fn run_progress(data_dir: &Path, check_only: bool) -> Result<()> {
    let words = load_words(data_dir)?;
    let (scanned, mut warnings) = scan_takes(data_dir)?;
    let existing = load_index(&data_dir.join(INDEX_FILE))?;
    let rows = merge_index(&scanned, &existing, &words, &mut warnings);

    for w in &warnings {
        eprintln!("警告: {}", w);
    }

    print_progress(&words, &rows);

    if check_only {
        println!("(--check のため index.tsv は更新していません)");
    } else {
        write_index(&data_dir.join(INDEX_FILE), &rows)?;
        println!("index.tsv を更新: {} 行", rows.len());
    }
    Ok(())
}

/// session サブコマンド: watch_dir を監視して自動取り込み
pub fn run_session(data_dir: &Path, watch_dir: &Path, no_hand_check: bool) -> Result<()> {
    if !watch_dir.is_dir() {
        anyhow::bail!("監視フォルダがありません: {}", watch_dir.display());
    }
    if watch_dir.canonicalize().ok() == data_dir.canonicalize().ok() {
        anyhow::bail!("--watch と --data-dir に同じフォルダは指定できません");
    }

    let words = load_words(data_dir)?;

    // 開始時に index を実ファイルと同期しておく
    let (scanned, mut warnings) = scan_takes(data_dir)?;
    let existing = load_index(&data_dir.join(INDEX_FILE))?;
    let mut rows = merge_index(&scanned, &existing, &words, &mut warnings);
    for w in &warnings {
        eprintln!("警告: {}", w);
    }
    write_index(&data_dir.join(INDEX_FILE), &rows)?;

    // 手検出チェッカー(Palm + Hand ONNX)。--no-hand-check なら読み込まない
    let mut checker = if no_hand_check {
        None
    } else {
        println!("手検出モデルを読み込み中...");
        Some(HandChecker::new()?)
    };

    // stdin からのコマンドを別スレッドで受ける
    // (監視ループを止めずにキー入力を拾うため、チャネル越しに渡す)
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(l) => {
                    if tx.send(l.trim().to_string()).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    println!("\n=== 撮影セッション開始 ===");
    println!("監視フォルダ : {}", watch_dir.display());
    println!("取り込み先   : {}", data_dir.display());
    println!("コマンド: s+Enter=現在の単語をスキップ / p+Enter=進捗表示 / q+Enter=終了");
    println!("(録画を止めてファイルが書き終わると自動で取り込みます)\n");
    print_progress(&words, &rows);

    let mut skipped: HashSet<String> = HashSet::new();
    // 開始時点で監視フォルダにあるファイルは取り込み対象にしない
    // (過去の録画を誤って飲み込まないため)
    // value = (サイズ, サイズが変わらなかった連続回数, 取り込み済みか)
    let mut tracked: HashMap<PathBuf, (u64, u32, bool)> = HashMap::new();
    for f in list_watch_files(watch_dir)? {
        tracked.insert(f, (0, 0, true));
    }

    announce_next(&words, &rows, &skipped);
    let mut done_announced = false;

    loop {
        // --- コマンド処理 ---
        while let Ok(cmd) = rx.try_recv() {
            match cmd.as_str() {
                "q" => {
                    write_index(&data_dir.join(INDEX_FILE), &rows)?;
                    println!("セッション終了。index.tsv を保存しました");
                    return Ok(());
                }
                "s" => {
                    if let Some(w) = next_word(&words, &rows, &skipped) {
                        println!("スキップ: {} {}", w.word_id, w.label_ja);
                        skipped.insert(w.word_id.clone());
                        announce_next(&words, &rows, &skipped);
                    } else {
                        println!("スキップ対象がありません");
                    }
                }
                "p" => print_progress(&words, &rows),
                "" => {}
                other => println!("不明なコマンド: {} (s/p/q)", other),
            }
        }

        // --- 監視フォルダの新ファイル検出 ---
        for f in list_watch_files(watch_dir)? {
            let size = std::fs::metadata(&f).map(|m| m.len()).unwrap_or(0);
            let entry = tracked.entry(f.clone()).or_insert((0, 0, false));
            if entry.2 {
                continue; // 取り込み済み or 開始前から存在
            }
            // 録画中はファイルが伸び続けるので、サイズが3回(約1.5秒)
            // 変わらなくなってから「書き終わった」とみなす
            if size > 0 && size == entry.0 {
                entry.1 += 1;
            } else {
                entry.0 = size;
                entry.1 = 0;
            }
            if entry.1 >= 3 {
                entry.2 = true;
                match next_word(&words, &rows, &skipped) {
                    Some(word) => {
                        let word = word.clone();
                        println!("\n[検出] {}", f.display());
                        match ingest(&f, &word, data_dir, &mut rows, checker.as_mut()) {
                            Ok(()) => {
                                write_index(&data_dir.join(INDEX_FILE), &rows)?;
                                announce_next(&words, &rows, &skipped);
                            }
                            Err(e) => eprintln!("取り込み失敗: {:#}", e),
                        }
                    }
                    None => {
                        println!(
                            "全単語が目標数に達しているため取り込みません: {}",
                            f.display()
                        );
                    }
                }
            }
        }

        if next_word(&words, &rows, &skipped).is_none() && !done_announced {
            println!("\n目標のテイク数にすべて到達しました(スキップ分を除く)。q+Enter で終了してください");
            done_announced = true;
        }

        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}

// ===== words.tsv =====

/// words.tsv を読み込む。無ければ50語テンプレートを書き出してから読む
fn load_words(data_dir: &Path) -> Result<Vec<WordSpec>> {
    let path = data_dir.join(WORDS_FILE);
    if !path.exists() {
        std::fs::create_dir_all(data_dir)
            .with_context(|| format!("create dir {}", data_dir.display()))?;
        std::fs::write(&path, WORDS_TEMPLATE)
            .with_context(|| format!("write {}", path.display()))?;
        println!(
            "words.tsv が無かったため50語テンプレートを作成しました: {}",
            path.display()
        );
        println!("(romaji / stage / target_takes は自由に編集してください。語彙はろう者協力者レビュー前の暫定です)");
    }

    let content = std::fs::read_to_string(&path)?;
    let mut words = Vec::new();
    for (lineno, line) in content.lines().enumerate() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') || line.starts_with("word_id") {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 5 {
            eprintln!("警告: words.tsv {}行目の列数が不足、無視します", lineno + 1);
            continue;
        }
        words.push(WordSpec {
            word_id: cols[0].to_string(),
            romaji: cols[1].to_string(),
            label_ja: cols[2].to_string(),
            stage: cols[3].parse().unwrap_or(99),
            target_takes: cols[4].parse().unwrap_or(1),
        });
    }
    if words.is_empty() {
        anyhow::bail!("words.tsv に単語がありません: {}", path.display());
    }
    Ok(words)
}

// ===== スキャンと index.tsv =====

/// `<word_id>_<romaji>` 形式のディレクトリ名を分解する。
/// word_id は数字のみ。最初の '_' で分割する(romaji 側の '-' は許容)
// [TEST向き] 純粋な文字列処理。境界('_'なし・数字でない・空)
fn parse_take_dir(name: &str) -> Option<(String, String)> {
    let pos = name.find('_')?;
    let (id, rest) = name.split_at(pos);
    let romaji = &rest[1..];
    if id.is_empty() || romaji.is_empty() {
        return None;
    }
    if !id.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((id.to_string(), romaji.to_string()))
}

/// テイクファイルの stem("01" など)をテイク番号にする。数字のみ許可
// [TEST向き] "01"→1、"1"→1、非数字→None
fn parse_rep_stem(stem: &str) -> Option<u32> {
    if stem.is_empty() || !stem.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    stem.parse().ok()
}

/// data_dir を走査して実在するテイクファイルを集める
fn scan_takes(data_dir: &Path) -> Result<(Vec<ScannedTake>, Vec<String>)> {
    let mut takes = Vec::new();
    let mut warnings = Vec::new();

    if !data_dir.is_dir() {
        return Ok((takes, warnings)); // まだ何も撮っていない状態は正常
    }

    for entry in std::fs::read_dir(data_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue; // words.tsv / index.tsv など
        }
        let dir_name = entry.file_name().to_string_lossy().to_string();
        let Some((word_id, romaji)) = parse_take_dir(&dir_name) else {
            warnings.push(format!(
                "ディレクトリ名が <word_id>_<romaji> 形式ではありません: {}",
                dir_name
            ));
            continue;
        };

        for f in std::fs::read_dir(&path)? {
            let f = f?;
            let fp = f.path();
            if !fp.is_file() {
                continue;
            }
            let ext_ok = fp
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| crate::VIDEO_EXTS.contains(&s.to_lowercase().as_str()))
                .unwrap_or(false);
            if !ext_ok {
                continue;
            }
            let stem = fp
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let Some(rep_idx) = parse_rep_stem(&stem) else {
                warnings.push(format!(
                    "テイク番号がファイル名から読めません(数字のみ想定): {}/{}",
                    dir_name,
                    fp.file_name().unwrap_or_default().to_string_lossy()
                ));
                continue;
            };
            let rel_path = format!(
                "{}/{}",
                dir_name,
                fp.file_name().unwrap_or_default().to_string_lossy()
            );
            takes.push(ScannedTake {
                rel_path,
                word_id: word_id.clone(),
                romaji: romaji.clone(),
                rep_idx,
                abs: fp,
            });
        }
    }
    takes.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    Ok((takes, warnings))
}

/// 既存 index.tsv を読む(無ければ空)。quality_flag / notes の保持が目的
fn load_index(path: &Path) -> Result<Vec<IndexRow>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let content = std::fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for line in content.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 9 {
            eprintln!("警告: index.tsv の行を読み飛ばしました(列数不足): {}", line);
            continue;
        }
        rows.push(IndexRow {
            file_path: cols[0].to_string(),
            word_id: cols[1].to_string(),
            word_romaji: cols[2].to_string(),
            word_label_ja: cols[3].to_string(),
            rep_idx: cols[4].parse().unwrap_or(0),
            recorded_at: cols[5].to_string(),
            duration_ms: cols[6].parse().unwrap_or(0),
            quality_flag: cols[7].to_string(),
            notes: cols[8].to_string(),
        });
    }
    Ok(rows)
}

/// スキャン結果と既存 index をマージする。
/// - 既存行: そのまま保持(quality_flag / notes を消さない)
/// - 新規ファイル: ffprobe して行を作る
/// - 実体が消えたファイルの行: 落とす(警告を出す)
fn merge_index(
    scanned: &[ScannedTake],
    existing: &[IndexRow],
    words: &[WordSpec],
    warnings: &mut Vec<String>,
) -> Vec<IndexRow> {
    let existing_map: BTreeMap<&str, &IndexRow> = existing
        .iter()
        .map(|r| (r.file_path.as_str(), r))
        .collect();
    let scanned_paths: HashSet<&str> = scanned.iter().map(|t| t.rel_path.as_str()).collect();
    let word_map: HashMap<&str, &WordSpec> =
        words.iter().map(|w| (w.word_id.as_str(), w)).collect();

    for r in existing {
        if !scanned_paths.contains(r.file_path.as_str()) {
            warnings.push(format!(
                "index にあるが実体が見つかりません(削除済み?): {}",
                r.file_path
            ));
        }
    }

    let mut rows = Vec::new();
    for t in scanned {
        if let Some(old) = existing_map.get(t.rel_path.as_str()) {
            rows.push((*old).clone());
            continue;
        }

        // words.tsv との照合(typo検出)
        let label = match word_map.get(t.word_id.as_str()) {
            Some(w) => {
                if w.romaji != t.romaji {
                    warnings.push(format!(
                        "romaji が words.tsv と一致しません: {} (words.tsv では {})",
                        t.rel_path, w.romaji
                    ));
                }
                w.label_ja.clone()
            }
            None => {
                warnings.push(format!(
                    "words.tsv に無い word_id です(typo?): {}",
                    t.rel_path
                ));
                String::new()
            }
        };

        // 新規ファイル: 長さと撮影日時を取得
        let (duration_ms, recorded_at) = match crate::probe_video(&t.abs) {
            Ok(info) => {
                let mtime = std::fs::metadata(&t.abs)
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::now());
                let dt: chrono::DateTime<chrono::Local> = mtime.into();
                (
                    (info.duration * 1000.0) as u64,
                    dt.format("%Y-%m-%d %H:%M:%S").to_string(),
                )
            }
            Err(e) => {
                warnings.push(format!("ffprobe 失敗(壊れたファイル?): {} ({})", t.rel_path, e));
                (0, String::new())
            }
        };

        if duration_ms > 0 && !(DURATION_MIN_MS..=DURATION_MAX_MS).contains(&duration_ms) {
            warnings.push(format!(
                "長さがプロトコル目安(3-4秒)から外れています: {} ({:.1}秒)",
                t.rel_path,
                duration_ms as f64 / 1000.0
            ));
        }

        rows.push(IndexRow {
            file_path: t.rel_path.clone(),
            word_id: t.word_id.clone(),
            word_romaji: t.romaji.clone(),
            word_label_ja: label,
            rep_idx: t.rep_idx,
            recorded_at,
            duration_ms,
            quality_flag: String::new(),
            notes: String::new(),
        });
    }
    rows.sort_by(|a, b| a.file_path.cmp(&b.file_path));
    rows
}

/// TSV に書けない文字(タブ・改行)を空白に潰す
// [TEST向き] 純粋関数
fn sanitize_tsv(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}

fn write_index(path: &Path, rows: &[IndexRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = String::new();
    out.push_str(INDEX_HEADER);
    out.push('\n');
    for r in rows {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            sanitize_tsv(&r.file_path),
            sanitize_tsv(&r.word_id),
            sanitize_tsv(&r.word_romaji),
            sanitize_tsv(&r.word_label_ja),
            r.rep_idx,
            sanitize_tsv(&r.recorded_at),
            r.duration_ms,
            sanitize_tsv(&r.quality_flag),
            sanitize_tsv(&r.notes),
        ));
    }
    std::fs::write(path, out).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

// ===== 進捗集計 =====

/// word_id ごとの (有効テイク数, NGテイク数)。
/// quality_flag が "ng" で始まる行は有効数に数えない
fn take_counts(rows: &[IndexRow]) -> HashMap<String, (u32, u32)> {
    let mut counts: HashMap<String, (u32, u32)> = HashMap::new();
    for r in rows {
        let e = counts.entry(r.word_id.clone()).or_default();
        if r.quality_flag.starts_with("ng") {
            e.1 += 1;
        } else {
            e.0 += 1;
        }
    }
    counts
}

fn print_progress(words: &[WordSpec], rows: &[IndexRow]) {
    let counts = take_counts(rows);

    let mut stages: BTreeMap<u32, Vec<&WordSpec>> = BTreeMap::new();
    for w in words {
        stages.entry(w.stage).or_default().push(w);
    }

    println!("=== 撮影進捗 ===");
    let mut total_done_words = 0usize;
    let mut total_takes = 0u32;
    let mut total_target = 0u32;

    for (stage, ws) in &stages {
        let mut stage_takes = 0u32;
        let mut stage_target = 0u32;
        println!("--- stage {} ---", stage);
        for w in ws {
            let (ok, ng) = counts.get(&w.word_id).copied().unwrap_or((0, 0));
            let status = if ok >= w.target_takes {
                total_done_words += 1;
                "完了".to_string()
            } else if ok > 0 {
                format!("あと{}", w.target_takes - ok)
            } else {
                "未着手".to_string()
            };
            let ng_str = if ng > 0 {
                format!(" (NG {})", ng)
            } else {
                String::new()
            };
            println!(
                "  {} {:<14} {}\t{}/{} {}{}",
                w.word_id, w.romaji, w.label_ja, ok, w.target_takes, status, ng_str
            );
            stage_takes += ok.min(w.target_takes);
            stage_target += w.target_takes;
        }
        println!("  stage {} 小計: {}/{}", stage, stage_takes, stage_target);
        total_takes += stage_takes;
        total_target += stage_target;
    }

    let pct = if total_target > 0 {
        total_takes as f32 / total_target as f32 * 100.0
    } else {
        0.0
    };
    println!(
        "合計: {}/{} テイク ({:.0}%)  完了 {}語 / 全{}語",
        total_takes,
        total_target,
        pct,
        total_done_words,
        words.len()
    );
}

// ===== session: 次の単語と取り込み =====

/// 次に撮るべき単語: stage 順 → words.tsv の行順で、
/// 有効テイク数が目標未満かつスキップされていない最初の単語
fn next_word<'a>(
    words: &'a [WordSpec],
    rows: &[IndexRow],
    skipped: &HashSet<String>,
) -> Option<&'a WordSpec> {
    let counts = take_counts(rows);
    let mut sorted: Vec<&WordSpec> = words.iter().collect();
    sorted.sort_by_key(|w| w.stage); // 同 stage 内は words.tsv の行順(安定ソート)
    sorted.into_iter().find(|w| {
        if skipped.contains(&w.word_id) {
            return false;
        }
        let (ok, _) = counts.get(&w.word_id).copied().unwrap_or((0, 0));
        ok < w.target_takes
    })
}

fn announce_next(words: &[WordSpec], rows: &[IndexRow], skipped: &HashSet<String>) {
    match next_word(words, rows, skipped) {
        Some(w) => {
            let counts = take_counts(rows);
            let (ok, _) = counts.get(&w.word_id).copied().unwrap_or((0, 0));
            println!(
                "\n次: {} {} ({}) {}/{} → 録画してください",
                w.word_id, w.label_ja, w.romaji, ok, w.target_takes
            );
        }
        None => println!("\n(次に撮る単語はありません)"),
    }
}

/// 既存テイクの最大番号 + 1(欠番があっても衝突しない)
// [TEST向き] 純粋関数
fn next_rep_idx(rows: &[IndexRow], word_id: &str) -> u32 {
    rows.iter()
        .filter(|r| r.word_id == word_id)
        .map(|r| r.rep_idx)
        .max()
        .unwrap_or(0)
        + 1
}

/// 監視フォルダ内の動画ファイル一覧(隠しファイル除外)
fn list_watch_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let ext_ok = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| crate::VIDEO_EXTS.contains(&s.to_lowercase().as_str()))
            .unwrap_or(false);
        if ext_ok {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

/// rename を試し、別ボリューム間(クロスデバイス)なら copy + remove にフォールバック
fn move_file(src: &Path, dest: &Path) -> Result<()> {
    if std::fs::rename(src, dest).is_ok() {
        return Ok(());
    }
    std::fs::copy(src, dest)
        .with_context(|| format!("copy {} -> {}", src.display(), dest.display()))?;
    std::fs::remove_file(src).with_context(|| format!("remove {}", src.display()))?;
    Ok(())
}

/// 1ファイルを取り込む: 命名・移動(必要なら mp4 へコンテナ詰め替え)・
/// ffprobe・手検出チェック・index 行追加
fn ingest(
    src: &Path,
    word: &WordSpec,
    data_dir: &Path,
    rows: &mut Vec<IndexRow>,
    checker: Option<&mut HandChecker>,
) -> Result<()> {
    let rep = next_rep_idx(rows, &word.word_id);
    let dir_name = format!("{}_{}", word.word_id, word.romaji);
    let dest_dir = data_dir.join(&dir_name);
    std::fs::create_dir_all(&dest_dir)?;

    let src_ext = src
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    let mut dest_name = format!("{:02}.mp4", rep);
    let mut dest = dest_dir.join(&dest_name);

    if src_ext == "mp4" {
        move_file(src, &dest)?;
    } else {
        // mkv(OBS既定)等はコンテナだけ mp4 に詰め替える(-c copy なので再エンコードなし・高速)
        let status = Command::new("ffmpeg")
            .args([
                "-y",
                "-loglevel",
                "error",
                "-i",
                &src.to_string_lossy(),
                "-c",
                "copy",
                &dest.to_string_lossy(),
            ])
            .status();
        match status {
            Ok(s) if s.success() => {
                std::fs::remove_file(src).ok();
            }
            _ => {
                // 詰め替え失敗時は元の拡張子のまま移動(取り込み自体は続行)
                eprintln!("  mp4への詰め替えに失敗、元の形式のまま取り込みます");
                let _ = std::fs::remove_file(&dest); // 失敗時の中途半端なファイルを掃除
                dest_name = format!("{:02}.{}", rep, src_ext);
                dest = dest_dir.join(&dest_name);
                move_file(src, &dest)?;
            }
        }
    }

    let info = crate::probe_video(&dest)?;
    let duration_ms = (info.duration * 1000.0) as u64;
    println!(
        "  → {}/{} に取り込み ({:.1}秒)",
        dir_name,
        dest_name,
        info.duration
    );
    if !(DURATION_MIN_MS..=DURATION_MAX_MS).contains(&duration_ms) {
        println!(
            "  注意: 長さがプロトコル目安(3-4秒)から外れています ({:.1}秒)",
            info.duration
        );
    }

    let mut quality_flag = String::new();
    if let Some(c) = checker {
        match c.coverage(&dest) {
            Ok((left, right)) => {
                let best = left.max(right);
                quality_flag = if best >= HAND_COVERAGE_THRESHOLD {
                    "ok".to_string()
                } else {
                    "ng_hands".to_string()
                };
                println!(
                    "  手検出: L={:.0}% R={:.0}% → {}",
                    left * 100.0,
                    right * 100.0,
                    quality_flag
                );
                if quality_flag == "ng_hands" {
                    println!("  → 撮り直し推奨(手がほとんど検出できていません。このテイクは NG フラグ付きで記録し、目標数には数えません)");
                }
            }
            Err(e) => eprintln!("  手検出チェック失敗(スキップ): {:#}", e),
        }
    }

    let now: chrono::DateTime<chrono::Local> = SystemTime::now().into();
    rows.push(IndexRow {
        file_path: format!("{}/{}", dir_name, dest_name),
        word_id: word.word_id.clone(),
        word_romaji: word.romaji.clone(),
        word_label_ja: word.label_ja.clone(),
        rep_idx: rep,
        recorded_at: now.format("%Y-%m-%d %H:%M:%S").to_string(),
        duration_ms,
        quality_flag,
        notes: String::new(),
    });
    Ok(())
}

// ===== 手検出チェック(段階4) =====

/// 取り込んだテイクに Palm + Hand 検出をサンプルフレームだけ実行して
/// 左右の手のカバレッジを返す(全フレームは回さないので数秒で終わる)
struct HandChecker {
    palm: Session,
    hand: Session,
    anchors: Vec<[f32; 2]>,
}

impl HandChecker {
    fn new() -> Result<Self> {
        let palm = Session::builder()?
            .commit_from_file(crate::PALM_MODEL)
            .with_context(|| format!("failed to load palm model: {}", crate::PALM_MODEL))?;
        let hand = Session::builder()?
            .commit_from_file(crate::HAND_MODEL)
            .with_context(|| format!("failed to load hand model: {}", crate::HAND_MODEL))?;
        Ok(Self {
            palm,
            hand,
            anchors: crate::load_palm_anchors(),
        })
    }

    /// 毎秒3フレーム程度をサンプリングして (左手カバレッジ, 右手カバレッジ) を返す
    fn coverage(&mut self, video: &PathBuf) -> Result<(f32, f32)> {
        let info = crate::probe_video(video)?;
        let step = ((info.fps / 3.0).round() as usize).max(1);

        let Self {
            palm,
            hand,
            anchors,
        } = self;

        let mut sampled = 0u32;
        let mut left = 0u32;
        let mut right = 0u32;
        crate::extract_frames(video, info.width, info.height, |idx, frame| {
            if idx % step != 0 {
                return Ok(());
            }
            sampled += 1;
            let palms = crate::run_palm_detection(palm, &frame, anchors)?;
            let mut l = false;
            let mut r = false;
            for p in &palms {
                if let Some(h) = crate::run_hand_landmark(hand, &frame, p)? {
                    if h.handedness >= 0.5 {
                        r = true;
                    } else {
                        l = true;
                    }
                }
            }
            if l {
                left += 1;
            }
            if r {
                right += 1;
            }
            Ok(())
        })?;

        if sampled == 0 {
            anyhow::bail!("フレームが1枚も取れませんでした");
        }
        Ok((
            left as f32 / sampled as f32,
            right as f32 / sampled as f32,
        ))
    }
}

// ===== テスト =====

#[cfg(test)]
mod tests {
    use super::*;

    // [TEST向き] ディレクトリ名分解の境界ケース
    #[test]
    fn parse_take_dir_cases() {
        // 期待値の根拠: 最初の '_' で分割、左は数字のみ
        assert_eq!(
            parse_take_dir("001_arigatou"),
            Some(("001".to_string(), "arigatou".to_string()))
        );
        // romaji 側の '-' や '_' は許容(最初の '_' で割るため)
        assert_eq!(
            parse_take_dir("011_kare-kanojo"),
            Some(("011".to_string(), "kare-kanojo".to_string()))
        );
        assert_eq!(parse_take_dir("abc_xyz"), None); // id が数字でない
        assert_eq!(parse_take_dir("001"), None); // '_' が無い
        assert_eq!(parse_take_dir("001_"), None); // romaji が空
        assert_eq!(parse_take_dir("_arigatou"), None); // id が空
    }

    // [TEST向き] テイク番号パース
    #[test]
    fn parse_rep_stem_cases() {
        assert_eq!(parse_rep_stem("01"), Some(1));
        assert_eq!(parse_rep_stem("1"), Some(1));
        assert_eq!(parse_rep_stem("12"), Some(12));
        assert_eq!(parse_rep_stem("take1"), None);
        assert_eq!(parse_rep_stem(""), None);
    }

    // [TEST向き] 欠番があっても max+1 で衝突しない
    #[test]
    fn next_rep_idx_uses_max() {
        let row = |word_id: &str, rep: u32| IndexRow {
            file_path: String::new(),
            word_id: word_id.to_string(),
            word_romaji: String::new(),
            word_label_ja: String::new(),
            rep_idx: rep,
            recorded_at: String::new(),
            duration_ms: 0,
            quality_flag: String::new(),
            notes: String::new(),
        };
        // 期待値の根拠: 02 を削除済みでも最大値 3 の次 = 4(01 の再利用はしない)
        let rows = vec![row("001", 1), row("001", 3), row("002", 1)];
        assert_eq!(next_rep_idx(&rows, "001"), 4);
        assert_eq!(next_rep_idx(&rows, "002"), 2);
        assert_eq!(next_rep_idx(&rows, "003"), 1); // 未撮影は 1 から
    }

    // [TEST向き] ng フラグは有効テイクに数えない
    #[test]
    fn take_counts_excludes_ng() {
        let row = |flag: &str| IndexRow {
            file_path: String::new(),
            word_id: "001".to_string(),
            word_romaji: String::new(),
            word_label_ja: String::new(),
            rep_idx: 1,
            recorded_at: String::new(),
            duration_ms: 0,
            quality_flag: flag.to_string(),
            notes: String::new(),
        };
        let rows = vec![row(""), row("ok"), row("ng_hands")];
        let counts = take_counts(&rows);
        // 期待値の根拠: ""(未チェック)と "ok" は有効、"ng_hands" は NG 側
        assert_eq!(counts.get("001"), Some(&(2, 1)));
    }

    #[test]
    fn sanitize_tsv_strips_separators() {
        assert_eq!(sanitize_tsv("a\tb\nc"), "a b c");
    }
}
