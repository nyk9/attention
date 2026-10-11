//! お題(次に撮る語)と進捗の読み取り。**読むだけで、何も書かない**。
//!
//! 規則は pose_extractor/src/progress.rs と同じにしてある(pose_extractor への依存は足さず、ここに小さく再実装):
//! - 語リスト: `words.tsv`(`load_words`)。空行・`#` で始まる行・`word_id` で始まる行(ヘッダ)は飛ばす。
//!   列が 5 つ未満の行は無視。stage が数でなければ 99、target_takes が数でなければ 1
//! - 撮影済みテイク: **ファイルが正**(`scan_takes` + `merge_index`)。`<word_id>_<romaji>/<数字>.<動画の拡張子>` を数え、
//!   `index.tsv` は quality_flag を引くためだけに使う。index に無い新しいファイルは quality_flag 空 = 有効に数える。
//!   index にあってもファイルが無い行は数えない
//! - 有効テイク数: quality_flag が `ng` で始まるものは数えない(`take_counts`)
//! - 次の語: stage の小さい順 → 同じ stage 内は words.tsv の行順で、有効テイク数が目標未満の最初の語(`next_word`)。
//!   撮影セッションの「スキップ」(その場限りの状態)は R0 には無い
//!
//! progress.rs との違い: progress.rs の `load_words` は words.tsv が無いとテンプレートを**書き出す**が、
//! ここでは書かずに「見つからない」と返す。index.tsv も更新しない(ffprobe も呼ばない。長さ・撮影日時は数に関係しない)。

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

const WORDS_FILE: &str = "words.tsv";
const INDEX_FILE: &str = "index.tsv";
/// 動画とみなす拡張子(pose_extractor/src/main.rs の `VIDEO_EXTS` と同じ)
const VIDEO_EXTS: &[&str] = &["mp4", "mov", "mkv", "avi", "webm", "m4v"];
/// データディレクトリを上書きする環境変数
pub const DATA_DIR_ENV: &str = "ATTENTION_DATA_DIR";

#[derive(Debug, Clone, PartialEq)]
pub struct WordSpec {
    pub word_id: String,
    pub romaji: String,
    pub label_ja: String,
    pub stage: u32,
    pub target_takes: u32,
}

/// 撮影済みテイク 1 本(数えるのに要る分だけ)
#[derive(Debug, Clone, PartialEq)]
pub struct Take {
    /// データディレクトリからの相対パス("001_konnichiwa/01.mp4")
    pub rel_path: String,
    pub word_id: String,
    pub quality_flag: String,
}

/// 次に撮るもの
#[derive(Debug, Clone, PartialEq)]
pub struct NextTake {
    pub word_id: String,
    pub label_ja: String,
    pub romaji: String,
    /// 有効テイク数(撮影済み)
    pub ok: u32,
    /// NG テイク数
    pub ng: u32,
    pub target: u32,
}

impl NextTake {
    /// 次が何本目か(有効テイク数 + 1)
    pub fn take_no(&self) -> u32 {
        self.ok + 1
    }
}

/// 画面に出す進捗のまとめ
#[derive(Debug, Clone, PartialEq)]
pub struct ProgressView {
    /// 次に撮る語。全部目標に達していれば None
    pub next: Option<NextTake>,
    /// 目標に数える撮影済みテイク(語ごとに目標で頭打ち)の合計
    pub done_takes: u32,
    pub target_takes: u32,
    pub done_words: usize,
    pub total_words: usize,
}

/// リポジトリ直下からのデータディレクトリの相対位置
const DEFAULT_DATA_SUBDIR: &str = "transformer_burn/data/raw_jsl";

/// データディレクトリがどの方法で決まったか(エラー文言に出す)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataDirOrigin {
    /// 環境変数 `ATTENTION_DATA_DIR` の指定
    Env,
    /// 既定の場所(`<リポジトリ>/transformer_burn/data/raw_jsl`)
    Default,
    /// git worktree の本体チェックアウト側の raw_jsl(既定の場所が無かったときのフォールバック)
    MainCheckout,
}

impl DataDirOrigin {
    pub fn label(self) -> &'static str {
        match self {
            DataDirOrigin::Env => "環境変数",
            DataDirOrigin::Default => "既定の場所",
            DataDirOrigin::MainCheckout => "本体へのフォールバック",
        }
    }
}

/// データディレクトリの決定結果。`tried` は決めるときに探した場所(順番どおり)
#[derive(Debug, Clone, PartialEq)]
pub struct DataDirChoice {
    pub path: PathBuf,
    pub origin: DataDirOrigin,
    pub tried: Vec<PathBuf>,
}

/// データディレクトリを決める(純関数。ディスクや git には `exists` と `main_checkout` 経由でしか触れない)。
/// 1. 環境変数 `ATTENTION_DATA_DIR` があれば最優先(存在確認も本体探しもしない)
/// 2. 既定の場所 `<repo_root>/transformer_burn/data/raw_jsl` が存在すればそれ
/// 3. 無ければ、git worktree の本体チェックアウト(`main_checkout()` が返す)の同じ相対位置が存在すればそれ
/// 4. どれも無ければ既定の場所を返す(読むときに「見つかりません」になる)
pub fn choose_data_dir(
    env_value: Option<&str>,
    repo_root: &Path,
    exists: impl Fn(&Path) -> bool,
    main_checkout: impl FnOnce() -> Option<PathBuf>,
) -> DataDirChoice {
    if let Some(v) = env_value.map(str::trim).filter(|v| !v.is_empty()) {
        let path = PathBuf::from(v);
        return DataDirChoice { tried: vec![path.clone()], path, origin: DataDirOrigin::Env };
    }
    let default = repo_root.join(DEFAULT_DATA_SUBDIR);
    if exists(&default) {
        return DataDirChoice { tried: vec![default.clone()], path: default, origin: DataDirOrigin::Default };
    }
    let mut tried = vec![default.clone()];
    if let Some(main) = main_checkout() {
        let candidate = main.join(DEFAULT_DATA_SUBDIR);
        if candidate != default {
            tried.push(candidate.clone());
            if exists(&candidate) {
                return DataDirChoice { path: candidate, origin: DataDirOrigin::MainCheckout, tried };
            }
        }
    }
    DataDirChoice { path: default, origin: DataDirOrigin::Default, tried }
}

/// データディレクトリが無いときの日本語メッセージ。どの方法で決まったかと、探した場所を併記する
pub fn missing_dir_message(choice: &DataDirChoice) -> String {
    let tried = choice
        .tried
        .iter()
        .map(|p| format!("  - {}", p.display()))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "撮影データのフォルダが見つかりません: {}\n決め方: {}\n探した場所:\n{tried}\n\
         worktree では raw_jsl は複製されません。本体のパスを {DATA_DIR_ENV} に指定してください\n\
         (撮影ガイドは読み取り専用なので、ファイルを作ったり直したりはしません)",
        choice.path.display(),
        choice.origin.label(),
    )
}

/// words.tsv の本文を読む(progress.rs の `load_words` の解析部分と同じ規則)
pub fn parse_words(content: &str) -> Vec<WordSpec> {
    let mut words = Vec::new();
    for line in content.lines() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') || line.starts_with("word_id") {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 5 {
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
    words
}

/// index.tsv の本文から (相対パス → quality_flag) を作る(progress.rs の `load_index` と同じ規則:
/// 1 行目はヘッダとして飛ばす、空行は飛ばす、列が 9 つ未満の行は無視)
pub fn parse_index_flags(content: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for line in content.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 9 {
            continue;
        }
        map.insert(cols[0].to_string(), cols[7].to_string());
    }
    map
}

/// `<word_id>_<romaji>` の分解(progress.rs の `parse_take_dir` と同じ)
pub fn parse_take_dir(name: &str) -> Option<(String, String)> {
    let pos = name.find('_')?;
    let (id, rest) = name.split_at(pos);
    let romaji = &rest[1..];
    if id.is_empty() || romaji.is_empty() || !id.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((id.to_string(), romaji.to_string()))
}

/// テイクファイル名の stem が数字だけか(progress.rs の `parse_rep_stem` と同じ条件)
pub fn is_rep_stem(stem: &str) -> bool {
    !stem.is_empty() && stem.chars().all(|c| c.is_ascii_digit()) && stem.parse::<u32>().is_ok()
}

fn is_video(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| VIDEO_EXTS.contains(&s.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// ファイル名の一覧(ディレクトリ名 → その中のファイル名)と index の quality_flag から、テイク一覧を作る(純関数)。
/// `scan_takes` + `merge_index` のうち、数えるのに要る部分
pub fn takes_from_listing(listing: &BTreeMap<String, Vec<String>>, flags: &HashMap<String, String>) -> Vec<Take> {
    let mut takes = Vec::new();
    for (dir, files) in listing {
        let Some((word_id, _romaji)) = parse_take_dir(dir) else { continue };
        for f in files {
            let p = Path::new(f);
            if !is_video(p) {
                continue;
            }
            let stem = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            if !is_rep_stem(&stem) {
                continue;
            }
            let rel_path = format!("{dir}/{f}");
            let quality_flag = flags.get(&rel_path).cloned().unwrap_or_default();
            takes.push(Take { rel_path, word_id: word_id.clone(), quality_flag });
        }
    }
    takes.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    takes
}

/// word_id ごとの (有効テイク数, NG テイク数)。quality_flag が "ng" で始まる行は有効に数えない
pub fn take_counts(takes: &[Take]) -> HashMap<String, (u32, u32)> {
    let mut counts: HashMap<String, (u32, u32)> = HashMap::new();
    for t in takes {
        let e = counts.entry(t.word_id.clone()).or_default();
        if t.quality_flag.starts_with("ng") {
            e.1 += 1;
        } else {
            e.0 += 1;
        }
    }
    counts
}

/// 次に撮る語(stage 順 → words.tsv の行順で、有効テイク数が目標未満の最初の語)
pub fn next_word<'a>(words: &'a [WordSpec], takes: &[Take]) -> Option<&'a WordSpec> {
    let counts = take_counts(takes);
    let mut sorted: Vec<&WordSpec> = words.iter().collect();
    sorted.sort_by_key(|w| w.stage); // 安定ソートなので、同じ stage 内は words.tsv の行順のまま
    sorted.into_iter().find(|w| {
        let (ok, _) = counts.get(&w.word_id).copied().unwrap_or((0, 0));
        ok < w.target_takes
    })
}

/// 進捗のまとめ(純関数)
pub fn summarize(words: &[WordSpec], takes: &[Take]) -> ProgressView {
    let counts = take_counts(takes);
    let next = next_word(words, takes).map(|w| {
        let (ok, ng) = counts.get(&w.word_id).copied().unwrap_or((0, 0));
        NextTake {
            word_id: w.word_id.clone(),
            label_ja: w.label_ja.clone(),
            romaji: w.romaji.clone(),
            ok,
            ng,
            target: w.target_takes,
        }
    });
    let mut done_takes = 0;
    let mut target_takes = 0;
    let mut done_words = 0;
    for w in words {
        let (ok, _) = counts.get(&w.word_id).copied().unwrap_or((0, 0));
        done_takes += ok.min(w.target_takes);
        target_takes += w.target_takes;
        if ok >= w.target_takes {
            done_words += 1;
        }
    }
    ProgressView { next, done_takes, target_takes, done_words, total_words: words.len() }
}

/// 画面に大きく出す一文
pub fn next_word_text(v: &ProgressView) -> String {
    match &v.next {
        Some(n) => format!("次の語: {}({})、{} 本目 / 目標 {}", n.label_ja, n.word_id, n.take_no(), n.target),
        None => "すべての語が目標の本数に達しています".to_string(),
    }
}

/// 画面で大きく出すための分割版: ("こんにちは(001)", "3 本目 / 目標 10")。全部終わっていれば None
pub fn next_word_parts(v: &ProgressView) -> Option<(String, String)> {
    v.next
        .as_ref()
        .map(|n| (format!("{}({})", n.label_ja, n.word_id), format!("{} 本目 / 目標 {}", n.take_no(), n.target)))
}

/// 補足の一文(全体の進み具合)
pub fn summary_text(v: &ProgressView) -> String {
    let ng = v.next.as_ref().map(|n| n.ng).unwrap_or(0);
    let ng_part = if ng > 0 { format!("(この語の NG {ng} 本は数えていません)") } else { String::new() };
    format!(
        "全体 {}/{} 本、完了 {}/{} 語{}",
        v.done_takes, v.target_takes, v.done_words, v.total_words, ng_part
    )
}

// ===== ディスクからの読み取り(読むだけ)=====

/// 決定結果から進捗を読む。フォルダが無いときは、決め方と探した場所を併記したメッセージにする
pub fn load_progress_for(choice: &DataDirChoice) -> Result<ProgressView, String> {
    if !choice.path.is_dir() {
        return Err(missing_dir_message(choice));
    }
    load_progress(&choice.path)
}

/// データディレクトリから進捗を読む。読めないときは、画面に出す日本語の理由を返す
pub fn load_progress(data_dir: &Path) -> Result<ProgressView, String> {
    let hint = format!(
        "(撮影ガイドは読み取り専用なので、ファイルを作ったり直したりはしません。場所は環境変数 {DATA_DIR_ENV} で変えられます)"
    );
    if !data_dir.is_dir() {
        return Err(format!("撮影データのフォルダが見つかりません: {}\n{hint}", data_dir.display()));
    }
    let words_path = data_dir.join(WORDS_FILE);
    let content = std::fs::read_to_string(&words_path)
        .map_err(|e| format!("words.tsv を読めません: {}({e})\n{hint}", words_path.display()))?;
    let words = parse_words(&content);
    if words.is_empty() {
        return Err(format!("words.tsv に語がありません: {}\n{hint}", words_path.display()));
    }
    // index.tsv は無くてもよい(progress.rs と同じく、無ければ quality_flag はすべて空)
    let index_path = data_dir.join(INDEX_FILE);
    let flags = if index_path.exists() {
        let text = std::fs::read_to_string(&index_path)
            .map_err(|e| format!("index.tsv を読めません: {}({e})\n{hint}", index_path.display()))?;
        parse_index_flags(&text)
    } else {
        HashMap::new()
    };
    let listing = list_take_dirs(data_dir).map_err(|e| format!("撮影データのフォルダを読めません: {e}\n{hint}"))?;
    let takes = takes_from_listing(&listing, &flags);
    Ok(summarize(&words, &takes))
}

/// データディレクトリ直下のディレクトリと、その中のファイル名の一覧(読むだけ)
fn list_take_dirs(data_dir: &Path) -> Result<BTreeMap<String, Vec<String>>, String> {
    let mut out = BTreeMap::new();
    let entries = std::fs::read_dir(data_dir).map_err(|e| format!("{}: {e}", data_dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if !path.is_dir() {
            continue; // words.tsv / index.tsv など
        }
        let dir_name = entry.file_name().to_string_lossy().to_string();
        let mut files = Vec::new();
        for f in std::fs::read_dir(&path).map_err(|e| format!("{}: {e}", path.display()))? {
            let f = f.map_err(|e| e.to_string())?;
            if f.path().is_file() {
                files.push(f.file_name().to_string_lossy().to_string());
            }
        }
        out.insert(dir_name, files);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORDS: &str = "word_id\tromaji\tlabel_ja\tstage\ttarget_takes\n\
        # コメント行\n\
        003\tkonbanwa\tこんばんは\t2\t2\n\
        001\tkonnichiwa\tこんにちは\t1\t3\n\
        002\tohayou\tおはよう\t1\t3\n\
        004\tbroken\tこわれ\t1\n\
        \n\
        005\tx\tエックス\tabc\tdef\n";

    fn listing(entries: &[(&str, &[&str])]) -> BTreeMap<String, Vec<String>> {
        entries.iter().map(|(d, fs)| (d.to_string(), fs.iter().map(|s| s.to_string()).collect())).collect()
    }

    #[test]
    fn parse_words_follows_progress_rs_rules() {
        let w = parse_words(WORDS);
        // ヘッダ・コメント・空行・列不足(004)は飛ばす。005 の stage/target は既定値(99 / 1)
        assert_eq!(w.iter().map(|w| w.word_id.as_str()).collect::<Vec<_>>(), vec!["003", "001", "002", "005"]);
        assert_eq!(w[3].stage, 99);
        assert_eq!(w[3].target_takes, 1);
        assert_eq!(w[1].label_ja, "こんにちは");
    }

    #[test]
    fn parse_index_flags_skips_header_and_short_rows() {
        let idx = "file_path\tword_id\tword_romaji\tword_label_ja\trep_idx\trecorded_at\tduration_ms\tquality_flag\tnotes\n\
            001_konnichiwa/01.mp4\t001\tkonnichiwa\tこんにちは\t1\t2026-07-02 01:06:27\t4233\tng_hands\t\n\
            001_konnichiwa/02.mp4\t001\tkonnichiwa\tこんにちは\t2\t\t4200\tok\tメモ\n\
            短い行\n";
        let f = parse_index_flags(idx);
        assert_eq!(f.len(), 2);
        assert_eq!(f["001_konnichiwa/01.mp4"], "ng_hands");
        assert_eq!(f["001_konnichiwa/02.mp4"], "ok");
    }

    #[test]
    fn takes_follow_files_not_the_index() {
        let l = listing(&[
            ("001_konnichiwa", &["01.mp4", "02.mp4", "03.MOV", "take4.mp4", "notes.txt", ".DS_Store"]),
            ("002_ohayou", &["01.mp4"]),
            ("misc", &["01.mp4"]),       // <id>_<romaji> 形式でない → 数えない
            ("abc_def", &["01.mp4"]),    // id が数字でない → 数えない
        ]);
        let mut flags = HashMap::new();
        flags.insert("001_konnichiwa/01.mp4".to_string(), "ng_hands".to_string());
        flags.insert("001_konnichiwa/02.mp4".to_string(), "ok".to_string());
        // index にだけあってファイルが無い行は数えない
        flags.insert("002_ohayou/09.mp4".to_string(), "ok".to_string());
        let t = takes_from_listing(&l, &flags);
        let paths: Vec<_> = t.iter().map(|t| t.rel_path.as_str()).collect();
        // 拡張子は大文字小文字を区別しない(03.MOV も数える)。take4 / txt は数えない
        assert_eq!(paths, vec!["001_konnichiwa/01.mp4", "001_konnichiwa/02.mp4", "001_konnichiwa/03.MOV", "002_ohayou/01.mp4"]);
        let c = take_counts(&t);
        // 01 は ng → NG 側。02 は ok、03 は index に無い(空)→ 有効
        assert_eq!(c["001"], (2, 1));
        assert_eq!(c["002"], (1, 0));
    }

    #[test]
    fn next_word_uses_stage_then_row_order_and_skips_completed() {
        let words = parse_words(WORDS);
        // 何も撮っていない: stage 1 の先頭(words.tsv の行順で 001)
        let v = summarize(&words, &[]);
        assert_eq!(v.next.as_ref().unwrap().word_id, "001");
        assert_eq!(next_word_text(&v), "次の語: こんにちは(001)、1 本目 / 目標 3");
        // 001 が 3 本(うち 1 本は ng)→ 有効 2 本なので、まだ 001 の 3 本目
        let take = |p: &str, id: &str, f: &str| Take { rel_path: p.into(), word_id: id.into(), quality_flag: f.into() };
        let mut takes = vec![
            take("001_konnichiwa/01.mp4", "001", "ng_hands"),
            take("001_konnichiwa/02.mp4", "001", "ok"),
            take("001_konnichiwa/03.mp4", "001", ""),
        ];
        let v = summarize(&words, &takes);
        assert_eq!(next_word_text(&v), "次の語: こんにちは(001)、3 本目 / 目標 3");
        assert_eq!(
            next_word_parts(&v),
            Some(("こんにちは(001)".to_string(), "3 本目 / 目標 3".to_string()))
        );
        assert_eq!(v.next.as_ref().unwrap().ng, 1);
        assert!(summary_text(&v).contains("NG 1 本は数えていません"), "{}", summary_text(&v));
        // 001 が目標に達したら、同じ stage 1 の 002(stage 2 の 003 は後回し)
        takes.push(take("001_konnichiwa/04.mp4", "001", "ok"));
        let v = summarize(&words, &takes);
        assert_eq!(v.next.as_ref().unwrap().word_id, "002");
        assert_eq!(v.done_words, 1);
        assert_eq!(v.done_takes, 3);
        // 目標の合計: 2 + 3 + 3 + 1
        assert_eq!(v.target_takes, 9);
        assert_eq!(summary_text(&v), "全体 3/9 本、完了 1/4 語");
    }

    #[test]
    fn all_done_message() {
        let words = parse_words("001\ta\tあ\t1\t1\n");
        let takes = vec![Take { rel_path: "001_a/01.mp4".into(), word_id: "001".into(), quality_flag: "ok".into() }];
        let v = summarize(&words, &takes);
        assert!(v.next.is_none());
        assert_eq!(next_word_text(&v), "すべての語が目標の本数に達しています");
        assert_eq!(next_word_parts(&v), None);
    }

    #[test]
    fn data_dir_env_wins_without_checking_anything() {
        let root = Path::new("/repo");
        let c = choose_data_dir(Some(" /data/x \n"), root, |_| panic!("環境変数指定では存在確認しない"), || panic!("本体も探さない"));
        assert_eq!(c.path, PathBuf::from("/data/x"));
        assert_eq!(c.origin, DataDirOrigin::Env);
    }

    #[test]
    fn data_dir_default_when_it_exists() {
        let root = Path::new("/repo");
        let want = PathBuf::from("/repo/transformer_burn/data/raw_jsl");
        for env in [None, Some("  ")] {
            let c = choose_data_dir(env, root, |p| p == want, || panic!("既定があれば本体は探さない"));
            assert_eq!((c.path.clone(), c.origin), (want.clone(), DataDirOrigin::Default));
        }
    }

    #[test]
    fn data_dir_falls_back_to_main_checkout() {
        let root = Path::new("/main/.claude/worktrees/w");
        let main_dir = PathBuf::from("/main/transformer_burn/data/raw_jsl");
        let c = choose_data_dir(None, root, |p| p == main_dir, || Some(PathBuf::from("/main")));
        assert_eq!(c.path, main_dir);
        assert_eq!(c.origin, DataDirOrigin::MainCheckout);
        assert_eq!(c.tried.len(), 2);
    }

    #[test]
    fn data_dir_missing_everywhere_keeps_the_default_for_the_original_error() {
        let root = Path::new("/repo");
        // 本体が見つからない / 本体にも無い / 本体が自分自身、のどれでも既定の場所を返す
        for main in [None, Some(PathBuf::from("/main")), Some(PathBuf::from("/repo"))] {
            let c = choose_data_dir(None, root, |_| false, || main.clone());
            assert_eq!(c.path, PathBuf::from("/repo/transformer_burn/data/raw_jsl"));
            assert_eq!(c.origin, DataDirOrigin::Default);
        }
        let c = choose_data_dir(None, root, |_| false, || Some(PathBuf::from("/main")));
        assert_eq!(c.tried.len(), 2);
        let msg = missing_dir_message(&c);
        assert!(msg.contains("撮影データのフォルダが見つかりません: /repo/transformer_burn/data/raw_jsl"), "{msg}");
        assert!(msg.contains("既定の場所"), "{msg}");
        assert!(msg.contains("/main/transformer_burn/data/raw_jsl"), "{msg}");
        assert!(msg.contains("worktree では raw_jsl は複製されません"), "{msg}");
        assert!(msg.contains(DATA_DIR_ENV), "{msg}");
    }

    #[test]
    fn missing_message_names_the_origin() {
        let env = DataDirChoice { path: PathBuf::from("/e"), origin: DataDirOrigin::Env, tried: vec![PathBuf::from("/e")] };
        let m = missing_dir_message(&env);
        assert!(m.contains("決め方: 環境変数") && m.contains("  - /e"), "{m}");
    }

    #[test]
    fn missing_dir_is_reported_in_japanese() {
        let err = load_progress(Path::new("/このフォルダは無い/raw_jsl")).unwrap_err();
        assert!(err.contains("撮影データのフォルダが見つかりません"), "{err}");
        assert!(err.contains(DATA_DIR_ENV), "{err}");
    }

    /// 実データでの確認(手動実行): ATTENTION_DATA_DIR のデータで「次の語」を出す。読むだけ。
    ///   ATTENTION_DATA_DIR=/path/to/raw_jsl cargo test --release real_data_next_word -- --ignored --nocapture
    #[test]
    #[ignore]
    fn real_data_next_word() {
        let Ok(dir) = std::env::var(DATA_DIR_ENV) else { return eprintln!("skip: {DATA_DIR_ENV} が無い") };
        let v = load_progress(Path::new(&dir)).unwrap();
        eprintln!("{}", next_word_text(&v));
        eprintln!("{}", summary_text(&v));
        eprintln!("{:?}", v.next);
    }
}
