//! 撮影(録画あり)の画面に出す文字と色の判断。**純関数だけで、ファイルには触れない**
//! (描画は main.rs、録画・保存はエンジン側 pose_extractor/src/live/capture.rs。
//! このファイルに書き込み系 API が無いことは capture_guide_tests.rs で検査している)。
//!
//! 画面の方針: コントラストと文字の大きさに配慮し、**色だけに頼らず文字でも示す**
//! (「● 録画中」「使えそう」「使えない」など。音には頼らない)。

use crate::capture_guide::progress::DataDirOrigin;
use crate::protocol::{CaptureWord, Verdict};
use crate::state::{CapturePhase, CaptureView, SavedView};

/// 見出しの色の種類(実際の色は main.rs)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    /// 待機中・判定中
    Neutral,
    /// 録画中(赤)
    Recording,
    /// カウントダウン(琥珀)
    Warning,
    /// 使えそう(緑)
    Good,
    /// 使えない(赤みの強い背景)
    Bad,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Banner {
    pub text: String,
    pub tone: Tone,
}

/// カウントダウンの大きな数字("3" "2" "1")。残りが 0 より大きければ切り上げ
// [TEST向き] 純関数
pub fn countdown_digit(left: f64) -> String {
    if left.is_finite() && left > 0.0 {
        format!("{}", left.ceil() as u32)
    } else {
        "0".to_string()
    }
}

/// 撮影の状態を 1 行で示す見出し
// [TEST向き] 純関数。段階ごとの文言と色の種類
pub fn banner(v: &CaptureView) -> Banner {
    match v.phase {
        CapturePhase::Idle => Banner { text: "待機中(録画していません)".into(), tone: Tone::Neutral },
        CapturePhase::Countdown => Banner {
            text: format!("録画開始まで {} 秒", countdown_digit(v.countdown_left)),
            tone: Tone::Warning,
        },
        CapturePhase::Recording => Banner {
            text: format!("● 録画中 {:.1} / {:.0} 秒", v.rec_secs, v.max_secs.max(1.0)),
            tone: Tone::Recording,
        },
        CapturePhase::Judging => Banner { text: "判定中…(数秒かかります)".into(), tone: Tone::Neutral },
        CapturePhase::Judged => match &v.verdict {
            Some(vd) if vd.usable => Banner { text: "使えそう".into(), tone: Tone::Good },
            Some(_) => Banner { text: "使えない".into(), tone: Tone::Bad },
            None => Banner { text: "判定済み".into(), tone: Tone::Neutral },
        },
    }
}

/// 判定の数字の説明(最長連続秒数・左右・長さ)
// [TEST向き] 純関数
pub fn verdict_detail(v: &Verdict) -> String {
    format!(
        "手が連続して検出できた最長: {:.2} 秒(左 {:.2} / 右 {:.2})  長さ: {:.1} 秒",
        v.longest_run_secs, v.left_run_secs, v.right_run_secs, v.duration_secs
    )
}

/// 録画の fps(エンジンが決めたもの)
pub fn fps_text(fps: Option<f64>) -> String {
    match fps {
        Some(f) if f.is_finite() && f > 0.0 => format!("{f:.1} fps"),
        _ => "- fps".to_string(),
    }
}

/// コマ落ちの警告(0 枚なら出さない)
pub fn dropped_text(dropped: u64) -> Option<String> {
    (dropped > 0).then(|| format!("コマ落ち {dropped} 枚(録画が間に合っていません。動きが少し速く記録されます)"))
}

/// 登録先の 1 行。常に画面に出す
pub fn dest_line(engine_data_dir: &str, fallback: &str) -> String {
    let d = if engine_data_dir.is_empty() { fallback } else { engine_data_dir };
    format!("登録先: {d}")
}

/// 登録名の見込み(`011_kare/04.mp4`)。採用の瞬間に番号は決め直されるので「見込み」
// [TEST向き] 純関数
pub fn planned_name(w: &CaptureWord) -> String {
    format!("{}_{}/{:02}.mp4", w.word_id, w.romaji, w.rep_idx)
}

/// 登録先がどう決まったかの注意。撮影済みの本物のデータに書き込むときは、そのことを明示する
// [TEST向き] 純関数
pub fn origin_note(origin: DataDirOrigin) -> String {
    match origin {
        DataDirOrigin::Env => {
            "環境変数 ATTENTION_DATA_DIR で指定された場所です(テスト用の場所でなければ、本物のデータに書き込みます)".into()
        }
        DataDirOrigin::Default => {
            "既定の場所です。撮影済みの本物のデータ(transformer_burn/data/raw_jsl)に書き込みます".into()
        }
        DataDirOrigin::MainCheckout => {
            "worktree から起動していますが、本体の raw_jsl(撮影済みの本物のデータ)に書き込みます".into()
        }
    }
}

/// 採用ボタンの文言。登録名を添えて、押す前に何が起きるか分かるようにする。
/// 使えないテイクは「NG として記録」と明示する(番号は進み、目標数には数えない)
// [TEST向き] 純関数
pub fn accept_label(v: &CaptureView) -> String {
    let name = v.word.as_ref().map(planned_name).unwrap_or_default();
    let usable = v.verdict.as_ref().is_some_and(|d| d.usable);
    if usable {
        format!("採用して登録({name})")
    } else {
        format!("NG として記録({name})")
    }
}

/// 使えないテイクを NG で記録したときの説明
pub const NG_ACCEPT_NOTE: &str = "NG として記録すると、番号は進み、目標のテイク数には数えられません";

/// 直近の登録結果の 1 行
// [TEST向き] 純関数
pub fn saved_text(s: &SavedView) -> String {
    format!(
        "登録しました: {}({})",
        s.file_path,
        crate::state::quality_flag_text(&s.quality_flag)
    )
}

/// 判定の理由(符号)→ 補足の一言。`summary` はエンジンが組み立てるので、これは補助
pub fn reason_hint(code: &str) -> &'static str {
    match code {
        "hands_short" => "手を枠の中で、もう少し長く見せてください",
        "too_short" => "もう少し長く撮ってください",
        "too_long" => "もう少し短く撮ってください",
        "judge_failed" => "判定できませんでした。登録するかは自分で決めてください",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(phase: CapturePhase) -> CaptureView {
        CaptureView { phase, max_secs: 10.0, ..CaptureView::default() }
    }

    fn verdict(usable: bool) -> Verdict {
        Verdict {
            usable,
            reasons: if usable { vec![] } else { vec!["hands_short".into()] },
            summary: String::new(),
            longest_run_secs: 1.67,
            left_run_secs: 0.0,
            right_run_secs: 1.67,
            duration_secs: 4.2,
            quality_flag: if usable { "ok".into() } else { "ng_hands".into() },
        }
    }

    fn word() -> CaptureWord {
        CaptureWord { word_id: "011".into(), romaji: "kare".into(), label: "かれ".into(), rep_idx: 4 }
    }

    #[test]
    fn countdown_digit_rounds_up() {
        assert_eq!(countdown_digit(3.0), "3");
        assert_eq!(countdown_digit(2.1), "3");
        assert_eq!(countdown_digit(0.2), "1");
        assert_eq!(countdown_digit(0.0), "0");
        assert_eq!(countdown_digit(f64::NAN), "0");
        assert_eq!(countdown_digit(-1.0), "0");
    }

    #[test]
    fn banner_texts_and_tones_for_each_phase() {
        assert_eq!(banner(&view(CapturePhase::Idle)), Banner { text: "待機中(録画していません)".into(), tone: Tone::Neutral });
        let mut c = view(CapturePhase::Countdown);
        c.countdown_left = 2.4;
        assert_eq!(banner(&c), Banner { text: "録画開始まで 3 秒".into(), tone: Tone::Warning });
        let mut r = view(CapturePhase::Recording);
        r.rec_secs = 3.24;
        let b = banner(&r);
        assert_eq!(b, Banner { text: "● 録画中 3.2 / 10 秒".into(), tone: Tone::Recording });
        assert!(b.text.contains("録画中"), "色だけに頼らず文字で示す");
        assert_eq!(banner(&view(CapturePhase::Judging)).tone, Tone::Neutral);
        let mut j = view(CapturePhase::Judged);
        j.verdict = Some(verdict(true));
        assert_eq!(banner(&j), Banner { text: "使えそう".into(), tone: Tone::Good });
        j.verdict = Some(verdict(false));
        assert_eq!(banner(&j), Banner { text: "使えない".into(), tone: Tone::Bad });
        j.verdict = None;
        assert_eq!(banner(&j).text, "判定済み");
    }

    #[test]
    fn detail_and_small_texts() {
        assert_eq!(
            verdict_detail(&verdict(true)),
            "手が連続して検出できた最長: 1.67 秒(左 0.00 / 右 1.67)  長さ: 4.2 秒"
        );
        assert_eq!(fps_text(Some(29.97)), "30.0 fps");
        assert_eq!(fps_text(None), "- fps");
        assert_eq!(dropped_text(0), None);
        assert!(dropped_text(3).unwrap().contains("3 枚"));
        assert_eq!(dest_line("/real/raw_jsl", "/guess"), "登録先: /real/raw_jsl");
        assert_eq!(dest_line("", "/guess"), "登録先: /guess", "エンジンから届くまでは画面側の解決結果を出す");
        assert_eq!(planned_name(&word()), "011_kare/04.mp4");
        assert!(reason_hint("hands_short").contains("長く"));
        assert_eq!(reason_hint("unknown"), "");
    }

    // 書き込み先が本物になる状況では、そのことを文字で明示する
    #[test]
    fn origin_notes_flag_real_data() {
        let d = origin_note(DataDirOrigin::Default);
        let m = origin_note(DataDirOrigin::MainCheckout);
        let e = origin_note(DataDirOrigin::Env);
        assert!(d.contains("本物のデータ"), "{d}");
        assert!(m.contains("本体の raw_jsl") && m.contains("本物のデータ"), "{m}");
        assert!(e.contains("ATTENTION_DATA_DIR"), "{e}");
    }

    #[test]
    fn accept_label_names_the_destination_and_marks_ng() {
        let mut v = view(CapturePhase::Judged);
        v.word = Some(word());
        v.verdict = Some(verdict(true));
        assert_eq!(accept_label(&v), "採用して登録(011_kare/04.mp4)");
        v.verdict = Some(verdict(false));
        assert_eq!(accept_label(&v), "NG として記録(011_kare/04.mp4)");
        assert!(NG_ACCEPT_NOTE.contains("目標") && NG_ACCEPT_NOTE.contains("番号"));
    }

    #[test]
    fn saved_text_shows_path_and_flag() {
        let s = SavedView {
            file_path: "011_kare/04.mp4".into(),
            label: "かれ".into(),
            rep_idx: 4,
            quality_flag: "ok".into(),
            usable: true,
        };
        assert_eq!(saved_text(&s), "登録しました: 011_kare/04.mp4(有効なテイク)");
    }
}
