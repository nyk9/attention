//! エンジン(`pose-extract --engine-json`)との JSON の会話。1 行 1 メッセージ。
//!
//! 仕様の出典: pose_extractor/src/live/json_engine.rs の `Event` / `Command` と、
//! その golden テスト(キー名はそれに合わせてある)。
//! 将来エンジン側に項目が増えても読めるよう、未知のキーは無視し、省略されうる項目には `default` を付ける。
//! 同じ理由で、**未知のイベント種別(`type`)も `Unknown` として読み、黙って読み捨てる**
//! (エンジンが新しい種類のイベントを足しても、画面に「解釈できない出力」の警告を出さない)。
//! 壊れた JSON や `type` の無い行は、従来どおりエラー(画面に警告が出る)。

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct VocabEntry {
    pub tag: String,
    pub label: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct RankedItem {
    pub tag: String,
    pub label: String,
    pub p: f64,
}

/// 撮影で録画する語
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct CaptureWord {
    pub word_id: String,
    #[serde(default)]
    pub romaji: String,
    pub label: String,
    /// 登録先になる見込みのテイク番号(採用の瞬間に決め直す)
    #[serde(default)]
    pub rep_idx: u32,
}

/// 録画の判定(エンジンが決めた結果。画面はそのまま見せる)
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Verdict {
    pub usable: bool,
    /// "hands_short" | "too_short" | "too_long" | "judge_failed"
    #[serde(default)]
    pub reasons: Vec<String>,
    /// 画面にそのまま出す 1 文
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub longest_run_secs: f64,
    #[serde(default)]
    pub left_run_secs: f64,
    #[serde(default)]
    pub right_run_secs: f64,
    #[serde(default)]
    pub duration_secs: f64,
    /// 採用したときに index.tsv へ書く値
    #[serde(default)]
    pub quality_flag: String,
}

/// 左右の手の 21 点の (x, y)。0〜1 に正規化済み(少し外れることがある)。未検出の枠は null。
/// 映像は反転していない座標で来る。
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct HandsPoints {
    #[serde(default)]
    pub left: Option<Vec<[f64; 2]>>,
    #[serde(default)]
    pub right: Option<Vec<[f64; 2]>>,
}

/// 検出範囲(Palm 検出に渡す範囲)。元フレームに対する正規化座標の矩形
#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
pub struct CropRect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineEvent {
    Ready {
        #[serde(default)]
        vocab: Vec<VocabEntry>,
        #[serde(default)]
        model_dir: String,
        #[serde(default)]
        palm_roi: String,
        #[serde(default)]
        frame_select: String,
        #[serde(default)]
        frames: usize,
        #[serde(default)]
        segmenter: String,
        #[serde(default)]
        preprocess: String,
    },
    Status {
        /// "idle"(待機)| "recording"(記録中)
        state: String,
        /// "L" | "R" | "LR" | "none"
        hands: String,
        fps: f64,
        rec_secs: f64,
    },
    Preview {
        width: u32,
        height: u32,
        jpeg_b64: String,
        #[serde(default)]
        hands: HandsPoints,
        crop: CropRect,
    },
    Result {
        tag: String,
        label: String,
        p: f64,
        #[serde(default)]
        ranked: Vec<RankedItem>,
        #[serde(default)]
        low_confidence: bool,
        #[serde(default)]
        span: Option<[usize; 2]>,
        #[serde(default)]
        buffered: usize,
        #[serde(default)]
        duration_secs: f64,
        #[serde(default)]
        selected_with_hand: usize,
        #[serde(default)]
        selected_total: usize,
    },
    Message {
        /// "info" | "warn"
        level: String,
        text: String,
    },
    Error {
        text: String,
    },
    Stopped {
        #[serde(default)]
        dropped_frames: u64,
    },
    Segmenter {
        /// "auto" | "manual"
        mode: String,
    },
    /// 撮影モード(録画あり)の状態。pose_extractor/src/live/json_engine.rs の `Event::CaptureState` と同じ形
    CaptureState {
        /// "idle" | "countdown" | "recording" | "judging" | "judged"
        phase: String,
        /// カウントダウンの残り秒数
        #[serde(default)]
        countdown_left: f64,
        /// 録画した長さ(秒)
        #[serde(default)]
        rec_secs: f64,
        /// 自動停止する長さ(秒)
        #[serde(default)]
        max_secs: f64,
        #[serde(default)]
        fps: Option<f64>,
        /// 録画が間に合わず捨てたコマ数
        #[serde(default)]
        dropped: u64,
        /// 録画する語。録画を始める前は null
        #[serde(default)]
        word: Option<CaptureWord>,
        /// 登録先(raw_jsl)。エンジンが実際に使うパス
        #[serde(default)]
        data_dir: String,
        /// 判定(judged のとき)
        #[serde(default)]
        verdict: Option<Verdict>,
        /// 画面に出す注意
        #[serde(default)]
        notes: Vec<String>,
    },
    /// 採用して登録した結果
    CaptureSaved {
        /// 登録先からの相対パス("011_kare/04.mp4")
        file_path: String,
        #[serde(default)]
        abs_path: String,
        #[serde(default)]
        word_id: String,
        #[serde(default)]
        label: String,
        #[serde(default)]
        rep_idx: u32,
        /// index.tsv に書いた値。使えると判定されたテイクしか登録されないので、常に "ok"
        #[serde(default)]
        quality_flag: String,
        #[serde(default)]
        longest_run_secs: f64,
        #[serde(default)]
        duration_secs: f64,
        #[serde(default)]
        data_dir: String,
    },
    /// 知らない `type` のイベント(エンジンの方が新しいとき)。呼び出し側は読み捨てる。
    /// Rust メモ: `#[serde(other)]` は、「タグ(type)が他のどの変種にも当てはまらなかったとき」に使う
    /// 変種を指定する属性。中身を持たない変種にしか付けられず、元のキーは捨てられる
    #[serde(other)]
    Unknown,
}

/// 1 行をイベントとして読む。空行は Ok(None)
pub fn parse_event(line: &str) -> Result<Option<EngineEvent>, String> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }
    serde_json::from_str::<EngineEvent>(line)
        .map(Some)
        .map_err(|e| e.to_string())
}

// ===== コマンド(アプリ → エンジン)=====

#[derive(Debug, Clone, PartialEq)]
pub enum SourceSpec {
    /// カメラ(デバイス番号は文字列)
    Camera { device: String },
    File { path: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmenterMode {
    Auto,
    Manual,
}

impl SegmenterMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(Self::Auto),
            "manual" => Some(Self::Manual),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Manual => "manual",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Start {
        source: SourceSpec,
        segmenter: SegmenterMode,
    },
    /// 撮影モード(録画あり)で起動する。`data_dir` は登録先(raw_jsl)。エンジンはここにだけ書く。
    /// 認識モデルは読み込まず、語の推論はしない
    StartCapture {
        source: SourceSpec,
        data_dir: String,
        /// 省略 = エンジンの既定(3 秒)
        countdown_secs: Option<f64>,
    },
    Toggle,
    SetSegmenter(SegmenterMode),
    Quit,
    /// 録画開始(カウントダウンのあと録画)。`word_id` は画面が出しているお題の語
    CaptureRecord { word_id: Option<String> },
    /// 録画停止(カウントダウン中なら取り消し)
    CaptureStop,
    /// 採用(登録先へ移して index.tsv に記帳)。使えると判定されたテイクだけ。使えないテイクはエンジンが拒否する
    /// (破棄か撮り直しだけ。以前あった `confirm_ng` は廃止)
    CaptureAccept,
    /// 破棄(一時ファイルを消す)
    CaptureDiscard,
}

impl Command {
    /// 標準入力へ書く 1 行(改行なし)
    pub fn to_line(&self) -> String {
        use serde_json::json;
        let v = match self {
            Command::Start { source, segmenter } => {
                let source = match source {
                    SourceSpec::Camera { device } => json!({"kind": "camera", "device": device}),
                    SourceSpec::File { path } => json!({"kind": "file", "path": path}),
                };
                // model_dir と preview は省略 = エンジンの既定(rec_t10_full / 10fps・幅 640)
                json!({"cmd": "start", "source": source, "segmenter": segmenter.name()})
            }
            Command::StartCapture { source, data_dir, countdown_secs } => {
                let source = match source {
                    SourceSpec::Camera { device } => json!({"kind": "camera", "device": device}),
                    SourceSpec::File { path } => json!({"kind": "file", "path": path}),
                };
                let mut v = json!({"cmd": "start", "mode": "capture", "source": source, "data_dir": data_dir});
                if let Some(c) = countdown_secs {
                    v["countdown_secs"] = json!(c);
                }
                v
            }
            Command::Toggle => json!({"cmd": "toggle"}),
            Command::SetSegmenter(mode) => json!({"cmd": "set_segmenter", "mode": mode.name()}),
            Command::Quit => json!({"cmd": "quit"}),
            Command::CaptureRecord { word_id } => match word_id {
                Some(id) => json!({"cmd": "capture_record", "word_id": id}),
                None => json!({"cmd": "capture_record"}),
            },
            Command::CaptureStop => json!({"cmd": "capture_stop"}),
            Command::CaptureAccept => json!({"cmd": "capture_accept"}),
            Command::CaptureDiscard => json!({"cmd": "capture_discard"}),
        };
        v.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // pose_extractor 側の golden テストと同じ形の行を読めること

    #[test]
    fn parse_ready() {
        let line = json!({
            "type": "ready",
            "vocab": [{"tag": "konnichiwa", "label": "こんにちは"}],
            "model_dir": "../transformer_burn/models/rec_demo_center_signspan",
            "palm_roi": "center-square",
            "frame_select": "sign-span(samples=16,max_gap=1)",
            "frames": 10,
            "segmenter": "auto",
            "preprocess": "default",
        })
        .to_string();
        match parse_event(&line).unwrap().unwrap() {
            EngineEvent::Ready { vocab, frames, segmenter, preprocess, .. } => {
                assert_eq!(vocab.len(), 1);
                assert_eq!(vocab[0].label, "こんにちは");
                assert_eq!(frames, 10);
                assert_eq!(segmenter, "auto");
                assert_eq!(preprocess, "default");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn parse_status_exact_line() {
        let line = r#"{"type":"status","state":"recording","hands":"LR","fps":15.2,"rec_secs":1.3}"#;
        assert_eq!(
            parse_event(line).unwrap().unwrap(),
            EngineEvent::Status {
                state: "recording".into(),
                hands: "LR".into(),
                fps: 15.2,
                rec_secs: 1.3
            }
        );
    }

    #[test]
    fn parse_preview_with_null_hand() {
        let line = json!({
            "type": "preview", "width": 640, "height": 427, "jpeg_b64": "AAAA",
            "hands": {"left": null, "right": vec![[0.1, 0.2]; 21]},
            "crop": {"x": 0.1667, "y": 0.0, "w": 0.6667, "h": 1.0},
        })
        .to_string();
        match parse_event(&line).unwrap().unwrap() {
            EngineEvent::Preview { width, height, jpeg_b64, hands, crop } => {
                assert_eq!((width, height), (640, 427));
                assert_eq!(jpeg_b64, "AAAA");
                assert!(hands.left.is_none());
                assert_eq!(hands.right.unwrap().len(), 21);
                assert_eq!(crop, CropRect { x: 0.1667, y: 0.0, w: 0.6667, h: 1.0 });
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn parse_result_with_and_without_span() {
        let line = json!({
            "type": "result", "tag": "konnichiwa", "label": "こんにちは", "p": 0.99,
            "ranked": [
                {"tag": "konnichiwa", "label": "こんにちは", "p": 0.99},
                {"tag": "ohayou", "label": "おはよう", "p": 0.006},
            ],
            "low_confidence": false, "span": [28, 59], "buffered": 71,
            "duration_secs": 4.6, "selected_with_hand": 7, "selected_total": 10,
        })
        .to_string();
        match parse_event(&line).unwrap().unwrap() {
            EngineEvent::Result { label, p, ranked, low_confidence, span, .. } => {
                assert_eq!(label, "こんにちは");
                assert_eq!(p, 0.99);
                assert_eq!(ranked.len(), 2);
                assert!(!low_confidence);
                assert_eq!(span, Some([28, 59]));
            }
            other => panic!("{other:?}"),
        }
        let line = r#"{"type":"result","tag":"x","label":"x","p":0.4,"ranked":[],"low_confidence":true,"span":null}"#;
        match parse_event(line).unwrap().unwrap() {
            EngineEvent::Result { span, low_confidence, .. } => {
                assert_eq!(span, None);
                assert!(low_confidence);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn parse_message_error_stopped_segmenter() {
        assert_eq!(
            parse_event(r#"{"type":"message","level":"warn","text":"手が検出されませんでした"}"#)
                .unwrap()
                .unwrap(),
            EngineEvent::Message { level: "warn".into(), text: "手が検出されませんでした".into() }
        );
        assert_eq!(
            parse_event(r#"{"type":"error","text":"x"}"#).unwrap().unwrap(),
            EngineEvent::Error { text: "x".into() }
        );
        assert_eq!(
            parse_event(r#"{"type":"stopped","dropped_frames":764}"#).unwrap().unwrap(),
            EngineEvent::Stopped { dropped_frames: 764 }
        );
        assert_eq!(
            parse_event(r#"{"type":"segmenter","mode":"manual"}"#).unwrap().unwrap(),
            EngineEvent::Segmenter { mode: "manual".into() }
        );
    }

    #[test]
    fn unknown_keys_and_unknown_type_are_tolerated_but_garbage_is_an_error() {
        // 余計なキーは無視(エンジンが項目を足しても壊れない)
        assert!(parse_event(r#"{"type":"stopped","dropped_frames":1,"extra":true}"#).unwrap().is_some());
        // 知らない type は Unknown(エラーにしない)。理由: エンジンが新しい種類のイベントを足したとき、
        // 古い画面が「解釈できない出力」の警告を出し続けないようにするため(以前は Err だった)。
        // 余計なキーを持っていても、中身が何であっても Unknown になる
        assert_eq!(parse_event(r#"{"type":"nope"}"#).unwrap(), Some(EngineEvent::Unknown));
        assert_eq!(
            parse_event(r#"{"type":"future_event","a":[1,2,3],"b":{"c":null}}"#).unwrap(),
            Some(EngineEvent::Unknown)
        );
        // 壊れた JSON・オブジェクトでない行・type が無い行・type が文字列でない行は、従来どおりエラー(警告が出る)
        assert!(parse_event("not json").is_err());
        assert!(parse_event(r#"{"type":"status""#).is_err(), "途中で切れた行");
        assert!(parse_event(r#"[1,2]"#).is_err());
        assert!(parse_event(r#"{"level":"warn"}"#).is_err(), "type が無い");
        assert!(parse_event(r#"{"type":5}"#).is_err(), "type が文字列でない");
        // 既知の type で必須項目が欠けているのは、Unknown ではなくエラー(壊れたイベントを見逃さない)
        assert!(parse_event(r#"{"type":"status","state":"idle"}"#).is_err());
        // 空行は無視
        assert_eq!(parse_event("   ").unwrap(), None);
    }

    // pose_extractor 側の golden テスト(golden_capture_state / golden_capture_saved)と同じ形の行を読めること
    #[test]
    fn parse_capture_state_and_saved() {
        let line = json!({
            "type": "capture_state",
            "phase": "judged",
            "countdown_left": 0.0,
            "rec_secs": 4.2,
            "max_secs": 10.0,
            "fps": 30.0,
            "dropped": 0,
            "word": {"word_id": "011", "romaji": "kare", "label": "かれ", "rep_idx": 4},
            "data_dir": "/data/raw_jsl",
            "verdict": {
                "usable": false,
                "reasons": ["hands_short"],
                "summary": "使えない: 手が 1.0 秒続いて検出されていません(最長 0.67 秒)",
                "longest_run_secs": 0.67,
                "left_run_secs": 0.0,
                "right_run_secs": 0.67,
                "duration_secs": 4.2,
                "quality_flag": "ng_hands",
            },
            "notes": ["注意"],
        })
        .to_string();
        match parse_event(&line).unwrap().unwrap() {
            EngineEvent::CaptureState { phase, rec_secs, word, data_dir, verdict, notes, fps, .. } => {
                assert_eq!(phase, "judged");
                assert_eq!(rec_secs, 4.2);
                assert_eq!(fps, Some(30.0));
                assert_eq!(word.unwrap().rep_idx, 4);
                assert_eq!(data_dir, "/data/raw_jsl");
                let v = verdict.unwrap();
                assert!(!v.usable);
                assert_eq!(v.reasons, vec!["hands_short"]);
                assert_eq!(v.quality_flag, "ng_hands");
                assert_eq!(notes.len(), 1);
            }
            other => panic!("{other:?}"),
        }
        // 待機(word と verdict は null、省略された項目は既定値)
        let line = r#"{"type":"capture_state","phase":"idle","word":null,"verdict":null,"data_dir":"/d"}"#;
        match parse_event(line).unwrap().unwrap() {
            EngineEvent::CaptureState { phase, word, verdict, notes, dropped, .. } => {
                assert_eq!(phase, "idle");
                assert!(word.is_none() && verdict.is_none() && notes.is_empty());
                assert_eq!(dropped, 0);
            }
            other => panic!("{other:?}"),
        }
        let line = json!({
            "type": "capture_saved", "file_path": "011_kare/04.mp4", "abs_path": "/data/raw_jsl/011_kare/04.mp4",
            "word_id": "011", "label": "かれ", "rep_idx": 4, "quality_flag": "ok",
            "longest_run_secs": 1.67, "duration_secs": 4.2, "data_dir": "/data/raw_jsl",
        })
        .to_string();
        match parse_event(&line).unwrap().unwrap() {
            EngineEvent::CaptureSaved { file_path, rep_idx, quality_flag, .. } => {
                assert_eq!(file_path, "011_kare/04.mp4");
                assert_eq!(rep_idx, 4);
                assert_eq!(quality_flag, "ok");
            }
            other => panic!("{other:?}"),
        }
        // phase が無い capture_state は壊れたイベント(Unknown ではなくエラー)
        assert!(parse_event(r#"{"type":"capture_state"}"#).is_err());
    }

    // 撮影モードのコマンドの形(pose_extractor 側の parse_capture_commands_and_start_mode と同じ)
    #[test]
    fn capture_command_lines() {
        let v: serde_json::Value = serde_json::from_str(
            &Command::StartCapture {
                source: SourceSpec::Camera { device: "0".into() },
                data_dir: "/a b/raw_jsl".into(),
                countdown_secs: None,
            }
            .to_line(),
        )
        .unwrap();
        assert_eq!(
            v,
            json!({"cmd": "start", "mode": "capture", "source": {"kind": "camera", "device": "0"}, "data_dir": "/a b/raw_jsl"})
        );
        let v: serde_json::Value = serde_json::from_str(
            &Command::StartCapture {
                source: SourceSpec::File { path: "/v.mp4".into() },
                data_dir: "/d".into(),
                countdown_secs: Some(0.0),
            }
            .to_line(),
        )
        .unwrap();
        assert_eq!(
            v,
            json!({"cmd": "start", "mode": "capture", "source": {"kind": "file", "path": "/v.mp4"}, "data_dir": "/d", "countdown_secs": 0.0})
        );
        assert_eq!(
            Command::CaptureRecord { word_id: Some("011".into()) }.to_line(),
            r#"{"cmd":"capture_record","word_id":"011"}"#
        );
        assert_eq!(Command::CaptureRecord { word_id: None }.to_line(), r#"{"cmd":"capture_record"}"#);
        assert_eq!(Command::CaptureStop.to_line(), r#"{"cmd":"capture_stop"}"#);
        assert_eq!(Command::CaptureAccept.to_line(), r#"{"cmd":"capture_accept"}"#, "confirm_ng は廃止した");
        assert_eq!(Command::CaptureDiscard.to_line(), r#"{"cmd":"capture_discard"}"#);
    }

    #[test]
    fn command_lines() {
        let v: serde_json::Value = serde_json::from_str(
            &Command::Start {
                source: SourceSpec::Camera { device: "0".into() },
                segmenter: SegmenterMode::Auto,
            }
            .to_line(),
        )
        .unwrap();
        assert_eq!(
            v,
            json!({"cmd": "start", "source": {"kind": "camera", "device": "0"}, "segmenter": "auto"})
        );
        let v: serde_json::Value = serde_json::from_str(
            &Command::Start {
                source: SourceSpec::File { path: "/a b/日本語.mp4".into() },
                segmenter: SegmenterMode::Manual,
            }
            .to_line(),
        )
        .unwrap();
        assert_eq!(
            v,
            json!({"cmd": "start", "source": {"kind": "file", "path": "/a b/日本語.mp4"}, "segmenter": "manual"})
        );
        assert_eq!(Command::Toggle.to_line(), r#"{"cmd":"toggle"}"#);
        assert_eq!(Command::Quit.to_line(), r#"{"cmd":"quit"}"#);
        assert_eq!(
            Command::SetSegmenter(SegmenterMode::Manual).to_line(),
            r#"{"cmd":"set_segmenter","mode":"manual"}"#
        );
        // 1 行であること(改行を含まない)
        assert!(!Command::Toggle.to_line().contains('\n'));
    }
}
