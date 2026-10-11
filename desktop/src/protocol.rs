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
    Toggle,
    SetSegmenter(SegmenterMode),
    Quit,
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
                // model_dir と preview は省略 = エンジンの既定(デモ用モデル / 10fps・幅 640)
                json!({"cmd": "start", "source": source, "segmenter": segmenter.name()})
            }
            Command::Toggle => json!({"cmd": "toggle"}),
            Command::SetSegmenter(mode) => json!({"cmd": "set_segmenter", "mode": mode.name()}),
            Command::Quit => json!({"cmd": "quit"}),
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
