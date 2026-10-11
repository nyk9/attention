//! 「このモデルは、どんな前処理で作った pose dict で学習したか」の記録(`preprocess.json`)。
//!
//! # なぜ必要か
//!
//! 認識モデルの精度は、推論時の特徴抽出が学習時の dict と同じ作り方であることに依存する
//! (Palm 検出に渡す領域 ROI、10 フレームの選び方)。ところがモデルディレクトリには
//! 従来これが残らず、たとえば中央正方クロップで作った dict で学習したモデルに
//! フレーム全体で抽出した特徴を食わせる、という不整合を静かに起こせた。
//! pose dict の metadata には `palm_roi` / `frame_select` が入っているので、
//! `--save` 時にそれをモデルと一緒にコピーしておき、推論側が読んで揃えられるようにする。
//!
//! # 値の意味(pose_extractor の `PalmRoiMode::label` / `FrameSelectMode::label` と同じ書式)
//!
//! - `palm_roi`: "full-frame" / "center-square" / "relative(...)" / "tiles(3)" / "person(...)"
//! - `frame_select`: "uniform" / "sign-span(samples=16,max_gap=1)"
//! - **キーが無い(null)= 古い dict**: `palm_roi` が無ければ full-frame、`frame_select` が
//!   無ければ uniform で作られている(それぞれ ROI 切り替え・サイン区間選択を足す前の挙動)。
//!   ここでは「無い」をそのまま `None` で持ち、解釈(既定値の補完)は読む側に任せる。
//!   `preprocess.json` 自体が無いモデル(古いモデル)とは区別できる。

use serde::{Deserialize, Serialize};
use std::path::Path;

const FILE_NAME: &str = "preprocess.json";

/// 学習に使った pose dict の前処理情報(dict の metadata から写した値)
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct PreprocessInfo {
    /// Palm 検出に渡した領域。None = dict に無い(古い dict = full-frame)
    #[serde(default)]
    pub palm_roi: Option<String>,
    /// 10 フレームの選び方。None = dict に無い(古い dict = uniform)
    #[serde(default)]
    pub frame_select: Option<String>,
    /// 1 サンプルのフレーム数 T(dict の metadata.frames)
    #[serde(default)]
    pub frames: Option<usize>,
}

/// dict の先頭 `metadata` だけを読むための最小スキーマ。
/// `tags`(巨大)は構造体に無いので、serde は読み飛ばす(値は保持しない)
#[derive(Deserialize)]
struct DictHead {
    metadata: PreprocessInfo,
}

impl PreprocessInfo {
    /// pose dict JSON の metadata から前処理情報を読む(学習データ本体は読まない)
    pub fn from_dict_file(dict_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(dict_path)
            .map_err(|e| format!("pose dict が読み込めません {}: {}", dict_path.display(), e))?;
        let head: DictHead = serde_json::from_str(&content)?;
        Ok(head.metadata)
    }

    /// モデルディレクトリに `preprocess.json` として保存する
    pub fn save(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(dir)?;
        let file = std::fs::File::create(dir.join(FILE_NAME))?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    /// `preprocess.json` を読む。ファイルが無ければ None(= 前処理が記録されていない古いモデル)
    // pose_extractor(lib 経由)から使う。transformer_burn の bin は mod を重複宣言しているので
    // bin 側では未使用扱いになる
    #[allow(dead_code)]
    pub fn load(dir: &Path) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let path = dir.join(FILE_NAME);
        if !path.exists() {
            return Ok(None);
        }
        let content = std::fs::read_to_string(&path)?;
        Ok(Some(serde_json::from_str(&content)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = PathBuf::from(format!("tests/temp_preprocess_{}", name));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_dict(dir: &Path, metadata: &str) -> PathBuf {
        let path = dir.join("dict.json");
        std::fs::write(
            &path,
            format!(r#"{{"metadata": {}, "tags": {{"a-01": {{"sequence": [[0.0]]}}}}}}"#, metadata),
        )
        .unwrap();
        path
    }

    // [TEST向き] dict の metadata から palm_roi / frame_select を写し、保存→読み込みで一致すること
    #[test]
    fn dict_metadata_roundtrips_through_preprocess_json() {
        let dir = temp_dir("roundtrip");
        let dict = write_dict(
            &dir,
            r#"{"frames": 10, "feature_dim": 126, "palm_roi": "center-square",
                "frame_select": "sign-span(samples=16,max_gap=1)", "tag_count": 1}"#,
        );
        let info = PreprocessInfo::from_dict_file(&dict).unwrap();
        assert_eq!(info.palm_roi.as_deref(), Some("center-square"));
        assert_eq!(info.frame_select.as_deref(), Some("sign-span(samples=16,max_gap=1)"));
        assert_eq!(info.frames, Some(10));

        let model_dir = dir.join("model");
        info.save(&model_dir).unwrap();
        assert_eq!(PreprocessInfo::load(&model_dir).unwrap(), Some(info));
        std::fs::remove_dir_all(&dir).ok();
    }

    // [TEST向き] 古い dict(キー無し)は None のまま写る。ファイルが無いモデルは load が None
    #[test]
    fn legacy_dict_keeps_none_and_missing_file_is_none() {
        let dir = temp_dir("legacy");
        let dict = write_dict(&dir, r#"{"frames": 10, "feature_dim": 126}"#);
        let info = PreprocessInfo::from_dict_file(&dict).unwrap();
        assert_eq!(info.palm_roi, None);
        assert_eq!(info.frame_select, None);

        // preprocess.json を持たないディレクトリ
        assert_eq!(PreprocessInfo::load(&dir).unwrap(), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    // 実 dict の metadata が読めること(dict は git 管理外なので、無ければスキップ)
    #[test]
    fn reads_real_dict_metadata_if_present() {
        let cases = [
            ("data/pose_dict_full_center_signspan.json", Some("center-square"), Some("sign-span(samples=16,max_gap=1)")),
            ("data/pose_dict_full_center_uniform.json", Some("center-square"), Some("uniform")),
            // 古い dict: キー自体が無い
            ("data/pose_dict_full.json", None, None),
        ];
        for (path, roi, select) in cases {
            let path = Path::new(path);
            if !path.exists() {
                eprintln!("skip: {} が無い", path.display());
                continue;
            }
            let info = PreprocessInfo::from_dict_file(path).unwrap();
            assert_eq!(info.palm_roi.as_deref(), roi, "{}", path.display());
            assert_eq!(info.frame_select.as_deref(), select, "{}", path.display());
        }
    }
}
