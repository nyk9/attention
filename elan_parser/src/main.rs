use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(name = "elan-eaf-parse", about = "Extract gloss sequences from ELAN .eaf files")]
struct Args {
    /// Path to input .eaf file
    input: PathBuf,

    /// Filter to a specific tier ID (extract only this tier if specified)
    #[arg(long)]
    tier: Option<String>,

    /// Output format
    #[arg(long, value_enum, default_value_t = Format::Tsv)]
    format: Format,

    /// Output file path (stdout if not specified)
    #[arg(long, short)]
    output: Option<PathBuf>,
}

#[derive(ValueEnum, Clone, Debug)]
enum Format {
    Tsv,
    Json,
}

#[derive(Serialize, Debug)]
struct AnnotationRecord {
    start_ms: i64,
    end_ms: i64,
    tier_id: String,
    value: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let xml = fs::read_to_string(&args.input)
        .with_context(|| format!("Failed to read {}", args.input.display()))?;

    let time_slots = parse_time_slots(&xml)?;
    let records = parse_annotations(&xml, &time_slots, args.tier.as_deref())?;

    write_output(&records, &args.format, args.output.as_ref())?;
    Ok(())
}

fn parse_time_slots(xml: &str) -> Result<HashMap<String, i64>> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut slots = HashMap::new();
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf)? {
            Event::Eof => break,
            Event::Empty(e) | Event::Start(e) => {
                if e.name().as_ref() == b"TIME_SLOT" {
                    let mut id: Option<String> = None;
                    let mut value: Option<i64> = None;
                    for attr_result in e.attributes() {
                        let attr = attr_result?;
                        let key = attr.key.as_ref();
                        let val = attr.unescape_value()?;
                        match key {
                            b"TIME_SLOT_ID" => id = Some(val.to_string()),
                            b"TIME_VALUE" => {
                                value = Some(val.parse::<i64>().with_context(|| {
                                    format!("Invalid TIME_VALUE: {val}")
                                })?);
                            }
                            _ => {}
                        }
                    }
                    if let (Some(id), Some(value)) = (id, value) {
                        slots.insert(id, value);
                    }
                }
            }
            _ => {}
        }
        buf.clear();
    }
    Ok(slots)
}

fn parse_annotations(
    xml: &str,
    time_slots: &HashMap<String, i64>,
    tier_filter: Option<&str>,
) -> Result<Vec<AnnotationRecord>> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut records = Vec::new();
    let mut buf = Vec::new();

    let mut current_tier_id: Option<String> = None;
    let mut in_target_tier = false;
    let mut pending_start_ms: Option<i64> = None;
    let mut pending_end_ms: Option<i64> = None;
    let mut in_annotation_value = false;
    let mut value_buf = String::new();

    loop {
        match reader.read_event_into(&mut buf)? {
            Event::Eof => break,
            Event::Start(e) => match e.name().as_ref() {
                b"TIER" => {
                    let mut tid: Option<String> = None;
                    for attr_result in e.attributes() {
                        let attr = attr_result?;
                        if attr.key.as_ref() == b"TIER_ID" {
                            tid = Some(attr.unescape_value()?.to_string());
                        }
                    }
                    in_target_tier = match (&tid, tier_filter) {
                        (Some(_), None) => true,
                        (Some(t), Some(f)) => t == f,
                        (None, _) => false,
                    };
                    current_tier_id = tid;
                }
                b"ALIGNABLE_ANNOTATION" if in_target_tier => {
                    let mut ref1: Option<String> = None;
                    let mut ref2: Option<String> = None;
                    for attr_result in e.attributes() {
                        let attr = attr_result?;
                        let key = attr.key.as_ref();
                        let val = attr.unescape_value()?;
                        match key {
                            b"TIME_SLOT_REF1" => ref1 = Some(val.to_string()),
                            b"TIME_SLOT_REF2" => ref2 = Some(val.to_string()),
                            _ => {}
                        }
                    }
                    let start = ref1
                        .as_ref()
                        .and_then(|r| time_slots.get(r))
                        .copied()
                        .ok_or_else(|| anyhow!("TIME_SLOT_REF1 unresolved: {:?}", ref1))?;
                    let end = ref2
                        .as_ref()
                        .and_then(|r| time_slots.get(r))
                        .copied()
                        .ok_or_else(|| anyhow!("TIME_SLOT_REF2 unresolved: {:?}", ref2))?;
                    pending_start_ms = Some(start);
                    pending_end_ms = Some(end);
                }
                b"ANNOTATION_VALUE" if in_target_tier => {
                    in_annotation_value = true;
                    value_buf.clear();
                }
                _ => {}
            },
            Event::Text(t) if in_annotation_value => {
                value_buf.push_str(&t.unescape()?);
            }
            Event::End(e) => match e.name().as_ref() {
                b"TIER" => {
                    current_tier_id = None;
                    in_target_tier = false;
                }
                b"ALIGNABLE_ANNOTATION" if in_target_tier => {
                    if let (Some(start), Some(end), Some(tid)) = (
                        pending_start_ms.take(),
                        pending_end_ms.take(),
                        current_tier_id.clone(),
                    ) {
                        records.push(AnnotationRecord {
                            start_ms: start,
                            end_ms: end,
                            tier_id: tid,
                            value: std::mem::take(&mut value_buf),
                        });
                    }
                }
                b"ANNOTATION_VALUE" => {
                    in_annotation_value = false;
                }
                _ => {}
            },
            _ => {}
        }
        buf.clear();
    }
    Ok(records)
}

fn write_output(
    records: &[AnnotationRecord],
    format: &Format,
    output: Option<&PathBuf>,
) -> Result<()> {
    let mut writer: Box<dyn Write> = match output {
        Some(p) => Box::new(fs::File::create(p)?),
        None => Box::new(io::stdout().lock()),
    };

    match format {
        Format::Tsv => {
            writeln!(writer, "start_ms\tend_ms\ttier_id\tvalue")?;
            for r in records {
                // ELAN の tier_id / 注釈値はタブや改行を含み得るため、
                // TSV を壊さないよう空白に置換してから出力する。
                writeln!(
                    writer,
                    "{}\t{}\t{}\t{}",
                    r.start_ms,
                    r.end_ms,
                    sanitize_tsv(&r.tier_id),
                    sanitize_tsv(&r.value)
                )?;
            }
        }
        Format::Json => {
            serde_json::to_writer_pretty(&mut writer, records)?;
            writeln!(writer)?;
        }
    }
    Ok(())
}

/// TSV を壊さないよう、タブ・改行・復帰を空白に置換する。
fn sanitize_tsv(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_tsv_replaces_separators() {
        assert_eq!(sanitize_tsv("a\tb\nc\rd"), "a b c d");
        assert_eq!(sanitize_tsv("plain"), "plain");
        assert_eq!(sanitize_tsv("multi\n\nline"), "multi  line");
    }
}
