use trakktor_core::asr::whisper::Segment;

use super::*;

/// A minimal segment carrying only the fields the writers read.
fn seg(id: usize, start: f64, end: f64, text: &str) -> Segment {
    Segment {
        id,
        seek: 0,
        start,
        end,
        text: text.to_string(),
        tokens: Vec::new(),
        words: Vec::new(),
        temperature: 0.0,
        avg_logprob: 0.0,
        compression_ratio: 0.0,
        no_speech_prob: 0.0,
    }
}

fn sample() -> Transcription {
    Transcription {
        text: " Hello world. Second one.".to_string(),
        segments: vec![
            seg(0, 0.0, 2.5, " Hello world."),
            seg(1, 2.5, 3661.75, " Second one."),
        ],
        language: "en".to_string(),
        duration: 3661.75,
    }
}

#[test]
fn format_timestamp_vtt_and_srt_markers() {
    // VTT: no hours field below an hour, dot marker, milliseconds rounded.
    assert_eq!(format_timestamp(2.5, false, '.'), "00:02.500");
    // SRT: hours always present, comma marker.
    assert_eq!(format_timestamp(2.5, true, ','), "00:00:02,500");
    // Past an hour the hours field appears even for VTT.
    assert_eq!(format_timestamp(3661.75, false, '.'), "01:01:01.750");
}

#[test]
fn txt_is_one_trimmed_segment_per_line() {
    assert_eq!(render_txt(&sample()), "Hello world.\nSecond one.\n");
}

#[test]
fn vtt_has_header_and_cues() {
    let vtt = render_vtt(&sample());
    assert!(vtt.starts_with("WEBVTT\n\n"));
    assert!(vtt.contains("00:00.000 --> 00:02.500\nHello world.\n\n"));
    assert!(vtt.contains("00:02.500 --> 01:01:01.750\nSecond one.\n\n"));
}

#[test]
fn srt_numbers_cues_from_one() {
    let srt = render_srt(&sample());
    assert!(
        srt.starts_with("1\n00:00:00,000 --> 00:00:02,500\nHello world.\n\n")
    );
    assert!(srt.contains("2\n00:00:02,500 --> 01:01:01,750\nSecond one.\n\n"));
}

#[test]
fn tsv_uses_integer_milliseconds() {
    let tsv = render_tsv(&sample());
    assert!(tsv.starts_with("start\tend\ttext\n"));
    assert!(tsv.contains("0\t2500\tHello world.\n"));
    assert!(tsv.contains("2500\t3661750\tSecond one.\n"));
}

#[test]
fn subtitle_text_neutralizes_arrows() {
    let t = Transcription {
        segments: vec![seg(0, 0.0, 1.0, "a --> b")],
        ..sample()
    };
    assert!(render_srt(&t).contains("a -> b"));
    assert!(!render_srt(&t).contains("a --> b"));
}

#[test]
fn all_writes_every_format_named_after_the_audio() {
    let dir = std::env::temp_dir()
        .join(format!("trakktor-writers-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let paths = write_outputs(
        &sample(),
        "tiny",
        std::path::Path::new("/audio/talk.mp3"),
        OutputFormatArg::All,
        &dir,
        false,
    )
    .expect("write_outputs");

    let names: Vec<String> = paths
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        ["talk.txt", "talk.vtt", "talk.srt", "talk.tsv", "talk.json"]
    );
    assert!(paths.iter().all(|p| p.exists()));
    // The json file is the full envelope.
    let json = std::fs::read_to_string(dir.join("talk.json")).unwrap();
    assert!(json.contains("\"engine\""));
    assert!(json.contains("\"model\":\"tiny\""));

    std::fs::remove_dir_all(&dir).unwrap();
}
