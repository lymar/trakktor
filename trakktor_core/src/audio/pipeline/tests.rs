use std::{
    io::{Read, Write},
    process::{Command, Stdio},
};

use super::*;

/// Deterministic full-entropy noise in (−0.5, 0.5) — keeps quantization
/// ties (where the reference's NEON and C paths disagree) out of the
/// comparison.
fn noise(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let v = ((state >> 40) as i64 - (1 << 23)) as f32 /
            f32::from(u16::MAX) /
            512.0;
        out.push(v);
    }
    out
}

fn planar_f32(channels: usize, frames: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..channels)
        .map(|ch| noise(frames, seed + ch as u64 * 7919))
        .collect()
}

#[test]
fn mono_s16_same_rate_is_identity() {
    let x: Vec<i16> = (-5000..5000).map(|v| v as i16).collect();
    let mut shaper =
        Shaper::new(16_000, 1, None, SampleFormat::S16, 16_000).unwrap();
    let mut out = shaper.feed(NativeBuf::S16(vec![x.clone()]));
    out.extend(shaper.finish());
    assert_eq!(out, x);
}

#[test]
fn stereo_s16_same_rate_downmixes_in_q15() {
    let l = vec![100i16, -100, 32767];
    let r = vec![300i16, -100, 32767];
    let mut shaper =
        Shaper::new(16_000, 2, None, SampleFormat::S16, 16_000).unwrap();
    let mut out = shaper.feed(NativeBuf::S16(vec![l, r]));
    out.extend(shaper.finish());
    // (16384·l + 16384·r + 16384) >> 15
    assert_eq!(out, vec![200, -100, 32767]);
}

#[test]
fn u8_input_resamples_on_the_integer_lane() {
    let x: Vec<u8> = noise(8_000, 3)
        .iter()
        .map(|v| ((v * 250.0) as i32 + 0x80) as u8)
        .collect();
    let mut shaper =
        Shaper::new(8_000, 1, None, SampleFormat::U8, 16_000).unwrap();
    let mut out = shaper.feed(NativeBuf::U8(vec![x]));
    out.extend(shaper.finish());
    assert_eq!(out.len(), 16_000);
}

/// Runs raw PCM through the local ffmpeg reference command.
fn ffmpeg_to_mono_s16(
    raw: &[u8],
    fmt: &str,
    channels: usize,
    in_rate: u32,
    out_rate: u32,
) -> Vec<i16> {
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-f", fmt, "-ac"])
        .arg(channels.to_string())
        .arg("-ar")
        .arg(in_rate.to_string())
        .args(["-i", "-", "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le"])
        .arg("-ar")
        .arg(out_rate.to_string())
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawning ffmpeg");
    let mut stdin = child.stdin.take().unwrap();
    let owned = raw.to_vec();
    let writer = std::thread::spawn(move || {
        stdin.write_all(&owned).expect("writing to ffmpeg");
    });
    let mut out = Vec::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_end(&mut out)
        .expect("reading");
    writer.join().unwrap();
    assert!(child.wait().unwrap().success());
    out.chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn interleave_bytes<T, F: Fn(&T) -> Vec<u8>>(
    planar: &[Vec<T>],
    enc: F,
) -> Vec<u8> {
    let frames = planar[0].len();
    let mut out = Vec::new();
    for i in 0..frames {
        for ch in planar {
            out.extend(enc(&ch[i]));
        }
    }
    out
}

fn run_ours(
    buf: NativeBuf,
    in_rate: u32,
    out_rate: u32,
    chunk_frames: usize,
) -> Vec<i16> {
    let channels = buf.channels();
    let format = buf.format();
    let mut shaper =
        Shaper::new(in_rate, channels, None, format, out_rate).unwrap();
    // Re-chunk to prove independence from the feeding pattern while
    // matching ffmpeg's one-shot output.
    let mut out = Vec::new();
    macro_rules! chunked {
        ($v:expr, $variant:ident) => {{
            let v = $v;
            let frames = v[0].len();
            let mut start = 0;
            while start < frames {
                let end = (start + chunk_frames).min(frames);
                let piece: Vec<Vec<_>> =
                    v.iter().map(|ch| ch[start..end].to_vec()).collect();
                out.extend(shaper.feed(NativeBuf::$variant(piece)));
                start = end;
            }
        }};
    }
    match buf {
        NativeBuf::U8(v) => chunked!(v, U8),
        NativeBuf::S16(v) => chunked!(v, S16),
        NativeBuf::S32(v) => chunked!(v, S32),
        NativeBuf::F32(v) => chunked!(v, F32),
        NativeBuf::F64(v) => chunked!(v, F64),
    }
    out.extend(shaper.finish());
    out
}

/// The full-pipeline differential against the pinned local ffmpeg: raw PCM
/// of every practical shape → mono s16 at 16 kHz, bit for bit.
#[test]
#[ignore = "requires ffmpeg on PATH; verifies against the local build"]
fn matches_ffmpeg_end_to_end() {
    let out_rate = 16_000u32;
    struct Case {
        name: &'static str,
        fmt: &'static str,
        channels: usize,
        rate: u32,
        make: fn(usize, usize) -> (NativeBuf, Vec<u8>),
    }
    fn f32_case(channels: usize, frames: usize) -> (NativeBuf, Vec<u8>) {
        let p = planar_f32(channels, frames, 42);
        let raw = interleave_bytes(&p, |v| v.to_le_bytes().to_vec());
        (NativeBuf::F32(p), raw)
    }
    fn f64_case(channels: usize, frames: usize) -> (NativeBuf, Vec<u8>) {
        let p: Vec<Vec<f64>> = planar_f32(channels, frames, 43)
            .into_iter()
            .map(|ch| ch.into_iter().map(f64::from).collect())
            .collect();
        let raw = interleave_bytes(&p, |v| v.to_le_bytes().to_vec());
        (NativeBuf::F64(p), raw)
    }
    fn s16_case(channels: usize, frames: usize) -> (NativeBuf, Vec<u8>) {
        let p: Vec<Vec<i16>> = planar_f32(channels, frames, 44)
            .into_iter()
            .map(|ch| ch.into_iter().map(|v| (v * 60000.0) as i16).collect())
            .collect();
        let raw = interleave_bytes(&p, |v| v.to_le_bytes().to_vec());
        (NativeBuf::S16(p), raw)
    }
    fn s32_case(channels: usize, frames: usize) -> (NativeBuf, Vec<u8>) {
        let p: Vec<Vec<i32>> = planar_f32(channels, frames, 45)
            .into_iter()
            .map(|ch| ch.into_iter().map(|v| (v * 4.0e9) as i32).collect())
            .collect();
        let raw = interleave_bytes(&p, |v| v.to_le_bytes().to_vec());
        (NativeBuf::S32(p), raw)
    }
    fn u8_case(channels: usize, frames: usize) -> (NativeBuf, Vec<u8>) {
        let p: Vec<Vec<u8>> = planar_f32(channels, frames, 46)
            .into_iter()
            .map(|ch| {
                ch.into_iter()
                    .map(|v| ((v * 250.0) as i32 + 0x80) as u8)
                    .collect()
            })
            .collect();
        let raw = interleave_bytes(&p, |v| vec![*v]);
        (NativeBuf::U8(p), raw)
    }

    let cases = [
        Case {
            name: "f32 stereo 44.1k",
            fmt: "f32le",
            channels: 2,
            rate: 44_100,
            make: f32_case,
        },
        Case {
            name: "f32 mono 44.1k",
            fmt: "f32le",
            channels: 1,
            rate: 44_100,
            make: f32_case,
        },
        Case {
            name: "f32 mono 8k (upsample)",
            fmt: "f32le",
            channels: 1,
            rate: 8_000,
            make: f32_case,
        },
        Case {
            name: "f32 stereo 16k (mix only)",
            fmt: "f32le",
            channels: 2,
            rate: 16_000,
            make: f32_case,
        },
        Case {
            name: "f32 2.1 48k",
            fmt: "f32le",
            channels: 3,
            rate: 48_000,
            make: f32_case,
        },
        Case {
            name: "f32 5.1 48k",
            fmt: "f32le",
            channels: 6,
            rate: 48_000,
            make: f32_case,
        },
        Case {
            name: "f32 7.1 44.1k",
            fmt: "f32le",
            channels: 8,
            rate: 44_100,
            make: f32_case,
        },
        Case {
            name: "s16 stereo 44.1k",
            fmt: "s16le",
            channels: 2,
            rate: 44_100,
            make: s16_case,
        },
        Case {
            name: "s16 stereo 16k (q15 mix)",
            fmt: "s16le",
            channels: 2,
            rate: 16_000,
            make: s16_case,
        },
        Case {
            name: "s16 mono 16k (copy)",
            fmt: "s16le",
            channels: 1,
            rate: 16_000,
            make: s16_case,
        },
        Case {
            name: "s16 mono 22.05k",
            fmt: "s16le",
            channels: 1,
            rate: 22_050,
            make: s16_case,
        },
        Case {
            name: "s32 stereo 48k",
            fmt: "s32le",
            channels: 2,
            rate: 48_000,
            make: s32_case,
        },
        Case {
            name: "u8 stereo 22.05k (int resample)",
            fmt: "u8",
            channels: 2,
            rate: 22_050,
            make: u8_case,
        },
        Case {
            name: "u8 mono 8k",
            fmt: "u8",
            channels: 1,
            rate: 8_000,
            make: u8_case,
        },
        Case {
            name: "f64 stereo 44.1k",
            fmt: "f64le",
            channels: 2,
            rate: 44_100,
            make: f64_case,
        },
    ];

    for case in &cases {
        let frames = case.rate as usize * 2 + 37;
        let (buf, raw) = (case.make)(case.channels, frames);
        let ours = run_ours(buf, case.rate, out_rate, 1152);
        let theirs = ffmpeg_to_mono_s16(
            &raw,
            case.fmt,
            case.channels,
            case.rate,
            out_rate,
        );
        assert_eq!(ours.len(), theirs.len(), "{}: length", case.name);
        let diff = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
        let max_delta = ours
            .iter()
            .zip(&theirs)
            .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
            .max()
            .unwrap_or(0);
        assert_eq!(
            diff,
            0,
            "{}: {diff}/{} samples differ (max |delta| = {max_delta})",
            case.name,
            ours.len()
        );
    }
}
