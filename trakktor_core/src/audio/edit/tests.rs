use super::{cut, split};
use crate::audio::{decode::DecodedAudio, pipeline::NativeBuf};

/// A one-channel s16 stream at `rate` Hz, so one second is `rate` frames.
fn mono_s16(samples: Vec<i16>, rate: u32) -> DecodedAudio {
    DecodedAudio::from_parts(
        rate,
        None,
        Some(16),
        NativeBuf::S16(vec![samples]),
    )
}

fn plane(audio: &DecodedAudio) -> Vec<i16> {
    match audio.samples() {
        NativeBuf::S16(planes) => planes[0].clone(),
        other => panic!("expected s16, got {:?}", other.format()),
    }
}

#[test]
fn cut_concatenates_ranges_verbatim_without_fade() {
    let audio = mono_s16((0..100).collect(), 100);
    let out = cut(&audio, &[(0.10, 0.30), (0.50, 0.60)], 0);
    assert_eq!(out.frames(), 30);
    let expected: Vec<i16> = (10..30).chain(50..60).collect();
    assert_eq!(plane(&out), expected);
}

#[test]
fn split_makes_one_clip_per_range() {
    let audio = mono_s16((0..100).collect(), 100);
    let clips = split(&audio, &[(0.10, 0.30), (0.50, 0.60)], 0);
    assert_eq!(clips.len(), 2);
    assert_eq!(clips[0].frames(), 20);
    assert_eq!(clips[1].frames(), 10);
    assert_eq!(plane(&clips[0]), (10..30).collect::<Vec<_>>());
    assert_eq!(plane(&clips[1]), (50..60).collect::<Vec<_>>());
}

#[test]
fn fade_attenuates_edges_and_keeps_the_interior() {
    // A constant signal, so every deviation from 1000 is the fade at work.
    let audio = mono_s16(vec![1000; 100], 100);
    // 100 ms at 100 Hz is a 10-frame ramp on each side.
    let out = cut(&audio, &[(0.0, 1.0)], 100);
    let samples = plane(&out);
    assert_eq!(samples.len(), 100);
    // The interior between the ramps is copied unchanged.
    assert!(samples[10..90].iter().all(|&v| v == 1000));
    // Both edges are pulled toward silence and ramp monotonically inward.
    assert!(samples[0] < samples[9] && samples[0] >= 0);
    assert!(samples[99] < samples[90] && samples[99] >= 0);
}

#[test]
fn empty_ranges_make_an_empty_buffer_of_the_same_shape() {
    let audio = mono_s16((0..100).collect(), 100);
    let out = cut(&audio, &[], 0);
    assert_eq!(out.frames(), 0);
    assert_eq!(out.channels(), 1);
}

#[test]
fn ranges_are_clamped_to_the_audio() {
    let audio = mono_s16((0..100).collect(), 100);
    // End past the audio is clamped to its length.
    let out = cut(&audio, &[(0.90, 5.0)], 0);
    assert_eq!(out.frames(), 10);
    assert_eq!(plane(&out), (90..100).collect::<Vec<_>>());
}
