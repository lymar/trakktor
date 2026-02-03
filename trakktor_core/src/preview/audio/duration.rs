use std::{io::Cursor, sync::Arc, time::Duration};

use anyhow::Context;
use symphonia::{
    core::{
        codecs::DecoderOptions, errors::Error, formats::FormatOptions,
        io::MediaSourceStream, meta::MetadataOptions, probe::Hint,
    },
    default::{get_codecs, get_probe},
};
pub fn audio_duration(
    data: &Arc<[u8]>,
    mime_type: &str,
) -> anyhow::Result<Duration> {
    let cur = Cursor::new(Arc::clone(data));
    let mss = MediaSourceStream::new(Box::new(cur), Default::default());

    let mut hint = Hint::new();
    hint.mime_type(mime_type);

    let probed = get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;

    let track = format.default_track().context("no default track")?;
    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let sample_rate =
        codec_params.sample_rate.context("unknown sample_rate")? as f64;

    // 1) Fast path: when n_frames is known.
    if let Some(n_frames) = codec_params.n_frames {
        let secs = (n_frames as f64) / sample_rate;
        return Ok(Duration::from_secs_f64(secs));
    }

    // 2) Reliable path: when n_frames is unknown, scan and count frames.
    let mut decoder =
        get_codecs().make(&codec_params, &DecoderOptions::default())?;

    let mut total_frames: u64 = 0;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(_)) => break, // EOF
            Err(e) => return Err(e.into()),
        };

        // If the container has multiple tracks, only process the desired one.
        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                // frames() is the number of samples per channel (exactly what
                // we need for duration).
                total_frames += audio_buf.frames() as u64;
            },
            // The decoder may occasionally error on corrupted frames — it is
            // usually safe to skip those.
            Err(Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        }
    }

    let secs = (total_frames as f64) / sample_rate;

    Ok(Duration::from_secs_f64(secs))
}
