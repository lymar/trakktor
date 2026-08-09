//! Packet-loss concealment, the detector half.
//!
//! There is no separate network for it. A call that lost packets arrives with
//! holes in it — runs of digital silence where 20 ms of speech should be — and
//! the pipeline treats each hole the way the encoder was pre-trained to treat a
//! masked span: the extractor's output at those frames is replaced by the
//! model's own learned mask embedding, and the transformer fills the gap from
//! the context on either side. So the whole of concealment, at inference time,
//! is deciding **which frames are holes**.
//!
//! The rule is the reference's: cut the window into 20 ms packets, and call a
//! packet lost when nearly all of its samples are at digital zero. Twenty
//! milliseconds at 16 kHz is 320 samples, which is exactly one encoder frame —
//! so a lost packet maps to one masked frame with no rounding anywhere.

use super::config::{HOP, PACKET_LOST_RATIO, PACKET_MS, PACKET_SILENCE};

/// Marks the frames that look like lost packets.
///
/// `samples` is one window at 16 kHz. The result has one flag per whole packet
/// in it — that is, per encoder frame — and is empty when the window is shorter
/// than a packet.
#[must_use]
pub fn lost_packets(samples: &[f32], sample_rate: u32) -> Vec<bool> {
    let packet = sample_rate as usize * PACKET_MS / 1000;
    if packet == 0 {
        return Vec::new();
    }
    samples
        .chunks_exact(packet)
        .map(|chunk| {
            let silent = chunk
                .iter()
                .filter(|value| value.abs() < PACKET_SILENCE)
                .count();
            silent as f32 / packet as f32 >= PACKET_LOST_RATIO
        })
        .collect()
}

/// The same, for the rate the pipeline works at, where a packet is a frame.
#[must_use]
pub fn lost_frames(samples: &[f32]) -> Vec<bool> {
    debug_assert_eq!(HOP, 16_000 * PACKET_MS / 1000);
    lost_packets(samples, 16_000)
}

#[cfg(test)]
mod tests;
