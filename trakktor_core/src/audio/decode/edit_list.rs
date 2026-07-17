//! Minimal MP4 edit-list reader for gapless playback.
//!
//! The mp4 demuxer of symphonia parses `elst` atoms but does not apply them
//! to the packet stream, so AAC priming samples (the encoder delay a
//! reference demuxer/decoder pair strips — "skip N samples due to side
//! data") would leak into the output. This walker extracts just enough to
//! reproduce that trim: the `media_time` of the first non-empty edit of the
//! first audio track, converted to sample frames via the track's `mdhd`
//! timescale.
//!
//! Anything unusual (no `moov`, no edit list, multiple edits, an empty
//! leading edit) yields `None` — the stream then plays untrimmed, which is
//! also what the reference does for containers without priming metadata.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

/// Returns the number of leading frames to drop, if the file is an MP4 with
/// a plain leading edit.
pub(super) fn start_trim_frames(path: &Path, sample_rate: u32) -> Option<u64> {
    let mut file = File::open(path).ok()?;
    let len = file.seek(SeekFrom::End(0)).ok()?;
    file.seek(SeekFrom::Start(0)).ok()?;
    let mut walker = Walker { file, end: len };
    // The file must lead with an ftyp to count as MP4 at all.
    let (fourcc, _, header) = walker.peek_box(0)?;
    if &fourcc != b"ftyp" {
        return None;
    }
    let _ = header;

    let moov = walker.find_box(0, len, b"moov")?;
    let mut pos = moov.0;
    while let Some(trak) = walker.find_box(pos, moov.1, b"trak") {
        if let Some(trim) = walker.audio_trak_trim(trak.0, trak.1, sample_rate)
        {
            return Some(trim);
        }
        pos = trak.1;
    }
    None
}

struct Walker {
    file: File,
    end: u64,
}

impl Walker {
    /// Reads a box header at `pos`: (fourcc, payload start, payload end).
    fn peek_box(&mut self, pos: u64) -> Option<([u8; 4], u64, u64)> {
        if pos + 8 > self.end {
            return None;
        }
        self.file.seek(SeekFrom::Start(pos)).ok()?;
        let mut head = [0u8; 8];
        self.file.read_exact(&mut head).ok()?;
        let size32 = u32::from_be_bytes(head[0..4].try_into().unwrap());
        let fourcc: [u8; 4] = head[4..8].try_into().unwrap();
        let (payload, size) = if size32 == 1 {
            let mut large = [0u8; 8];
            self.file.read_exact(&mut large).ok()?;
            (pos + 16, u64::from_be_bytes(large))
        } else if size32 == 0 {
            (pos + 8, self.end - pos)
        } else {
            (pos + 8, u64::from(size32))
        };
        let end = pos.checked_add(size)?;
        if end > self.end || payload > end {
            return None;
        }
        Some((fourcc, payload, end))
    }

    /// Finds the first `name` box in `[pos, end)`: (payload start, box end).
    fn find_box(
        &mut self,
        mut pos: u64,
        end: u64,
        name: &[u8; 4],
    ) -> Option<(u64, u64)> {
        while pos < end {
            let (fourcc, payload, box_end) = self.peek_box(pos)?;
            if &fourcc == name {
                return Some((payload, box_end));
            }
            pos = box_end;
        }
        None
    }

    /// If the trak is an audio track with a plain leading edit, returns its
    /// start trim in frames.
    fn audio_trak_trim(
        &mut self,
        trak_pos: u64,
        trak_end: u64,
        sample_rate: u32,
    ) -> Option<u64> {
        let mdia = self.find_box(trak_pos, trak_end, b"mdia")?;
        let hdlr = self.find_box(mdia.0, mdia.1, b"hdlr")?;
        let mut buf = [0u8; 12];
        self.file.seek(SeekFrom::Start(hdlr.0)).ok()?;
        self.file.read_exact(&mut buf).ok()?;
        if &buf[8..12] != b"soun" {
            return None;
        }

        let mdhd = self.find_box(mdia.0, mdia.1, b"mdhd")?;
        self.file.seek(SeekFrom::Start(mdhd.0)).ok()?;
        let mut version = [0u8; 4];
        self.file.read_exact(&mut version).ok()?;
        let timescale = if version[0] == 1 {
            self.file.seek(SeekFrom::Current(16)).ok()?;
            let mut ts = [0u8; 4];
            self.file.read_exact(&mut ts).ok()?;
            u32::from_be_bytes(ts)
        } else {
            self.file.seek(SeekFrom::Current(8)).ok()?;
            let mut ts = [0u8; 4];
            self.file.read_exact(&mut ts).ok()?;
            u32::from_be_bytes(ts)
        };
        if timescale == 0 {
            return None;
        }

        let edts = self.find_box(trak_pos, trak_end, b"edts")?;
        let elst = self.find_box(edts.0, edts.1, b"elst")?;
        self.file.seek(SeekFrom::Start(elst.0)).ok()?;
        let mut head = [0u8; 8];
        self.file.read_exact(&mut head).ok()?;
        let version = head[0];
        let entry_count = u32::from_be_bytes(head[4..8].try_into().unwrap());
        if entry_count != 1 {
            // Multi-edit timelines are beyond this reader; play untrimmed.
            return None;
        }
        let media_time: i64 = if version == 1 {
            let mut entry = [0u8; 16];
            self.file.read_exact(&mut entry).ok()?;
            i64::from_be_bytes(entry[8..16].try_into().unwrap())
        } else {
            let mut entry = [0u8; 8];
            self.file.read_exact(&mut entry).ok()?;
            i64::from(i32::from_be_bytes(entry[4..8].try_into().unwrap()))
        };
        if media_time <= 0 {
            return None;
        }
        // media_time is in mdhd timescale units; convert to frames.
        let frames = if timescale == sample_rate {
            media_time as u64
        } else {
            (media_time as u128 * u128::from(sample_rate) /
                u128::from(timescale)) as u64
        };
        Some(frames)
    }
}
