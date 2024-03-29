use std::{fs::File, io::Write, path::Path};

#[derive(Debug, Clone)]
pub struct DecodingResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    pub temperature: f64,
    pub compression_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub start: f64,
    pub duration: f64,
    pub dr: DecodingResult,
}

pub trait SpeechRecognitionOutputProvider {
    fn start(&mut self) -> anyhow::Result<()>;
    fn add_segment(&mut self, s: Segment) -> anyhow::Result<()>;
    fn finish(&mut self) -> anyhow::Result<()>;
}

#[derive(Debug, Clone, Copy)]
pub enum TimestampFormat {
    Start,
    StartEnd,
}

impl TimestampFormat {
    pub fn format(&self, start: f64, duration: f64) -> String {
        fn hms(t: f64) -> (f64, f64, f64) {
            let h = (t / 3600.0).floor();
            let m = ((t - h * 3600.0) / 60.0).floor();
            let s = (t - h * 3600.0 - m * 60.0).floor();
            (h, m, s)
        }
        let s = hms(start);
        match self {
            TimestampFormat::Start => {
                format!("[{:02.0}:{:02.0}:{:02.0}]", s.0, s.1, s.2)
            },
            TimestampFormat::StartEnd => {
                let e = hms(start + duration);
                format!(
                    "[{:02.0}:{:02.0}:{:02.0}-{:02.0}:{:02.0}:{:02.0}]",
                    s.0, s.1, s.2, e.0, e.1, e.2
                )
            },
        }
    }
}

#[test]
fn test_timestamp_format() {
    assert_eq!("[00:00:00]", TimestampFormat::Start.format(0.111, 1.0));
    assert_eq!("[00:00:01]", TimestampFormat::Start.format(1.111, 1.0));
    assert_eq!("[00:01:00]", TimestampFormat::Start.format(60.0, 1.0));
    assert_eq!("[00:30:00]", TimestampFormat::Start.format(1800.0, 1.0));
    assert_eq!("[01:00:01]", TimestampFormat::Start.format(3601.0, 1.0));
    assert_eq!(
        "[00:00:00-00:00:01]",
        TimestampFormat::StartEnd.format(0.111, 1.0)
    );
    assert_eq!(
        "[00:00:00-01:00:01]",
        TimestampFormat::StartEnd.format(0.111, 3601.0)
    );
}

pub struct TimestampedTextOutputProvider {
    file: File,
    format: TimestampFormat,
}

impl TimestampedTextOutputProvider {
    pub fn new(
        file_name: impl AsRef<Path>,
        format: TimestampFormat,
    ) -> std::io::Result<Self> {
        Ok(Self {
            file: File::create(file_name)?,
            format,
        })
    }
}

impl SpeechRecognitionOutputProvider for TimestampedTextOutputProvider {
    fn start(&mut self) -> anyhow::Result<()> { Ok(()) }

    fn add_segment(&mut self, s: Segment) -> anyhow::Result<()> {
        writeln!(
            &mut self.file,
            "{} {}",
            self.format.format(s.start, s.duration),
            s.dr.text.trim()
        )?;
        self.file.flush()?;
        Ok(())
    }

    fn finish(&mut self) -> anyhow::Result<()> {
        self.file.flush()?;
        Ok(())
    }
}

pub struct TextOutputProvider {
    file: File,
}

impl TextOutputProvider {
    pub fn new(file_name: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self {
            file: File::create(file_name)?,
        })
    }
}

impl SpeechRecognitionOutputProvider for TextOutputProvider {
    fn start(&mut self) -> anyhow::Result<()> { Ok(()) }

    fn add_segment(&mut self, s: Segment) -> anyhow::Result<()> {
        writeln!(&mut self.file, "{}", s.dr.text.trim())?;
        self.file.flush()?;
        Ok(())
    }

    fn finish(&mut self) -> anyhow::Result<()> {
        self.file.flush()?;
        Ok(())
    }
}
