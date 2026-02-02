use std::{cmp::min, io::Cursor, sync::Arc, time::Duration};

use crossterm::event::{Event, EventStream, KeyCode};
use mime_guess::Mime;
use ratatui::{
    DefaultTerminal,
    buffer::Buffer,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Style, Stylize, palette::tailwind},
    symbols::scrollbar,
    text::{Line, Span},
    widgets::{
        Block, Gauge, Padding, Paragraph, ScrollDirection, Scrollbar,
        ScrollbarOrientation, ScrollbarState, StatefulWidget, Widget,
    },
};
use rodio::Source;
use tokio_stream::StreamExt;

use crate::{logger::mute_log, preview::PreviewData};

const CUSTOM_LABEL_COLOR: Color = tailwind::SLATE.c200;
const GAUGE2_COLOR: Color = tailwind::INDIGO.c800;

pub async fn render_audio_preview(preview: PreviewData) -> anyhow::Result<()> {
    log::info!("Rendering audio preview");

    let _mute = mute_log();

    let mut stream_handle = rodio::OutputStreamBuilder::open_default_stream()?;
    stream_handle.log_on_drop(false);

    let sink = rodio::Sink::connect_new(stream_handle.mixer());

    let terminal = ratatui::init();

    let res = AudioPreview {
        sink,
        should_close: false,
        audio_data: preview.data,
        mime: preview.mime,
        text: preview.text,
        total_duration: None,
        current_position: Duration::ZERO,
        text_scrollbar_state: Default::default(),
    }
    .run(terminal)
    .await;
    ratatui::restore();

    drop(_mute);

    log::info!("Audio preview stopped.");

    res
}

struct AudioPreview {
    sink: rodio::Sink,
    should_close: bool,
    audio_data: Arc<[u8]>,
    mime: Mime,
    text: Option<String>,
    total_duration: Option<Duration>,
    current_position: Duration,
    text_scrollbar_state: ScrollbarState,
}

impl AudioPreview {
    const FRAMES_PER_SECOND: f32 = 30.0;

    pub async fn run(
        mut self,
        mut terminal: DefaultTerminal,
    ) -> anyhow::Result<()> {
        let period = Duration::from_secs_f32(1.0 / Self::FRAMES_PER_SECOND);
        let mut interval = tokio::time::interval(period);
        let mut events = EventStream::new();

        // self.sink.set_volume(0.0);

        let mime = self.mime.to_string();
        {
            let cur = Cursor::new(Arc::clone(&self.audio_data));
            let decoder = rodio::Decoder::builder()
                .with_data(cur)
                .with_mime_type(&mime)
                .build()?;
            self.total_duration = decoder.total_duration();
            self.sink.append(decoder);
        }

        while !self.should_close {
            tokio::select! {
                _ = interval.tick() => {
                    if self.sink.empty() {
                        let cur = Cursor::new(Arc::clone(&self.audio_data));
                        let decoder = rodio::Decoder::builder()
                            .with_data(cur)
                            .with_mime_type(&mime)
                            .build()?;
                        self.sink.append(decoder);
                    }

                    self.current_position = self.sink.get_pos();
                    terminal.draw(|frame|
                        frame.render_widget(&mut self, frame.area()))?;
                },
                Some(Ok(event)) = events.next() => self.handle_event(&event),
            }
        }
        Ok(())
    }

    fn handle_event(&mut self, event: &Event) {
        if let Some(key) = event.as_key_press_event() {
            match key.code {
                KeyCode::Char('x') | KeyCode::Esc => self.should_close = true,
                KeyCode::Char('j') | KeyCode::Down => {
                    self.text_scrollbar_state.scroll(ScrollDirection::Forward)
                },
                KeyCode::Char('k') | KeyCode::Up => {
                    self.text_scrollbar_state.scroll(ScrollDirection::Backward)
                },
                KeyCode::Char(' ') => {
                    if self.sink.is_paused() {
                        self.sink.play();
                    } else {
                        self.sink.pause();
                    }
                },
                _ => {},
            }
        }
    }
}

impl Widget for &mut AudioPreview {
    #[allow(clippy::similar_names)]
    fn render(self, area: Rect, buf: &mut Buffer) {
        use Constraint::{Length, Min, Percentage};

        if self.text.is_none() {
            let layout = Layout::vertical([
                Percentage(100 - 61),
                Length(5),
                Percentage(61),
                Length(1),
            ]);
            let [_, gauge_area, _, footer_area] = layout.areas(area);

            self.render_gauge(gauge_area, buf);
            self.render_bottom_bar(footer_area, buf);
        } else {
            let layout = Layout::vertical([Min(1), Length(5), Length(1)]);
            let [header_area, gauge_area, footer_area] = layout.areas(area);

            self.render_text(header_area, buf);

            self.render_gauge(gauge_area, buf);
            self.render_bottom_bar(footer_area, buf);
        }
    }
}

impl AudioPreview {
    fn render_text(&mut self, area: Rect, buf: &mut Buffer) {
        use Constraint::{Length, Min};

        let Some(text) = &self.text else {
            return;
        };

        let layout = Layout::horizontal([Min(1), Length(1)]).vertical_margin(1);
        let [text_area, scroll_area] = layout.areas(area);

        let text_width = min(text_area.width, 80) as usize;

        let lines = textwrap::wrap(text, text_width);

        let content_length =
            (lines.len() as u16).saturating_sub(text_area.height) as usize;

        self.text_scrollbar_state =
            self.text_scrollbar_state.content_length(content_length + 1);
        if self.text_scrollbar_state.get_position() > content_length {
            self.text_scrollbar_state =
                self.text_scrollbar_state.position(content_length);
        }

        // todo: cache !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        let lines = lines.join("\n");

        Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .symbols(scrollbar::VERTICAL)
            .render(scroll_area, buf, &mut self.text_scrollbar_state);

        let text_centered_area = text_area.centered_horizontally(Length(80));

        Paragraph::new(lines)
            .bold()
            .alignment(Alignment::Left)
            .fg(CUSTOM_LABEL_COLOR)
            .scroll((self.text_scrollbar_state.get_position() as u16, 0))
            .render(text_centered_area, buf);
    }

    fn render_bottom_bar(&self, area: Rect, buf: &mut Buffer) {
        let space_icon = if self.sink.is_paused() { "▶" } else { "⏸" };

        let keys = [
            ("K/↑", "Scroll up"),
            ("J/↓", "Scroll down"),
            ("Space", space_icon),
            ("X/Esc", "Close"),
        ];
        let spans = keys
            .iter()
            .flat_map(|(key, desc)| {
                let key = Span::styled(
                    format!(" {key} "),
                    Style::new().fg(tailwind::BLACK).bg(tailwind::GRAY.c300),
                );
                let desc = Span::styled(
                    format!(" {desc} "),
                    Style::new().fg(tailwind::GRAY.c300).bg(tailwind::BLACK),
                );
                [key, desc]
            })
            .collect::<Vec<_>>();
        Line::from(spans).centered().render(area, buf);
    }

    fn render_gauge(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::bordered().padding(Padding::new(1, 1, 0, 0));

        let total = self.total_duration.unwrap_or(self.current_position);
        let with_minutes = total.as_secs() >= 60;

        let label = Span::styled(
            format!(
                "{} / {}",
                format_duration(self.current_position, with_minutes),
                format_duration(total, with_minutes)
            ),
            Style::new().italic().bold().fg(CUSTOM_LABEL_COLOR),
        );
        Gauge::default()
            .gauge_style(GAUGE2_COLOR)
            .ratio(self.current_position.as_secs_f64() / total.as_secs_f64())
            .label(label)
            .block(block)
            .render(area, buf);
    }
}

fn format_duration(d: Duration, with_minutes: bool) -> String {
    let total_ms = d.as_millis();

    let minutes = total_ms / 60_000;
    let seconds = (total_ms / 1_000) % 60;
    let millis = total_ms % 1_000;

    if with_minutes {
        return format!("{:02}:{:02}:{:03}", minutes, seconds, millis);
    } else {
        return format!("{}.{:03}", seconds, millis);
    }
}
