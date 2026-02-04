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
use tokio_stream::StreamExt;

use crate::{logger::mute_log, preview::PreviewData};

mod duration;

pub async fn render_audio_preview(preview: PreviewData) -> anyhow::Result<()> {
    log::info!("Rendering audio preview");

    let _mute = mute_log();

    let mut stream_handle = rodio::OutputStreamBuilder::open_default_stream()?;
    stream_handle.log_on_drop(false);

    let sink = rodio::Sink::connect_new(stream_handle.mixer());

    let theme = ColorScheme::init();

    let terminal = ratatui::init();

    let total_duration =
        duration::audio_duration(&preview.data, &preview.mime.to_string()).ok();

    let res = AudioPreview {
        sink,
        should_close: false,
        audio_data: preview.data,
        mime: preview.mime,
        text: preview.text,
        total_duration,
        current_position: Duration::ZERO,
        text_scrollbar_state: Default::default(),
        text_wrap_cache: None,
        theme,
    }
    .run(terminal)
    .await;
    ratatui::restore();

    drop(_mute);

    log::info!("Audio preview stopped.");

    res
}

struct ColorScheme {
    text_color: Color,
    gauge_color: Color,
    gauge_text_color: Color,
    bottom_bar_color_1: Color,
    bottom_bar_color_2: Color,
}

impl ColorScheme {
    fn dark_mode() -> Self {
        Self {
            text_color: tailwind::SLATE.c200,
            gauge_color: tailwind::SKY.c900,
            gauge_text_color: tailwind::SKY.c200,
            bottom_bar_color_1: tailwind::INDIGO.c900,
            bottom_bar_color_2: tailwind::INDIGO.c200,
        }
    }

    fn light_mode() -> Self {
        Self {
            text_color: tailwind::SLATE.c900,
            gauge_color: tailwind::SKY.c300,
            gauge_text_color: tailwind::SKY.c800,
            bottom_bar_color_1: tailwind::INDIGO.c200,
            bottom_bar_color_2: tailwind::INDIGO.c800,
        }
    }

    fn init() -> Self {
        let theme = termbg::theme(Duration::from_millis(100))
            .unwrap_or(termbg::Theme::Dark);
        match theme {
            termbg::Theme::Dark => Self::dark_mode(),
            termbg::Theme::Light => Self::light_mode(),
        }
    }
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
    text_wrap_cache: Option<TextWrapCache>,
    theme: ColorScheme,
}

struct TextWrapCache {
    width: usize,
    lines_count: usize,
    joined_lines: String,
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

        self.sink_append()?;

        while !self.should_close {
            tokio::select! {
                _ = interval.tick() => {
                    if self.sink.empty() {
                        self.sink_append()?;
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

    fn sink_append(&mut self) -> anyhow::Result<()> {
        let cur = Cursor::new(Arc::clone(&self.audio_data));
        let decoder = rodio::Decoder::builder()
            .with_data(cur)
            .with_mime_type(&self.mime.to_string())
            .build()?;

        self.sink.append(decoder);
        Ok(())
    }

    fn restart(&mut self) -> anyhow::Result<()> {
        let is_paused = self.sink.is_paused();

        self.sink.stop();
        self.sink_append()?;

        if is_paused {
            self.sink.pause();
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
                KeyCode::Char('r') => {
                    if let Err(e) = self.restart() {
                        log::error!("Failed to restart audio: {}", e);
                    }
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
    fn get_text_wrap<'a>(
        &'a mut self,
        curr_width: usize,
    ) -> Option<&'a TextWrapCache> {
        let Some(text) = &self.text else {
            return None;
        };

        let cache_hit = self
            .text_wrap_cache
            .as_ref()
            .is_some_and(|c| c.width == curr_width);
        if cache_hit {
            return self.text_wrap_cache.as_ref();
        }

        let lines = textwrap::wrap(text, curr_width);
        let joined_lines = lines.join("\n");

        self.text_wrap_cache = Some(TextWrapCache {
            width: curr_width,
            lines_count: lines.len(),
            joined_lines,
        });

        self.text_wrap_cache.as_ref()
    }

    fn render_text(&mut self, area: Rect, buf: &mut Buffer) {
        use Constraint::{Length, Min};

        let layout =
            Layout::horizontal([Length(1), Min(1), Length(1), Length(1)])
                .vertical_margin(1);
        let [_, text_area, _, scroll_area] = layout.areas(area);

        let text_width = min(text_area.width, 80) as usize;

        let wrap_width = min(area.width as usize, text_width);
        let Some(lines_count) =
            self.get_text_wrap(wrap_width).map(|c| c.lines_count)
        else {
            return;
        };

        let content_length =
            (lines_count as u16).saturating_sub(text_area.height) as usize;

        self.text_scrollbar_state =
            self.text_scrollbar_state.content_length(content_length + 1);
        if self.text_scrollbar_state.get_position() > content_length {
            self.text_scrollbar_state =
                self.text_scrollbar_state.position(content_length);
        }

        Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .symbols(scrollbar::VERTICAL)
            .render(scroll_area, buf, &mut self.text_scrollbar_state);

        let text_centered_area = text_area.centered_horizontally(Length(80));

        let scroll_pos = self.text_scrollbar_state.get_position() as u16;

        let text_color = self.theme.text_color;
        let joined_lines = {
            let Some(txt_cache) = self.get_text_wrap(wrap_width) else {
                return;
            };
            txt_cache.joined_lines.clone()
        };

        Paragraph::new(joined_lines)
            .alignment(Alignment::Left)
            .fg(text_color)
            .scroll((scroll_pos, 0))
            .render(text_centered_area, buf);
    }

    fn render_bottom_bar(&self, area: Rect, buf: &mut Buffer) {
        let space_icon = if self.sink.is_paused() { "▶" } else { "⏸" };

        let keys = [
            ("K/↑", "Scroll up"),
            ("J/↓", "Scroll down"),
            ("Space", space_icon),
            ("R", "Restart"),
            ("X/Esc", "Close"),
        ];
        let spans = keys
            .iter()
            .flat_map(|(key, desc)| {
                let key = Span::styled(
                    format!(" {key} "),
                    Style::new()
                        .fg(self.theme.bottom_bar_color_1)
                        .bg(self.theme.bottom_bar_color_2),
                );
                let desc = Span::styled(
                    format!(" {desc} "),
                    Style::new()
                        .fg(self.theme.bottom_bar_color_2)
                        .bg(self.theme.bottom_bar_color_1),
                );
                [key, desc]
            })
            .collect::<Vec<_>>();
        Line::from(spans).centered().render(area, buf);
    }

    fn render_gauge(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::bordered()
            .padding(Padding::new(1, 1, 0, 0))
            .border_style(Style::new().fg(self.theme.gauge_text_color));

        let total = self.total_duration.unwrap_or(self.current_position);
        let with_minutes = total.as_secs() >= 60;

        let label = Span::styled(
            format!(
                "{} / {}",
                format_duration(self.current_position, with_minutes),
                format_duration(total, with_minutes)
            ),
            Style::new().italic().bold().fg(self.theme.gauge_text_color),
        );
        let mut ratio =
            self.current_position.as_secs_f64() / total.as_secs_f64();
        if ratio > 1.0 || ratio < 0.0 || ratio.is_nan() {
            ratio = 0.0;
        }

        Gauge::default()
            .gauge_style(self.theme.gauge_color)
            .ratio(ratio)
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
        return format!("{:02}:{:02}.{:03}", minutes, seconds, millis);
    } else {
        return format!("{}.{:03}", seconds, millis);
    }
}
