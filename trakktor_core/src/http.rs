//! HTTP client and policy shared across features.
//!
//! Implements the cross-cutting HTTP rules: allowed schemes, fixed User-Agent,
//! connect/request timeouts, redirect limit, transparent compression, and a
//! hard cap on the response body size. The parameters are hard-coded for now;
//! making them configurable is left for later.

use std::{io::Read, time::Duration};

/// User-Agent sent with every request: `trakktor/<version>`.
const USER_AGENT: &str = concat!("trakktor/", env!("CARGO_PKG_VERSION"));
/// Connection timeout.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Overall request timeout.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Maximum number of redirects to follow.
const MAX_REDIRECTS: usize = 10;
/// Maximum accepted response body size: 20 MiB.
const MAX_RESPONSE_BYTES: u64 = 20 * 1024 * 1024;

/// Transport- and protocol-level HTTP errors.
///
/// These map to the stable error codes `invalid_url`, `fetch_failed`,
/// `http_error`, and `too_large`; the mapping itself lives at the CLI
/// boundary.
#[derive(Debug, thiserror::Error)]
pub enum HttpError {
    /// The URL does not parse or uses a scheme other than `http`/`https`.
    #[error("invalid URL: {0}")]
    InvalidUrl(String),
    /// A network/transport failure (DNS, connection, timeout, TLS).
    ///
    /// The underlying cause is preserved as the error source; it may be a
    /// `reqwest::Error` or an [`std::io::Error`] from reading the body.
    #[error("request failed: {0}")]
    Fetch(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// The server returned a non-2xx status code.
    #[error("HTTP status {0}")]
    Status(u16),
    /// The response body exceeded [`MAX_RESPONSE_BYTES`].
    #[error("response exceeds maximum size of {max} bytes")]
    TooLarge {
        /// The configured limit, in bytes.
        max: u64,
    },
}

/// A blocking HTTP client configured with the shared HTTP policy.
pub struct HttpClient {
    inner: reqwest::blocking::Client,
}

impl HttpClient {
    /// Builds a client with the shared HTTP policy.
    ///
    /// # Errors
    ///
    /// Returns [`HttpError::Fetch`] if the underlying client cannot be built.
    pub fn new() -> Result<Self, HttpError> {
        let inner = reqwest::blocking::Client::builder()
            .user_agent(USER_AGENT)
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .redirect(reqwest::redirect::Policy::limited(MAX_REDIRECTS))
            .gzip(true)
            .deflate(true)
            .brotli(true)
            .build()
            .map_err(|e| HttpError::Fetch(Box::new(e)))?;
        Ok(Self { inner })
    }

    /// Fetches `url` and returns the response body, enforcing the shared HTTP
    /// policy.
    ///
    /// Redirects are followed up to the configured limit; the body is read
    /// with a hard size cap so a hostile server cannot exhaust memory.
    ///
    /// # Errors
    ///
    /// - [`HttpError::InvalidUrl`] if the scheme is not `http`/`https`.
    /// - [`HttpError::Fetch`] on transport failures.
    /// - [`HttpError::Status`] on a non-2xx response.
    /// - [`HttpError::TooLarge`] if the body exceeds the size cap.
    pub fn get_bytes(&self, url: &str) -> Result<Vec<u8>, HttpError> {
        validate_url(url)?;

        let mut resp = self
            .inner
            .get(url)
            .send()
            .map_err(|e| HttpError::Fetch(Box::new(e)))?;

        let status = resp.status();
        if !status.is_success() {
            return Err(HttpError::Status(status.as_u16()));
        }

        // Early rejection when the server advertises an oversized body.
        if let Some(len) = resp.content_length() &&
            len > MAX_RESPONSE_BYTES
        {
            return Err(HttpError::TooLarge {
                max: MAX_RESPONSE_BYTES,
            });
        }

        // Read at most MAX_RESPONSE_BYTES + 1 so we can detect an overrun
        // without buffering an unbounded amount of data.
        let mut buf = Vec::new();
        (&mut resp)
            .take(MAX_RESPONSE_BYTES + 1)
            .read_to_end(&mut buf)
            .map_err(|e| HttpError::Fetch(Box::new(e)))?;
        if buf.len() as u64 > MAX_RESPONSE_BYTES {
            return Err(HttpError::TooLarge {
                max: MAX_RESPONSE_BYTES,
            });
        }

        Ok(buf)
    }
}

/// Validates that `url` parses and uses an allowed scheme.
///
/// # Errors
///
/// Returns [`HttpError::InvalidUrl`] when the URL does not parse or its scheme
/// is neither `http` nor `https`.
pub fn validate_url(url: &str) -> Result<(), HttpError> {
    let parsed = url::Url::parse(url)
        .map_err(|_| HttpError::InvalidUrl(url.to_string()))?;
    match parsed.scheme() {
        "http" | "https" => Ok(()),
        other => Err(HttpError::InvalidUrl(format!(
            "unsupported scheme: {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_http_and_https() {
        assert!(validate_url("http://example.com/feed.xml").is_ok());
        assert!(validate_url("https://example.com/feed.xml").is_ok());
    }

    #[test]
    fn rejects_other_schemes() {
        assert!(matches!(
            validate_url("ftp://example.com/feed.xml"),
            Err(HttpError::InvalidUrl(_))
        ));
        assert!(matches!(
            validate_url("file:///etc/passwd"),
            Err(HttpError::InvalidUrl(_))
        ));
    }

    #[test]
    fn rejects_unparseable() {
        assert!(matches!(
            validate_url("not a url"),
            Err(HttpError::InvalidUrl(_))
        ));
    }
}
