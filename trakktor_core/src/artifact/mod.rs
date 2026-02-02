use std::fmt;

use boa_engine::JsData;
use boa_gc::{Finalize, Trace};

#[derive(Trace, Finalize, bon::Builder, JsData)]
pub struct Artifact {
    data: Vec<u8>,
    mime: Option<String>,
    original_name: Option<String>,
}

impl fmt::Debug for Artifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bs = bytesize::ByteSize::b(self.data.len() as u64);
        f.debug_struct("Artifact")
            .field("data", &bs)
            .field("mime", &self.mime)
            .field("original_name", &self.original_name)
            .finish()
    }
}

// // src/artifact/types.rs
// pub enum ArtifactKind {
//     Image(ImageKind),
//     Audio(AudioKind),
//     Text(TextKind),
//     Binary, // на будущее
// }

// pub struct Artifact {
//     pub kind: ArtifactKind,
//     pub source: ArtifactSource, // path/bytes/etc
//     pub name: Option<String>,
//     pub mime: Option<String>,
// }

// pub enum ArtifactSource {
//     Path(std::path::PathBuf),
//     Bytes(std::sync::Arc<[u8]>),
//     // Url(String) // если понадобится
// }

// pub enum TextKind {
//     Plain,
//     Markdown,
//     Json,
// }
// pub enum ImageKind { Png, Jpeg, Webp, Unknown }
// pub enum AudioKind { Mp3, Wav, Ogg, Unknown }
