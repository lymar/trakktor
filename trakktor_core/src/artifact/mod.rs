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
