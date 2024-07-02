use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};

pub fn get_hash_value(data: impl AsRef<[u8]>) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(data.as_ref());
    let hash = hasher.finalize();
    URL_SAFE_NO_PAD.encode(&hash.as_bytes())
}
