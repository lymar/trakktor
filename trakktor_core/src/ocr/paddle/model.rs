//! The model catalog: which artifacts exist, where they come from, and which
//! language each recognizer covers.
//!
//! Every entry pins a **repository revision**, not a branch: the weights live
//! on a moving branch upstream, and a silent re-publish would change what
//! trakktor runs without changing a line of this repository. The sizes and
//! BLAKE3 digests are what the shared downloader verifies against.
//!
//! The catalog holds two detectors, one text-line orientation classifier and
//! twelve recognizers. The recognizers all share an architecture and differ
//! only in the alphabet they were trained on and, with it, the width of the
//! final projection — so covering a new script costs a catalog entry, not a
//! port.
//!
//! **A recognizer can only ever emit characters its own dictionary carries**,
//! and the dictionaries are not supersets of one another: the English one has
//! no en dash, the Latin one has none either but has `ß` and `ā`, the Eastern
//! Slavic one has the dash. A character outside the dictionary comes out
//! missing rather than wrong, which reads like a recognition failure but is
//! not one.

use crate::ocr::error::OcrError;

/// What a model does in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Detection,
    Recognition,
    Orientation,
}

/// One file of a published model.
#[derive(Debug, Clone, Copy)]
pub struct File {
    pub name: &'static str,
    pub size: u64,
    pub blake3: &'static str,
}

/// A published model.
#[derive(Debug, Clone, Copy)]
pub struct Model {
    /// Upstream's name, which is also the repository name and the directory
    /// the artifacts are cached under.
    pub name: &'static str,
    pub kind: Kind,
    /// The repository revision the digests below were taken at.
    pub revision: &'static str,
    pub files: &'static [File],
}

impl Model {
    /// Total download size, in bytes.
    pub fn size(&self) -> u64 { self.files.iter().map(|f| f.size).sum() }
}

/// The classic engine's default detector: small, fast, and script-independent
/// — it looks for text as such, so it finds lines in scripts no recognizer in
/// the catalog can read.
pub const DEFAULT_DETECTION: &str = "PP-OCRv5_mobile_det";

/// The other detector: the same job at nineteen times the size, and slower by
/// about as much.
///
/// What it buys is the class of line the small one drops — short lines, and
/// lines carrying a superscript. Upstream makes it the default for every
/// language but Chinese and Japanese; here it is the default of the generative
/// engine, where a missed line costs a whole block rather than one line, and an
/// option in the classic one.
pub const SERVER_DETECTION: &str = "PP-OCRv5_server_det";

/// The default text-line orientation classifier.
pub const DEFAULT_ORIENTATION: &str = "PP-LCNet_x1_0_textline_ori";

/// The default language when none is given.
pub const DEFAULT_LANGUAGE: &str = "ru";

/// Every model trakktor can run.
pub const MODELS: &[Model] = &[
    Model {
        name: "PP-OCRv5_mobile_det",
        kind: Kind::Detection,
        revision: "0d63e78e2b680928f6b1747d76a08db6e645efb7",
        // 4.9 MB
        files: &[
            File { name: "config.json", size: 2871, blake3: "e2a3aa2ffc6ca10a5afd876acd571d96817a0a4bf34be0672a849f7bc6d08715" },
            File { name: "inference.json", size: 229777, blake3: "c037fd8c9a08b92e9d162ff6af81bd3fa822e1fc32e71cd4e636871073adfd82" },
            File { name: "inference.pdiparams", size: 4692937, blake3: "c88bebcdf86c201f116a3ce254197ceaf69c3e23ca4d9b6de32ae066eec760e6" },
        ],
    },
    Model {
        name: "PP-OCRv5_server_det",
        kind: Kind::Detection,
        revision: "ca867c897ecbca8873081573a802ad70d499cb94",
        // 88.3 MB
        files: &[
            File { name: "config.json", size: 2871, blake3: "7d1b56cf253bef87222114f635f187a060c64e5461570b8907c6c1669ac510dd" },
            File { name: "inference.json", size: 402480, blake3: "b02f6b74bddb2d22687af3e4fa3a6199adb4d88a289d36833cc64bdd9f3d4e55" },
            File { name: "inference.pdiparams", size: 87932887, blake3: "2e0f1121cedc00d689cd48e9ec3839d1df637f284615dcfe22ed654f5fdd38f3" },
        ],
    },
    Model {
        name: "PP-LCNet_x1_0_textline_ori",
        kind: Kind::Orientation,
        revision: "cd237a44b0e359d4fe38310a416203cf7403faa5",
        // 6.9 MB
        files: &[
            File { name: "config.json", size: 2487, blake3: "3cb6180c751213c71ea787e036fb0d2bd66f4ebc266067f689320ee9e44bbb89" },
            File { name: "inference.json", size: 104399, blake3: "3e95836851f6fbe6c95fdcce3f14c0712b009db07493dceb7f9fc534fa368f40" },
            File { name: "inference.pdiparams", size: 6743918, blake3: "3c959e02c708768d3ca9e0f14b8716d9ca59f31ae7917cc75ffa1d8aab5fbd46" },
        ],
    },
    Model {
        name: "eslav_PP-OCRv5_mobile_rec",
        kind: Kind::Recognition,
        revision: "7553801264d3379d8d2e854971989e5e26c22e03",
        // 8.0 MB
        files: &[
            File { name: "config.json", size: 11920, blake3: "771cb0138d601976f2a99001eeaa8b21592037124d6acb1811bbdf4d704deab9" },
            File { name: "inference.json", size: 217712, blake3: "e78388054b8ebdd1eac3d0bd352b19cf0664d268d027ccf01c2adf26e7bcd481" },
            File { name: "inference.pdiparams", size: 7811519, blake3: "7c57941305c04e467ed7d7641a046386bb9e808303ae41a13c6b0720527acd1c" },
        ],
    },
    Model {
        name: "cyrillic_PP-OCRv5_mobile_rec",
        kind: Kind::Recognition,
        revision: "712d2d65556ccc1ea7b5d2bb232b018838b6a3ab",
        // 8.2 MB
        files: &[
            File { name: "config.json", size: 18036, blake3: "7f4d1b53daf4adc98c24feb1faea961dbea53769be0db915f1c0037a9cf91353" },
            File { name: "inference.json", size: 217712, blake3: "a12de5d7fc458bfdc13d4e2984389cbcfbb2f274d86a98d2d5e5e198d42e1201" },
            File { name: "inference.pdiparams", size: 7972691, blake3: "e93da6ba99961115cc09864288a05a520aac6fa074a53c7959f21cf59c061499" },
        ],
    },
    Model {
        name: "latin_PP-OCRv5_mobile_rec",
        kind: Kind::Recognition,
        revision: "ab2cd5cc5fa6309be2e5acdfe66eca2c2c127d57",
        // 8.2 MB
        files: &[
            File { name: "config.json", size: 17742, blake3: "801edd0229e651cfb3041ab2b6ac11cb9ac1d21ef23d1f89cc62a236dfac5409" },
            File { name: "inference.json", size: 217712, blake3: "41c0d85759e1d2a590ae4d1976ee3a68af7a9ab27f6e8c39de02a0422585f83d" },
            File { name: "inference.pdiparams", size: 7965915, blake3: "8da39434236fd27125df8a65fd8f1fea40319c051b3cd6d52892634bc0b4d717" },
        ],
    },
    Model {
        name: "en_PP-OCRv5_mobile_rec",
        kind: Kind::Recognition,
        revision: "267c36e24c331595590fe7bd72bde2436fd286f2",
        // 8.0 MB
        files: &[
            File { name: "config.json", size: 10455, blake3: "c9e4518e6c435de760ad6bc2c3cb4bca9807a77fce1c524c4788d164168cfed0" },
            File { name: "inference.json", size: 217712, blake3: "4f1fa8c72ef8cc5e51c1e12ee59cc26df2a65c50defe7872ced20eab4ab75477" },
            File { name: "inference.pdiparams", size: 7772315, blake3: "a1c6b3773c75e7484025e8bf17de85ef46e1e7c4309ecd5bf4e3b6862e12556f" },
        ],
    },
    Model {
        name: "arabic_PP-OCRv5_mobile_rec",
        kind: Kind::Recognition,
        revision: "33d91636a65dca87f5562cc48860332ae367ee1b",
        // 8.2 MB
        files: &[
            File { name: "config.json", size: 16077, blake3: "7160d511fc59c7baab6be4a96f9b8a09c35fc9a36774521a1b36e16f95cb4efa" },
            File { name: "inference.json", size: 217712, blake3: "5c4ef1b6a9bb671c1df4caea82057c0d95d328f009a444a9a838b65494d3b492" },
            File { name: "inference.pdiparams", size: 7922839, blake3: "2e5f05f21103ef283710f7cc23b258f0133ba08d808f3567d53a094d3ca69232" },
        ],
    },
    Model {
        name: "devanagari_PP-OCRv5_mobile_rec",
        kind: Kind::Recognition,
        revision: "99dcce6d196bd4aaf268c7a5c72c3cc9f3ea4932",
        // 8.1 MB
        files: &[
            File { name: "config.json", size: 12970, blake3: "ba2173acc403b31b6f5184606af4b69ebda41dd8a7e80a05c3694b5ace2925e4" },
            File { name: "inference.json", size: 217712, blake3: "6ee21be24240d603161aca56f6db1cc8e0889f22c3c5b1f23ca9333bc5ade5a7" },
            File { name: "inference.pdiparams", size: 7836203, blake3: "61fa5727d9a0651bbda4efcb6ffef147c3c9f88e0dd607adfac2df55f3d7784e" },
        ],
    },
    Model {
        name: "korean_PP-OCRv5_mobile_rec",
        kind: Kind::Recognition,
        revision: "c02ecaf1f22bfd1c618cce154fd19185b47e663a",
        // 13.8 MB
        files: &[
            File { name: "config.json", size: 229129, blake3: "3d92d568ca325d38bfe62d61b14a64e8c7abd96af02a3c1c1e33941362a12b54" },
            File { name: "inference.json", size: 217724, blake3: "7bb35e7199baa13d56c839015b6a73335f18f873da595e1666125ed0e5975663" },
            File { name: "inference.pdiparams", size: 13342671, blake3: "0634de9035e25e8ec7081ff034e6b2bc5cc74c0715a8efa0d9653be04a105981" },
        ],
    },
    Model {
        name: "th_PP-OCRv5_mobile_rec",
        kind: Kind::Recognition,
        revision: "9e25080455925d3943f5db885fddc79db6a07ca3",
        // 8.0 MB
        files: &[
            File { name: "config.json", size: 12126, blake3: "996cda5c7a522254248b2f8bb363c371391172918a611e2f3f5ede10be12a189" },
            File { name: "inference.json", size: 217712, blake3: "53ffb337e4e5ad43859359d3624e15793a86fb99e2539623d2079cf0542a556c" },
            File { name: "inference.pdiparams", size: 7814907, blake3: "935ae430ff63f8c8d2cf8130035dad03cf0a1f1076bab4d06dd933813821b061" },
        ],
    },
    Model {
        name: "el_PP-OCRv5_mobile_rec",
        kind: Kind::Recognition,
        revision: "b15bcf2467a03a0c079baa4b0347da51e89aaaba",
        // 8.0 MB
        files: &[
            File { name: "config.json", size: 8911, blake3: "db646bb8f23f5ac558658529aacb0ba57a5d60720ad5be8282c7ee0f7c7f2ec8" },
            File { name: "inference.json", size: 217712, blake3: "b88ecc978903335f4eed8a4c10d375cb42e604476c98ae2ab822afc13a6a539f" },
            File { name: "inference.pdiparams", size: 7732627, blake3: "9ef974949233303672de983dcb383512a40e72d2c9201dfe6618c52a8379bd76" },
        ],
    },
    Model {
        name: "te_PP-OCRv5_mobile_rec",
        kind: Kind::Recognition,
        revision: "151ab3b1c2f2a058f07a944416b92e9eaec6bf36",
        // 8.1 MB
        files: &[
            File { name: "config.json", size: 12430, blake3: "e0aa976c089963b6a804bc16914bdee600b235935c31904d5eab121230abae63" },
            File { name: "inference.json", size: 217712, blake3: "1ee9e9dc832dbaaa6f4c4aa487a75036a52492a6abce4ce4f6f5dba538718a4d" },
            File { name: "inference.pdiparams", size: 7822651, blake3: "36d4533038c6ab9e395be6b469fb34fd753455265fdf6220976d826211a00bed" },
        ],
    },
    Model {
        name: "ta_PP-OCRv5_mobile_rec",
        kind: Kind::Recognition,
        revision: "1bb164dad1d8eb23c7f7a382827e5305b37868d4",
        // 8.0 MB
        files: &[
            File { name: "config.json", size: 11917, blake3: "3120c61ab77308030897079d1640beed814be686a0de5535d1e592b9370e26b8" },
            File { name: "inference.json", size: 217712, blake3: "9a0ca9e63a1b41d6ff8b773021eac3834d2490595f95b8daecbf7ecfd9ba0958" },
            File { name: "inference.pdiparams", size: 7809583, blake3: "98837689fbb555be3116e26ce887bb070736c53e03acb13d7038f2c709b3cc28" },
        ],
    },
    Model {
        name: "PP-OCRv5_mobile_rec",
        kind: Kind::Recognition,
        revision: "682f20538d8c086cb2128e5cfac775e6c4904e85",
        // 17.0 MB
        files: &[
            File { name: "config.json", size: 352253, blake3: "830c103e977007d85d5d719809fb6454a7943106c45a493db856fe85c8de8982" },
            File { name: "inference.json", size: 217724, blake3: "e032c4bb50b3df61ccb0efd13c03a6156256363ed0164766939b35df6856aea4" },
            File { name: "inference.pdiparams", size: 16458665, blake3: "f0953764c8da3d40f7716b8853b8e194be9f12f12eae3d26b122ee2ba76e2781" },
        ],
    },
];

/// Language code to recognizer. The order matters where a language belongs to
/// two scripts: Eastern Slavic wins over the broader Cyrillic model for
/// Russian, Belarusian and Ukrainian (it is trained on exactly those three and
/// reads them better), and Kurdish is treated as Latin rather than Arabic —
/// both are upstream's own resolution order.
///
/// Codes are ISO 639-1 where one exists; the handful that upstream names
/// differently (`chinese_cht`) are accepted under both spellings.
pub const LANGUAGES: &[(&str, &str)] = &[
    // Eastern Slavic.
    ("ru", "eslav_PP-OCRv5_mobile_rec"),
    ("be", "eslav_PP-OCRv5_mobile_rec"),
    ("uk", "eslav_PP-OCRv5_mobile_rec"),
    // English has its own model; the Latin one covers the rest.
    ("en", "en_PP-OCRv5_mobile_rec"),
    ("af", "latin_PP-OCRv5_mobile_rec"),
    ("az", "latin_PP-OCRv5_mobile_rec"),
    ("bs", "latin_PP-OCRv5_mobile_rec"),
    ("ca", "latin_PP-OCRv5_mobile_rec"),
    ("cs", "latin_PP-OCRv5_mobile_rec"),
    ("cy", "latin_PP-OCRv5_mobile_rec"),
    ("da", "latin_PP-OCRv5_mobile_rec"),
    ("de", "latin_PP-OCRv5_mobile_rec"),
    ("es", "latin_PP-OCRv5_mobile_rec"),
    ("et", "latin_PP-OCRv5_mobile_rec"),
    ("eu", "latin_PP-OCRv5_mobile_rec"),
    ("fi", "latin_PP-OCRv5_mobile_rec"),
    ("fr", "latin_PP-OCRv5_mobile_rec"),
    ("ga", "latin_PP-OCRv5_mobile_rec"),
    ("gl", "latin_PP-OCRv5_mobile_rec"),
    ("hr", "latin_PP-OCRv5_mobile_rec"),
    ("hu", "latin_PP-OCRv5_mobile_rec"),
    ("id", "latin_PP-OCRv5_mobile_rec"),
    ("is", "latin_PP-OCRv5_mobile_rec"),
    ("it", "latin_PP-OCRv5_mobile_rec"),
    ("ku", "latin_PP-OCRv5_mobile_rec"),
    ("la", "latin_PP-OCRv5_mobile_rec"),
    ("lb", "latin_PP-OCRv5_mobile_rec"),
    ("lt", "latin_PP-OCRv5_mobile_rec"),
    ("lv", "latin_PP-OCRv5_mobile_rec"),
    ("mi", "latin_PP-OCRv5_mobile_rec"),
    ("ms", "latin_PP-OCRv5_mobile_rec"),
    ("mt", "latin_PP-OCRv5_mobile_rec"),
    ("nl", "latin_PP-OCRv5_mobile_rec"),
    ("no", "latin_PP-OCRv5_mobile_rec"),
    ("oc", "latin_PP-OCRv5_mobile_rec"),
    ("pl", "latin_PP-OCRv5_mobile_rec"),
    ("pt", "latin_PP-OCRv5_mobile_rec"),
    ("qu", "latin_PP-OCRv5_mobile_rec"),
    ("rm", "latin_PP-OCRv5_mobile_rec"),
    ("ro", "latin_PP-OCRv5_mobile_rec"),
    ("sk", "latin_PP-OCRv5_mobile_rec"),
    ("sl", "latin_PP-OCRv5_mobile_rec"),
    ("sq", "latin_PP-OCRv5_mobile_rec"),
    ("sv", "latin_PP-OCRv5_mobile_rec"),
    ("sw", "latin_PP-OCRv5_mobile_rec"),
    ("tl", "latin_PP-OCRv5_mobile_rec"),
    ("tr", "latin_PP-OCRv5_mobile_rec"),
    ("uz", "latin_PP-OCRv5_mobile_rec"),
    ("vi", "latin_PP-OCRv5_mobile_rec"),
    // Cyrillic beyond the Eastern Slavic three.
    ("ba", "cyrillic_PP-OCRv5_mobile_rec"),
    ("bg", "cyrillic_PP-OCRv5_mobile_rec"),
    ("ce", "cyrillic_PP-OCRv5_mobile_rec"),
    ("cv", "cyrillic_PP-OCRv5_mobile_rec"),
    ("kk", "cyrillic_PP-OCRv5_mobile_rec"),
    ("ky", "cyrillic_PP-OCRv5_mobile_rec"),
    ("mk", "cyrillic_PP-OCRv5_mobile_rec"),
    ("mn", "cyrillic_PP-OCRv5_mobile_rec"),
    ("os", "cyrillic_PP-OCRv5_mobile_rec"),
    ("sah", "cyrillic_PP-OCRv5_mobile_rec"),
    ("sr", "cyrillic_PP-OCRv5_mobile_rec"),
    ("tg", "cyrillic_PP-OCRv5_mobile_rec"),
    ("tt", "cyrillic_PP-OCRv5_mobile_rec"),
    ("tyv", "cyrillic_PP-OCRv5_mobile_rec"),
    ("udm", "cyrillic_PP-OCRv5_mobile_rec"),
    ("xal", "cyrillic_PP-OCRv5_mobile_rec"),
    // Arabic script.
    ("ar", "arabic_PP-OCRv5_mobile_rec"),
    ("fa", "arabic_PP-OCRv5_mobile_rec"),
    ("ps", "arabic_PP-OCRv5_mobile_rec"),
    ("sd", "arabic_PP-OCRv5_mobile_rec"),
    ("ug", "arabic_PP-OCRv5_mobile_rec"),
    ("ur", "arabic_PP-OCRv5_mobile_rec"),
    // Devanagari.
    ("bho", "devanagari_PP-OCRv5_mobile_rec"),
    ("hi", "devanagari_PP-OCRv5_mobile_rec"),
    ("mai", "devanagari_PP-OCRv5_mobile_rec"),
    ("mr", "devanagari_PP-OCRv5_mobile_rec"),
    ("ne", "devanagari_PP-OCRv5_mobile_rec"),
    ("sa", "devanagari_PP-OCRv5_mobile_rec"),
    // Scripts with a model of their own.
    ("el", "el_PP-OCRv5_mobile_rec"),
    ("ko", "korean_PP-OCRv5_mobile_rec"),
    ("ta", "ta_PP-OCRv5_mobile_rec"),
    ("te", "te_PP-OCRv5_mobile_rec"),
    ("th", "th_PP-OCRv5_mobile_rec"),
    // The Chinese/Japanese model also carries the Latin alphabet and digits.
    ("ja", "PP-OCRv5_mobile_rec"),
    ("zh", "PP-OCRv5_mobile_rec"),
    ("chinese_cht", "PP-OCRv5_mobile_rec"),
];

/// Looks a model up by name.
pub fn model(name: &str) -> Result<&'static Model, OcrError> {
    MODELS.iter().find(|m| m.name == name).ok_or_else(|| {
        OcrError::UnknownModel {
            name: name.to_string(),
            known: MODELS.iter().map(|m| m.name).collect::<Vec<_>>().join(", "),
        }
    })
}

/// Picks the recognizer for a language code.
pub fn recognizer_for(lang: &str) -> Result<&'static Model, OcrError> {
    let name = LANGUAGES
        .iter()
        .find(|(code, _)| *code == lang)
        .map(|(_, model)| *model)
        .ok_or_else(|| OcrError::UnsupportedLanguage {
            lang: lang.to_string(),
            known: languages().join(", "),
        })?;
    model(name)
}

/// Every language code the catalog covers, in the order it is declared.
pub fn languages() -> Vec<&'static str> {
    LANGUAGES.iter().map(|(code, _)| *code).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_language_names_a_catalogued_recognizer() {
        for (code, name) in LANGUAGES {
            let model = model(name)
                .unwrap_or_else(|_| panic!("`{code}` names unknown `{name}`"));
            assert_eq!(model.kind, Kind::Recognition, "{name}");
        }
    }

    #[test]
    fn the_defaults_are_in_the_catalog() {
        assert_eq!(model(DEFAULT_DETECTION).unwrap().kind, Kind::Detection);
        assert_eq!(model(SERVER_DETECTION).unwrap().kind, Kind::Detection);
        assert_eq!(model(DEFAULT_ORIENTATION).unwrap().kind, Kind::Orientation);
        assert!(recognizer_for(DEFAULT_LANGUAGE).is_ok());
    }

    #[test]
    fn language_codes_are_unique() {
        let mut seen: Vec<&str> = LANGUAGES.iter().map(|(c, _)| *c).collect();
        seen.sort_unstable();
        let count = seen.len();
        seen.dedup();
        assert_eq!(seen.len(), count, "duplicate language code");
    }

    #[test]
    fn an_unknown_language_lists_the_known_ones() {
        let error = recognizer_for("tlh").unwrap_err();
        let message = error.to_string();
        assert!(message.contains("tlh"));
        assert!(message.contains("ru"));
    }

    #[test]
    fn every_model_carries_the_three_published_files() {
        for model in MODELS {
            let names: Vec<&str> = model.files.iter().map(|f| f.name).collect();
            assert!(names.contains(&"config.json"), "{}", model.name);
            assert!(names.contains(&"inference.json"), "{}", model.name);
            assert!(names.contains(&"inference.pdiparams"), "{}", model.name);
            assert_eq!(model.revision.len(), 40, "{}", model.name);
        }
    }
}
