use super::Synthesizer;

#[test]
fn a_supported_language_is_accepted_however_it_is_spelled() {
    for language in
        [None, Some("auto"), Some("russian"), Some("RU"), Some("rus")]
    {
        Synthesizer::check_language(language).expect("should accept");
    }
}

#[test]
fn another_language_is_refused_with_a_pointer_to_the_other_engine() {
    let error = Synthesizer::check_language(Some("english"))
        .expect_err("should refuse");
    let message = error.to_string();
    assert!(message.contains("Russian"), "{message}");
    assert!(message.contains("qwen3-tts"), "{message}");
}
