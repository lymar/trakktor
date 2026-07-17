use super::*;

#[test]
fn stereo_to_mono_is_exact_half() {
    let m = build_matrix(ChannelLayout::STEREO, ChannelLayout::MONO).unwrap();
    assert_eq!(m, vec![vec![0.5, 0.5]]);
}

#[test]
fn mono_to_stereo_is_sqrt_half() {
    let m = build_matrix(ChannelLayout::MONO, ChannelLayout::STEREO).unwrap();
    assert_eq!(m, vec![vec![FRAC_1_SQRT_2], vec![FRAC_1_SQRT_2]]);
}

#[test]
fn five_one_to_mono_drops_lfe_and_normalizes() {
    // 5.1(back): FL FR FC LFE BL BR.
    let m = build_matrix(
        ChannelLayout::default_for(6).unwrap(),
        ChannelLayout::MONO,
    )
    .unwrap();
    assert_eq!(m.len(), 1);
    let row = &m[0];
    assert_eq!(row.len(), 6);
    // LFE (index 3) is dropped with the default mix level of 0.
    assert_eq!(row[3], 0.0);
    // FL = FR, BL = BR, FC is the loudest; the row is normalized to sum 1.
    assert_eq!(row[0], row[1]);
    assert_eq!(row[4], row[5]);
    assert!(row[2] > row[0] && row[0] > row[4]);
    let sum: f64 = row.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12);
}

#[test]
fn default_layouts_match_reference_guesses() {
    assert_eq!(ChannelLayout::default_for(1).unwrap(), ChannelLayout::MONO);
    assert_eq!(
        ChannelLayout::default_for(2).unwrap(),
        ChannelLayout::STEREO
    );
    assert_eq!(ChannelLayout::default_for(3).unwrap().0, 0b1011); // 2.1
    assert_eq!(ChannelLayout::default_for(6).unwrap().0, 0x3F); // 5.1(back)
    assert_eq!(ChannelLayout::default_for(8).unwrap().0, 0x63F); // 7.1
    assert!(ChannelLayout::default_for(9).is_none());
}

#[test]
fn single_front_left_cleans_to_mono() {
    let one_fl = ChannelLayout(1 << 0);
    let m = build_matrix(one_fl, ChannelLayout::MONO).unwrap();
    assert_eq!(m, vec![vec![1.0]]); // identity after clean_layout
}

#[test]
fn apply_float_stereo_downmix() {
    let m = build_matrix(ChannelLayout::STEREO, ChannelLayout::MONO).unwrap();
    let mix = Rematrix::<f32>::new(&m);
    let l = [1.0f32, -0.5, 0.25];
    let r = [0.0f32, 0.5, 0.25];
    let out = mix.apply(&[&l, &r]);
    assert_eq!(out, vec![vec![0.5, 0.0, 0.25]]);
}

#[test]
fn apply_q15_stereo_downmix() {
    let m = build_matrix(ChannelLayout::STEREO, ChannelLayout::MONO).unwrap();
    let mix = Rematrix::<i16>::new(&m);
    let l = [32767i16, -32768, 1];
    let r = [32767i16, -32768, 0];
    let out = mix.apply(&[&l, &r]);
    // (16384·a + 16384·b + 16384) >> 15
    assert_eq!(out, vec![vec![32767, -32768, 1]]);
}

#[test]
fn identity_channel_copies_verbatim() {
    let m = build_matrix(ChannelLayout::MONO, ChannelLayout::MONO).unwrap();
    let mix = Rematrix::<f32>::new(&m);
    let x = [0.1f32, 0.2];
    assert_eq!(mix.apply(&[&x]), vec![vec![0.1, 0.2]]);
}
