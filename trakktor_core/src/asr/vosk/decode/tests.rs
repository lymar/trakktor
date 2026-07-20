use super::{argmax, log_add, log_softmax};

#[test]
fn log_add_basics() {
    // log(e^0 + e^0) = ln 2.
    assert!((log_add(0.0, 0.0) - 2.0f64.ln()).abs() < 1e-12);
    // Adding -inf is the identity.
    assert_eq!(log_add(f64::NEG_INFINITY, -1.5), -1.5);
    assert_eq!(log_add(-1.5, f64::NEG_INFINITY), -1.5);
    // Symmetric.
    assert_eq!(log_add(-3.0, -1.0), log_add(-1.0, -3.0));
}

#[test]
fn log_softmax_normalizes() {
    let mut row = [1.0f32, 2.0, 3.0];
    log_softmax(&mut row);
    let sum: f32 = row.iter().map(|v| v.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-6);
    // Order preserved.
    assert!(row[0] < row[1] && row[1] < row[2]);
}

#[test]
fn argmax_first_on_ties() {
    assert_eq!(argmax(&[0.5, 2.0, 2.0, 1.0]), 1);
    assert_eq!(argmax(&[3.0]), 0);
}
