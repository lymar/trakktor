use super::*;

#[test]
fn the_schedule_runs_from_zero_to_one_without_turning_back() {
    let times = schedule(16);
    assert_eq!(times.len(), 17);
    assert!((times[0] - 0.0).abs() < 1e-12);
    assert!((times[16] - 1.0).abs() < 1e-12);
    for pair in times.windows(2) {
        assert!(pair[1] > pair[0], "{pair:?} is not increasing");
    }
}

#[test]
fn half_the_trajectory_is_covered_in_the_first_quarter_of_the_time() {
    // The property the base is chosen for, checked on a grid where the quarter
    // point is a grid point.
    let times = schedule(4);
    assert!(
        (times[1] - 0.5).abs() < 1e-9,
        "the quarter point is at {}",
        times[1]
    );
}

#[test]
fn a_budget_buys_the_steps_the_method_can_pay_for() {
    assert_eq!(Method::Euler.steps(64), 64);
    assert_eq!(Method::Midpoint.steps(64), 32);
    assert_eq!(Method::Rk4.steps(64), 16);
    // The reference truncates rather than rounds up.
    assert_eq!(Method::Midpoint.steps(63), 31);
}

#[test]
fn a_single_evaluation_falls_back_to_the_only_method_that_fits() {
    assert_eq!(Method::Midpoint.resolve(1), Method::Euler);
    assert_eq!(Method::Rk4.resolve(1), Method::Euler);
    assert_eq!(Method::Midpoint.resolve(2), Method::Midpoint);
}
