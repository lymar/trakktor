use candle_core::{Device, Tensor};

use super::*;

fn tensor(data: Vec<f32>, shape: (usize, usize, usize, usize)) -> Tensor {
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

#[test]
fn reflection_mirrors_without_repeating_the_edge() {
    // Five rows of 0..5, padded by two: the pad reads inward from the border
    // and never doubles it, along both axes.
    let row: Vec<f32> = (0..5).map(|v| v as f32).collect();
    let x = tensor(row.repeat(5), (1, 1, 5, 5));
    let padded = reflect_pad(&x, 2).unwrap();
    assert_eq!(padded.dims(), &[1, 1, 9, 9]);

    let mirrored = vec![2., 1., 0., 1., 2., 3., 4., 3., 2.];
    // Every row carries the columns reflected...
    assert_eq!(
        padded.i((0, 0, 0)).unwrap().to_vec1::<f32>().unwrap(),
        mirrored
    );
    // ...and the rows, all identical here, are reflected the same way, so the
    // padded rows are the padded row.
    assert_eq!(
        padded.i((0, 0, 8)).unwrap().to_vec1::<f32>().unwrap(),
        mirrored
    );
}

#[test]
fn reflection_along_the_rows_reads_inward_too() {
    // Columns that all count down the rows, so the row reflection shows and
    // the column reflection cannot be mistaken for it.
    let column: Vec<f32> = (0..5)
        .flat_map(|y| std::iter::repeat_n(y as f32, 5))
        .collect();
    let x = tensor(column, (1, 1, 5, 5));
    let padded = reflect_pad(&x, 2).unwrap();
    assert_eq!(padded.dims(), &[1, 1, 9, 9]);
    let down: Vec<f32> = (0..9)
        .map(|y| padded.i((0, 0, y, 0)).unwrap().to_vec0::<f32>().unwrap())
        .collect();
    assert_eq!(down, vec![2., 1., 0., 1., 2., 3., 4., 3., 2.]);
}

#[test]
fn reflection_of_nothing_is_the_same_tensor() {
    let x = tensor(vec![1., 2., 3., 4.], (1, 1, 2, 2));
    let same = reflect_pad(&x, 0).unwrap();
    assert_eq!(
        same.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        vec![1., 2., 3., 4.]
    );
}

#[test]
fn reflection_wider_than_the_axis_is_refused() {
    let x = tensor(vec![1., 2.], (1, 1, 1, 2));
    assert!(reflect_pad(&x, 3).is_err());
}

#[test]
fn the_activation_bends_only_the_negative_half() {
    let x = tensor(vec![-2., -1., 0., 1., 2.], (1, 1, 1, 5));
    let weight = Tensor::from_vec(vec![0.25f32], 1, &Device::Cpu).unwrap();
    let y = prelu(&x, &weight).unwrap();
    assert_eq!(
        y.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        vec![-0.5, -0.25, 0., 1., 2.]
    );
}

use candle_core::IndexOp;
