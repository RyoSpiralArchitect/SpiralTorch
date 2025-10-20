use st_tensor::{
    fractal::{FractalPatch, UringFractalScheduler},
    Tensor,
};
use st_vision::{CanvasProjector, FractalCanvas};

fn gradient_tensor(width: usize, height: usize) -> Tensor {
    let mut values = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            values.push(((x + y) as f32) / (width.max(height) as f32));
        }
    }
    Tensor::from_vec(height, width, values).expect("valid gradient tensor")
}

#[test]
fn canvas_projector_refreshes_rgba_and_vectors() {
    let scheduler = UringFractalScheduler::new(4).expect("scheduler");
    let mut projector = CanvasProjector::new(scheduler.clone(), 128, 128).expect("projector");

    let relation = gradient_tensor(128, 128);
    let patch = FractalPatch::new(relation, 1.0, 1.0, 0).expect("patch");
    scheduler.push(patch).expect("push patch");

    let (rgba, vectors) = projector
        .refresh_with_vectors()
        .expect("refresh with vectors");

    assert_eq!(rgba.len(), 128 * 128 * 4);
    assert_eq!(vectors.vectors().len(), 128 * 128);
    assert!(vectors
        .vectors()
        .iter()
        .all(|vector| vector.iter().all(|component| component.is_finite())));
}

#[test]
fn fractal_canvas_exposes_scheduler_and_refresh() {
    let mut canvas = FractalCanvas::new(8, 64, 64).expect("canvas");
    let relation = gradient_tensor(64, 64);
    canvas
        .push_patch(relation, 0.75, 0.5, 1)
        .expect("push patch");

    assert_eq!(canvas.width(), 64);
    assert_eq!(canvas.height(), 64);

    let (_, vectors) = canvas.refresh_with_vectors().expect("refresh with vectors");
    assert_eq!(vectors.vectors().len(), 64 * 64);
}
