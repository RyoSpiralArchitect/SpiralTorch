#[cfg(feature = "wgpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use st_backend_wgpu::transform::{
        CenterCropConfig, GeometryCommand, HorizontalFlipConfig, ImageGeometry, ResizeConfig,
        TransformDispatcher,
    };
    use std::time::Instant;

    let dispatcher = TransformDispatcher::new_default_gpu()?;
    let initial = ImageGeometry {
        channels: 3,
        height: 512,
        width: 512,
    };
    let resize = ResizeConfig {
        channels: 3,
        src_height: 512,
        src_width: 512,
        dst_height: 384,
        dst_width: 384,
    };
    let flip_before = HorizontalFlipConfig {
        channels: 3,
        height: 384,
        width: 384,
        apply: true,
    };
    let crop = CenterCropConfig {
        channels: 3,
        src_height: 384,
        src_width: 384,
        crop_height: 320,
        crop_width: 320,
    };
    let flip_after = HorizontalFlipConfig {
        channels: 3,
        height: 320,
        width: 320,
        apply: true,
    };
    let commands = [
        GeometryCommand::Resize(resize),
        GeometryCommand::HorizontalFlip(flip_before),
        GeometryCommand::CenterCrop(crop),
        GeometryCommand::HorizontalFlip(flip_after),
    ];
    let input: Vec<f32> = (0..3 * 512 * 512)
        .map(|index| ((index * 37 + 17) % 257) as f32 / 256.)
        .collect();

    let mut separate = Vec::new();
    let mut resident = Vec::new();
    for iteration in 0..9 {
        let mut outputs = [Vec::new(), Vec::new()];
        for offset in 0..2 {
            let route = (iteration + offset) % 2;
            let start = Instant::now();
            outputs[route] = if route == 0 {
                let resized = dispatcher.resize(&input, resize)?;
                let flipped = dispatcher.horizontal_flip(&resized, flip_before)?;
                let cropped = dispatcher.center_crop(&flipped, crop)?;
                dispatcher.horizontal_flip(&cropped, flip_after)?
            } else {
                dispatcher
                    .run_geometry_sequence(&input, initial, &commands)?
                    .0
            };
            if iteration >= 2 {
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
                if route == 0 {
                    separate.push(elapsed_ms);
                } else {
                    resident.push(elapsed_ms);
                }
            }
        }
        assert_eq!(outputs[0].len(), outputs[1].len());
        for (&left, &right) in outputs[0].iter().zip(&outputs[1]) {
            assert!(left.is_finite() && right.is_finite());
            assert!((left - right).abs() <= 1e-5, "{left} != {right}");
        }
    }
    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    println!("separate_ms={separate:?}");
    println!("resident_ms={resident:?}");
    println!("separate_median_ms={:.3}", median(&mut separate));
    println!("resident_median_ms={:.3}", median(&mut resident));
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("Build with --features wgpu to run the native transform benchmark");
}
