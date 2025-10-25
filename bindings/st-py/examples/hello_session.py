# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.
"""Showcase the high-level SpiralTorch Python bindings.

This example keeps everything self-contained and exercises the features that are
currently implemented in the lightweight bindings:

* `SpiralSession` describes device backends and produces rank plans, including
  SpiralK FFT helpers, HIP probing, and plan batch synthesis utilities.
* `ModuleTrainer` performs a tiny linear regression while demonstrating pacing
  helpers, N-accis packers, and loss metrics such as masked MSE and InfoNCE.
* Tensor utilities cover barycenters, biomes, temporal resonance buffers, and
  hyperbolic tensor algebra features like hypergrads and row-softmax helpers.
* Extras such as the golden ratio helpers, LanguageWave encoders, and tensor
  capture utilities are all demonstrated so the script doubles as a smoke-test
  when run under CI.
"""

from __future__ import annotations

from pprint import pprint

import spiraltorch as st


def banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def describe_backend(session: st.SpiralSession) -> None:
    banner("Device description & rank planning")
    print(f"Selected backend: {session.backend} (device label: {session.device})")
    device_info = st.describe_device(session.backend, lane_width=32, subgroup=True)
    print("Device capabilities snapshot:")
    pprint(device_info)

    plan = session.plan_topk(rows=128, cols=256, k=32)
    print("\nRank-plan summary:")
    print(
        "kind=", plan.kind,
        "merge=", plan.merge_strategy,
        "detail=", plan.merge_detail,
        "lanes=", plan.lanes,
        "workgroup=", plan.workgroup,
        "fft_tile=", plan.fft_tile,
    )
    print("Latency window:", plan.latency_window())
    print("SpiralK hint:\n", plan.fft_spiralk_hint())

    hip_info = st.hip_probe()
    print("\nHIP probe:")
    pprint(hip_info)

    fft_plan = st.plan(
        kind="fft",
        rows=256,
        cols=256,
        k=64,
        backend=session.backend,
        lane_width=32,
        subgroup=True,
    )
    print("\nFFT plan latency window:", fft_plan.latency_window())
    spiralk_fft = st.SpiralKFftPlan.from_rank_plan(fft_plan)
    print(
        "SpiralKFftPlan radix:",
        getattr(spiralk_fft, "radix", lambda: None)()
        if callable(getattr(spiralk_fft, "radix", None))
        else spiralk_fft.radix,
    )
    print("SpiralKFftPlan workgroup size:", spiralk_fft.workgroup_size())
    wgsl_fn = getattr(spiralk_fft, "wgsl", None) or getattr(spiralk_fft, "emit_wgsl", None)
    if wgsl_fn:
        wgsl_src = wgsl_fn()
        print("SpiralKFftPlan WGSL snippet:")
        print(wgsl_src.splitlines()[0])
    hint_fn = getattr(spiralk_fft, "spiralk_hint", None) or getattr(spiralk_fft, "emit_spiralk_hint", None)
    if hint_fn:
        print("SpiralKFftPlan hint:")
        print(hint_fn())

    unison_lines = fft_plan.to_unison_script().splitlines()
    print("Unison script (first 3 lines):")
    for line in unison_lines[:3]:
        print("  ", line)

    batches = st.generate_plan_batch_ex(
        n=3,
        total_steps=24,
        base_radius=1.0,
        radial_growth=0.3,
        base_height=0.2,
        meso_gain=0.4,
        micro_gain=0.25,
        seed=0xBEE,
    )
    print("\nGenerated plan batch (first entry):")
    pprint(batches[0])


def train_small_linear_model() -> None:
    banner("ModuleTrainer demo")
    st.set_global_seed(0xC0FFEE)
    pacing = st.fibonacci_pacing(12)
    chunks = st.pack_tribonacci_chunks(12)
    print("Fibonacci pacing:", pacing)
    print("Tribonacci chunks:", chunks)
    print("Pack-4-nacci chunks:", st.pack_nacci_chunks(4, 12))
    print("Tetranacci chunks:", st.pack_tetranacci_chunks(12))

    trainer = st.ModuleTrainer(input_dim=2, output_dim=2)
    inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    targets = [
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
    loss = trainer.train_epoch(inputs, targets, learning_rate=0.1, batch_size=2)
    print(f"Average loss after one epoch: {loss:.6f}")

    predictions = trainer.predict(inputs)
    print("Predictions:")
    pprint(predictions.tolist())

    tensor_inputs = st.capture(inputs)
    tensor_predictions = trainer.predict_tensor(tensor_inputs)
    print("Predictions (tensor API):")
    pprint(tensor_predictions.tolist())

    weights = trainer.weights().tolist()
    bias = trainer.bias()
    print("Weights matrix:")
    pprint(weights)
    print("Bias vector:", bias)

    eval_loss = trainer.evaluate(inputs, targets)
    print(f"Evaluation MSE: {eval_loss:.6f}")

    mse = st.mean_squared_error(predictions, st.Tensor(len(targets), 2, [v for row in targets for v in row]))
    print(f"Standalone mean_squared_error: {mse:.6f}")

    masked = st.masked_mse(predictions.tolist(), targets, [[1], [0], [1], [0]])
    print("Masked MSE result:")
    pprint(masked)

    nce = st.info_nce(
        anchors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        positives=[[0.9, 0.05, 0.05], [0.1, 0.75, 0.15]],
        temperature=0.7,
        normalize=True,
    )
    print("InfoNCE summary:")
    pprint(nce)


def explore_barycenters() -> None:
    banner("Barycenter and tensor biome utilities")
    densities = [
        st.Tensor(2, 2, [0.9, 0.1, 0.3, 0.7]),
        st.Tensor(2, 2, [0.2, 0.8, 0.4, 0.6]),
    ]
    bary = st.z_space_barycenter(
        weights=[0.6, 0.4],
        densities=densities,
        entropy_weight=0.2,
        beta_j=0.8,
    )
    print(
        f"Barycenter objective={bary.objective:.6f} entropy={bary.entropy:.6f} "
        f"effective_weight={bary.effective_weight:.3f}"
    )
    canopy = bary.density().tolist()
    print("Barycenter density matrix:")
    pprint(canopy)

    intermediates = bary.intermediates()
    if intermediates:
        head = intermediates[0]
        print(
            "First interpolation stage: interp=",
            head.interpolation,
            "KL=",
            head.kl_energy,
            "entropy=",
            head.entropy,
        )

        hyper = st.Hypergrad(curvature=-0.9, learning_rate=0.05, rows=2, cols=2)
        hyper.accumulate_barycenter_path(intermediates)
        summary = hyper.summary()
        print(
            "Hypergrad summary:",
            {
                "l2": summary.l2(),
                "count": summary.count(),
                "mean_abs": summary.mean_abs(),
            },
        )
        hyper.scale_learning_rate(0.5)
        print("Hypergrad learning rate scaled; gradient length:", len(hyper.gradient()))

    topos = st.OpenCartesianTopos(curvature=-1.0, tolerance=0.25, saturation=0.4, max_depth=4, max_volume=16)
    biome = st.TensorBiome(topos)
    for density in densities:
        biome.absorb_weighted(density, 0.5)
    print("Biome canopy:")
    pprint(biome.canopy().tolist())
    print("Biome total weight:", biome.total_weight())


def tensor_algebra_and_losses() -> None:
    banner("Tensor algebra & loss helpers")
    captured = st.capture([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    random = st.Tensor.rand(3, 2, seed=123)
    product = captured.matmul(random)
    print("Matmul result shape:", product.shape())
    print("Row softmax of captured tensor:")
    pprint(captured.row_softmax().tolist())

    scaled = captured.scale(0.5)
    cat = st.Tensor.cat_rows([captured, scaled])
    print("Concatenated tensor shape:", cat.shape())

    distance = captured.project_to_poincare(curvature=-1.0).hyperbolic_distance(
        scaled.project_to_poincare(curvature=-1.0),
        curvature=-1.0,
    )
    print(f"Hyperbolic distance between projections: {distance:.6f}")

    l2 = product.squared_l2_norm()
    print(f"Squared L2 norm of product: {l2:.6f}")

    dot_masked = st.masked_mse(
        predictions=[[0.2, 0.8], [0.6, 0.4]],
        targets=[[0.0, 1.0], [1.0, 0.0]],
        mask_indices=[[1], [0]],
    )
    print("Secondary masked MSE:")
    pprint(dot_masked)


def resonance_and_vision_demo() -> None:
    banner("Temporal resonance & vision accumulation")
    buffer = st.TemporalResonanceBuffer(capacity=3, alpha=0.6)
    volume_a = [
        [[0.1, 0.2], [0.3, 0.4]],
        [[0.5, 0.6], [0.7, 0.8]],
    ]
    volume_b = [
        [[0.2, 0.1], [0.4, 0.3]],
        [[0.6, 0.5], [0.8, 0.7]],
    ]
    for idx, vol in enumerate([volume_a, volume_b, volume_a]):
        state = buffer.update(vol)
        print(f"Buffer update {idx}: state[0][0][:2] =", state[0][0][:2])
    print("Buffer history length:", len(buffer.history()))
    print("Buffer state dict keys:", sorted(buffer.state_dict().keys()))

    vision = st.SpiralTorchVision(depth=2, height=2, width=2, alpha=0.7, window="hann", temporal=3)
    vision.accumulate(volume_a, weight=1.0)
    vision.accumulate(volume_b, weight=0.5)
    projection = vision.project(normalise=True)
    print("Vision projection (normalised):")
    pprint(projection)
    print("Vision volume energy:", vision.volume_energy())
    temporal_state = vision.temporal_state()
    if temporal_state:
        print("Temporal state sample:", temporal_state[0][0])


def wave_encoding_showcase() -> None:
    banner("Language wave encoder & extras")
    encoder = st.LanguageWaveEncoder(curvature=-1.0, temperature=0.5)
    wave = encoder.encode_wave("hello SpiralTorch session")
    wave_data = wave.data()
    print("Encoded wave (first four complex samples):")
    pprint(wave_data[:4])

    z_space = encoder.encode_z_space("tensor spiral")
    print("Z-space tensor shape:", z_space.shape())

    print("Golden ratio:", st.golden_ratio())
    print("Golden angle:", st.golden_angle())

    rows, cols = z_space.shape()
    hyper = st.Hypergrad(curvature=-0.85, learning_rate=0.02, rows=rows, cols=cols)
    hyper.absorb_text(encoder, "hyperbolic gradients are fun")
    hyper.accumulate_complex_wave(wave)
    hyper.accumulate_pair(z_space, z_space)
    summary = hyper.summary()
    print(
        "Hypergrad after wave encoding:",
        {
            "rms": summary.rms(),
            "l1": summary.l1(),
            "linf": summary.linf(),
        },
    )
    hyper.reset()
    print("Hypergrad reset; count:", hyper.summary().count())


if __name__ == "__main__":
    session = st.SpiralSession(backend="wgpu")
    describe_backend(session)
    train_small_linear_model()
    tensor_algebra_and_losses()
    explore_barycenters()
    resonance_and_vision_demo()
    wave_encoding_showcase()
    session.close()
