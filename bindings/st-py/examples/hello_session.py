# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.
"""Showcase the high-level SpiralTorch Python bindings.

This example keeps everything self-contained and exercises the features that are
currently implemented in the lightweight bindings:

* `SpiralSession` describes device backends and produces rank plans.
* `ModuleTrainer` performs a tiny linear regression and exposes model weights.
* Tensor utilities cover barycenters, biomes, and gradient helpers.
* Extras such as the golden ratio helpers and SpiralK plan annotations are all
  demonstrated so the script doubles as a smoke-test when run under CI.
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


def train_small_linear_model() -> None:
    banner("ModuleTrainer demo")
    st.set_global_seed(0xC0FFEE)
    pacing = st.fibonacci_pacing(12)
    chunks = st.pack_tribonacci_chunks(12)
    print("Fibonacci pacing:", pacing)
    print("Tribonacci chunks:", chunks)

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

    weights = trainer.weights().tolist()
    bias = trainer.bias()
    print("Weights matrix:")
    pprint(weights)
    print("Bias vector:", bias)

    eval_loss = trainer.evaluate(inputs, targets)
    print(f"Evaluation MSE: {eval_loss:.6f}")

    mse = st.mean_squared_error(predictions, st.Tensor(len(targets), 2, [v for row in targets for v in row]))
    print(f"Standalone mean_squared_error: {mse:.6f}")


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

    topos = st.OpenCartesianTopos(curvature=-1.0, tolerance=0.25, saturation=0.4, max_depth=4, max_volume=16)
    biome = st.TensorBiome(topos)
    for density in densities:
        biome.absorb_weighted(density, 0.5)
    print("Biome canopy:")
    pprint(biome.canopy().tolist())
    print("Biome total weight:", biome.total_weight())


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


if __name__ == "__main__":
    session = st.SpiralSession(backend="wgpu")
    describe_backend(session)
    train_small_linear_model()
    explore_barycenters()
    wave_encoding_showcase()
    session.close()
