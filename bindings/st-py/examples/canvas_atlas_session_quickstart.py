from __future__ import annotations

import time


def main(steps: int = 8) -> None:
    import spiraltorch as st

    session = st.amegagrad_session(
        (1, 4),
        curvature=-0.9,
        hyper_learning_rate=0.03,
        real_learning_rate=0.02,
        telemetry=True,
        telemetry_bound=128,
    )
    projector = st.canvas.CanvasProjector(width=4, height=1, palette="turbo")

    print("build:", st.build_info())
    print("initial weights:", [float(x) for x in session.weights.tolist()[0]])

    for step in range(int(steps)):
        projector.push_patch(
            session.weights,
            coherence=1.0,
            tension=1.0,
            depth=step,
        )
        patch = projector.emit_zspace_patch(coherence=0.9, tension=1.1, depth=step)
        session.step_wave(patch["relation"], note=f"loop {step}")

        if session.route is not None:
            frame = projector.emit_atlas_frame(prefix="canvas", refresh=False, timestamp=time.time())
            assert frame.metric_value("canvas.palette.id") is not None
            assert frame.metric_value("canvas.normalizer.alpha") is not None
            assert frame.metric_value("canvas.trail.energy.mean") is not None
            assert frame.metric_value("canvas.fft_db.mean") is not None
            session.route.push_bounded(frame, bound=128)

        print(f"step {step:02d} weights:", [float(x) for x in session.weights.tolist()[0]])

    if session.route is not None:
        summary = session.route.summary()
        print("atlas frames:", summary["frames"])

        surface = session.route.perspective_for("Surface", focus_prefixes=["canvas."])
        if surface is not None:
            print("surface:", surface["guidance"])
        substrate = session.route.perspective_for("Substrate", focus_prefixes=["realgrad."])
        if substrate is not None:
            print("substrate:", substrate["guidance"])

        beacons = session.route.beacons(limit=3)
        print("beacons:", [beacon["metric"] for beacon in beacons])


if __name__ == "__main__":
    main()
