from __future__ import annotations


def main(steps: int = 6) -> None:
    import spiraltorch as st

    session = st.amegagrad_session(
        (1, 4),
        curvature=-0.9,
        hyper_learning_rate=0.03,
        real_learning_rate=0.02,
        telemetry=True,
        telemetry_bound=64,
    )

    target = [0.4, -0.35, 0.2, 0.1]
    print("build:", st.build_info())
    print("initial weights:", [float(x) for x in session.weights.tolist()[0]])

    for step in range(int(steps)):
        current = [float(x) for x in session.weights.tolist()[0]]
        signal = [t - c for t, c in zip(target, current)]
        session.step_wave(st.Tensor(1, 4, signal), note=f"step {step}")
        print(f"step {step:02d} weights:", [float(x) for x in session.weights.tolist()[0]])

    if session.route is not None:
        perspective = session.route.perspective_for("Substrate", focus_prefixes=["realgrad."])
        if perspective is not None:
            print("realgrad perspective:", perspective["guidance"])


if __name__ == "__main__":
    main()
