from __future__ import annotations


def main() -> None:
    import spiraltorch as st

    metrics = {
        "psi.total": 1.25,
        "psi.spiral.mu_eff0": 0.42,
        "canvas.energy": 0.9,
        "tensor.matmul.ms": 3.2,
        "realgrad_step": 0.05,
    }

    frame = st.telemetry.AtlasFrame.from_metrics(metrics)
    print("timestamp:", frame.timestamp)

    for district in frame.districts():
        name = district["name"]
        mean = district["mean"]
        span = district["span"]
        print(f"- {name}: mean={mean:.3f} span={span:.3f}")
        for metric in district["metrics"]:
            print(
                f"  - {metric.name}={metric.value:.3f} district={metric.district!r}"
            )

    route = st.telemetry.AtlasRoute()
    for step in range(6):
        mut = dict(metrics)
        mut["psi.total"] = metrics["psi.total"] + 0.08 * step
        mut["canvas.energy"] = metrics["canvas.energy"] - 0.04 * step
        mut["tensor.matmul.ms"] = metrics["tensor.matmul.ms"] + (0.15 if step % 2 == 0 else -0.05)
        route.push_bounded(
            st.telemetry.AtlasFrame.from_metrics(mut, timestamp=float(step)),
            bound=32,
        )

    summary = route.summary()
    print("route frames:", summary["frames"])

    psi = route.perspective_for("Concourse", focus_prefixes=["psi."])
    if psi is not None:
        print("psi perspective guidance:", psi["guidance"])

    beacons = route.beacons(limit=3)
    print("beacons:")
    for beacon in beacons:
        print(f"- {beacon['metric']} ({beacon['trend']}) {beacon['narrative']}")


if __name__ == "__main__":
    main()
