from __future__ import annotations

import math
from pathlib import Path


def write_ppm(path: Path, width: int, height: int, rgba: bytes) -> None:
    if len(rgba) != width * height * 4:
        raise ValueError(
            f"expected {width}x{height} RGBA bytes, got {len(rgba)} bytes"
        )
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        for offset in range(0, len(rgba), 4):
            f.write(rgba[offset : offset + 3])


def main() -> None:
    import spiraltorch as st

    width = 96
    height = 96

    print("available palettes:", st.canvas.available_palettes())
    projector = st.canvas.CanvasProjector(width=width, height=height, palette="turbo")

    values: list[float] = []
    for y in range(height):
        ny = (y / (height - 1)) * 2.0 - 1.0
        for x in range(width):
            nx = (x / (width - 1)) * 2.0 - 1.0
            radius = math.sqrt(nx * nx + ny * ny)
            value = math.cos(6.0 * math.pi * radius) * max(0.0, 1.0 - radius)
            values.append(value)

    relation = st.Tensor(height, width, values)
    projector.push_patch(relation, coherence=1.0, tension=0.5, depth=0)

    rgba = projector.refresh_rgba()
    out = Path("spiraltorch_canvas.ppm")
    write_ppm(out, width, height, rgba)
    print(f"wrote {out.resolve()}")

    patch = projector.emit_zspace_patch(coherence=0.9, tension=1.1, depth=1)
    print(
        "loopback patch:",
        "shape",
        patch["relation"].shape(),
        "weight",
        patch["weight"],
    )
    projector.push_patch(
        patch["relation"],
        coherence=patch["coherence"],
        tension=patch["tension"],
        depth=patch["depth"],
    )
    loopback = projector.refresh_rgba()
    out_loop = Path("spiraltorch_canvas_loopback.ppm")
    write_ppm(out_loop, width, height, loopback)
    print(f"wrote {out_loop.resolve()}")

    trail = projector.emit_wasm_trail(curvature=1.5)
    print("wasm trail samples tensor shape:", trail["samples"].shape())

    spectrum_db = projector.refresh_vector_fft_power_db_tensor()
    print("vector FFT power(dB) tensor shape:", spectrum_db.shape())


if __name__ == "__main__":
    main()
