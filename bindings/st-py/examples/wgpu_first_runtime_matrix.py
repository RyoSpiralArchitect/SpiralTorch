# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
"""Write a tiny WGPU-first runtime matrix report as JSON.

The script probes the everyday runtime entrypoints for ``auto``, ``wgpu``,
``mps``, and ``cpu``. Each backend request is isolated so one unavailable route
does not hide the rest of the matrix.
"""

from __future__ import annotations

import argparse
from pprint import pprint

import spiraltorch as st


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="wgpu_first_runtime_matrix.json",
        help="JSON artifact path to write",
    )
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        help="Backend label to include; repeat to override the default matrix",
    )
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args()

    matrix = st.write_wgpu_first_runtime_matrix(
        args.out,
        args.backends or ("auto", "wgpu", "mps", "cpu"),
        rows=args.rows,
        cols=args.cols,
        k=args.k,
    )

    print(f"Wrote {matrix['artifact_path']}")
    pprint(matrix["summary"])


if __name__ == "__main__":
    main()
