#!/usr/bin/env python3
"""Generic Hugging Face fine-tuning bridge with SpiralTorch Z-Space preflight."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from hf_gpt2_finetune_bridge import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    raise SystemExit(main())
