#!/usr/bin/env python3
"""Wait for a long Hugging Face FT handoff, then launch the next command."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from hf_gpt2_finetune_wait_launch import *  # noqa: F401,F403,E402
from hf_gpt2_finetune_wait_launch import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
