#!/usr/bin/env python3
"""Summarize SpiralTorch GPT-2 fine-tuning run status JSONL history."""

from __future__ import annotations

from spiraltorch.hf_ft_status import (
    _load_history,
    history_lines,
    main,
    parse_args,
    summarize_history,
)


if __name__ == "__main__":
    raise SystemExit(main())
