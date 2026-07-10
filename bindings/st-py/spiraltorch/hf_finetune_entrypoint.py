"""Portable module entrypoint for SpiralTorch Hugging Face fine-tuning."""

from __future__ import annotations

from .hf_cli import finetune_bridge_main


def main() -> int:
    return finetune_bridge_main()


if __name__ == "__main__":
    raise SystemExit(main())
