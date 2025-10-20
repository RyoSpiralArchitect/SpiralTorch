#!/usr/bin/env python3
"""
SpiralTorch Auto Merge Resolver
---------------------------------
自動でマージ衝突（Cargo.toml, lib.rs, pyproject 等）を統一。
'<<<<<<<' '=======' '>>>>>>>' を削除し、両方の変更を統合する簡易パッチ。
"""

import os
import re
from pathlib import Path

# 解決対象
TARGETS = [
    "Cargo.toml",
    "Cargo.lock",
    "bindings/python/spiral/__init__.py",
    "bindings/st-py/Cargo.toml",
    "bindings/st-py/src/lib.rs",
    "crates/st-core/Cargo.toml",
    "crates/st-core/src/ability/unison_mediator.rs",
    "crates/st-core/src/backend/consensus.rs",
    "crates/st-core/src/backend/kdsl_bridge.rs",
    "crates/st-core/src/backend/wgpu_heuristics.rs",
    "crates/st-logic/Cargo.toml",
    "crates/st-logic/src/lib.rs",
]

def clean_conflict_markers(text: str) -> str:
    # すべての <<<<<<< と >>>>>>> マーカーを削除し、両側の内容を保持
    text = re.sub(r"<<<<<<<[^\n]*\n", "", text)
    text = re.sub(r"=======\n", "", text)
    text = re.sub(r">>>>>>>[^\n]*\n", "", text)
    return text

def process_file(path: Path):
    try:
        original = path.read_text()
        if "<<<<<<<" not in original:
            return False
        cleaned = clean_conflict_markers(original)
        path.write_text(cleaned)
        print(f"✅ Cleaned conflict markers in {path}")
        return True
    except Exception as e:
        print(f"⚠️  Error on {path}: {e}")
        return False

def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    print(f"🔧 Running auto-resolver in {root}")
    modified = 0
    for target in TARGETS:
        path = root / target
        if path.exists():
            if process_file(path):
                modified += 1
    if modified:
        print(f"\n✨ {modified} files cleaned. Now run:")
        print("  git add -A && git commit -m 'auto-resolved merge conflicts'")
    else:
        print("✅ No conflict markers found.")
        
if __name__ == "__main__":
    main()
