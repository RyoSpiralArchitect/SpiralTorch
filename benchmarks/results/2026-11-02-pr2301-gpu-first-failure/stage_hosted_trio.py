"""Build only a Git tree for the diagnostic branch. No keys, refs or remote writes."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

from hashlib import sha256

BASE = "fb936f7d11ff8e57ba794f6faf9324343a658120"  # already-published probe branch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCOPE = HERE.relative_to(ROOT).as_posix()
SOURCES = ("run_embedding_trio.py", "test_run_embedding_trio.py", "compare_embedding_bits.py",
           "test_compare_embedding_bits.py", "hosted_embedding_environment.py", "test_hosted_embedding_environment.py")
WORKFLOW = ".github/workflows/byte_study_replay.yml"
PACKET = ".github/diagnostics/pr2301-embedding/wheel.age"


def render(template, recipient, digest):
    if not re.fullmatch(r"age1[023456789acdefghjklmnpqrstuvwxyz]{58}", recipient):
        raise ValueError("not an age X25519 public recipient")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("expected SHA-256")
    for marker, value in (("__RESULT_RECIPIENT__", recipient), ("__CIPHERTEXT_SHA256__", digest)):
        if template.count(marker) != 1:
            raise ValueError("unexpected workflow marker count")
        template = template.replace(marker, value)
    return template


def stage(packet, recipient):
    if packet.is_symlink() or not (13030711 < packet.stat().st_size < 14 * 1024**2):
        raise ValueError("expected the bounded, encrypted arm64 wheel")
    sealed = packet.read_bytes()
    if not sealed.startswith(b"age-encryption.org/v1\n-> X25519 "):
        raise ValueError("refusing to stage plaintext or another recipient format")
    with tempfile.TemporaryDirectory() as temp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp)/"index"))
        def git(*args, data=None):
            return subprocess.check_output(["git", *args], input=data, cwd=ROOT, env=env)
        template = git("show", f"HEAD:{SCOPE}/hosted_workflow.yml.in").decode()
        replacements = {WORKFLOW: render(template, recipient, sha256(sealed).hexdigest()).encode(), PACKET: sealed}
        git("read-tree", BASE)
        for path in SOURCES:
            dst = f"{SCOPE}/{path}"
            blob = git("rev-parse", f"HEAD:{dst}").decode().strip()
            git("update-index", "--add", "--cacheinfo", f"100644,{blob},{dst}")
        for dst, data in replacements.items():
            blob = git("hash-object", "-w", "--stdin", data=data).decode().strip()
            git("update-index", "--add", "--cacheinfo", f"100644,{blob},{dst}")
        tree = git("write-tree").decode().strip()
        actual = set(git("diff-tree", "--no-commit-id", "--name-only", "-r", BASE, tree).decode().splitlines())
        if actual != set(replacements) | {f"{SCOPE}/{name}" for name in SOURCES}:
            raise ValueError("diagnostic tree differs from its exact file allowlist")
        return tree


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel_ciphertext", type=Path)
    parser.add_argument("result_recipient")
    args = parser.parse_args()
    print(stage(args.wheel_ciphertext, args.result_recipient))
