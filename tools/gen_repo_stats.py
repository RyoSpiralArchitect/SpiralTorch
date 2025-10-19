# tools/gen_repo_stats.py
#!/usr/bin/env python3
import json, subprocess, sys, re
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
BADGE_DIR = ROOT / "docs" / "badges"
README = ROOT / "README.md"

def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    return p.stdout.strip()

def load_tokei() -> dict:
    # tokei JSON 形式の差異に強めにパース
    raw = run(["tokei", ".", "-o", "json"])
    data = json.loads(raw)
    # 代表的な形: {"Rust": {...}, "Total": {...}, ...}
    return data

def pick_lang(stats: dict, name: str) -> dict | None:
    for k, v in stats.items():
        if k.lower() == name.lower():
            return v
    return None

def count_files(lang_block: dict) -> int:
    # tokei の各言語ブロックは "reports": [{"name": "...", "stats": {...}}, ...] を持つ
    # Markdown 内の埋め込みもあるが、Rust 本体は言語 "Rust" 側で OK
    reps = lang_block.get("reports") or []
    return len(reps)

def cargo_dep_count() -> int:
    meta = json.loads(run(["cargo", "metadata", "--format-version", "1"]))
    return len(meta.get("packages", []))

def digits(n: int) -> str:
    return f"{n:,}"

def pick_color(value: int, bands: list[tuple[int,str]]) -> str:
    # 値に応じて色分け（左から閾値昇順）
    color = "lightgrey"
    for th, col in bands:
        if value >= th: color = col
    return color

def badge_svg(label: str, value: str, color: str) -> str:
    # シンプルな Shields 風 SVG（依存なし）
    label_w = 6*len(label)+40
    value_w = 6*len(value)+40
    total_w = label_w + value_w
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="20" role="img" aria-label="{label}: {value}">
  <linearGradient id="a" x2="0" y2="100%">
    <stop offset="0" stop-color="#fff" stop-opacity=".7"/>
    <stop offset=".1" stop-opacity=".1"/>
    <stop offset=".9" stop-opacity=".3"/>
    <stop offset="1" stop-opacity=".5"/>
  </linearGradient>
  <mask id="m"><rect width="{total_w}" height="20" rx="3" fill="#fff"/></mask>
  <g mask="url(#m)">
    <rect width="{label_w}" height="20" fill="#555"/>
    <rect x="{label_w}" width="{value_w}" height="20" fill="#{color}"/>
    <rect width="{total_w}" height="20" fill="url(#a)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="{label_w/2:.1f}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{label_w/2:.1f}" y="14">{label}</text>
    <text x="{label_w + value_w/2:.1f}" y="15" fill="#010101" fill-opacity=".3">{value}</text>
    <text x="{label_w + value_w/2:.1f}" y="14">{value}</text>
  </g>
</svg>"""

def update_readme_section(section_md: str) -> None:
    content = README.read_text(encoding="utf-8")
    start = "<!-- STATS:START -->"
    end = "<!-- STATS:END -->"
    if start not in content or end not in content:
        print("README markers not found; skipping README update.", file=sys.stderr)
        return
    new_block = f"{start}\n{section_md.strip()}\n{end}"
    content = re.sub(
        re.compile(rf"{re.escape(start)}.*?{re.escape(end)}", re.S),
        new_block,
        content,
    )
    README.write_text(content, encoding="utf-8")

def main():
    BADGE_DIR.mkdir(parents=True, exist_ok=True)

    stats = load_tokei()
    rust = pick_lang(stats, "Rust") or {}
    total = pick_lang(stats, "Total") or {}

    rust_code = int(rust.get("code", 0))
    rust_files = count_files(rust)
    total_code = int(total.get("code", 0) or sum(v.get("code",0) for k,v in stats.items() if k!="Total"))

    deps = cargo_dep_count()

    # バッジ作成
    rust_color = pick_color(rust_code, [
        (10_000, "4c1"),
        (25_000, "97CA00"),
        (50_000, "28a3f0"),
        (75_000, "fb8c00"),
        (100_000,"e05d44"),
    ])
    deps_color = pick_color(deps, [
        (50, "4c1"),
        (100, "97CA00"),
        (200, "28a3f0"),
        (300, "fb8c00"),
        (400, "e05d44"),
    ])
    total_color = pick_color(total_code, [
        (20_000, "4c1"),
        (50_000, "97CA00"),
        (100_000,"28a3f0"),
        (150_000,"fb8c00"),
        (200_000,"e05d44"),
    ])

    (BADGE_DIR/"rust-loc.svg").write_text(badge_svg("rust loc", digits(rust_code), rust_color), encoding="utf-8")
    (BADGE_DIR/"deps.svg").write_text(badge_svg("crates", digits(deps), deps_color), encoding="utf-8")
    (BADGE_DIR/"total-code.svg").write_text(badge_svg("total code", digits(total_code), total_color), encoding="utf-8")

    # README セクション更新
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    section = f"""
> _auto-generated: {now}_

| Metric | Value |
|---|---:|
| Rust code LOC | **{digits(rust_code)}** |
| Rust files | {digits(rust_files)} |
| Total code LOC (all langs) | {digits(total_code)} |
| Workspace+deps crates | {digits(deps)} |

<p>
<img src="docs/badges/rust-loc.svg" alt="rust loc" />
<img src="docs/badges/total-code.svg" alt="total code" />
<img src="docs/badges/deps.svg" alt="crates" />
</p>
""".strip()

    update_readme_section(section)

if __name__ == "__main__":
    main()
