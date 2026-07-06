"""Export executed tutorial notebooks to the Markdown pages mkdocs publishes.

The docs site serves ``docs/tutorials/*.md`` (plus their ``*_files/`` image
directories). The ``.ipynb`` files are the executable authoring sources; they
are excluded from the site build via ``exclude_docs`` in ``mkdocs.yml``.

After editing or re-executing a notebook (``poe notebooks`` does both), this
script regenerates the Markdown:

1. ``jupyter nbconvert --to markdown`` for each notebook, which extracts
   output images into ``<name>_files/``.
2. Re-inject per-image alt text. nbconvert's markdown exporter writes
   ``![png](...)`` for every image/png output, discarding the alt text
   authors store in ``output.metadata["image/png"]["alt"]``. We walk the
   notebook's code-cell outputs in render order (the same order nbconvert
   emits images) and replace the k-th ``![png](...)`` with ``![<alt>](...)``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TUTORIALS = REPO_ROOT / "docs" / "tutorials"
IMG_RE = re.compile(r"!\[png\]\(([^)]+)\)")


def collect_alts(nb_path: Path) -> list[str]:
    """Ordered alt texts for each image/png output ('' where unset)."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    alts: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if "image/png" not in out.get("data", {}):
                continue
            alt = (out.get("metadata", {}).get("image/png", {}) or {}).get("alt", "")
            alts.append(alt)
    return alts


def inject_alts(md_path: Path, alts: list[str]) -> int:
    """Rewrite ![png](...) images with alt text; returns count injected."""
    text = md_path.read_text(encoding="utf-8")
    counter = {"i": 0, "injected": 0}

    def sub(m: re.Match[str]) -> str:
        i = counter["i"]
        counter["i"] += 1
        alt = alts[i] if i < len(alts) else ""
        if alt:
            counter["injected"] += 1
            return f"![{alt.replace(']', chr(92) + ']')}]({m.group(1)})"
        return m.group(0)

    md_path.write_text(IMG_RE.sub(sub, text), encoding="utf-8")
    return counter["injected"]


def main() -> int:
    notebooks = sorted(TUTORIALS.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {TUTORIALS}", file=sys.stderr)
        return 1
    status = 0
    for nb_path in notebooks:
        result = subprocess.run(
            [sys.executable, "-m", "nbconvert", "--to", "markdown", str(nb_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"FAIL {nb_path.name}: {result.stderr.strip()}", file=sys.stderr)
            status = 1
            continue
        md_path = nb_path.with_suffix(".md")
        alts = collect_alts(nb_path)
        n_alts = inject_alts(md_path, alts)
        n_imgs = len(IMG_RE.findall(md_path.read_text(encoding="utf-8"))) + n_alts
        missing = n_imgs - n_alts
        note = f", {missing} image(s) missing alt text" if missing else ""
        print(f"{md_path.name}: {n_imgs} images, {n_alts} alts{note}")
    return status


if __name__ == "__main__":
    sys.exit(main())
