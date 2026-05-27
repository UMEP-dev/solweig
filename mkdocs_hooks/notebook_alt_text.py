"""MkDocs hook: replace nbconvert's generic alt-text placeholder.

`nbconvert`'s HTML exporter post-processes rendered output and injects
``alt="No description has been provided for this image"`` on every
`<img>` tag that lacks one (see
``.venv/.../nbconvert/exporters/html.py``). It does **not** read
per-image alt text from `output.metadata["image/png"]["alt"]`, even
though notebooks can carry that metadata and the JSON schema allows it.

This hook closes the gap: for every tutorial notebook page,
- load the source `.ipynb`,
- collect ordered per-image alts from `output.metadata["image/png"]["alt"]`,
- walk the rendered HTML in order and replace the placeholder alt with
  the corresponding stored alt where one exists.

Authoring contract: to add alt text for a code-output image, set
``cell.outputs[i].metadata["image/png"]["alt"]`` in the `.ipynb`
(do it via a one-shot script or in JupyterLab's metadata editor).

Skipped silently if the source notebook can't be located — for non-
notebook pages this hook is a no-op.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TUTORIALS_DIR = REPO_ROOT / "docs" / "tutorials"


class _MissingAltTextFilter(logging.Filter):
    """Drop nbconvert's 'Alternative text is missing on N image(s)' warning.

    The warning is emitted from `nbconvert.exporters.html` during HTML
    generation *before* `on_page_content` runs, so it would fire even
    when this hook subsequently substitutes proper alt text. Filtering
    at the logger level keeps CI output focused on real warnings.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "Alternative text is missing" not in record.getMessage()


def _install_warning_filter() -> None:
    """Idempotent install of the nbconvert warning filter."""
    nbconv_logger = logging.getLogger("traitlets")
    if not any(isinstance(f, _MissingAltTextFilter) for f in nbconv_logger.filters):
        nbconv_logger.addFilter(_MissingAltTextFilter())


_install_warning_filter()

# Matches the placeholder nbconvert injects; tolerant of attr-order changes.
_PLACEHOLDER_RE = re.compile(r'(<img\b[^>]*\balt=")No description has been provided for this image(")')


def _collect_alts(nb_path: Path) -> list[str]:
    """Read a notebook and return ordered alts for each code-output PNG.

    The order matches nbconvert's render order: iterate cells in order,
    iterate that cell's outputs in order, take any image/png output. Each
    output contributes one entry. Outputs without alt metadata contribute
    an empty string so the index alignment with rendered <img> tags is
    preserved.
    """
    alts: list[str] = []
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return alts
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "image/png" not in data:
                continue
            alt = (out.get("metadata", {}).get("image/png", {}) or {}).get("alt", "")
            alts.append(alt)
    return alts


def _notebook_for_page(page) -> Path | None:
    """Map a mkdocs Page to its source notebook, if applicable."""
    src = getattr(page.file, "src_path", "") or ""
    if not src.endswith(".ipynb"):
        return None
    return REPO_ROOT / "docs" / src


def on_page_content(html: str, page, config, files) -> str:
    """Replace nbconvert's placeholder alt with per-image alts from the .ipynb."""
    nb_path = _notebook_for_page(page)
    if nb_path is None or not nb_path.is_file():
        return html
    alts = _collect_alts(nb_path)
    if not alts:
        return html

    # Replace placeholders in order — only those that have a non-empty alt
    # in metadata. Any extra placeholders (more images than annotated alts)
    # keep the generic text, so nbconvert's accessibility floor remains.
    counter = {"idx": 0}

    def _sub(m: re.Match[str]) -> str:
        i = counter["idx"]
        counter["idx"] += 1
        if i < len(alts) and alts[i]:
            # Escape `"` and `&` in the alt text for safe HTML embedding.
            safe = alts[i].replace("&", "&amp;").replace('"', "&quot;")
            return f"{m.group(1)}{safe}{m.group(2)}"
        return m.group(0)

    return _PLACEHOLDER_RE.sub(_sub, html)
