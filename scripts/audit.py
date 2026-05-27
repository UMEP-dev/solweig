#!/usr/bin/env python3
"""Repository audit — produces AUDIT.md from measured signals.

Runs ~8 axis measurements over the repo and writes a single AUDIT.md at the
repo root. Designed to be re-run periodically (`poe audit`); the report is
checked in so `git diff AUDIT.md` shows drift between snapshots.

Axes:
  1. Rust panic surface
  2. Python type strictness
  3. Test coverage (pysrc/)
  4. CI vs local task gap
  5. Public API discipline
  6. Docstring coverage on public API
  7. Hot files & TODO/FIXME density
  8. Dependency freshness

The script is intentionally simple — single file, stdlib + tomllib + a few
existing dev deps. No external services, no databases, no daemons. If a
measurement cannot run (e.g. tool not installed), it prints "n/a" with the
reason rather than failing the whole run.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYSRC = REPO / "pysrc" / "solweig"
RUST_SRC = REPO / "rust" / "src"
TESTS = REPO / "tests"
WORKFLOWS = REPO / ".github" / "workflows"
PYPROJECT = REPO / "pyproject.toml"
REPORT = REPO / "AUDIT.md"

# ── Thresholds (single source of truth) ─────────────────────────────────────
# Adjust here; the report shows status vs these targets.
THRESHOLDS = {
    "rust_unwrap_per_kloc": 5.0,  # non-test unwrap/expect/panic per 1000 lines
    "ty_suppressions": 3,  # broad rule-level "ignore" entries
    "test_coverage_pct": 80.0,
    "ci_gap_count": 1,  # poe tasks not wired into CI (test_quick/full are in CI; others may be intentional)
    "public_api_documented_pct": 90.0,
    "hot_file_max_lines": 700,
    "py_outdated_count": 10,
}


@dataclass
class AxisResult:
    name: str
    summary: str  # one-line headline
    detail: list[str] = field(default_factory=list)  # bullet/table rows for the body
    status: str = "info"  # "ok" | "warn" | "fail" | "info" | "n/a"
    metric: str | None = None  # raw metric for trend tracking


def _run(cmd: list[str], cwd: Path = REPO, timeout: int = 120) -> tuple[int, str, str]:
    """Run a command, return (returncode, stdout, stderr). Never raises."""
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"


def _files_under(root: Path, suffix: str) -> list[Path]:
    return sorted(p for p in root.rglob(f"*{suffix}") if "__pycache__" not in p.parts)


def _count_lines(p: Path) -> int:
    try:
        return sum(1 for _ in p.open(encoding="utf-8", errors="replace"))
    except OSError:
        return 0


# ── Axis 1: Rust panic surface ──────────────────────────────────────────────


# Distinguish .unwrap() (no rationale) from .expect("...") (documented panic).
# .unwrap() with no message is the real smell; .expect() with a message is
# defensive documentation and acceptable.
_UNWRAP_PAT = re.compile(r"\.unwrap\(\)")
_EXPECT_PAT = re.compile(r"\.expect\(")
_PANIC_PAT = re.compile(r"\bpanic!\s*\(|\bunreachable!\s*\(")


def axis_rust_panics() -> AxisResult:
    """Count panic surface in Rust, distinguishing undocumented vs documented.

    Heuristic limitations (acceptable in practice, documented here so the
    measurement is honest):
    - "Inside a test block" is detected by tracking brace depth from the
      first `#[cfg(test)]` or `mod tests {`. Single-function `#[test]` items
      at the top level of a non-test module are NOT recognised as tests
      (none exist in this codebase today).
    - `.expect(...)` calls are counted as "documented" without verifying the
      message is informative. A bare `.expect("")` would slip through.
    - String parsing only — no syntactic awareness of comments / strings,
      so `// .unwrap()` in a comment would be counted. Manual grep shows
      no such case in this codebase.
    """
    files = _files_under(RUST_SRC, ".rs")
    total_lines = 0
    undocumented = 0  # .unwrap() — no rationale
    documented = 0  # .expect(...) / panic!(...) — documented
    test_only = 0
    per_file: dict[str, int] = {}  # non-test undocumented per file

    for fp in files:
        try:
            src = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = src.splitlines()
        total_lines += len(lines)
        in_test = False
        brace_depth = 0
        test_start_depth = -1
        file_undoc = 0
        for line in lines:
            stripped = line.strip()
            if not in_test and ("#[cfg(test)]" in stripped or "mod tests" in stripped):
                in_test = True
                test_start_depth = brace_depth
            if in_test:
                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= test_start_depth and stripped.endswith("}"):
                    in_test = False
                    test_start_depth = -1
                if _UNWRAP_PAT.search(line) or _EXPECT_PAT.search(line) or _PANIC_PAT.search(line):
                    test_only += 1
            else:
                brace_depth += line.count("{") - line.count("}")
                if _UNWRAP_PAT.search(line):
                    undocumented += 1
                    file_undoc += 1
                if _EXPECT_PAT.search(line) or _PANIC_PAT.search(line):
                    documented += 1
        if file_undoc:
            per_file[str(fp.relative_to(REPO))] = file_undoc

    undoc_rate = undocumented / (total_lines / 1000) if total_lines else 0.0
    total_rate = (undocumented + documented) / (total_lines / 1000) if total_lines else 0.0
    # The threshold applies to undocumented unwraps — those are the real smell.
    status = "ok" if undoc_rate <= THRESHOLDS["rust_unwrap_per_kloc"] else "warn"
    detail = [
        f"Undocumented `.unwrap()` sites: **{undocumented}** across {len(per_file)} files",
        f"Documented `.expect(...)` / `panic!(...)` sites: {documented}",
        f"Test-only panic sites (acceptable): {test_only}",
        f"Total Rust LOC: {total_lines:,}",
        f"Undocumented rate: **{undoc_rate:.2f}** / kloc (threshold ≤ {THRESHOLDS['rust_unwrap_per_kloc']})",
        f"Total panic-site rate: {total_rate:.2f} / kloc (informational)",
        "",
        "Top files by undocumented `.unwrap()` count:",
    ]
    for fp, n in sorted(per_file.items(), key=lambda x: -x[1])[:10]:
        detail.append(f"- `{fp}` — {n}")
    if not per_file:
        detail.append("- _(none)_")
    return AxisResult(
        name="Rust panic surface",
        summary=f"{undocumented} undocumented `.unwrap()` ({undoc_rate:.2f}/kloc), {documented} documented",
        detail=detail,
        status=status,
        metric=f"{undoc_rate:.2f}",
    )


# ── Axis 2: Python type strictness ──────────────────────────────────────────


def axis_python_type_strictness() -> AxisResult:
    py = tomllib.loads(PYPROJECT.read_text())
    ty_rules = py.get("tool", {}).get("ty", {}).get("rules", {})
    ignored = [k for k, v in ty_rules.items() if v == "ignore"]

    # Count `type:ignore` suppression comments and `cast(Any` calls in pysrc/.
    type_ignore = 0
    cast_any = 0
    for fp in _files_under(PYSRC, ".py"):
        text = fp.read_text(encoding="utf-8", errors="replace")
        type_ignore += len(re.findall(r"#\s*type:\s*ignore", text))
        cast_any += len(re.findall(r"\bcast\s*\(\s*Any\b", text))

    status = "warn" if len(ignored) > THRESHOLDS["ty_suppressions"] else "ok"
    detail = [
        f"Project-wide `ty` rule suppressions: **{len(ignored)}** (threshold ≤ {THRESHOLDS['ty_suppressions']})",
        f"In-source `# type: ignore` comments in pysrc/: {type_ignore}",
        f"`cast(Any, ...)` calls in pysrc/: {cast_any}",
        "",
        "Suppressed rules:",
    ]
    for r in ignored:
        detail.append(f"- `{r}` = `ignore`")
    if not ignored:
        detail.append("- _(none)_")
    return AxisResult(
        name="Python type strictness",
        summary=f"{len(ignored)} ty rules suppressed, {type_ignore} `# type: ignore`",
        detail=detail,
        status=status,
        metric=f"{len(ignored)}",
    )


# ── Axis 3: Test coverage (pysrc/) ──────────────────────────────────────────


def axis_test_coverage() -> AxisResult:
    rc, _, _ = _run(["uv", "run", "python", "-c", "import coverage"], timeout=10)
    if rc != 0:
        return AxisResult(
            name="Test coverage (pysrc/)",
            summary="`coverage` not installed (add to dev group)",
            detail=["Add `coverage[toml]>=7.0` to `[dependency-groups] dev` to enable."],
            status="n/a",
        )

    rc, _, _ = _run(
        [
            "uv",
            "run",
            "coverage",
            "run",
            "--source=pysrc/solweig",
            "-m",
            "pytest",
            "tests/",
            "-m",
            "not slow",
            "-q",
            "--no-header",
            "-x",
        ],
        timeout=600,
    )
    if rc not in (0, 5):  # 5 = no tests collected after filter
        return AxisResult(
            name="Test coverage (pysrc/)",
            summary="coverage run failed (see CI for fast-test status)",
            detail=[f"`coverage run` exit code: {rc}"],
            status="n/a",
        )

    rc, out, _ = _run(["uv", "run", "coverage", "report", "--format=total"], timeout=30)
    if rc != 0:
        return AxisResult(
            name="Test coverage (pysrc/)",
            summary="coverage report failed",
            status="n/a",
        )
    try:
        pct = float(out.strip())
    except ValueError:
        return AxisResult(name="Test coverage (pysrc/)", summary=f"unparseable output: {out!r}", status="n/a")

    # Per-file breakdown for the lowest-covered modules.
    _, breakdown, _ = _run(["uv", "run", "coverage", "report", "--skip-covered", "--sort=Cover"], timeout=30)
    rows = [ln for ln in breakdown.splitlines() if ln.startswith("pysrc/")][:10]

    status = "ok" if pct >= THRESHOLDS["test_coverage_pct"] else "warn"
    target_pct = THRESHOLDS["test_coverage_pct"]
    detail = [
        f"Line coverage on pysrc/solweig (fast tests only): **{pct:.1f}%** (target ≥ {target_pct:.0f}%)",
        "",
        "Lowest-covered modules:",
        "```",
        *rows,
        "```",
    ]
    return AxisResult(
        name="Test coverage (pysrc/)",
        summary=f"{pct:.1f}% line coverage on pysrc/solweig (fast tests)",
        detail=detail,
        status=status,
        metric=f"{pct:.1f}",
    )


# ── Axis 4: CI vs local task gap ────────────────────────────────────────────


def axis_ci_gap() -> AxisResult:
    py = tomllib.loads(PYPROJECT.read_text())
    poe_tasks_raw: dict = py.get("tool", {}).get("poe", {}).get("tasks", {})
    poe_tasks = list(poe_tasks_raw.keys())

    # poe task `shell:` strings — the actual command each task runs.
    poe_shells: dict[str, str] = {}
    for name, defn in poe_tasks_raw.items():
        if isinstance(defn, dict) and "shell" in defn:
            poe_shells[name] = defn["shell"]
        elif isinstance(defn, str):
            poe_shells[name] = defn

    # Parse each workflow as YAML so we look at actual `run:` commands rather
    # than raw text — avoids false positives from comments mentioning a task.
    run_commands: list[str] = []
    if WORKFLOWS.exists():
        try:
            import yaml
        except ImportError:
            yaml = None  # type: ignore[assignment]
        for fp in WORKFLOWS.glob("*.yml"):
            if yaml is None:
                # Fallback: regex-extract `run: ...` blocks
                for m in re.finditer(r"^\s*run:\s*(.+)$", fp.read_text(), re.MULTILINE):
                    run_commands.append(m.group(1))
                continue
            try:
                doc = yaml.safe_load(fp.read_text())
            except Exception:
                continue
            for job in (doc or {}).get("jobs", {}).values():
                for step in (job or {}).get("steps", []) or []:
                    if isinstance(step, dict) and "run" in step:
                        run_commands.append(str(step["run"]))
    ci_text = "\n".join(run_commands)

    # A poe task is "covered" if:
    #   (a) the workflow runs `poe <name>` directly, OR
    #   (b) the workflow runs the same underlying command the task wraps.
    referenced: set[str] = set()
    for t in poe_tasks:
        if re.search(rf"\bpoe\s+{re.escape(t)}\b", ci_text):
            referenced.add(t)
            continue
        # Heuristic: if the workflow runs the same `pytest tests/...` path the
        # poe task wraps, count it as covered.
        shell = poe_shells.get(t, "")
        if shell.startswith("pytest "):
            # Extract the path/marker fragment after `pytest `.
            fragment = shell[len("pytest ") :].strip()
            # The most distinctive part is the tests/ path.
            first_path = next((tok for tok in fragment.split() if tok.startswith("tests/")), "")
            if first_path and first_path in ci_text:
                referenced.add(t)
        elif shell.startswith("ruff "):
            if "ruff check" in ci_text:
                referenced.add(t)
        elif shell.startswith("ty "):
            if re.search(r"\bty\s+check\b", ci_text):
                referenced.add(t)
        elif shell.startswith("mkdocs "):
            if "mkdocs build" in ci_text or "mkdocs serve" in ci_text:
                referenced.add(t)

    # Tasks intentionally not in CI. Document the reason for each so it's clear
    # that absence is a choice, not an oversight.
    INTENTIONALLY_SKIPPED = {
        "docs": "interactive (mkdocs serve)",
        "notebooks": "developer-only notebook execution",
        "verify_project": "aggregate of lint+typecheck+tests; CI runs each separately",
        "test_gpu_gates": "requires GPU; standard GitHub Actions runners are CPU-only",
        "test_gpu_perf_gate": "requires GPU; standard GitHub Actions runners are CPU-only",
    }

    missing = sorted(set(poe_tasks) - referenced - set(INTENTIONALLY_SKIPPED))
    status = "warn" if len(missing) > THRESHOLDS["ci_gap_count"] else "ok"
    detail = [
        f"poe tasks defined: {len(poe_tasks)} ({', '.join(poe_tasks)})",
        f"Wired to CI (direct or by command): {len(referenced)}",
        f"Intentionally not in CI: {len(INTENTIONALLY_SKIPPED)} ({', '.join(sorted(INTENTIONALLY_SKIPPED))})",
        f"Unintended gap: **{len(missing)}** (threshold ≤ {THRESHOLDS['ci_gap_count']})",
        "",
        "Gap:",
    ]
    for t in missing:
        detail.append(f"- `{t}`")
    if not missing:
        detail.append("- _(none)_")
    if INTENTIONALLY_SKIPPED:
        detail += ["", "Intentionally skipped (with reason):"]
        for t, reason in sorted(INTENTIONALLY_SKIPPED.items()):
            detail.append(f"- `{t}` — {reason}")
    return AxisResult(
        name="CI vs local task gap",
        summary=f"{len(missing)} unintended poe task(s) not in CI",
        detail=detail,
        status=status,
        metric=f"{len(missing)}",
    )


# ── Axis 5: Public API discipline ───────────────────────────────────────────


def axis_public_api() -> AxisResult:
    init = PYSRC / "__init__.py"
    if not init.exists():
        return AxisResult(name="Public API discipline", summary="no __init__.py", status="n/a")

    tree = ast.parse(init.read_text())
    all_list: list[str] = []
    exposed: set[str] = set()

    # IMPORTANT: only inspect the module's TOP-LEVEL body. Walking the whole
    # AST (`ast.walk`) recurses into function bodies and treats local
    # variables as "exposed", producing false positives. The module's public
    # surface is what's bound at module scope — imports, top-level functions,
    # top-level classes, and top-level assignments.
    # Protocol-hook dunders that are part of the Python data model but never
    # imported as user-facing public API. These get filtered from the
    # "exposed" set so they don't show as `leaked from __all__`.
    _PROTOCOL_DUNDERS = frozenset(
        {
            "__getattr__",  # PEP 562 module-level attribute hook
            "__dir__",  # PEP 562 module-level dir() hook
            "__init__",
            "__class__",
            "__doc__",
            "__name__",
            "__file__",
            "__path__",
            "__package__",
            "__loader__",
            "__spec__",
            "__builtins__",
        }
    )

    def _is_public(name: str) -> bool:
        if name in _PROTOCOL_DUNDERS:
            return False
        # Other dunders are part of the public data model (e.g. __version__).
        if name.startswith("__") and name.endswith("__"):
            return True
        return not name.startswith("_")

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if not isinstance(t, ast.Name):
                    continue
                if t.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                    for el in node.value.elts:
                        if isinstance(el, ast.Constant) and isinstance(el.value, str):
                            all_list.append(el.value)
                elif _is_public(t.id):
                    exposed.add(t.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if _is_public(node.name):
                exposed.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                if name != "*" and _is_public(name):
                    exposed.add(name)
        elif isinstance(node, ast.Import):
            # `import foo` exposes `foo` at module level.
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if _is_public(name):
                    exposed.add(name)
        elif isinstance(node, ast.Try):
            # `try: from X import Y` is the standard optional-import idiom.
            for sub in node.body:
                if isinstance(sub, ast.ImportFrom):
                    for alias in sub.names:
                        name = alias.asname or alias.name
                        if name != "*" and _is_public(name):
                            exposed.add(name)
                elif isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name) and _is_public(t.id):
                            exposed.add(t.id)

    in_all = set(all_list)
    leaked = sorted(exposed - in_all)
    missing_from_exposure = sorted(in_all - exposed)
    ratio = (len(in_all) / len(exposed) * 100) if exposed else 0.0

    status = "ok" if not leaked and not missing_from_exposure else "warn"
    detail = [
        f"`__all__` size: **{len(all_list)}** entries",
        f"Top-level public names: {len(exposed)}",
        f"Coverage of public names by `__all__`: **{ratio:.0f}%**",
        f"Public names NOT in `__all__` (leaked surface): {len(leaked)}",
        f"`__all__` entries NOT exposed at top level (stale): {len(missing_from_exposure)}",
    ]
    if leaked:
        detail += ["", "Leaked (in module but not declared public):"]
        detail += [f"- `{n}`" for n in leaked[:20]]
        if len(leaked) > 20:
            detail.append(f"- … and {len(leaked) - 20} more")
    if missing_from_exposure:
        detail += ["", "Stale `__all__` entries:"]
        detail += [f"- `{n}`" for n in missing_from_exposure[:20]]
    return AxisResult(
        name="Public API discipline",
        summary=f"{len(all_list)} in __all__, {len(leaked)} leaked, {len(missing_from_exposure)} stale",
        detail=detail,
        status=status,
        metric=f"{len(leaked)}",
    )


# ── Axis 6: Docstring coverage on public API ────────────────────────────────


def axis_docstring_coverage() -> AxisResult:
    """% of public functions/classes (no leading _) with a docstring.

    Symbols are counted as "public" only when neither the symbol itself nor
    any enclosing class is private (leading underscore). Methods of a
    private class like `_BooleanArray.all()` are NOT public API and don't
    count toward the denominator. Nested public-in-private also doesn't
    count — if the outer wrapper isn't public, neither is the inner detail.
    """
    total = 0
    documented = 0
    undocumented_examples: list[str] = []

    def _walk(node, parent_is_private: bool, fp):
        nonlocal total, documented
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                child_private = parent_is_private or child.name.startswith("_")
                if not child_private:
                    total += 1
                    if ast.get_docstring(child):
                        documented += 1
                    elif len(undocumented_examples) < 15:
                        rel = fp.relative_to(REPO)
                        undocumented_examples.append(f"{rel}:{child.lineno} `{child.name}`")
                _walk(child, child_private, fp)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_private = parent_is_private or child.name.startswith("_")
                if not fn_private:
                    total += 1
                    if ast.get_docstring(child):
                        documented += 1
                    elif len(undocumented_examples) < 15:
                        rel = fp.relative_to(REPO)
                        undocumented_examples.append(f"{rel}:{child.lineno} `{child.name}`")
                # No need to recurse into function bodies for docstring coverage.
            else:
                _walk(child, parent_is_private, fp)

    for fp in _files_under(PYSRC, ".py"):
        try:
            tree = ast.parse(fp.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        _walk(tree, parent_is_private=False, fp=fp)

    pct = (documented / total * 100) if total else 100.0
    status = "ok" if pct >= THRESHOLDS["public_api_documented_pct"] else "warn"
    detail = [
        f"Public symbols (functions/classes, no leading _): **{total}**",
        f"With docstring: {documented}",
        f"Coverage: **{pct:.1f}%** (target ≥ {THRESHOLDS['public_api_documented_pct']:.0f}%)",
    ]
    if undocumented_examples:
        detail += ["", "Undocumented (sample of first 15):"]
        detail += [f"- {ex}" for ex in undocumented_examples]
    return AxisResult(
        name="Docstring coverage (public API)",
        summary=f"{pct:.1f}% of {total} public symbols documented",
        detail=detail,
        status=status,
        metric=f"{pct:.1f}",
    )


# ── Axis 7: Hot files & TODO/FIXME density ──────────────────────────────────

_TODO_PAT = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")


def axis_hot_files() -> AxisResult:
    big_py: list[tuple[int, str]] = []
    for fp in _files_under(PYSRC, ".py"):
        n = _count_lines(fp)
        if n >= THRESHOLDS["hot_file_max_lines"]:
            big_py.append((n, str(fp.relative_to(REPO))))
    for fp in _files_under(RUST_SRC, ".rs"):
        n = _count_lines(fp)
        if n >= THRESHOLDS["hot_file_max_lines"]:
            big_py.append((n, str(fp.relative_to(REPO))))
    big_py.sort(reverse=True)

    todo_count = 0
    todo_examples: list[str] = []
    for root in (PYSRC, RUST_SRC):
        for fp in _files_under(root, ".py") + _files_under(root, ".rs"):
            try:
                for i, line in enumerate(fp.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if _TODO_PAT.search(line):
                        todo_count += 1
                        if len(todo_examples) < 15:
                            todo_examples.append(f"{fp.relative_to(REPO)}:{i}  {line.strip()[:120]}")
            except OSError:
                continue

    status = "warn" if big_py or todo_count > 30 else "ok"
    detail = [
        f"Files ≥ {THRESHOLDS['hot_file_max_lines']} lines: **{len(big_py)}**",
        f"`TODO`/`FIXME`/`XXX`/`HACK` markers in pysrc + rust: **{todo_count}**",
    ]
    if big_py:
        detail += ["", "Large files:"]
        detail += [f"- {n:>5} lines — `{p}`" for n, p in big_py]
    if todo_examples:
        detail += ["", "TODO sample (first 15):"]
        detail += [f"- {ex}" for ex in todo_examples]
    return AxisResult(
        name="Hot files & TODO density",
        summary=f"{len(big_py)} files ≥ {THRESHOLDS['hot_file_max_lines']} lines, {todo_count} TODOs",
        detail=detail,
        status=status,
        metric=f"{todo_count}",
    )


# ── Axis 8: Dependency freshness ────────────────────────────────────────────


def axis_dependency_freshness() -> AxisResult:
    detail: list[str] = []
    py_outdated = "n/a"
    rust_outdated = "n/a"

    rc, out, _ = _run(["uv", "lock", "--upgrade", "--dry-run"], timeout=90)
    if rc == 0:
        # Lines that contain ` -> ` indicate an upgrade.
        upgrade_lines = [ln.strip() for ln in out.splitlines() if " -> " in ln]
        py_outdated = str(len(upgrade_lines))
        detail.append(f"Python deps with available upgrades: **{py_outdated}**")
        if upgrade_lines:
            detail.append("")
            detail.append("Available upgrades (first 20):")
            for ln in upgrade_lines[:20]:
                detail.append(f"- `{ln}`")
    else:
        detail.append(f"`uv lock --upgrade --dry-run` failed (rc={rc}); skipping Python dep check")

    rc, _, _ = _run(["cargo", "outdated", "--version"], timeout=10)
    if rc == 0:
        rc, out, _ = _run(["cargo", "outdated", "-R", "--manifest-path", "rust/Cargo.toml"], timeout=120)
        if rc == 0:
            lines = [ln for ln in out.splitlines() if ln and not ln.startswith("Name") and not ln.startswith("----")]
            rust_outdated = str(len(lines))
            detail.append("")
            detail.append(f"Rust deps with available upgrades: **{rust_outdated}**")
    else:
        detail.append("")
        detail.append("`cargo outdated` not installed — `cargo install cargo-outdated` to enable Rust dep check")

    status = "info"
    if py_outdated.isdigit() and int(py_outdated) > THRESHOLDS["py_outdated_count"]:
        status = "warn"
    summary = f"Python: {py_outdated} upgrade(s) available, Rust: {rust_outdated}"
    return AxisResult(
        name="Dependency freshness",
        summary=summary,
        detail=detail,
        status=status,
        metric=f"{py_outdated}/{rust_outdated}",
    )


# ── Report rendering ────────────────────────────────────────────────────────


_STATUS_ICON = {"ok": "✅", "warn": "⚠️ ", "fail": "❌", "info": "ℹ️ ", "n/a": "—"}


def _git_commit() -> str:
    rc, out, _ = _run(["git", "rev-parse", "--short", "HEAD"], timeout=5)
    return out.strip() if rc == 0 else "unknown"


def _git_branch() -> str:
    rc, out, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=5)
    return out.strip() if rc == 0 else "unknown"


def render(results: list[AxisResult]) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append("# Repository audit snapshot")
    lines.append("")
    lines.append(
        f"_Generated by `scripts/audit.py` on {now}_  ·  commit `{_git_commit()}`  ·  branch `{_git_branch()}`"
    )
    lines.append("")
    lines.append(
        "Re-run with `poe audit`. Thresholds live at the top of "
        "[`scripts/audit.py`](scripts/audit.py); adjust there, not here."
    )
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Axis | Status | Headline | Metric |")
    lines.append("|------|:------:|----------|:------:|")
    for r in results:
        lines.append(f"| {r.name} | {_STATUS_ICON.get(r.status, '?')} | {r.summary} | `{r.metric or 'n/a'}` |")
    lines.append("")

    # Per-axis sections
    for r in results:
        lines.append(f"## {r.name}")
        lines.append("")
        lines.append(f"**Status:** {_STATUS_ICON.get(r.status, '?')} {r.status}  ·  **{r.summary}**")
        lines.append("")
        lines.extend(r.detail)
        lines.append("")

    # Legend
    lines.append("---")
    lines.append("")
    lines.append("**Legend:** ✅ ok · ⚠️ warn (above threshold) · ❌ fail · ℹ️ info-only · — n/a")
    return "\n".join(lines) + "\n"


def main() -> int:
    axes = [
        axis_rust_panics,
        axis_python_type_strictness,
        axis_test_coverage,
        axis_ci_gap,
        axis_public_api,
        axis_docstring_coverage,
        axis_hot_files,
        axis_dependency_freshness,
    ]
    results: list[AxisResult] = []
    for fn in axes:
        print(f"[audit] {fn.__name__} …", file=sys.stderr)
        try:
            results.append(fn())
        except Exception as e:  # pragma: no cover — defensive
            results.append(AxisResult(name=fn.__name__, summary=f"axis crashed: {e!r}", status="fail"))

    REPORT.write_text(render(results), encoding="utf-8")
    print(f"[audit] wrote {REPORT.relative_to(REPO)}", file=sys.stderr)

    # Optional: dump a JSON sidecar for trend tracking by other tools.
    if os.environ.get("AUDIT_JSON"):
        sidecar = REPO / "audit-metrics.json"
        payload = {
            "commit": _git_commit(),
            "branch": _git_branch(),
            "generated_at": datetime.now(UTC).isoformat(),
            "axes": {r.name: {"status": r.status, "metric": r.metric, "summary": r.summary} for r in results},
        }
        sidecar.write_text(json.dumps(payload, indent=2))
        print(f"[audit] wrote {sidecar.relative_to(REPO)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
