# Repository audit

The repository carries a self-audit script
([`scripts/audit.py`](https://github.com/UMEP-dev/solweig/blob/main/scripts/audit.py))
that measures eight signals and writes a single
[`AUDIT.md`](https://github.com/UMEP-dev/solweig/blob/main/AUDIT.md) at the repo root.

```bash
poe audit
```

The report is checked in so ``git diff AUDIT.md`` shows drift between
snapshots. CI runs the audit as an informational job and uploads the
generated `AUDIT.md` as a workflow artifact (the job is intentionally
non-blocking — warnings are signals to act on, not gates).

## The eight axes

| Axis | What it measures | Why it matters |
| --- | --- | --- |
| **Rust panic surface** | Count of `.unwrap()` (undocumented) vs `.expect("...")` / `panic!(…)` (documented), excluding `#[cfg(test)]` blocks | `unwrap()` with no message is a code smell; `expect(...)` with a rationale is defensive documentation |
| **Python type strictness** | Count of `[tool.ty.rules]` suppressions, in-source `type:ignore` comments, `cast(Any, …)` calls | Each suppression hides a latent typing issue |
| **Test coverage** | Two-track: **full-suite** (slow + golden + validation) and **fast-tests only** (per-PR CI) | Full is the honest "is the code reached" number; fast is what runs on every PR |
| **CI vs local task gap** | Parses each `poe` task and each `.github/workflows/*.yml` `run:` step, computes which tasks are not in CI. Intentionally-skipped tasks have a documented reason | Catches local-only checks drifting away from what CI enforces |
| **Public API discipline** | Compares `solweig.__all__` to the actual top-level public symbols. Reports leaks (in module but not in `__all__`) and stale entries (in `__all__` but not exposed) | The published API contract should match what's actually exported |
| **Docstring coverage** | % of public functions and classes (excluding methods of leading-underscore classes) with a docstring | The bar for "public" includes top-level + nested public symbols, but skips nested-in-private |
| **Hot files & TODO density** | Files ≥ 700 lines + count of `TODO`/`FIXME`/`XXX`/`HACK` markers | Large files are decomposition candidates |
| **Dependency freshness** | `uv lock --upgrade --dry-run` upgrade count for Python; `cargo outdated` for Rust if installed | Stale deps accumulate latent vulnerabilities + miss fixes |

## Thresholds

The pass/fail thresholds live at the top of
[`scripts/audit.py`](https://github.com/UMEP-dev/solweig/blob/main/scripts/audit.py)
in the `THRESHOLDS` dict. They are deliberately tunable rather than
hard-coded values throughout the script — adjust there, not in the
generated `AUDIT.md`.

```python
THRESHOLDS = {
    "rust_unwrap_per_kloc": 5.0,
    "ty_suppressions": 3,
    "test_coverage_pct": 80.0,
    "ci_gap_count": 1,
    "public_api_documented_pct": 90.0,
    "hot_file_max_lines": 700,
    "py_outdated_count": 10,
}
```

## When to run the audit

- **Locally, before submitting a PR:** ``poe audit`` regenerates
  `AUDIT.md`. `git diff AUDIT.md` shows what your change affected.
- **In CI:** the `audit` workflow job runs on every push to `main` /
  `dev` and on every PR, uploading the generated report as an artifact.
  It does NOT block merging — the report is feedback, not a gate.
- **After landing a substantial refactor:** the audit will often shift
  multiple axes at once (rust panic count, hot-file count, docstring
  coverage). Compare to the previous `AUDIT.md` to confirm the change
  improved things.

## When to update a threshold

When you've made structural improvement that raises the bar
permanently — e.g. you've driven `.unwrap()` count to 0 and want the
audit to fail if a regression sneaks back in. Update the threshold
in `audit.py` so the audit fails (`status: warn`) on regression rather
than silently accepting it.

When you have **not** made improvement and the threshold is just
inconvenient — leave it alone. The whole point of the audit is to
keep visible the things that the project hasn't gotten to yet.

## Limitations (honestly documented)

These are intentionally accepted in the current implementation:

- **Rust panic detector** uses string parsing, not full Rust parsing.
  `// .unwrap()` in a comment would be miscounted (manual check: there
  are no such cases today). Test-block detection is heuristic on brace
  depth from `#[cfg(test)]` / `mod tests {` — single `#[test]` items at
  the top level of a non-test module aren't recognised.
- **CI gap detector** uses YAML parsing of `.github/workflows/*.yml`
  and matches each `run:` line against `poe <name>` or the underlying
  command (`pytest tests/…`, `ruff check`, `ty check`, `mkdocs build`).
  Other invocation styles (Makefile, shell script) aren't recognised.
- **Public API audit** treats anything from leading-underscore as
  private. Protocol-hook dunders (`__getattr__`, `__init__`, etc.) are
  in an explicit exclusion list so they don't show as leaked.
- **Test coverage** runs the test suite twice (fast + full), which is
  slow (~3-5 min). The cached `.coverage` file from the second run is
  what feeds the per-module breakdown.

If any of these limits start mattering in practice, the audit script
is the single source of truth — adjust the offending heuristic there
and document why.
