# Contributing

Contributions to SOLWEIG are welcome.

## Development Setup

### Prerequisites

- Python 3.11–3.13
- Rust toolchain (for building extensions)
- uv (package manager)

### Clone and Install

```bash
git clone https://github.com/UMEP-dev/solweig.git
cd solweig

# Install dependencies (including dev and test groups)
uv sync --group test --group dev

# Build Rust extension (must be --release; the test suite rejects debug builds)
maturin develop --release
```

### Verify Installation

```bash
# Run tests
pytest ./tests

# Full verification (format, lint, typecheck, test)
poe verify_project
```

## Development Workflow

### Making Changes

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make changes
3. Run verification: `poe verify_project`
4. Commit with clear messages
5. Open a pull request

### Code Style

The following tools are used (configured in `pyproject.toml`):

| Tool | Purpose |
| ---- | ------- |
| **ruff** | Linting and formatting |
| **ty** | Type checking |
| **pytest** | Testing |

Run all checks:

```bash
poe verify_project
```

### Testing

Tests are located in `tests/`:

- `tests/spec/` — Physical property tests (shadows, SVF, radiation)
- `tests/golden/` — Reference data validation
- `tests/test_api.py` — Integration tests

Add tests for new functionality:

```bash
# Run a specific test file
pytest tests/test_api.py

# Run with coverage
uv run coverage run -m pytest tests/
uv run coverage report
```

## Project Structure

```
pysrc/solweig/          # Python source
  api.py                # Public API
  models/               # Dataclasses (SurfaceData, Weather, etc.)
  components/           # Modular calculation functions
  computation.py        # Core orchestration
rust/                   # Rust extensions
specs/                  # Module specifications
tests/                  # Test suite
docs/                   # Documentation (MkDocs)
```

## Types of Contributions

### Bug Reports

Open an issue with:

- A description of the bug
- Steps to reproduce
- Expected vs. actual behaviour
- Version information

### Feature Requests

Open an issue describing:

- The use case
- Proposed approach
- Alternatives considered

### Code Contributions

1. Check existing issues for related work
2. Discuss major changes in an issue first
3. Follow the code style guidelines
4. Add tests for new functionality
5. Update documentation as needed

## Acknowledgements

SOLWEIG is adapted from the original UMEP (Urban Multi-scale Environmental Predictor) code by Fredrik Lindberg, Ting Sun, Sue Grimmond, Yihao Tang, and Nils Wallenberg.

If you use SOLWEIG in research, please cite:

> Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services. Environmental Modelling and Software 99, 70-87 [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

## License

By contributing, you agree that your contributions will be licensed under the GNU General Public License v3.0 (GPL-3.0).
