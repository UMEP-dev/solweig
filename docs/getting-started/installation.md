# Installation

## Install from PyPI

```bash
pip install solweig
```

Verify the installation:

```bash
python -c "import solweig; print(solweig.__version__)"
```

## Install from Source (for development)

To modify the code or contribute:

- **Python 3.11+**
- **Rust toolchain** — required to compile the compute core ([install Rust](https://rustup.rs/))
- **uv** — Python package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))

```bash
git clone https://github.com/UMEP-dev/solweig.git
cd solweig
uv sync --group test --group dev    # Install Python dependencies (dev + test groups)
maturin develop --release           # Compile Rust extension and link it
```

## Dependencies

Core geospatial dependencies are installed automatically with the package:

| Package | Purpose |
| ------- | ------- |
| `rasterio` | Loading/saving GeoTIFF rasters (install dependency) |
| `pyproj` | CRS handling and coordinate conversion (install dependency) |
| `affine` | Geospatial coordinate transforms (pulled in transitively by rasterio) |
| `geopandas` | Rasterising vector data, e.g. tree polygons to a canopy grid (optional, install separately) |

`rasterio` is a hard install dependency even for pure numpy-array workflows.
The alternative GDAL backend (no rasterio) exists only for running inside
QGIS, where the plugin uses the QGIS-bundled GDAL instead.

## GPU Acceleration

SOLWEIG uses GPU acceleration (via wgpu/Metal/Vulkan) when available. No additional setup is required.

```python
import solweig

print(f"GPU available: {solweig.is_gpu_available()}")
print(f"Backend: {solweig.get_compute_backend()}")  # "gpu" or "cpu"
```

If no GPU is found, computation falls back to CPU. To force CPU mode:

```python
solweig.disable_gpu()
```

## Troubleshooting

### `maturin: command not found`

Install via uv or pip:

```bash
uv tool install maturin
# or
pip install maturin
```

### Build errors on macOS

Ensure Xcode command line tools are installed:

```bash
xcode-select --install
```

### `import solweig` fails after `maturin develop`

Ensure the same Python environment created by `uv sync` is active. If using uv:

```bash
uv run python -c "import solweig; print('OK')"
```
