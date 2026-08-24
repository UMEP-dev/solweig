# Tutorials

The tutorials walk through complete SOLWEIG workflows on real city data
that ships with the repository. Each page below is the rendered output of
a Jupyter notebook, so you can read them here without installing
anything.

- [Athens Quick Start](01-athens-quickstart.md) — from GeoTIFFs to a Tmrt
  map in a few lines.
- [Timeseries Analysis](02-timeseries-analysis.md) — multi-day runs,
  summary grids, and charts.
- [Thermal Comfort](03-thermal-comfort.md) — UTCI and PET outputs.
- [Terrain Shadows (Bilbao)](04-bilbao-terrain-shadows.md) — hilly
  terrain and DEM handling.
- [Ground Scheme (experimental)](05-ground-scheme-experimental.md) — the
  opt-in UMEP 2026a ground-surface scheme, with caveats.

## Running the tutorials yourself

The notebooks live in `docs/tutorials/*.ipynb` in the repository, and the
demo data they use (Athens, Gothenburg, Bilbao) is included in
`demos/data/`, so a clone is all the setup they need:

```bash
git clone https://github.com/UMEP-dev/solweig.git
cd solweig
pip install solweig[geo] matplotlib jupyter
jupyter lab docs/tutorials/01-athens-quickstart.ipynb
```

Each notebook resolves the repository root itself, so it runs whether
Jupyter is started from the repository root or from `docs/tutorials/`.
Outputs are written to the gitignored `temp/` directory.

The Madrid large-raster demo is not part of the tutorials because its
LiDAR inputs are too large to ship; `scripts/fetch_madrid_data.py`
downloads them from IGN's open data, and `demos/madrid-demo.py` runs the
computation.

Note for contributors: the Markdown pages on this site are exported from
the notebooks by `poe notebooks` — edit the `.ipynb` sources, not the
`.md` exports.
