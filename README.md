# AEP

AEP fits a smooth neural map from high-dimensional data (for example, single-cell expression
profiles) to three dimensions. The map is trained so that the embedded points lie on a
two-dimensional **surface** in 3D. The surface keeps each point's neighborhood undistorted and also
preserves distances between points.

The README has two parts:

- **[Part 1 — For humans](#part-1--for-humans-what-aep-does)**: what the method is meant to do,
  and what each parameter of `fit_aep` controls.
- **[Part 2 — For coding agents](#part-2--for-coding-agents-clone-install-verify)**: how to clone,
  install and check the repository.

---

## Part 1 — For humans: what AEP does

### The idea

Many datasets are high-dimensional but have a low-dimensional structure. Cells along a
differentiation trajectory or the points of a rolled-up sheet are two examples. Common
visualizations such as PCA, t-SNE and UMAP each have a limitation:

- **PCA** is linear, so it flattens curved structure.
- **t-SNE and UMAP** preserve neighborhoods, but they tear, shrink or stretch the space between
  groups.

AEP takes a different approach. It learns a function `f: R^d -> R^3`, a small multilayer
perceptron, that places the data on a smooth 2D surface in 3D. A surface can be rotated, viewed
from any angle and read like a curved map. Three terms in the training objective shape it:

1. **Local isometry (as rigid as possible).** At every point, AEP estimates the local 2D tangent
   plane from the point's `k` nearest neighbors. The map's Jacobian is pushed to act on that plane
   as a rotation: distances in the neighborhood are kept, with no local stretching or squashing.
2. **Rank-2 penalty.** The smallest singular value of the Jacobian is pushed to zero. Locally, the
   map then collapses to a plane, so the image is a surface and not a 3D cloud.
3. **Distance-preserving stress.** Within each training batch, distances in the embedding are
   matched to distances in the input. Pairs are weighted by `distance^-p`, so with the default
   `p = 2` the stress measures *relative* error. This keeps the global layout faithful without
   letting large distances dominate.

The map is a function, not a lookup table, so it also embeds **new points** that were not used
for training (`model(X_new)`).

**Replicates.** If the data come from several samples or batches, pass their labels with
`replicate=`. AEP then fits **one shared map** to all of them, so the replicates land on the same
surface and can be compared directly. Neighbors, tangent planes and stress pairs are computed
within each replicate, so no cross-replicate distances are assumed. The input should already be
batch-corrected.

### Basic use

```python
import numpy as np
from aep import fit_aep, AEPModel

X = ...                                  # (n, D) array: points in rows, features in columns
Y, model, info = fit_aep(X)              # Y: (n, 3) AEP coordinates

Y_new = model(X_new)                     # embed new points with the same D features
model.save("aep_model.pt")
model = AEPModel.load("aep_model.pt")
```

With replicates: `Y, model, info = fit_aep(X, replicate=sample_labels)`.

See [`notebooks/swiss_roll_tutorial.ipynb`](notebooks/swiss_roll_tutorial.ipynb) for a complete
example. It generates a swiss roll in R^4, fits AEP and shows interactive plotly views.

### Parameters of `fit_aep`

All parameters except `X_raw` are optional. The defaults work well in most cases; the ones most
worth changing are marked with ★.

| Parameter | Default | What it does |
|---|---|---|
| `X_raw` | — | Data matrix `(n, D)`: points in rows, features in columns. |
| `replicate` ★ | `None` | One label per row (sample, donor, batch). One map is fit to all replicates; neighbors, tangent planes and stress pairs stay within each replicate, and each training batch is drawn from one replicate. `None` means all rows form one replicate. |
| `pca` | `True` | Project the data onto its top principal components before fitting. This removes noise and speeds up fitting. Set to `False` if `X_raw` is already a low-dimensional representation, such as PC scores or a latent space. |
| `n_pca` ★ | `min(D, 10)` | Number of PCs `d` used as the working space. Must satisfy `3 ≤ d ≤ D`. |
| `pca_solver` | `"auto"` | `"full"` (exact SVD) or `"randomized"` (fast, seeded). `"auto"` uses exact SVD when `min(n, D) ≤ 500`. |
| `k` ★ | `20` | Neighborhood size (the point plus its `k − 1` nearest neighbors) used to estimate local tangent planes. Larger values give smoother, more global surfaces; smaller values follow finer structure. |
| `knn_method` | `"auto"` | `"exact"` or `"approx"` (NN-descent). `"auto"` is exact for replicates with up to 20,000 unique rows. |
| `stress_power` | `2.0` | Power `p` in the stress pair weights `w_ij = δ_ij^−p`. With `p = 2` (the default), errors are relative; `p = 0` gives ordinary metric stress, dominated by large distances. |
| `lambda_rank` | `1.0` | Weight of the rank-2 penalty. Higher values flatten the image more strictly onto a surface. |
| `lambda_s` ★ | `5.0` | Weight of the stress term relative to the local-isometry term. Higher values preserve global distances more; lower values favor locally undistorted neighborhoods. |
| `tau` | `1e-12` | Small numerical guard in the isometry term. Rarely needs changing. |
| `jacobian_route` | `"auto"` | `"explicit"` or `"factorized"`: how Jacobians are computed. The results are identical and `"auto"` picks the faster one. |
| `steps` | `5000` | Number of training updates (Adam optimizer, cosine learning-rate decay, exponential moving average of the weights). |
| `batch_size` | `256` | Points per training batch. All pairs within a batch enter the stress term. |
| `lr` | `1e-3` | Initial learning rate. |
| `seed` | `0` | Random seed (initialization, batch sampling, randomized SVD, NN-descent). With the same seed and the same input, results are reproducible. |
| `run_name` | `None` | If given, saves `{out_dir}/logs/{run_name}.csv` (training log), `{out_dir}/embeddings/{run_name}.npy` (coordinates) and `{out_dir}/embeddings/{run_name}.pt` (model). |
| `out_dir` | `"results"` | Output folder used with `run_name`. |

**Returns** `(Y, model, info)`:

- `Y` is an `(n, 3)` array of AEP coordinates.
- `model` is an `AEPModel`. Call it on new data (`model(X_new)`); save it with
  `model.save(path)` and restore it with `AEPModel.load(path)`.
- `info` is a dictionary. It holds the settings used, the PCA variance retained, the replicate
  sizes, timings and the loss terms of the last batch.

The network is fixed: `d → 64 → 64 → 64 → 3`, fully connected with SiLU activations. Training
runs on the CPU. Fitting the tutorial's 10,000-point swiss roll with the defaults takes about
20 seconds on a 16-core machine.

---

## Part 2 — For coding agents: clone, install, verify

### Clone and install

```bash
git clone git@github.com:sdzhao/aep.git      # or https://github.com/sdzhao/aep.git
cd aep

# Preferred: uv (installs CPU-only PyTorch through the index in pyproject.toml)
uv sync --extra dev --extra tutorial

# Alternative: pip in a virtual environment (installs the default PyTorch build)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,tutorial]"
```

- Python ≥ 3.10.
- Runtime dependencies: `numpy`, `scikit-learn`, `torch` and `pynndescent`.
- Extras:
  - `tutorial` adds `plotly`, `jupyter`, `ipykernel` and `nbformat`.
  - `dev` adds `pytest` and `scipy`.

### Verify

```bash
uv run pytest -q                                   # unit tests, under a minute
uv run python -c "import numpy as np, aep; Y,_,_ = aep.fit_aep(np.random.default_rng(0).normal(size=(500, 8)), steps=50); print(Y.shape)"
uv run jupyter nbconvert --to notebook --execute notebooks/swiss_roll_tutorial.ipynb --output /tmp/tutorial_check.ipynb
```

Without uv, drop the `uv run` prefix inside the activated virtual environment.

### Repository layout

```
src/aep/
  __init__.py     public API: fit_aep, AEPModel
  api.py          fit_aep: preprocessing -> training -> embedding (entry point)
  preprocess.py   PCA working space (PCATransform), replicate encoding, per-replicate dedup + k-NN + tangents
  knn.py          exact blockwise k-NN and approximate k-NN (pynndescent)
  tangents.py     top-2 local principal directions (tangent bases)
  cotangent.py    Jacobian computation routes (explicit / factorized)
  losses.py       isometry term, rank penalty, power-weighted stress, stress normalizer
  model.py        SurfaceMLP network and AEPModel (out-of-sample embedding, save/load)
  train.py        TrainConfig, ReplicateBatchSampler, training loop (Adam, cosine decay, EMA)
tests/            pytest unit tests
notebooks/        swiss_roll_tutorial.ipynb
```

### Conventions to keep

- The public API is `aep.fit_aep(X_raw, ...) -> (Y, model, info)` and `aep.AEPModel`. Keep the
  signature and defaults stable; the README parameter table documents them.
- Training uses float32 tensors. PCA, the stress pair weights and the normalizers are computed in
  float64.
- Results are deterministic for a fixed `seed` on the CPU (`setup_torch` seeds torch, numpy and
  random). Don't add unseeded randomness.
- Preprocessing (deduplication, k-NN, tangent planes, stress pairs) is always within a replicate.
  Each batch comes from a single replicate.
- No GPU code paths: the package targets the CPU.
- Writing files happens only when `run_name` is given.

### License

MIT — see [LICENSE](LICENSE).
