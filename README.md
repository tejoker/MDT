# Deep Diffusion Maps

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A novel implementation of diffusion maps using deep learning (neural networks). This repository contains the code for the experiments described in our [paper](https://arxiv.org/abs/2505.06087).

## Overview

Diffusion Maps is a dimensionality reduction technique that preserves the diffusion distance between data points. This implementation extends the traditional diffusion maps algorithm by using neural networks to learn the embedding, which allows for:

1. **Out-of-sample extension**: Apply the learned embedding to new data points without recomputing the entire diffusion map
2. **Scalability**: Handle larger datasets more efficiently than traditional methods
3. **Integration with deep learning pipelines**: Easily incorporate diffusion maps into existing neural network architectures

Our approach combines the theoretical guarantees of diffusion maps with the flexibility and scalability of neural networks.

## Key Features

- **TensorFlow Implementation**: Custom loss function (`DiffusionLoss`) implemented as a TensorFlow Keras loss
- **Comparative Analysis**: Experiments comparing traditional diffusion maps, deep diffusion maps, and Nyström extension
- **Multiple Datasets**: Implementations for synthetic datasets (Swiss Roll, S-Curve, Helix) and real-world datasets (MNIST, Phoneme)
- **Comprehensive Metrics**: Evaluation using mean absolute error (MAE) and mean relative error (MRE) of pairwise distances
- **Visualization Tools**: Utilities for visualizing embeddings, eigenvalue decay, and error metrics

## Installation

To install all the dependencies, create a conda environment using:

```bash
conda env create -f experiments/environment.yml
```

This will create an environment named `ddm` with all the required packages.

The multi-view (MDT) work added in this fork — `experiments/gnn_mdt`,
`experiments/graph_mdt`, `experiments/mvbench`, `tests/` — is PyTorch on Python
3.13 and needs a second environment, since TensorFlow 2.17 has no 3.13 wheels:

```bash
conda env create -f experiments/environment-mdt.yml   # creates env `mdt`
```

Use `ddm` for the original experiments plus `experiments/mvmat` and
`experiments/helix_mv`; use `mdt` for everything else.

Additionally, you may need to install LaTeX to create publication-quality plots.

## Usage

### Basic Usage

1. Activate the conda environment:

```bash
conda activate ddm
```

2. Run an experiment (e.g., Swiss Roll):

```bash
python -m experiments.swiss_roll.experiment -c experiments/swiss_roll/config.yml
python -m experiments.swiss_roll.plot_results -c experiments/swiss_roll/config.yml
```

3. Run all experiments:

```bash
bash experiments/run_experiments.sh
```

### Custom Implementation

To use the `DiffusionLoss` in your own projects:

```python
import tensorflow as tf
from diffusionloss import DiffusionLoss

# Prepare your data
X_train = ...  # Your training data

# Create the loss function
diffusion_loss = DiffusionLoss(
    X=X_train,
    sigma=0.1,  # Kernel bandwidth
    steps=10,   # Number of diffusion steps
    alpha=1.0   # Alpha parameter for normalization
)

# Create a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)  # Output dimension
])

# Compile with the diffusion loss
model.compile(optimizer='adam', loss=diffusion_loss)

# Train the model
# Note: y_train should be the indices of the training samples
indices = np.arange(len(X_train))
model.fit(X_train, indices, epochs=100, batch_size=32)

# Use the trained model for embedding new data
X_new = ...  # New data
embeddings = model.predict(X_new)
```

## Experiments

The repository includes experiments on several datasets. Each experiment compares three approaches:
1. **Traditional Diffusion Maps**: The standard algorithm
2. **Deep Diffusion Maps**: Our neural network implementation
3. **Nyström Extension**: A method for out-of-sample extension with traditional diffusion maps

### Datasets

#### Synthetic Datasets

- **Swiss Roll**: A 2D manifold embedded in 3D space, commonly used to evaluate manifold learning algorithms
- **S-Curve**: Another 2D manifold embedded in 3D space with a different topology
- **Helix**: A 1D manifold embedded in 3D space, forming a helix shape

#### Real-world Datasets

- **MNIST**: Handwritten digit dataset, used to evaluate the method on image data
- **Phoneme**: Speech dataset, used to evaluate the method on sequential data

### Neural Network Architectures

The repository includes several neural network architectures for different types of data:

- **MLP**: For tabular data (used in synthetic datasets)
- **CNN**: For image data (used in MNIST)
- **RNN/LSTM**: For sequential data (used in Phoneme)

## Configuration

Each experiment has a configuration file (`config.yml`) with the following parameters:

```yaml
data:
  npoints: 2000        # Number of data points
  split: 0.5           # Train/test split ratio
  noise: 0.0           # Noise level
  seed: 123            # Random seed
diffusion_maps:
  n_components: 2      # Number of dimensions in the embedding
  quantile: 5.0e-3     # Quantile for sigma estimation
  alpha: 1             # Alpha parameter for normalization
  steps: 100           # Number of diffusion steps
encoder:
  architecture: # This section may change depending on the data type
    units: 128         # Number of units in the hidden layers
    use_bn: False  # Whether to use batch normalization
  optimizer:
    learning_rate: 0.01  # Learning rate for the optimizer
  training:
    epochs: 5000       # Number of training epochs
    batch_size: 512    # Batch size
    validation_split: 0.1  # Validation split
    shuffle: True      # Whether to shuffle the data
    verbose: 2         # Verbosity level
output_dir: /path/to/output  # Output directory
```

You can modify these parameters to customize the experiments.

## Results

The experiments produce several outputs:

- **Embeddings**: The reduced-dimensional representations of the data
- **Eigenvalues and Log-likelihood**: Plots showing the eigenvalue decay and log-likelihood curves
- **Error Metrics**: MAE and MRE of pairwise distances, binned by distance deciles
- **Training History**: Loss curves for the neural network training

### Example visualizations

#### Swiss Roll
<div align="center">
  <table>
    <tr>
      <td align="center" width="25%">
        <img src="results/swiss_roll/original_test.png" alt="Original Test" width="100%"><br>
        <b>Original (test)</b>
      </td>
      <td align="center" width="25%">
        <img src="results/swiss_roll/projection_diffusion_maps_test_dims_1_2.png" alt="Diffusion Maps" width="100%"><br>
        <b>Diffusion Maps</b>
      </td>
      <td align="center" width="25%">
        <img src="results/swiss_roll/projection_deep_diffusion_maps_test_dims_1_2.png" alt="Deep Diffusion Maps" width="100%"><br>
        <b>Deep Diffusion Maps</b>
      </td>
      <td align="center" width="25%">
        <img src="results/swiss_roll/projection_nystrom_test_dims_1_2.png" alt="Nyström" width="100%"><br>
        <b>Nyström</b>
      </td>
    </tr>
  </table>
</div>

#### S Curve
<div align="center">
  <table>
    <tr>
      <td align="center" width="25%">
        <img src="results/s_curve/original_test.png" alt="Original Test" width="100%"><br>
        <b>Original (test)</b>
      </td>
      <td align="center" width="25%">
        <img src="results/s_curve/projection_diffusion_maps_test_dims_1_2.png" alt="Diffusion Maps" width="100%"><br>
        <b>Diffusion Maps</b>
      </td>
      <td align="center" width="25%">
        <img src="results/s_curve/projection_deep_diffusion_maps_test_dims_1_2.png" alt="Deep Diffusion Maps" width="100%"><br>
        <b>Deep Diffusion Maps</b>
      </td>
      <td align="center" width="25%">
        <img src="results/s_curve/projection_nystrom_test_dims_1_2.png" alt="Nyström" width="100%"><br>
        <b>Nyström</b>
      </td>
    </tr>
  </table>
</div>

#### Helix
<div align="center">
  <table>
    <tr>
      <td align="center" width="25%">
        <img src="results/helix/original_test.png" alt="Original Test" width="100%"><br>
        <b>Original (test)</b>
      </td>
      <td align="center" width="25%">
        <img src="results/helix/projection_diffusion_maps_test_dims_1_2.png" alt="Diffusion Maps" width="100%"><br>
        <b>Diffusion Maps</b>
      </td>
      <td align="center" width="25%">
        <img src="results/helix/projection_deep_diffusion_maps_test_dims_1_2.png" alt="Deep Diffusion Maps" width="100%"><br>
        <b>Deep Diffusion Maps</b>
      </td>
      <td align="center" width="25%">
        <img src="results/helix/projection_nystrom_test_dims_1_2.png" alt="Nyström" width="100%"><br>
        <b>Nyström</b>
      </td>
    </tr>
  </table>
</div>

#### Phoneme
<div align="center">
  <table>
    <tr>
      <td align="center" width="25%">
        <img src="results/phoneme/original_test.png" alt="Original Test" width="100%"><br>
        <b>Original (test)</b>
      </td>
      <td align="center" width="25%">
        <img src="results/phoneme/projection_diffusion_maps_test_dims_1_2.png" alt="Diffusion Maps" width="100%"><br>
        <b>Diffusion Maps</b>
      </td>
      <td align="center" width="25%">
        <img src="results/phoneme/projection_deep_diffusion_maps_test_dims_1_2.png" alt="Deep Diffusion Maps" width="100%"><br>
        <b>Deep Diffusion Maps</b>
      </td>
      <td align="center" width="25%">
        <img src="results/phoneme/projection_nystrom_test_dims_1_2.png" alt="Nyström" width="100%"><br>
        <b>Nyström</b>
      </td>
    </tr>
  </table>
</div>

#### MNIST
<div align="center">
  <table>
    <tr>
      <td align="center" width="25%">
        <img src="results/mnist_d_2/original_test.png" alt="Original Test" width="100%"><br>
        <b>Original (test)</b>
      </td>
      <td align="center" width="25%">
        <img src="results/mnist_d_2/projection_diffusion_maps_test_dims_1_2.png" alt="Diffusion Maps" width="100%"><br>
        <b>Diffusion Maps</b>
      </td>
      <td align="center" width="25%">
        <img src="results/mnist_d_2/projection_deep_diffusion_maps_test_dims_1_2.png" alt="Deep Diffusion Maps" width="100%"><br>
        <b>Deep Diffusion Maps</b>
      </td>
      <td align="center" width="25%">
        <img src="results/mnist_d_2/projection_nystrom_test_dims_1_2.png" alt="Nyström" width="100%"><br>
        <b>Nyström</b>
      </td>
    </tr>
  </table>
</div>

## Multi-view extension (MDT)

This repository also includes an experimental extension that applies the deep
out-of-sample idea to **Multi-view Diffusion Trajectories (MDT)** ([Debaussart-Joniec
& Kalogeratos, 2025](https://arxiv.org/abs/2512.01484)). MDT builds a diffusion
operator `W = W_t ··· W_1` as a product of per-view, row-stochastic operators
(a *trajectory* through the views); its embedding is the truncated SVD of `W`.
MDT is transductive — embedding new points means recomputing the operator — so
the natural question is whether the Deep Diffusion Maps encoder can extend it
out-of-sample, and whether trajectories can be selected automatically.

### Surgical integration: how MDT maps onto DDM

The integration is deliberately minimal. DDM's encoder never computes an
eigendecomposition — its loss simply matches the Gram matrix of the
`√π`-scaled embeddings to a fixed, symmetric, positive-semidefinite target
(`‖ diag(√π) Γ Γᵀ diag(√π) − T ‖²_F`). MDT slots in by **changing only that
target operator and the encoder's fan-in**; the loss formulation, the index
trick, and the training loop are untouched.

| DDM (single view) | MDT (multi-view) | location |
|---|---|---|
| kernel `K` from one `X`; symmetric `A = D⁻½ K_α D⁻½` | per-view `Pᵥ = Dᵥ⁻¹Kᵥ`; operator `W = Wₜ···W₁` (asymmetric) | `src/mdt_operators.py` |
| `π = d_W / Σ d_W` (closed form) | `π_t` = left Perron vector of `W`, by power iteration | `MVDiffusionLoss._stationary_distribution` |
| target `T = A²ᵗ − √π√πᵀ` (symmetric `A` ⇒ a power works) | target `T = AAᵀ − √π√πᵀ`, with `A = Π½ W Π⁻½` | `MVDiffusionLoss.__init__` |
| loss `‖diag(√π) ΓΓᵀ diag(√π) − T‖²` | **identical** | `call()` (line-for-line the same) |
| single-input MLP/CNN encoder | per-view branch → concat → **same `EncoderHead`** | `build_mv_encoder` |
| `model.fit(x=X, y=indices)` | `model.fit(x=[view₁…viewᵥ], y=indices)` | `deep_mdt_experiment` |

**What stayed byte-for-byte identical:** the Gram-matching loss body, the
`y = sample indices` trick with per-batch sub-block `gather`, the `EncoderHead`,
and the `fit`/`predict` loop. The diff against DDM is essentially one matrix
construction (`__init__`) plus a multi-view input head.

**What the surgery forced (two non-obvious points):**

1. **`π_t` weighting is load-bearing, not cosmetic.** Because `W` is asymmetric,
   the naive target `WWᵀ − σ₁²u₁u₁ᵀ` (plain SVD, the form MDT itself ships) makes
   the parametric encoder **collapse to a constant** — that trivial solution
   isn't penalised when the removed mode `u₁ ≠ constant`. Symmetrising via
   `A = Π½ W Π⁻½` makes `√π` the exact top singular vector, so a constant
   embedding maps onto the removed `√π√πᵀ` term and collapse becomes costly.
   This is the same mechanism DDM relies on in the symmetric case; here it
   becomes essential. (`A²ᵗ` ↔ `AAᵀ` is the same correspondence: both are the
   symmetric-PSD Gram whose factorisation *is* the embedding.)
2. **BatchNorm head + lr ≈ 0.01.** The Gram loss is quadratic in `Γ`, so `Γ=0`
   is a saddle; a linear head initialised near zero gets stuck on multi-cluster
   data. `use_bn: True` on the embedding head escapes it reliably.

### What was added

- `src/mdt_operators.py` — per-view transition matrices and the MDT operator
  (`random`, `circulant`, or `contrastive`/learned convex fusion).
- `src/mvdiffusionloss.py` — `MVDiffusionLoss`, the multi-view analogue of
  `DiffusionLoss`. Target `G = A Aᵀ − √π√πᵀ` with `A = Π^½ W Π^{-½}` and
  `√π`-scaled embeddings. **The `π_t` weighting is essential**: without it the
  parametric encoder collapses to a constant on multi-cluster data (the constant
  embedding must map onto the removed `√π√πᵀ` mode for collapse to be penalised).
  A BatchNorm embedding head (`use_bn: True`) is also required to escape the
  `Ψ=0` saddle.
- `experiments/mvmat/` — runs on real multi-view `.mat` datasets from
  [ChuanbinZhang/Multi-view-datasets](https://github.com/ChuanbinZhang/Multi-view-datasets),
  evaluated by clustering AMI: `experiment.py` (deep OOS), `oos_compare.py`
  (deep vs Nyström), `path_selection.py` and `beam_path.py` (trajectory selection).
- `build_mv_encoder` (`experiments/utils/models.py`) and `deep_mdt_experiment`
  (`experiments/utils/experiments.py`).

The `contrastive` trajectory and the MDT operator construction reuse the
[mixed-diffusion-trajectory](https://github.com/Gwendal-Debaussart/mixed-diffusion-trajectory)
package (vendored where needed); set `data.path` to a local clone of the dataset repo.

### Findings (honest)

The deep encoder **reproduces** the classical MDT embedding and extends it
out-of-sample with little degradation given enough training data (Handwritten,
1400 train: deep OOS AMI 0.88 vs classical 0.89). **However, it does not beat
Nyström extension**, the simple training-free baseline:

| Dataset | Nyström OOS | Deep OOS | full-recompute |
|---|---|---|---|
| MSRC-v5 | 0.689 | 0.642 | 0.629 |
| Handwritten | 0.885 | 0.862 | 0.882 |
| UCI | 0.836 | 0.839 | 0.852 |
| OutdoorScene | 0.482 | 0.482 | 0.514 |
| Caltech101-7 | 0.522 | **0.555** | 0.576 |

Nyström wins or ties most datasets; the deep encoder's only clear win is the
highly heterogeneous Caltech101-7. **Conclusion: deep OOS for MDT is not a
general win** — Nyström is cheaper and as good or better.

**Trajectory selection** is the part that holds up. Comparing label-free
criteria for ranking trajectories against oracle AMI (mean Spearman ρ over 10
datasets): **Silhouette 0.48**, CH (the MDT paper's choice) 0.25, Davies-Bouldin
0.25, spectral entropy 0.22, MDT contrastive-Q 0.20. Silhouette wins on 9/10 and
selects within ~0.1 AMI of the oracle on well-clustering data. Ranking a sampled
pool by silhouette beats beam search (which prunes away good paths). On data
without cluster structure no criterion works. Practical recommendation:
**select MDT trajectories by silhouette over a sampled pool** (a drop-in
improvement over CH).

### GNN closure follow-up

A controlled GraphSAGE/GAE/DGI follow-up is documented in
[`docs/GNN_RESEARCH_LEDGER.md`](docs/GNN_RESEARCH_LEDGER.md). The six-dataset screen was
followed by a locked five-seed validation on six untouched datasets. The screen
gains from the GNN teacher and GAE message passing did not replicate reliably.
Across all 12 datasets, teacher distillation still beats Nyström (+0.048 test
AMI), but the GNN adds -0.0049 over a matched MLP: the useful signal is
parametric OOS extension, not graph message passing. GAE as an MDT replacement,
learned global view weights, DGI, a GNN fidelity advantage, and a compute
advantage are closed as dead ends under the tested formulation. Raw rows and
aggregate verdicts are in `results/gnn_mdt/metrics.jsonl`,
`results/gnn_mdt/summary.json`, `results/gnn_mdt/validation_metrics.jsonl`, and
`results/gnn_mdt/validation_summary.json`.

### Out-of-sample benchmark against the multi-view literature

`experiments/mvbench/` puts MDT next to the methods the MDT paper compares
itself to — Alternating Diffusion, powered AD, Integrated Diffusion, Multi-View
Diffusion Maps, Cross-Diffusion, Composite Diffusion — plus GCCA and two
controls, and asks the question the paper cannot: **which multi-view operator
still separates classes on points that were not in the operator?** Every
compared method is transductive, so all of them get the same treatment: the
same per-view Gaussian kNN transitions, the same truncated-SVD embedding, and
the same Nyström extension, obtained by writing each operator as
`W = Σ_v P_v S_v` and substituting the test→train transition `B_v` for the
leftmost factor. `bench.py --smoke` asserts that identity to machine precision
for every method before anything is measured.

Two controls carry the interpretation. `uniform_fused` is the *no-trajectory*
ablation — the uniform mean operator raised to `t` — and `features` is raw
concatenated views with no fusion at all. `ENCOMPASSED` records which
competitors paper Sec. 3.5 proves *are* MDT trajectories (AD, ID, p-AD): a win
over those is a statement about trajectory choice, not about the framework.

Two arms do not fit the `(W, suffix)` shape and are run separately, because
they need no Nyström rule at all — they are inductive by construction:

* `gcca` / `dgcca` — Generalised CCA and its deep form (Benton et al. 2017).
* `specrage` — **SpecRaGE** (Yacobi et al., TMLR 2025, `arXiv:2411.02138`), a
  parametric map approximating the joint diagonalisation of the per-view graph
  Laplacians. This is the closest published relative of this repo's MDT → DDM
  pipeline and the strongest available competitor, so it is run on the
  *identical* split via `experiments/mvbench/specrage_arm.py`. See that
  module's docstring for the leakage and batch-size traps upstream contains.

Result on **19 datasets × 10 seeds = 190 cells per arm**, mean inductive AMI
(train-fitted KMeans applied to the extended test embedding):

| method | MDT special case? | inductive AMI | PRR | train s |
|---|---|---|---|---|
| `specrage` (TMLR 2025) | no | **0.417** | 1.10 | 47.4 |
| `uniform_fused` (no trajectory) | yes | 0.413 | 1.09 | 0.1 |
| `dgcca` | no | 0.381 | 1.00 | 2.9 |
| `mdt_cvx_rand` | — (baseline) | 0.380 | 1.00 | 0.1 |
| `features` (concat, no fusion) | — | 0.355 | 0.93 | 0.0 |
| `mdt_selected` (silhouette search) | yes | 0.338 | 0.89 | 11.0 |
| `mdt_rand` | yes | 0.333 | 0.88 | 0.1 |
| `gcca` | no | 0.323 | 0.85 | 0.3 |
| `p_ad` | yes | 0.322 | 0.85 | 1.4 |
| `mvd` | no | 0.320 | 0.84 | 1.4 |
| `ad` | yes | 0.320 | 0.84 | 0.1 |
| `cr_diff` | no | 0.317 | 0.83 | 0.8 |
| `id` | yes | 0.309 | 0.81 | 0.8 |

All five MDT variants from the paper's Tab. 2 are present. `mdt_direct` uses
`scipy.optimize.direct`, which *is* Jones 2001 (the paper's ref. [18]), so the
reference repo's `gob` dependency is unnecessary; `mdt_cst` reuses the
contrastive loss already vendored in `src/mdt_operators.py`.

| MDT variant | space | index | optimiser | inductive AMI | PRR vs MDT-RAND | paper's PRR |
|---|---|---|---|---|---|---|
| `mdt_cvx_rand` | convex | — | — | **0.380** | **1.141** | 1.07 |
| `mdt_direct` | convex | `Q_CH` | DIRECT | 0.361 | 1.084 | 1.08 |
| `mdt_cst` | convex | `Q_X` | ADAM | 0.348 | 1.044 | — |
| `mdt_selected` | mixed | silhouette | pool rank | 0.338 | 1.014 | — |
| `mdt_bsc` | discrete | `Q_CH` | beam | 0.333 | 1.000 | 1.03 |
| `mdt_rand` | discrete | — | — | 0.333 | 1.000 | 1.00 |

**Measured against the paper's own Eq. 21 baseline (MDT-RAND), the paper's PRR
ordering reproduces out of sample almost exactly** — MDT-DIRECT lands at 1.084
against the published 1.08. That is a genuine, non-trivial reproduction of
Fig. 6 on unseen points, and it is the strongest thing this benchmark says in
the paper's favour.

**But trajectory search buys nothing over a single random convex draw, and
costs 180x.** `mdt_cvx_rand` beats every optimised variant: DIRECT on 13 of 19
datasets, CST 16/19, silhouette 16/19, beam 15/19. Three different internal
indices (`Q_CH`, contrastive `Q_X`, silhouette) and three different search
strategies (DIRECT, beam, full-pool ranking) all land below one Dirichlet
sample. The transductive ordering *inverts*: the paper has DIRECT >= CVX-RAND,
out of sample CVX-RAND > DIRECT. That inversion is the signature of the
internal index overfitting the training operator — it is selected on the train
embedding and does not transfer, which is exactly what an out-of-sample
protocol is for. Only the DIRECT gap is small enough to be non-significant
(13-6, `p_sign = .167`); the CST, silhouette and beam losses are individually
significant.

Three statistics per comparison, because a t-interval over 19 heterogeneous
datasets assumes those 19 numbers are roughly normal and there is no reason
they should be: the t-interval on dataset means, a 20k-draw percentile
**bootstrap** over datasets, and an exact **sign test** on how many datasets
each side wins. **Holm-Bonferroni** across the 13-comparison family, because
one baseline against 13 competitors makes the chance that at least one
uncorrected 95% interval clears zero by luck about `1 − 0.95¹³ ≈ 49%`.

| MDT vs | effect | t-CI95 | boot-CI95 | W–L | p_sign | p_holm(sign) |
|---|---|---|---|---|---|---|
| `id` | +0.071 | `[+0.015,+0.126]` | `[+0.026,+0.127]` | 15–4 | .019 | .173 |
| `cr_diff` | +0.063 | `[+0.032,+0.094]` | `[+0.036,+0.092]` | 16–3 | .004 | .053 |
| `ad` | +0.060 | `[+0.016,+0.104]` | `[+0.024,+0.104]` | 15–4 | .019 | .173 |
| `mvd` | +0.060 | `[+0.016,+0.103]` | `[+0.024,+0.103]` | 17–2 | **.0007** | **.0095** |
| `p_ad` | +0.058 | `[+0.013,+0.104]` | `[+0.019,+0.104]` | 16–3 | .004 | .053 |
| `gcca` | +0.057 | `[-0.017,+0.131]` | `[-0.005,+0.129]` | 15–4 | .019 | .173 |
| `dgcca` | −0.001 | `[-0.033,+0.031]` | `[-0.032,+0.026]` | 10–9 | 1.00 | 1.00 |
| `uniform_fused` | **−0.033** | `[-0.058,-0.009]` | `[-0.057,-0.014]` | 4–15 | .019 | .173 |
| `specrage` | −0.037 | `[-0.086,+0.012]` | `[-0.082,+0.007]` | 6–13 | .167 | .501 |

**MDT beats every published diffusion competitor, directionally and
robustly** — 15 to 17 of 19 datasets, +0.058 to +0.071, with both the t and
bootstrap intervals excluding zero in all five cases. But only MVD survives
family-wise correction at α=0.05 (`p_holm = 0.0095`); CR-DIFF and p-AD sit on
the line at 0.053. State it as a consistent direction across 19 datasets, not
as five independently significant wins.

**The uniform mean operator still beats MDT, and this is the robust result.**
Effect −0.033 with *both* intervals excluding zero, winning **15 of 19**
datasets. It is not one dataset carrying a mean: MDT's only four wins are
3Sources, Movies, Prokaryotic and Reuters-1200 — all datasets where every
method is at the AMI floor (< 0.07) *for MDT*. On every dataset with real signal, the
uniform blend wins. Trajectory search makes it worse, not better
(`mdt_selected` loses to a single Dirichlet draw 16–3).

**SpecRaGE is not significantly ahead** — 6–13, `p_sign = 0.167`, both
intervals straddling zero, at **395× the compute** (47.4 s vs 0.12 s). Its
wins are concentrated exactly where MDT's fixed kernel collapses (Wikipedia
+0.293, Reuters-1200 +0.176, 3Sources +0.153, ProteinFold +0.090) and it loses
where the kernel is good (Yale −0.190, MNIST-10k −0.078). It also diverged to
`nan` loss from epoch 1 on 3 of 10 BBCSport seeds (3183-dim sparse text,
reproducible across re-runs, recovered on a second init) — a robustness cost
MDT does not pay. `attempts_used` and `diverged` are recorded per row.

Scaling from 6 to 19 datasets changed two conclusions, which is the reason to
distrust the pilot: GCCA went from tying MDT to losing 15–4, and SpecRaGE's
lead shrank from +0.060 to a non-significant +0.037. Absolute AMI fell across
the board (MDT 0.59 → 0.38) because the 13 added datasets are harder. They are
*not* simply uninformative: only 3 of 19 (Movies, NUS-WIDE, ProteinFold) are near
the floor for every method. On 5 others — 3Sources, BBCSport, Prokaryotic,
Reuters-1200, WebKB — MDT is below 0.11 while another method clears 0.15, so
those are MDT failures, not dataset failures.

Raw rows and verdicts: `results/mvbench/metrics.jsonl` (Nyström arms),
`results/mvbench/specrage.jsonl`, merged `results/mvbench/all_metrics.jsonl`
and `results/mvbench/all_summary.json`.

### Usage

```bash
git clone https://github.com/ChuanbinZhang/Multi-view-datasets   # set data.path to this
python -m experiments.mvmat.experiment     -c experiments/mvmat/config.yml            # deep OOS
python -m experiments.mvmat.oos_compare    -c experiments/mvmat/config_handwritten.yml # deep vs Nyström
python -m experiments.mvmat.path_selection -c experiments/mvmat/config_handwritten.yml # criterion comparison
python -m experiments.mvbench.bench --smoke                                           # OOS suite self-check
python -m experiments.mvbench.bench -c experiments/mvbench/config.yml                 # MDT vs the literature, OOS
git clone https://github.com/shaham-lab/SpecRaGE                                      # strongest inductive competitor
python -m experiments.mvbench.specrage_arm --repo ./SpecRaGE                          # same split, same metrics
```

## References

- Deep Diffusion Maps paper: [García-Heredia, S., Fernández, Á., & Alaíz, C. M. (2025). Deep Diffusion Maps. arXiv preprint arXiv:2505.06087](https://arxiv.org/abs/2505.06087)
- Multi-view Diffusion Trajectories: [Debaussart-Joniec, G., & Kalogeratos, A. (2025). Multi-view diffusion geometry using intertwined diffusion trajectories. arXiv:2512.01484](https://arxiv.org/abs/2512.01484)
- Multi-view datasets: [ChuanbinZhang/Multi-view-datasets](https://github.com/ChuanbinZhang/Multi-view-datasets)
- Diffusion Maps code implementation: [diffusion-maps-with-nystrom](https://github.com/sgh14/diffusion-maps-with-nystrom.git)
- Original Diffusion Maps paper: Coifman, R. R., & Lafon, S. (2006). Diffusion maps. Applied and computational harmonic analysis, 21(1), 5-30.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

If you have any questions or comments, please do not hesitate to contact us.
