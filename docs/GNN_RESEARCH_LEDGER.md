# MDT × GNN research ledger

This ledger turns the GNN continuation into falsifiable tests. Labels are never
used for fitting or model selection; they are read only after a run to compute
AMI. Every claim is evaluated over the same data splits and seeds.

## Why a GNN is worth one controlled pass

The previous MLP study already closes any claim that a neural factorisation of
a fixed MDT Gram matrix can exceed its truncated SVD: at the global optimum it
is the same rank-k factorisation. A GNN does not change that theorem. It can,
however, change two questions that the theorem does not answer:

1. Does message passing give a better inductive map for unseen samples than a
   feature-only encoder and Nyström?
2. Does a graph-native self-supervised objective learn a more cluster-useful
   geometry than fixed MDT consensus, without using labels?

The suite in `experiments/gnn_mdt/closure.py` tests both questions and isolates
multi-view fusion from extra model capacity.

## Hypotheses and closure rules

| ID | hypothesis | decisive comparisons | continue only if |
|---|---|---|---|
| G1 | A GNN distils fixed MDT more faithfully than an MLP. | `teacher_gnn` vs `teacher_mlp`; Procrustes fidelity and train AMI. | Fidelity improves materially and is stable over 5 seeds. This is an engineering result, not an accuracy result. |
| G2 | Graph message passing improves MDT out-of-sample extension. | `teacher_gnn` vs `mdt_nystrom` and `teacher_mlp`; test and inductive AMI. | Mean paired gain ≥0.02 and the gain is not driven by one dataset/seed. |
| G3 | A graph autoencoder escapes the fixed-SVD ceiling usefully. | `gae_uniform` vs `mdt_consensus`, `features`, and `gae_mlp`. | Mean paired train/test gain ≥0.02 and wins on at least 4/6 datasets. |
| G4 | Infomax, rather than edge reconstruction, is the missing objective. | `dgi_uniform` vs `mdt_consensus` and `gae_uniform`. | Same threshold as G3; otherwise close this objective. |
| G5 | Learning view importance solves heterogeneous-view dilution. | learned vs uniform variants on Wikipedia/Caltech plus all datasets. | ≥0.02 mean gain, stable attention weights, and no regression on homogeneous datasets. |
| G6 | GNN computation is preferable to sparse truncated SVD. | graph build + training + inference vs graph build + `svds`; report both. | End-to-end time wins, or a measured repeated-query break-even point is realistic. |
| A2a | AlphaFold-style recycling improves the parametric OOS map. | `teacher_mlp_recycle` vs `teacher_mlp`, inductive AMI primary. | ≥0.02 mean paired gain with the interval excluding zero, and the extra inference cost justified. |
| A2b | An AlphaFold-style pair target beats coordinate regression. | `teacher_mlp_pair` vs `teacher_mlp`; interaction read from `teacher_mlp_pair_recycle`. | Same threshold; a fidelity loss is acceptable only if inductive AMI clears it. |

The automated verdict uses paired effects and a 95% interval clustered by
dataset: the five seeds are averaged first, so they are not treated as 30
independent benchmark tasks. Continue when the mean clears the threshold and
the interval excludes zero; stop when the interval excludes the minimum useful
effect; otherwise remain inconclusive. It refuses to issue a verdict from
fewer than five datasets. A paper-facing positive result still needs the locked
configuration and evaluation on untouched data.

## Fairness constraints

- Identical train/test splits, per-view standardisation, kNN graphs, embedding
  dimension, and KMeans protocol for all methods.
- `teacher_*` uses the fixed MDT SVD as its only target. `gae_*` and `dgi_*`
  never see the SVD target or labels.
- `*_mlp` removes message passing while retaining comparable projection
  capacity. `*_uniform` freezes relation weights; `*_learned` changes only
  those weights.
- Nyström is the OOS baseline; sparse truncated `svds` is the compute baseline;
  MDT consensus is the strongest surviving unsupervised geometry baseline.
- Report both test-only KMeans AMI (compatible with the existing paper) and the
  stricter inductive AMI obtained by assigning test points to train-fitted
  centroids.

## Method: the two distillation students, `teacher_gnn` and `teacher_mlp`

Both variants are parametric inductive maps. They learn a function from raw
multi-view features to the MDT embedding, so that an unseen sample can be
embedded by a forward pass instead of by redoing the spectral decomposition.
They are trained on exactly the same target with exactly the same architecture,
and differ in one line of code. Everything below is implemented in
`experiments/gnn_mdt/closure.py`.

### The teacher they distil

The target is built once per (dataset, seed), before any network exists.

1. **Graphs.** For each view `v`, standardise the features, take a Gaussian
   kernel over the `k = floor(log n_train)` nearest neighbours with a global
   median bandwidth estimated on at most 200 points, symmetrise with
   `max(W, Wᵀ)`, and row-normalise to get a stochastic matrix `P_v`
   (`build_graphs`, `knn_transitions`, `:107-148`). The self-neighbour is
   included, so `P_v` has non-zero diagonal. One graph per view, same node set.
2. **Trajectory.** Draw `T = 4` weight vectors from a Dirichlet(1, …, 1) over
   the views, seeded by the run seed (`trajectory`, `:151-152`). Row `t` gives
   the mixing coefficients `α_t,v` for diffusion step `t`.
3. **Operator.** Compose the steps: `W = (Σ_v α_T,v P_v) ⋯ (Σ_v α_1,v P_v)`
   (`mdt_operator`, `:155-160`). This is a single MDT trajectory, not the
   8-trajectory consensus used by `mdt_consensus`.
4. **Embedding.** Sparse truncated SVD of `W`, keep components 1…k of the
   left factor scaled by their singular values, discarding component 0 as the
   trivial stationary direction (`svd_embedding`, `:177-185`). `k` equals the
   number of classes, which is the only place class information enters, and it
   enters as a dimension count rather than as labels.

The result, `svd_train` (`n_train × k`), is the sole supervision signal for
both students (`:461`). Before the loss it is standardised per column
(`:335`), so every retained direction carries equal weight and the students are
not asked to reproduce the spectral decay. Fidelity is later measured after the
same standardisation, so the two are consistent.

### Shared architecture

`MultiViewSAGE` (`:229-287`), a two-hop relation-aware GraphSAGE:

| stage | operation | shape |
|---|---|---|
| per-view input | `h0_v = relu(W_v x_v + b_v)`, one `Linear` per view | `n × 128` |
| view fusion | `h0 = Σ_v α_v h0_v`, `α = softmax(fusion_logits)` | `n × 128` |
| hop 1 | `h1 = relu(layer1[ h0 ‖ Σ_v α_v (P_v h0_v) ])`, `layer1: 256 → 128` | `n × 128` |
| hop 2 | `z = layer2[ h1 ‖ Σ_v α_v (P_v h1) ]`, `layer2: 256 → k`, no activation | `n × k` |

Three details matter. Hop 1 diffuses each view's own `h0_v` on that view's own
graph before fusing, whereas hop 2 diffuses the single shared `h1` on every
graph (`h1_views = [h1 for _ in graphs]`, `:263`) — the per-view identity of the
representation survives only one hop. The concatenation keeps an untransformed
copy of the node's own state alongside the neighbourhood average, which is what
separates GraphSAGE from GCN, where the two are summed. And for the two teacher
variants `fusion_logits` is frozen (`requires_grad` is set only for
`*_learned`, `:239-241`), so fusion is uniform `1/V`; learned fusion is tested
separately under G5.

For MSRC-v5 (5 views of 24/576/512/256/254 dimensions, `k = 7`) this is 242,951
trained parameters: 208,256 in the input projections, 32,896 in `layer1`, 1,799
in `layer2`. The `Bilinear` discriminator (`:242`) exists on the module but is
used only by the DGI objective; under the teacher loss it receives no gradient.

### The single difference

`_aggregate` returns a zero tensor when `message_passing` is false
(`:251-256`):

```python
def _aggregate(self, graphs, values):
    if not self.message_passing:
        return torch.zeros_like(values[0])
    return self._fuse([torch.sparse.mm(graph, value)
                       for graph, value in zip(graphs, values)])
```

So `teacher_mlp` computes `h1 = relu(layer1[ h0 ‖ 0 ])` and
`z = layer2[ h1 ‖ 0 ]`. The right half of both weight matrices still exists and
still receives weight decay; it simply multiplies zeros. Parameter count,
optimiser, learning rate, epoch budget, patience, seed, splits, graphs and
evaluation protocol are identical. The variant is selected by a suffix test,
`message_passing = not variant.endswith("mlp")` (`:325`). This is what makes
the contrast attributable to message passing rather than to capacity — a
comparison against a smaller or differently shaped MLP would confound the two.

The match is on *nominal* parameters, not on effective capacity: weights that
multiply zeros compute no function, so a GNN win over this control is still
open to the reading "the extra live parameters helped". Matching effective
capacity while removing message passing is impossible, because the hop is the
capacity under test. Amendment A3 closes that gap from the other side, with a
control that keeps every weight live and destroys only the neighbourhoods.

### Training

Full-batch, no minibatching, no validation split, no labels (`fit_model`,
`:318-385`):

- loss `MSE(z, teacher)` against the standardised target;
- Adam, learning rate 3e-3, weight decay 1e-4;
- gradient-norm clipping at 5.0;
- up to 100 epochs, early stopping after 40 epochs without a new best
  *training* loss, then the best state is restored before inference.

Early stopping on training loss rather than on a held-out score is deliberate:
a held-out selection signal would need labels or a second target, and the
ledger's rule is that nothing is selected on labels. The cost is that the
epoch budget is not tuned per dataset, which is one reason the timings in G6
are not a tuned-implementation benchmark.

### Out-of-sample inference

Test points are never in the training graph. They are attached by a directed
row-normalised bipartite graph `B_v` of shape `n_test × n_train`, built with the
same bandwidth and the same `k` (`:128-134`); test points connect only to
training points, never to each other, and each test row is independent of the
rest of the test set.

`forward_test` (`:267-287`) then mirrors the training forward pass with the
train-side states as the message source:

    h0_test  = Σ_v α_v relu(W_v x_test,v)
    h1_test  = relu(layer1[ h0_test ‖ Σ_v α_v (B_v h0_train,v) ])
    z_test   = layer2[ h1_test ‖ Σ_v α_v (B_v h1_train) ]

For `teacher_mlp` both aggregation terms are zero, so the map collapses to
`z_test = f(x_test)`: a pure feature-to-embedding function that needs neither
the graph nor the training set at query time. This asymmetry is the substantive
difference between the two at deployment, and it is why the G6 break-even
question is well posed for the MLP and not for the GNN.

Both are compared against `mdt_nystrom`, which uses the same `B_v` but no
learned parameters: it replays the MDT trajectory with the final step replaced
by the bipartite operator and projects onto the teacher's right singular
vectors (`mdt_oos_operator`, `:163-174`; `:425`).

### What is recorded per run

`fidelity` (Procrustes agreement with the teacher on train), `train_ami`,
`test_ami` (KMeans refit on the test embedding), `inductive_ami` (test points
assigned to train-fitted centroids), `graph_seconds`, `train_seconds`,
`inference_seconds`, the converged loss, and the softmax fusion weights. Labels
touch none of this until AMI is computed after the run.

## Commands

```bash
python -m unittest tests.test_gnn_mdt
python -m experiments.gnn_mdt.closure --smoke
python -m experiments.gnn_mdt.closure --config experiments/gnn_mdt/config.yml
python -m experiments.gnn_mdt.closure --datasets MSRC-v5 Handwritten --seeds 0 1 --epochs 100 --output /tmp/gnn-pilot.jsonl
python -m experiments.gnn_mdt.closure --config experiments/gnn_mdt/config.yml --resume
python -m experiments.gnn_mdt.closure --summarize results/gnn_mdt/metrics.jsonl
python -m experiments.gnn_mdt.closure --summarize results/gnn_mdt/metrics.jsonl --summary-output results/gnn_mdt/summary.json
python -m experiments.gnn_mdt.closure --summarize results/gnn_mdt/validation_metrics.jsonl --summary-output results/gnn_mdt/validation_summary.json
python -m experiments.gnn_mdt.closure --variants teacher_mlp teacher_mlp_recycle teacher_mlp_pair teacher_mlp_pair_recycle --output results/gnn_mdt/af_metrics.jsonl --summary-output results/gnn_mdt/af_summary.json
python -m experiments.gnn_mdt.closure --output results/gnn_mdt/control_metrics.jsonl --summary-output results/gnn_mdt/control_summary.json
python -m experiments.gnn_mdt.closure --datasets MSRC-v5 Handwritten Wikipedia UCI OutdoorScene Caltech101-7 3Sources BBCSport Prokaryotic Reuters-1200 NUS-WIDE MNIST-4 --output results/gnn_mdt/power_metrics.jsonl --summary-output results/gnn_mdt/power_summary.json
```

Do not read `results/gnn_mdt/summary.json` as the screen result. It holds the
*validation* numbers — `mdt_nystrom` test AMI mean 0.2495, which is the
validation six, against 0.6026 on `metrics.jsonl` — and it is equal to
`validation_summary.json` on every field. The `teacher_gnn − teacher_mlp` test
effect reads -0.0215 there and +0.0118 on the screen grid, so the two disagree
on sign. Always recompute with `--summarize` against the file you mean.

The full configuration is intentionally five-seed and six-dataset. During
development, reduce `epochs`, the seed list, and the dataset list in a copied
configuration; do not use those reduced runs for a verdict.

## Locked run: 6 datasets × 5 seeds

Run completed with the checked-in configuration (800 training and 400 test
samples maximum, 100 epochs, embedding dimension equal to class count). The
raw 300 rows are in `results/gnn_mdt/metrics.jsonl`. Intervals below are paired
95% t intervals across the six dataset means (five seeds are averaged within
each dataset).

| question | paired AMI effect | evidence | verdict |
|---|---:|---|---|
| GNN teacher vs Nyström, test | +0.0359 [0.0153, 0.0564] | wins 25/30; inductive effect +0.0349 | **continue to untouched-data validation** |
| GNN teacher vs MLP teacher, test | +0.0118 [-0.0097, 0.0333] | inductive effect +0.0195 [-0.0003, 0.0392] | **inconclusive / graph-specific gain not established** |
| GNN vs MLP teacher fidelity | -0.0009 [-0.0036, 0.0017] | mean fidelity 0.9895 vs 0.9904 | **dead end as a fidelity claim** |
| GAE message passing vs GAE-MLP, test | +0.0402 [0.0258, 0.0545] | wins 26/30 | **continue: the graph is useful** |
| learned GAE vs MDT consensus, test | +0.0215 [-0.0412, 0.0841] | wins 19/30; −0.080 on Wikipedia | **inconclusive / not a general replacement** |
| learned vs uniform GAE fusion, train | +0.0008 [-0.0069, 0.0085] | mean attention deviation from uniform only 0.0071 | **dead end** |
| DGI vs MDT consensus, train | -0.3011 [-0.3988, -0.2033] | loses 30/30 although its loss converges | **dead end: objective misalignment** |
| GNN teacher compute vs sparse SVD | 12.1× slower training | 2.535 s vs 0.209 s mean, excluding shared graph build | **dead end for compute** |

The screen left two provisional survivors: parametric SVD distillation for OOS
and graph-autoencoder message passing. Their required untouched-data validation
is reported below.

## Untouched validation and final closure

The locked configuration was evaluated without tuning on six additional
datasets: 100Leaves, Yale, 3Sources, BBCSport, Cora, and CiteSeer. Five seeds
produced 240 complete, unique, finite result rows in
`results/gnn_mdt/validation_metrics.jsonl`. Cora and CiteSeer have almost no
recoverable cluster signal for any evaluated method; they are retained in the
primary analysis rather than removed post hoc.

| question | validation paired AMI effect | replication result | final decision |
|---|---:|---|---|
| teacher GNN vs Nyström, test | +0.0602 [-0.0454, 0.1659] | 22/30 wins, but the interval crosses zero and the largest gain is on 100Leaves | **close as a general GNN claim** |
| teacher GNN vs MLP, test | -0.0215 [-0.0872, 0.0442] | 14/30 wins; inductive +0.0042 [-0.0308, 0.0393] | **dead end: message passing is not the source of OOS gain** |
| GNN vs MLP teacher fidelity | +0.0065 [-0.0103, 0.0233] | no replication of a fidelity advantage | **dead end** |
| GAE message passing vs GAE-MLP, test | +0.0107 [-0.0803, 0.1018] | 18/30 wins and strong dataset heterogeneity | **close: screen gain did not replicate** |
| learned GAE vs MDT consensus, test | -0.0099 [-0.0719, 0.0520] | 12/30 wins | **dead end as an MDT replacement** |
| learned vs uniform GAE fusion, train | +0.0058 [-0.0069, 0.0185] | below the +0.02 minimum useful effect | **dead end, replicated** |
| GNN teacher compute vs sparse SVD | 9.8× slower training | excludes shared graph construction | **dead end, replicated** |

Pooling the screen and validation as a secondary 12-dataset analysis clarifies
the only apparent positive result. Teacher distillation beats Nyström by
+0.0480 test AMI [0.0034, 0.0927], but the GNN-minus-MLP effect is -0.0049
[-0.0352, 0.0254]. Thus **parametric OOS extension remains worth studying, but
it is an MLP/DDM result rather than evidence for a GNN**. The pooled GAE
message-passing effect is +0.0254 [-0.0134, 0.0643], so it also fails the
stability rule.

All launched GNN hypotheses are therefore closed under this formulation:

- G1 fidelity: dead end.
- G2 graph-specific OOS extension: dead end; retain feature-only parametric OOS.
- G3 graph autoencoder as an MDT replacement: dead end.
- G4 DGI: dead end from objective misalignment.
- G5 learned global view weights: dead end.
- G6 compute advantage: dead end.

Reopening a GNN branch now requires a materially different hypothesis, such as
sample-dependent view gating or genuinely relational side information. More
seeds, hidden-width sweeps, or another global fusion scalar do not address the
observed failure modes.

## Detailed records: G1, G2, G6

The three hypotheses below are expanded because they carry the load-bearing
negative results. Every number is recomputed from the two checked-in result
files with the same estimator the automated summary uses: the five seeds are
averaged within a dataset first, then a paired 95% Student interval is taken
across the dataset means. Win counts remain seed-level and are reported only as
a consistency diagnostic, never as evidence on their own.

### G1 — a GNN distils fixed MDT more faithfully than an MLP

**Methodology.** Both `teacher_gnn` and `teacher_mlp` regress one identical
target: the rank-k factor `U_k Σ_k` of a single fixed MDT operator, obtained by
sparse truncated SVD of the train graphs and passed as the only supervision
signal (`run_one`, `closure.py:420-422` and `:461`). The target is standardised
per column before the loss, so the two variants optimise the same objective on
the same scale. They also share the whole architecture in `MultiViewSAGE`:
per-view input projections to 128 hidden units, a softmax fusion over views, and
two `Linear(2·hidden, ·)` layers. The single difference is that `_aggregate`
returns zeros for the `*_mlp` variant (`closure.py:253-254`), so the neighbour
half of each concatenation is dead while the parameter count, the optimiser,
the epoch budget and the early-stopping patience are unchanged. That matches
nominal parameters but not effective capacity, since dead weights compute no
function; the residual gap is closed by the shuffled-graph control in A3, whose
teacher arm agrees with the verdict below. Fidelity is Procrustes agreement with the teacher after per-column
standardisation of both matrices, `1 - ||AR - B||²_F / ||B||²_F`
(`procrustes_fidelity`, `closure.py:403-408`); it equals 1 when the student
recovers the teacher up to rotation. `mdt_nystrom` is assigned fidelity 1.0 by
definition because it *is* the teacher, and it must therefore be excluded from
this comparison.

**Evidence.** Per-dataset means over five seeds.

| dataset | classes | fidelity GNN | fidelity MLP | Δ fidelity | Δ train AMI |
|---|---:|---:|---:|---:|---:|
| MSRC-v5 | 7 | 0.9991 | 0.9999 | -0.0008 | +0.0045 |
| Handwritten | 10 | 0.9936 | 0.9935 | +0.0001 | +0.0024 |
| Wikipedia | 10 | 0.9732 | 0.9701 | +0.0031 | -0.0095 |
| UCI | 10 | 0.9905 | 0.9920 | -0.0015 | -0.0022 |
| OutdoorScene | 8 | 0.9837 | 0.9883 | -0.0046 | +0.0082 |
| Caltech101-7 | 7 | 0.9970 | 0.9989 | -0.0019 | +0.0078 |
| 100Leaves | 100 | 0.5959 | 0.5567 | +0.0392 | +0.0066 |
| Yale | 15 | 0.9997 | 0.9999 | -0.0002 | -0.0005 |
| 3Sources | 6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| BBCSport | 5 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| Cora | 7 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| CiteSeer | 6 | 1.0000 | 1.0000 | 0.0000 | -0.0003 |

**Results.** One estimate over the twelve datasets of the table, screen and
validation pooled: **+0.0028 [-0.0046, 0.0102]**, 16/60 seed-level wins. The
upper limit sits at a fifth of the +0.02 minimum useful effect, so this is a
decided negative rather than an inconclusive one, and the screen and validation
halves are not reported separately because neither reaches a different verdict.
Two structural facts explain why. First, the measurement has no headroom on
most of the suite: 89 of the 120 teacher rows exceed 0.99 fidelity, and on
3Sources, BBCSport, Cora and CiteSeer both students hit 1.0000 to four
decimals. Second, the only dataset with real headroom is 100Leaves, whose
100-class target makes the regression genuinely hard (0.60 versus 0.56); the
GNN does win there by +0.0392, and that single dataset is what lifts the
validation mean. That fidelity win converts to only +0.0066 train AMI, so even
where message passing demonstrably fits the teacher better, the extra fidelity
buys no usable cluster structure.

**Verdict.** Dead end, and dead in the strong direction. The finding is not
"the GNN failed to distil"; it is that a feature-only MLP already reproduces
the fixed MDT embedding to within rotation on ten of twelve datasets, leaving
nothing for message passing to recover. A3 corroborates this from the capacity
side: destroying the graph outright costs the student only +0.0029 fidelity
[-0.0014, 0.0072], so the neighbourhoods were carrying no fidelity to begin
with. A distillation-fidelity argument for a
GNN cannot be rescued by more seeds or wider hidden layers, because the
baseline is at the ceiling. It could only be reopened on targets with the shape
of 100Leaves — high embedding rank relative to sample count — and even there
the AMI consequence would have to be demonstrated separately.

### G2 — graph message passing improves MDT out-of-sample extension

**Methodology.** Three inductive maps are compared on identical splits, graphs
and KMeans protocol. (i) `mdt_nystrom`, training-free: test rows are attached
to train columns by a directed row-normalised Gaussian kNN bipartite graph
(`knn_transitions`, `closure.py:128-134`), the MDT trajectory is replayed with
its final step replaced by that bipartite operator (`mdt_oos_operator`), and the
result is projected onto the train right singular vectors (`closure.py:425`).
(ii) `teacher_mlp`, parametric and feature-only: at inference the map is
`x_test → z`, with no graph involved. (iii) `teacher_gnn`, parametric and
graph-aware: test hidden states additionally aggregate over the same bipartite
graph, so an unseen point borrows the hidden states of its train neighbours
(`forward_test`, `closure.py:267-287`). Both parametric maps distil the same
fixed SVD target, so any difference between them is attributable to message
passing alone.

Two accuracy metrics are reported. `test_ami` refits KMeans on the test
embedding, which measures only whether the geometry is clusterable.
`inductive_ami` assigns test points to centroids fitted on the train embedding
(`closure.py:394-399`), which additionally requires the OOS map to land in the
same frame as the train embedding; it is the stricter and more deployment-like
metric. The decisive quantity is the decomposition

    (GNN − Nyström) = (MLP − Nyström) + (GNN − MLP)

whose first term isolates parametric distillation and whose second term
isolates the graph. G2 is a claim about the second term only.

**Evidence.** Per-dataset means over five seeds; `nys`, `mlp`, `gnn` are
absolute AMI, not differences.

| dataset | test nys | test mlp | test gnn | induct nys | induct mlp | induct gnn |
|---|---:|---:|---:|---:|---:|---:|
| MSRC-v5 | 0.6202 | 0.6892 | 0.6789 | 0.5989 | 0.6176 | 0.6726 |
| Handwritten | 0.8698 | 0.8756 | 0.8823 | 0.8715 | 0.8792 | 0.8835 |
| Wikipedia | 0.2923 | 0.2955 | 0.3458 | 0.2768 | 0.2783 | 0.2958 |
| UCI | 0.7818 | 0.8242 | 0.8283 | 0.8047 | 0.8275 | 0.8305 |
| OutdoorScene | 0.5232 | 0.5352 | 0.5495 | 0.4999 | 0.5291 | 0.5470 |
| Caltech101-7 | 0.5283 | 0.5404 | 0.5459 | 0.5050 | 0.5176 | 0.5367 |
| 100Leaves | 0.6581 | 0.8591 | 0.8849 | 0.6375 | 0.8618 | 0.8758 |
| Yale | 0.5398 | 0.4698 | 0.4895 | 0.5144 | 0.4523 | 0.5004 |
| 3Sources | 0.0799 | 0.3535 | 0.2093 | 0.0755 | 0.1316 | 0.0776 |
| BBCSport | 0.1670 | 0.2231 | 0.2026 | 0.1210 | 0.1143 | 0.1303 |
| Cora | 0.0266 | 0.0246 | 0.0256 | 0.0000 | 0.0000 | 0.0000 |
| CiteSeer | 0.0258 | 0.0576 | 0.0466 | 0.0082 | 0.0075 | 0.0089 |

Paired effects for the decomposition:

| contrast | metric | screen | validation | pooled (12) |
|---|---|---|---|---|
| GNN − Nyström | test | +0.0359 [0.0153, 0.0564] | +0.0602 [-0.0454, 0.1659] | +0.0480 [0.0034, 0.0927] |
| GNN − Nyström | inductive | +0.0349 [0.0113, 0.0584] | +0.0394 [-0.0632, 0.1420] | +0.0372 [-0.0058, 0.0801] |
| MLP − Nyström | test | +0.0241 [-0.0033, 0.0515] | +0.0817 [-0.0545, 0.2180] | +0.0529 [-0.0069, 0.1128] |
| MLP − Nyström | inductive | +0.0154 [0.0047, 0.0261] | +0.0352 [-0.0697, 0.1401] | +0.0253 [-0.0182, 0.0688] |
| **GNN − MLP** | test | +0.0118 [-0.0097, 0.0333] | -0.0215 [-0.0872, 0.0442] | **-0.0049 [-0.0352, 0.0254]** |
| **GNN − MLP** | inductive | +0.0195 [-0.0003, 0.0392] | +0.0042 [-0.0308, 0.0393] | **+0.0119 [-0.0053, 0.0290]** |

**Results.** The composite contrast against Nyström is the only one that ever
clears the threshold, and the decomposition shows it is not a graph effect. On
the pooled twelve datasets the parametric term carries +0.0529 test AMI while
the graph term is -0.0049, an interval that excludes the minimum useful effect
in both directions of interest. The inductive term, +0.0119 [-0.0053, 0.0290],
is the most favourable number message passing produces anywhere in the suite
and it still fails the rule. Sign instability across datasets is severe rather
than incidental: the graph helps most on Wikipedia (+0.0503 test) and hurts most
on 3Sources (-0.1442 test), and the screen-to-validation flip of the mean from
+0.0118 to -0.0215 is driven by that single dataset. Note also that
`inductive_ami` is uniformly lower than `test_ami` for every method, confirming
that part of the apparent OOS quality in the paper-compatible metric comes from
refitting KMeans on test rather than from the map itself.

**Verdict.** G2 is a dead end. Message passing is not the source of the OOS
gain, and the surviving positive result — parametric distillation beating
Nyström — belongs to the MLP/DDM line, where it should be pursued without a
graph. Three limitations bound how far even this negative generalises, and all
three point at the same missing ingredient. First, every graph in the suite is
built by kNN on the features themselves (`build_graphs` runs unconditionally),
so it encodes no information the encoder cannot already read; the experiment
never tested relational side information. Second, Cora and CiteSeer ship
genuine citation structure only as an n×n adjacency *view*, which is then
standardised and re-kNN'd like any other feature block, and `cap_train: 800`
subsamples them from 2708 and 3312 nodes, discarding most of their edges before
the graph is even built. Third, both retain almost no recoverable cluster
signal for any method — Cora's inductive AMI is exactly 0.0000 across all three
maps — so they contribute two near-zero rows that tighten the interval without
carrying information. A reopened G2 would need graphs that are given rather
than derived, and full graphs rather than subsampled ones.

### G6 — GNN computation is preferable to sparse truncated SVD

**Methodology.** Three timers are recorded per row. `graph_seconds` covers kNN
search, Gaussian weighting, symmetrisation and row normalisation over all
views; it is shared by every method and is therefore excluded from the ratio.
`train_seconds` covers `svds` plus the OOS projection for `mdt_nystrom`, and
the complete training loop with early stopping for the neural variants.
`inference_seconds` covers the forward pass over the test split. The hypothesis
allows two routes to a positive verdict: an end-to-end wall-clock win, or a
realistic repeated-query break-even point where amortised inference recovers
the training cost.

**Evidence.** Per-dataset means over five seeds, in seconds.

| dataset | graph (shared) | svds | GNN train | MLP train | GNN inference | GNN/svds |
|---|---:|---:|---:|---:|---:|---:|
| MSRC-v5 | 0.090 | 0.015 | 1.231 | 0.830 | 0.0073 | 82.4× |
| Handwritten | 0.510 | 0.251 | 3.305 | 1.296 | 0.0247 | 13.2× |
| Wikipedia | 0.305 | 0.248 | 1.551 | 0.851 | 0.0106 | 6.3× |
| UCI | 0.063 | 0.193 | 1.924 | 0.922 | 0.0145 | 10.0× |
| OutdoorScene | 0.162 | 0.277 | 2.858 | 1.282 | 0.0219 | 10.3× |
| Caltech101-7 | 0.340 | 0.268 | 4.342 | 2.350 | 0.0366 | 16.2× |
| 100Leaves | 0.079 | 1.168 | 1.832 | 0.976 | 0.0129 | 1.6× |
| Yale | 0.928 | 0.007 | 2.003 | 1.796 | 0.0111 | 268.7× |
| 3Sources | 0.287 | 0.011 | 1.603 | 1.374 | 0.0084 | 147.0× |
| BBCSport | 0.316 | 0.046 | 2.210 | 1.745 | 0.0164 | 47.7× |
| Cora | 0.674 | 0.249 | 5.304 | 3.696 | 0.0476 | 21.3× |
| CiteSeer | 0.498 | 0.210 | 3.584 | 2.834 | 0.0340 | 17.1× |

**Results.** Training is 12.1× slower than `svds` on the screen (2.535 s versus
0.209 s) and 9.8× slower on validation (2.756 s versus 0.282 s). Including the
shared graph build, end-to-end fitting is 6.1× slower on the screen and 4.3×
slower on validation, 5.0× pooled. The ratio is extremely heterogeneous
(1.6× on 100Leaves, 268.7× on Yale) because it is dominated by how cheap `svds`
happens to be: the SVD scales with sample count and requested rank, so it is
near-free on the 115-to-380-sample datasets and costliest on 100Leaves, where a
rank-100 decomposition of 800 points takes 1.168 s. 100Leaves is the only case
where the two are within one order of magnitude, and even there the GNN loses.
The neural cost is not a message-passing artefact: `teacher_mlp` is itself 6.8×
slower than `svds` pooled, so the deficit is the training loop, not the
aggregation.

**Verdict.** Dead end on the end-to-end route, and the break-even route was
never actually evaluated — a gap that should be recorded rather than glossed.
`mdt_nystrom` writes `inference_seconds: 0.0` unconditionally
(`closure.py:430`); its OOS projection cost is folded into `train_seconds` and
never separately timed, so no amortisation curve can be drawn from these
result files. The break-even argument nevertheless fails on structure rather
than on missing measurement. Message passing requires the test-to-train
bipartite graph at inference time (`forward_test` consumes `test_graphs`), so a
GNN query still pays the kNN search against the training set — exactly the cost
Nyström pays. The GNN therefore removes no term from the query path while
adding a 2.5-to-2.8 s fit; its ~0.02 s forward pass cannot amortise against a
baseline it has not made cheaper. Only `teacher_mlp` drops the graph at
inference and so is the only variant for which a break-even question is even
well posed. If that question is ever pursued, it belongs to the MLP/DDM line
and requires instrumenting Nyström inference first.

## Amendment A3 — the capacity-matched null for message passing

**Why this was run at all.** `### The single difference` argues that the
zero-input MLP makes every message-passing contrast a message-passing contrast
rather than a capacity contrast. That holds for nominal parameter count and
fails for effective capacity: in the `*_mlp` variants the neighbour half of
`layer1` and `layer2` multiplies zeros, so those weights compute no function.
Any GNN win over that control therefore still admits the reading "the extra
live parameters helped, not the graph". The control cannot be fixed by matching
capacity, because the hop under test *is* the capacity. A3 attacks from the
other side: keep every weight live and destroy only the correspondence between
neighbourhoods and features.

**Methodology.** `shuffle_graphs` (`closure.py:167-181`) draws one permutation
of the training node set per split and relabels every graph — `P_v[order][:,
order]` on train, `B_v[:, order]` out of sample — while the features stay in
place. Preserved exactly: row stochasticity, `nnz`, `k`, the multiset of
degrees, the kernel weight values, and the spectrum of each `P_v` (relabelling
is a similarity transform by a permutation matrix). Preserved in the model: all
weights active, `message_passing` true, since the selector is
`endswith("mlp")`. Destroyed: which node's features arrive at which node. A
single permutation is shared across views, matching the real setting where
every view indexes the same node set. `teacher_shuffled` and `gae_shuffled` are
dispatched in `run_one` (`:498-500`) and are otherwise identical to
`teacher_gnn` and `gae_uniform`: same seed, splits, bandwidths, `k`,
trajectories, epoch budget, patience, KMeans protocol, frozen uniform fusion.

The GAE arm needs one extra care. Its reconstruction target is the union of the
*true* per-view kNN graphs, so `fit_model` takes a separate `edge_graphs`
argument (`:355`, `:373`) and only the aggregation graphs are shuffled.
Shuffling the target as well would make it unlearnable, and that collapse would
be misread as evidence for message passing.

**Grids.** Both were produced by the code as it stood before the α parameter
existed, which is the plain row-stochastic construction the screen used; the
rows carry no `alpha` field. Teacher and Nyström rows reproduce
`results/gnn_mdt/metrics.jsonl` to 0.0 maximum absolute difference, so the
shuffled machinery is inert on the unshuffled arms.

- *Control grid*: the six screen datasets × five seeds, 360 rows,
  `results/gnn_mdt/control_metrics.jsonl` — directly comparable to the locked
  run.
- *Power grid*: twelve datasets, 612 rows,
  `results/gnn_mdt/power_metrics.jsonl`, summary rebuilt into
  `power_summary.json`.

The twelve are the six screen datasets plus 3Sources, BBCSport, Prokaryotic,
Reuters-1200, NUS-WIDE and MNIST-4. **This is not the ledger's pooled twelve.**
3Sources and BBCSport belong to the untouched-validation six and are already
spent; Prokaryotic, Reuters-1200, NUS-WIDE and MNIST-4 are new to this ledger;
100Leaves, Yale, Cora and CiteSeer are absent. Those four were excluded before
the run on stated grounds — Cora and CiteSeer carry an adjacency-as-feature view
whose columns misalign once `cap_train` subsamples the rows, and 100Leaves puts
100 classes against 800 training points — but the ledger's own rule retained
them rather than dropping them post hoc. The A3 mix is therefore the more
favourable one and cannot overturn a validation-set closure. It is a power and
control analysis, not a fresh validation.

**Evidence.** Per-dataset means on the power grid; GAE columns are train AMI,
teacher columns are test AMI.

| dataset | gae unif | gae mlp | gae shuf | Δ mp | Δ signal | teacher gnn | teacher shuf | Δ signal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3Sources | 0.0712 | 0.0584 | 0.0550 | +0.0128 | +0.0162 | 0.2184 | 0.2818 | -0.0634 |
| BBCSport | 0.1965 | 0.1121 | 0.1240 | +0.0844 | +0.0724 | 0.1868 | 0.2374 | -0.0506 |
| Caltech101-7 | 0.5750 | 0.5161 | 0.5469 | +0.0589 | +0.0281 | 0.5565 | 0.5510 | +0.0054 |
| Handwritten | 0.8511 | 0.7960 | 0.7810 | +0.0551 | +0.0700 | 0.8823 | 0.8768 | +0.0055 |
| MNIST-4 | 0.7850 | 0.7841 | 0.7519 | +0.0008 | +0.0331 | 0.7071 | 0.7076 | -0.0005 |
| MSRC-v5 | 0.6731 | 0.6776 | 0.6590 | -0.0045 | +0.0140 | 0.6789 | 0.7155 | -0.0365 |
| NUS-WIDE | 0.1196 | 0.1157 | 0.1151 | +0.0039 | +0.0045 | 0.1092 | 0.1126 | -0.0034 |
| OutdoorScene | 0.5698 | 0.5445 | 0.5204 | +0.0254 | +0.0494 | 0.5520 | 0.5480 | +0.0040 |
| Prokaryotic | 0.2219 | 0.1684 | 0.1310 | +0.0535 | +0.0909 | 0.0812 | 0.0385 | +0.0427 |
| Reuters-1200 | 0.1522 | 0.1174 | 0.1166 | +0.0348 | +0.0356 | 0.0931 | 0.1021 | -0.0090 |
| UCI | 0.8465 | 0.8132 | 0.7882 | +0.0334 | +0.0584 | 0.8245 | 0.8138 | +0.0106 |
| Wikipedia | 0.3928 | 0.3630 | 0.2908 | +0.0298 | +0.1020 | 0.3458 | 0.3057 | +0.0400 |

Paired effects with the ledger's estimator: seeds averaged within a dataset,
then a paired 95% Student interval across the dataset means.

| contrast | metric | control, 6 datasets | power, 12 datasets |
|---|---|---:|---:|
| `gae_uniform` − `gae_shuffled` | train | +0.0543 [+0.0199, 0.0887], 26/30 | +0.0479 [+0.0280, 0.0678], 44/51 |
| `gae_uniform` − `gae_shuffled` | test | +0.0603 [+0.0326, 0.0879] | +0.0559 [+0.0300, 0.0817] |
| `gae_uniform` − `gae_shuffled` | fidelity | +0.1641 [+0.0580, 0.2702] | +0.0949 [+0.0361, 0.1536] |
| `gae_uniform` − `gae_mlp` | train | +0.0303 [+0.0083, 0.0523], 25/30 | +0.0324 [+0.0152, 0.0495], 40/51 |
| `gae_shuffled` − `gae_mlp` | train | -0.0240 [-0.0584, 0.0103], 12/30 | -0.0155 [-0.0323, 0.0012], 23/51 |
| `gae_shuffled` − `gae_mlp` | test | -0.0189 [-0.0397, 0.0018] | -0.0317 [-0.0576, -0.0058] |
| `teacher_gnn` − `teacher_shuffled` | test | +0.0049 [-0.0208, 0.0307], 16/30 | -0.0046 [-0.0250, 0.0158], 24/51 |
| `teacher_gnn` − `teacher_shuffled` | inductive | +0.0147 [-0.0085, 0.0378] | +0.0109 [-0.0012, 0.0230] |
| `gae_uniform` − `mdt_consensus` | test | +0.0168 [-0.0422, 0.0758] | +0.0340 [-0.0089, 0.0769] |
| `gae_learned` − `mdt_consensus` | test | +0.0227 [-0.0387, 0.0840] | +0.0328 [-0.0120, 0.0776] |

**Results.** The capacity reading of the GAE effect is eliminated.
`gae_shuffled` runs the full architecture with every weight live and loses
0.0479 train AMI [+0.0280, 0.0678] to the identical model on true graphs, on 44
of 51 pairs. Had the `gae_uniform` − `gae_mlp` gain come from the extra live
parameters, the shuffled arm would have matched `gae_uniform`; it does not come
close. The same contrast on Procrustes fidelity, +0.0949 [+0.0361, 0.1536],
shows the structure the GAE exploits is MDT-aligned rather than incidental.

The stronger reading — wrong neighbours are worse than *no* neighbours — is
directionally consistent everywhere but established on only one metric. The
mean ordering puts `gae_shuffled` last on both grids, and the contrast against
`gae_mlp` excludes zero on test AMI at twelve datasets (-0.0317 [-0.0576,
-0.0058]) while crossing zero on train AMI on both grids (-0.0240 [-0.0584,
0.0103] and -0.0155 [-0.0323, 0.0012]). Report it as a tendency, not a result.
The load-bearing claim does not depend on it.

On the teacher arm the graph is worth nothing, and A3 upgrades that from a
failure to replicate into a positive null: destroying the graph entirely costs
-0.0046 test AMI [-0.0250, 0.0158] over twelve datasets, an upper limit below
the +0.02 minimum useful effect, hence *dead end* under the suite's own rule.
The mechanism is the one G1 already identified from the other direction — the
regression target *is* the MDT embedding, so the graph is already inside the
label and message passing has nothing left to add. The inductive metric is the
one place a residue survives, +0.0109 [-0.0012, 0.0230], which is the same
sub-threshold hint G2 recorded and is not enough to reopen anything.

`gae_vs_consensus` does not resolve, and not for want of power. The effect is
stable across the two grids (+0.0227 → +0.0328 learned; +0.0168 → +0.0340
uniform) while the half-width fell from 0.061 to 0.045, so it sits on the
+0.02 threshold rather than drifting toward it. Compute is 14.1× the sparse SVD
on the power grid. A real but threshold-sized gain at fourteen times the cost
is what the minimum-effect rule exists to reject.

**Reproducibility.** Teacher, Nyström and consensus rows are bit-exact across
identical invocations. GAE rows are not: the same dataset/seed/method cell moves
by up to 0.0205 train AMI between two runs of the same six datasets. The cause
is early stopping on `value < best_loss - 1e-6` applied to a loss whose sparse
matrix products are not bit-reproducible, so an infinitesimal difference shifts
the stopping epoch and the restored best state. Aggregates absorb it — the
`gae_uniform` − `gae_mlp` effect moved 0.0315 → 0.0303 between the two runs of
the screen six — so cite GAE aggregates and never a single GAE row.

**Completeness.** The power grid is 612 of 720 rows: seeds 0-3 finished on all
twelve datasets, seed 4 only on MSRC-v5, Handwritten and Wikipedia, because the
process was killed rather than failing. Nine of the twelve dataset means
therefore average four seeds instead of five. `--resume` would finish it; given
that every effect above is stable from six to twelve datasets, no verdict is
expected to move.

**Verdict.** A3 succeeds as a control and reopens nothing. It removes one
specific escape route — "the GNN only won because more of its weights were
alive" — and in doing so converts the GAE message-passing gain into a
demonstrated graph-structure effect on this dataset mix, and the teacher arm's
graph term into a demonstrated null. Neither changes a decision: G2 stays a dead
end and is now dead for a stated reason rather than for a missing replication,
and G3 stays closed because a threshold-sized advantage over `mdt_consensus` at
14× the compute still fails the useful-effect test. The lasting instruction is
procedural: a zero-input ablation is not a capacity control on its own, and any
future message-passing claim in this line should ship its shuffled-graph arm in
the same run.

## Amendment A2 — the AlphaFold family on the OOS map

**Scope, and what was deliberately not built.** The GNN closure leaves one live
result: parametric out-of-sample extension by a feature-only map. The question
here is whether the DeepMind Alpha-family design ideas improve that map. Only
two of them survive contact with this problem, and the ledger's own numbers are
the reason:

- **Recycling** (AF2's iterative refinement) is tested. It attacks the observed
  failure mode directly — `inductive_ami` is below `test_ami` for every method in
  the suite, meaning the OOS map lands in a slightly wrong frame — and it adds
  almost no parameters, so a gain could not be a capacity artefact.
- **Pair-representation targets** (AF predicts pair geometry rather than
  coordinates) are tested. This is the one place where the Alpha-family framing
  says something structural about MDT: the student currently regresses `U_k Σ_k`,
  whose frame is arbitrary, which is exactly why `procrustes_fidelity` has to
  rotate before measuring. A Gram target removes that gauge freedom.
- **Evoformer stacks, triangle multiplicative updates, axial attention, MSA and
  template modules are not built.** Templates are retrieval over a training set
  and are structurally what `mdt_nystrom` and message passing already do, closed
  under G2. The rest is capacity, and capacity is not the binding constraint on a
  rank-k SVD target that `teacher_mlp` already reproduces at 0.9904 mean
  Procrustes fidelity with 89 of 120 rows above 0.99. Adding an attention stack
  here would repeat the confound the shuffled-graph null exists to catch.

**Methodology.** A 2×2 factorial over {coordinate, pair} × {no recycling,
3 cycles}, all feature-only, all distilling the same fixed MDT SVD teacher, on
the six screen datasets × five seeds; 210 rows in
`results/gnn_mdt/af_metrics.jsonl`, verdicts in `results/gnn_mdt/af_summary.json`.
Recycling re-enters the previous cycle's embedding into the fused input through a
projection that is **zero-initialised** (`MultiViewSAGE.__init__`,
`closure.py:276-281`, applied at `:303` and `:319-321`), so an untrained recycled
model is bit-identical to `teacher_mlp` and the baseline is nested inside the
candidate; all cycles but the last run under `no_grad`, as in AF2, so the added
cost is forward-only (`closure.py:410-416`, inference loop `:447-453`). The pair
arm replaces `MSE(z, F)` with `MSE(z zᵀ, G)` where `G = F Fᵀ` scaled to unit RMS
(`closure.py:395-396`, loss at `:418`); it adds no parameters at all. Variant
flags are substring tests, so `teacher_mlp_pair_recycle` composes both
(`:377-380`), and `recycles` defaults to 3 (`:545`). Everything else
is untouched: same splits, seeds, graphs, epoch budget, patience, optimiser and
KMeans protocol, `inductive_ami` pre-declared as primary because it is the metric
that punishes landing in the wrong frame.

**Evidence.** Per-dataset means over five seeds. `mlp` is the current survivor,
`nys` the training-free baseline.

| dataset | induct nys | induct mlp | + recycle | + pair | + pair & recycle |
|---|---:|---:|---:|---:|---:|
| MSRC-v5 | 0.5989 | 0.6176 | 0.6231 | 0.6578 | 0.6448 |
| Handwritten | 0.8715 | 0.8792 | 0.8793 | 0.8594 | 0.8583 |
| Wikipedia | 0.2768 | 0.2783 | 0.2647 | 0.3014 | 0.3041 |
| UCI | 0.8047 | 0.8275 | 0.8312 | 0.8045 | 0.8084 |
| OutdoorScene | 0.4999 | 0.5291 | 0.5266 | 0.5224 | 0.5368 |
| Caltech101-7 | 0.5050 | 0.5176 | 0.5199 | 0.5216 | 0.5350 |

| dataset | test nys | test mlp | + recycle | + pair | + pair & recycle |
|---|---:|---:|---:|---:|---:|
| MSRC-v5 | 0.6202 | 0.6892 | 0.6554 | 0.6999 | 0.6932 |
| Handwritten | 0.8698 | 0.8756 | 0.8855 | 0.8579 | 0.8661 |
| Wikipedia | 0.2923 | 0.2955 | 0.2921 | 0.3178 | 0.3081 |
| UCI | 0.7818 | 0.8242 | 0.8231 | 0.8148 | 0.8114 |
| OutdoorScene | 0.5232 | 0.5352 | 0.5331 | 0.5488 | 0.5501 |
| Caltech101-7 | 0.5283 | 0.5404 | 0.5413 | 0.5440 | 0.5428 |

Paired effects, seeds averaged within a dataset then a paired 95% Student
interval across the six dataset means:

| arm | metric | vs `teacher_mlp` | vs `mdt_nystrom` |
|---|---|---:|---:|
| recycle | inductive | -0.0008 [-0.0080, 0.0064] | +0.0147 [-0.0012, 0.0305] |
| recycle | test | -0.0049 [-0.0206, 0.0107] | +0.0192 [+0.0025, 0.0358] |
| recycle | train | -0.0032 [-0.0075, 0.0011], 6/30 wins | +0.0062 [+0.0005, 0.0120] |
| pair | inductive | +0.0029 [-0.0231, 0.0290] | +0.0184 [-0.0072, 0.0440] |
| pair | test | +0.0038 [-0.0119, 0.0196] | +0.0279 [-0.0034, 0.0592] |
| pair & recycle | inductive | +0.0063 [-0.0163, 0.0290] | +0.0218 [-0.0015, 0.0450] |
| pair & recycle | test | +0.0019 [-0.0099, 0.0138] | +0.0260 [-0.0011, 0.0531] |
| pair | fidelity | **-0.0850 [-0.1395, -0.0305]**, 0/30 wins | -0.0945 [-0.1550, -0.0340] |
| recycle | fidelity | +0.0003 [-0.0000, 0.0006] | -0.0093 [-0.0205, 0.0019] |

**Results.** A2a is a decided negative: recycling's inductive effect is
-0.0008 with an upper limit of 0.0064, a third of the minimum useful effect, so
the rule labels it *dead end* rather than inconclusive. It is mildly harmful
where it is measurable at all — train AMI -0.0032 with 6/30 wins, test AMI
-0.0049 — and it is not free: training goes from 1.001 s to 1.623 s and
inference from 6.93 ms to 20.11 ms, a 2.9× query cost for a negative effect. The
mechanism is not mysterious. AF2 recycling refines a *structure* prediction
against a geometry module that can be re-evaluated; here the student regresses a
fixed linear-algebraic target in one shot, so the second and third passes have
nothing new to condition on and the recycled input acts as noise the model must
learn to ignore — which, given the zero-initialised projection, is exactly what
it mostly does.

A2b does not clear the bar either: +0.0029 inductive, +0.0038 test, both
intervals straddling zero, and the combination with recycling adds nothing beyond
the pair target alone (+0.0063 inductive), so there is no interaction to exploit.
Its dataset heterogeneity is severe and structured: the pair target gains on the
two datasets where the coordinate map is weakest relative to Nyström (MSRC-v5
+0.0402, Wikipedia +0.0231 inductive) and loses on the two where the coordinate
map is already strong (Handwritten -0.0198, UCI -0.0230). That is a real pattern,
but it is a per-dataset trade, not an improvement.

The one unambiguous finding is a negative that matters more than either
hypothesis. The pair arm loses **0.0850 Procrustes fidelity** to the teacher,
0/30 wins, while its AMI is statistically indistinguishable from the coordinate
arm's. MSE over the Gram weights each direction by roughly the square of its
singular value, so the pair arm fits the dominant directions and neglects
low-variance ones, which column-standardised Procrustes counts equally — hence
fidelity collapses to 0.86 on Handwritten and 0.85 on UCI and Wikipedia while
clustering quality does not move. G1 already showed that a fidelity *advantage*
buys no AMI; A2b shows the converse, that a large fidelity *loss* costs no AMI
either. Procrustes fidelity to the fixed MDT target is therefore not a proxy for
anything the suite cares about, in either direction, and should be reported as a
distillation diagnostic only.

**Verdict.** Both arms closed on the screen, so no untouched-data validation was
spent. Recycling is a dead end with a cost attached; the pair target is a
dataset-dependent trade with no mean gain and a large fidelity penalty. The
Alpha-family transplant does not improve MDT out-of-sample extension, and the
best arm against the training-free baseline (+0.0218 inductive for pair &
recycle) is not better than the plain feature-only map that was already the
survivor. The honest reading is that this suite has now tested three different
mechanisms on top of parametric OOS distillation — message passing, iterative
refinement, and gauge-free targets — and none of them beats the two-layer
feature-only map, which is consistent with the theorem the ledger opens with: at
the global optimum the target is a rank-k factorisation, and the remaining
headroom is in *what target is chosen*, not in the machinery that fits it. A
reopened Alpha-family branch would need to change the teacher — for instance a
target that is itself refined across cycles, which is what AF2 actually does and
what none of these arms did.

## Literature anchors

- Hamilton, Ying & Leskovec, *Inductive Representation Learning on Large
  Graphs* (GraphSAGE), <https://arxiv.org/abs/1706.02216>.
- Kipf & Welling, *Variational Graph Auto-Encoders*,
  <https://arxiv.org/abs/1611.07308>.
- Veličković et al., *Deep Graph Infomax*,
  <https://arxiv.org/abs/1809.10341>.
- Jumper et al., *Highly accurate protein structure prediction with AlphaFold*,
  Nature 596 (2021) — recycling and the pair representation, the two mechanisms
  transplanted in A2, <https://doi.org/10.1038/s41586-021-03819-2>.
- Abramson et al., *Accurate structure prediction of biomolecular interactions
  with AlphaFold 3*, Nature 630 (2024) — pair-geometry prediction retained with a
  diffusion decoder, <https://doi.org/10.1038/s41586-024-07487-w>.
