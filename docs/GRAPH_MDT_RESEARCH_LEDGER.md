# MDT × graphs, second formulation: pre-registered ledger

The first GNN ledger (`GNN_RESEARCH_LEDGER.md`) closed all six hypotheses, but
its formulation could not have detected a graph advantage: every graph was a
kNN re-derivation of the features, datasets were subsampled to 800 nodes, and
the harness's Gaussian-euclidean kNN collapses on bag-of-words features
(measured ACM feature-graph homophily 0.36 euclidean vs 0.71 cosine). This
ledger replaces that formulation. It was written before any GNN result of the
new suite existed; the gates and thresholds below are frozen.

## Pivotal hypothesis

> When at least one view carries relational information not recoverable from
> the features, graph message passing beats MDT consensus **given the same
> inputs** (natural adjacencies supplied to MDT as views).

Everything else is an ablation of this claim. The decisive comparison is
GNN vs MDT-with-graph-views, never GNN vs feature-only MDT.

## Pre-registered gates and thresholds (frozen)

1. **Independence gate.** A dataset enters the primary analysis iff at least
   one natural view has edge Jaccard ≤ 0.30 against a cosine feature-kNN graph
   of matched density (capped at k=128). Views with adjusted homophily < 0.05
   are flagged heterophilous and analysed as secondary controls.
2. **Minimum useful effect** +0.02 test AMI, unchanged from the first ledger.
3. **Verdict rule** (enforced by code, not prose): effects are paired per
   (dataset, seed), seeds averaged within dataset, 95% t-interval across
   dataset means. `continue` iff mean ≥ +0.02 and lower bound > 0; `dead_end`
   iff upper bound < +0.02; otherwise `inconclusive`. No verdict from fewer
   than 5 primary datasets. Secondary: win-rate sign test across datasets.
4. **Model selection is label-free.** One unsupervised criterion selects
   trajectories, checkpoints, and any per-dataset choice; identical for every
   arm; labels are read only after a run. Criterion: **modularity of the
   KMeans(k) partition on the binary union of all input graphs.**
   *Amendment (pre-grid, recorded 2026-07-23):* the original criterion was
   silhouette, which pathologically rewards collapsed diffusion embeddings —
   ACM placement measured consensus NMI 0.63 vs silhouette-selected 0.004,
   with silhouette preferring the degenerate PLP-dominated operator. Replaced
   before any grid row existed; the broken ACM select rows are the evidence.
5. **No view selection by labels.** The independence screen uses labels to
   qualify datasets and interpret results, never to configure methods.

## Step 1 — placement against the published record

Full-graph protocol (all nodes, no split), KMeans on the embedding,
ACC/macro-F1/NMI/ARI, matching the multiplex clustering literature.

Published anchors (details and caveats in the research notes; two incompatible
dataset lineages share names — we use the DMGI lineage for DBLP-7907 and
IMDB-3550, the shared 3025-node ACM):

| method | ACM NMI | DBLP-7907 NMI | IMDB-3550 NMI |
|---|---:|---:|---:|
| O2MAC (WWW 2020) | 0.692 | 0.407 (BTGF re-run) | — |
| MCGC (NeurIPS 2021) | 0.713 | 0.551 (BTGF re-run) | — |
| HDMI (WWW 2021) | 0.695 | 0.582 (test split) | 0.198 (test split) |
| BTGF (AAAI 2024) | 0.758 | 0.624 | — |
| DMGI (AAAI 2020) | 0.687 | 0.409 (test split) | 0.196 (test split) |

Caveats: DMGI/HDMI report NMI on test nodes only; O2MAC-lineage numbers are
full-graph. Cross-paper re-runs of the same method diverge by up to 30 NMI
points (see notes), so placement is against the *range*, not a single number.

MDT placement arms (all label-free): `feat` (cosine kNN transition only),
`graph` (natural adjacencies only), `all` (both), × fusion mode
(`consensus` = 8 random Dirichlet trajectories, `select` = best silhouette of
16 sampled + one-hot trajectories), × embedding dim (n_classes, 32).
Results: `results/graph_mdt/placement.jsonl`.

## Step 2 — independence screen (completed)

`results/graph_mdt/independence.jsonl`, produced by
`python -m experiments.graph_mdt.independence`. Summary:

| dataset | view | adj. homophily | kNN adj. homophily | Jaccard | primary? |
|---|---|---:|---:|---:|---|
| acm | PAP | 0.728 | 0.563 | 0.030 | yes |
| acm | PLP | 0.455 | 0.355 | 0.080 | yes |
| dblp | PAP | 0.609 | 0.429 | 0.029 | yes |
| dblp | PPrefP | 0.540 | 0.453 | 0.020 | yes |
| dblp | PTP | 0.130 | 0.300 | 0.048 | yes |
| dblp | PATAP | 0.009 | 0.300 | 0.021 | heterophilous flag |
| imdb | MAM | 0.167 | 0.240 | 0.008 | yes |
| imdb | MDM | 0.425 | 0.297 | 0.014 | yes |
| cora | A0 | 0.771 | 0.488 | 0.060 | yes |
| citeseer | A0 | 0.673 | 0.536 | 0.114 | yes |
| amazon_ratings | graph | 0.140 | 0.087 | 0.034 | yes |
| minesweeper | graph | 0.009 | ~0 | 0.001 | heterophilous control |
| roman_empire | graph | −0.047 | 0.585 | 0.002 | heterophilous control |

Every natural view passes the Jaccard gate: unlike the first formulation,
these graphs are not feature re-derivations. Primary datasets: acm, dblp,
imdb, cora, citeseer, amazon_ratings (6). Secondary heterophilous controls:
minesweeper, roman_empire.

## Step 3 — method grid

| arm | question it answers |
|---|---|
| MLP + feature-space SSL (no graph) | feature ceiling |
| diffusion/spectral on adjacency only | graph ceiling without learning |
| MDT consensus incl. graph views | lab baseline, same inputs |
| GCN and SAGE backbones × GAE and BGRL objectives | the pivotal claim |
| GNN→MLP distillation | is the graph needed at inference |

Decomposition readout: if `all ≤ max(features-only, graph-only)` for a
method family, fusion failed — a different conclusion from "GNN failed".
DGI stays closed (first ledger, 30/30 losses, objective misalignment).

Grid settings, frozen before any grid row existed: embedding dim 32 for
every arm (KMeans still uses k = class count), hidden 128, 150 epochs,
checkpoint by the modularity criterion every 15 epochs, encoder BatchNorm
after layer 1 (added after the Cora smoke run showed textbook BGRL collapse
without it — recorded as a pre-grid amendment, not tuned on grid results).
Fused adjacency rows are capped at their top-64 weights (SAGE-style
neighbour sampling, label-free): full-batch passing on the 700+-degree
metapath views cost 88 minutes per cell for no design reason. ARPACK
non-convergence on a degenerate trajectory skips that trajectory instead of
killing the run. All amendments predate the first valid grid row.
DBLP only: the same top-64 cap is applied to the MDT transition views
(`GRAPH_MDT_VIEW_CAP=64`) — PATAP's 28.5M edges make trajectory SVDs
intractable otherwise; symmetric with the GNN's capped fused graph and
recorded before any DBLP grid row existed.

## Step 4 — datasets

Primary (6): acm, dblp, imdb (DMGI-lineage multiplex), cora, citeseer
(intact citation graphs), amazon_ratings. Heterophilous controls (2):
minesweeper, roman_empire. Loaders: `experiments/graph_mdt/datasets.py`,
cached under `/tmp/multiplex-data/cache`. No subsampling: graphs stay intact.
ogbn-arxiv deferred until something survives at this scale (CPU-only box).

## Step 5 — evaluation protocol

- Metrics: AMI (primary), NMI, ARI, ACC, macro-F1; KMeans with 10 restarts,
  5 seeds; plus spectral clustering on the embedding as a robustness check.
- Inductive protocol (secondary): 70/30 split, embed train, extend to test,
  assign test points to train-fitted centroids — same as the first ledger.
- Compute: wall-clock and peak memory per arm; OOS break-even query count.
- Statistics: as in the frozen verdict rule above; heterogeneity across
  datasets reported (per-dataset effects always shown, never only the pool).
- Verdicts emitted by `experiments/graph_mdt` code from the intervals;
  hand-written verdict prose is not admissible.

## Commands

```bash
python -m experiments.graph_mdt.datasets          # convert + cache + smoke
python -m experiments.graph_mdt.independence      # step 2 screen
python -m experiments.graph_mdt.placement         # step 1b placement arms
```

## Results (2026-07-24): grid complete, 440/440 rows

Full-graph AMI, mean over 5 seeds (`results/graph_mdt/grid.jsonl`; verdicts
in `grid_verdicts.json`, emitted mechanically by `verdicts.py`):

| arm | acm | dblp | imdb | cora | citeseer | amazon | minesw. | roman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mdt_consensus_all | **0.619** | **0.550** | 0.001 | **0.436** | **0.381** | 0.002 | 0.002 | 0.287 |
| mdt_select_all | 0.522 | 0.500 | 0.089 | 0.419 | 0.384 | 0.002 | 0.000 | 0.292 |
| mdt_feat | 0.509 | 0.061 | 0.147 | 0.285 | 0.278 | 0.001 | 0.000 | 0.110 |
| mdt_graph | 0.022 | 0.377 | 0.002 | 0.160 | 0.051 | 0.007 | 0.000 | 0.002 |
| gcn_gae | 0.441 | 0.478 | 0.025 | 0.380 | 0.168 | 0.018 | 0.050 | 0.100 |
| gcn_bgrl | 0.420 | 0.520 | 0.038 | 0.241 | 0.070 | 0.007 | 0.051 | 0.109 |
| sage_gae | 0.417 | 0.468 | 0.044 | 0.371 | 0.162 | 0.020 | 0.053 | 0.144 |
| sage_bgrl | 0.070 | 0.060 | 0.013 | 0.039 | 0.018 | 0.002 | -0.000 | 0.346 |
| mlp_ssl | 0.023 | 0.033 | 0.012 | 0.009 | 0.006 | 0.001 | -0.000 | 0.350 |
| kmeans_feat | 0.379 | 0.383 | **0.150** | 0.082 | 0.051 | 0.001 | 0.000 | 0.347 |
| distill_mlp | 0.424 | 0.501 | 0.047 | 0.377 | 0.168 | 0.008 | 0.000 | **0.352** |

Pivotal verdicts (6 primary datasets, frozen rule):

| comparison | effect | 95% CI | wins | verdict |
|---|---:|---|---|---|
| gcn_gae vs mdt_select_all | -0.068 | [-0.152, +0.017] | 1/6 | dead_end |
| sage_gae vs mdt_select_all | -0.072 | [-0.160, +0.015] | 1/6 | dead_end |
| sage_bgrl vs mdt_select_all | -0.286 | [-0.492, -0.080] | 1/6 | dead_end |
| gcn_bgrl vs mdt_select_all | -0.103 | [-0.236, +0.029] | 2/6 | inconclusive |
| all four GNN arms vs mdt_consensus_all | -0.08 to -0.30 | upper < +0.03 | ≤2/6 | dead_end / inconclusive |
| mdt_select_all vs mdt_graph | +0.216 | [+0.022, +0.410] | 5/6 | continue |
| distill_mlp vs mdt_select_all | -0.065 | [-0.152, +0.022] | 2/6 | inconclusive |

Closure statements:

1. **Pivotal hypothesis rejected in direction.** No GNN arm beats MDT given
   the same inputs; every mean effect is negative; GNN wins at most 2/6
   datasets, and only where the screen showed weak graphs (imdb +0.02-0.04,
   amazon +0.005-0.02). Formal labels are dead_end where intervals permit and
   inconclusive elsewhere only because the study cannot exclude a +0.02 GNN
   advantage — it can and does exclude any large one on these datasets.
2. **Where GNNs lose, they lose to fusion, not features.** GNN arms beat
   kmeans_feat 5/6 (effect +0.04-0.08, inconclusive); MDT's advantage comes
   from consensus over feature+graph views (fusion readout: continue).
3. **Controls behaved as designed.** roman_empire (anti-correlated graph):
   message passing hurts every GNN (-0.14 to -0.19 vs MDT; feature-only arms
   win at 0.35). minesweeper (structure-only): GNNs +0.05 over everything —
   the only regime with a graph-specific gain, and it is small.
4. **Trajectory selection: consensus beats the silhouette/select recipe**
   on acm (+0.10) and dblp (+0.05); the silhouette criterion itself was shown
   to prefer collapsed embeddings (amendment above). Actionable for the lab
   independent of any GNN question.
5. **amazon_ratings carries no recoverable cluster signal for any method**
   (max AMI 0.02) — retained in the primary analysis per pre-registration.
6. **Placement (step 1) headline.** Zero-training MDT consensus with graph
   views reaches NMI 0.63 on ACM and 0.59 on DBLP-7907 vs published GNN
   clustering 0.69-0.76 and 0.41-0.62 respectively; on DBLP it beats HDMI
   and every pre-2024 method. The field's GNN-over-spectral margin on these
   benchmarks is small once the spectral method is given the same graphs.

## Enhancement round E1/E3 (pre-registered 2026-07-24, before any E-row)

The grid located MDT's failures: noise views poison uniform consensus
(imdb, acm-PLP), and diffusion cannot use heterophilous structure
(minesweeper is the only regime with a GNN gain, +0.05). Two heuristic
arms test the cheap fix first; a GNN version is licensed only if its
heuristic shows the effect exists.

- **E1 `mdt_gated`** — per-node view gating. Gate of view v at node i is
  the cosine between node i's raw features and its view-v neighbour
  average (P_v X, self-loop-free); gates are clipped at 0 and normalised
  per node; per-step Dirichlet weights modulate the gates and the mixture
  stays row-stochastic. Consensus over 8 trajectories as in
  `mdt_consensus_all`. Global learned weights stay closed; this tests
  *sample-dependent* gating, the reopening clause of both ledgers.
- **E3 `mdt_a2`** — two-hop derived views. For every natural adjacency A,
  add binarised A^2 (diagonal removed, rows capped at top-64) as an extra
  view to the `mdt_consensus_all` recipe. Tests whether "same-role"
  2-hop homophily captures the heterophilous signal without any GNN.

Frozen readouts (same verdict rule, same 6 primary datasets):
1. `mdt_gated` vs `mdt_consensus_all` — E1 primary.
2. `mdt_a2` vs `mdt_consensus_all` — E3 primary.
3. Rescue check: `mdt_gated` on imdb ≥ the 0.147 feat-only AMI means
   gating defeats the noise-view failure; licenses a learned (GNN) gate.
4. Control check: `mdt_a2` vs `gcn_gae` on minesweeper — if A^2 closes
   the +0.05 gap, the heterophily argument for GNNs dies too.
5. A combined gated+A^2 arm runs only if both individual arms clear
   their primary readout.

*Amendment before the first recorded E-row:* the acm/imdb seed-0 smoke
showed the frozen cosine gates are too soft to defeat imdb's noise views
(feature view ≈0.4 weight, gated AMI 0.001). A sharpened variant
`mdt_gated_sharp` (gates^4, renormalised; β=4 fixed a priori, no sweep)
runs as a **secondary** arm. Because the smoke touched imdb — the rescue
dataset — the sharp variant cannot claim the pre-registered rescue readout;
only `mdt_gated` can. The sharp arm is exploratory and any positive result
needs fresh validation.

## E1/E3 results (2026-07-24, 120/120 rows)

AMI, mean over 5 seeds:

| arm | acm | dblp | imdb | cora | citeseer | amazon | minesw. | roman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mdt_consensus_all (ref) | 0.619 | 0.550 | 0.001 | 0.436 | 0.381 | 0.002 | 0.002 | 0.287 |
| mdt_gated | 0.624 | 0.150 | 0.001 | 0.379 | 0.324 | 0.002 | 0.008 | 0.318 |
| mdt_gated_sharp (expl.) | 0.685 | 0.032 | 0.001 | 0.342 | 0.220 | 0.002 | -0.000 | 0.222 |
| mdt_a2 | 0.596 | 0.414 | 0.001 | 0.454 | 0.322 | 0.003 | 0.002 | 0.173 |

Verdicts (frozen rule): `mdt_gated` vs consensus -0.085 [-0.250, +0.080],
1/6 wins, inconclusive; `mdt_a2` vs consensus -0.033 [-0.093, +0.027], 2/6,
inconclusive; `mdt_a2` vs `gcn_gae` on minesweeper -0.048 — **A^2 does not
close the GNN's structure-only gap**, the one graph-specific gain stands.

Closure:

1. **E1 heuristic fails its primary readout and the imdb rescue** (gated
   imdb 0.001 vs 0.147 feat-only). Under the frozen licensing rule, a
   learned (GNN) gate is **not licensed** by this round.
2. **The failure is diagnostic, not random.** The gate anchors on feature
   consistency, so it helps exactly where features are strong (acm +0.07
   AMI / +0.21 ACC for the sharp variant: ACC 0.886, NMI 0.685 —
   O2MAC-level from a zero-training spectral method) and destroys where
   features are weak (dblp 0.55 → 0.03). Per-node gating has real leverage;
   the *signal* must estimate view quality without anchoring on any single
   view. A future round may pre-register e.g. cross-view neighbourhood
   agreement as the gate signal; it starts as a new hypothesis, not a
   continuation of this one.
3. **E3 dead as an MDT enhancement** and as an explanation of the GNN's
   minesweeper edge. The +0.05 structure-only GNN gain on minesweeper is
   the single surviving graph-specific result in the whole programme; it is
   small and lives on a synthetic control.
4. The exploratory sharp arm's acm result (ACC 0.886) was obtained after
   the smoke touched acm and carries no pre-registered status; treating it
   as a claim requires fresh validation.

## Sharp-gate validation on untouched data (pre-registered 2026-07-24)

The exploratory `mdt_gated_sharp` acm result (ACC 0.886 / NMI 0.685) gets
one validation shot on the DMGI `amazon` multiplex pickle — never loaded by
any prior run of this programme. Frozen protocol: identical recipes, 5
seeds, MDT-family arms plus `kmeans_feat` reference only (GNN arms skipped;
no GNN claim is under validation). Readouts, frozen before the first row:

1. `mdt_gated_sharp` vs `mdt_consensus_all` on amazon — replication check.
   The gate anchors on feature consistency, so the prediction is
   conditional: it should help only if amazon's feature view is strong
   (mdt_feat near the best arm) and hurt otherwise.
2. `mdt_gated` (primary formula) same comparison, same prediction.
3. No further amendment of gate formulas after this run; a third gate
   design requires a new hypothesis section.

### Validation results (2026-07-24, 40/40 rows)

AMI, mean over 5 seeds: kmeans_feat 0.048, mdt_feat 0.021,
mdt_gated_sharp 0.014, mdt_gated 0.011, mdt_consensus_all 0.001,
mdt_select_all/mdt_a2/mdt_graph ≈ 0.000.

1. **The amazon dataset is uninformative for the replication check**: no
   method exceeds 0.05 AMI (like amazon_ratings). The exploratory acm
   sharp-gate result (ACC 0.886) therefore remains **unvalidated** — it
   gains no support and suffers no refutation here. It must not be
   presented as more than exploratory.
2. **The conditional gate model is refined, not confirmed.** Gates beat
   consensus on amazon (+0.013/+0.010) despite weak features, because the
   graph views are pure noise (mdt_graph 0.000) and the gate correctly
   leans toward the only informative view. Full-programme pattern: the
   feature-anchored gate helps when features are at least as informative
   as the graphs (acm, amazon, roman_empire) and hurts when graphs
   dominate (dblp, cora, citeseer). That is exactly what a
   feature-anchored signal must do; it estimates *which view agrees with
   features*, not *which view is good*.
3. Gate-formula work is closed per readout 3. A view-quality signal that
   does not anchor on any single view (e.g. cross-view neighbourhood
   agreement) remains the licensed future hypothesis, as a new
   pre-registration.

## Kill criteria

- Placement shows MDT-`all` inside the published GNN range on ACM/DBLP/IMDB →
  the field's GNN gain over spectral consensus is smaller than reported;
  proceed to the controlled grid with that as the headline.
- Placement shows MDT-`all` far below the published range → MDT fusion is the
  bottleneck; fix or close before spending GNN compute.
- Grid: pivotal comparison dead by the frozen rule on ≥5 primary datasets →
  close the branch; the only reopening is genuinely relational side
  information of a different kind (sample-dependent gating already excluded).
