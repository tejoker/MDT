# GNN × MDT: what was done, why, what it showed, and what comes next

Nicolas Bigeard — report for A. Kalogeratos, 2026-07-24.
Full evidence: `GRAPH_MDT_RESEARCH_LEDGER.md/.pdf`, `results/graph_mdt/`
(600 result rows, raw), `GNN_RESEARCH_LEDGER.md` (first study).

---

## 1. Starting point and why a second study existed at all

The first GNN study (`experiments/gnn_mdt/`, ledger G1-G6) concluded that
GNNs add nothing to MDT: distillation fidelity, graph-specific
out-of-sample extension, graph autoencoders, DGI, learned view weights and
compute were all closed as dead ends after a 6-dataset screen and a
6-dataset untouched validation (540 rows). Its one surviving result:
parametric out-of-sample extension beats Nyström (+0.048 pooled test AMI),
but the effect belongs to the MLP, not to message passing.

Before presenting that as "we should not do GNN", I audited it and found
the conclusion was not defensible in that form, for five reasons:

1. **Every graph was a kNN re-derivation of the features.** The graph could
   not, by construction, contain information the feature encoder lacked.
   A GNN's null result on such graphs is near-tautological.
2. **Datasets were subsampled to 800 train / 400 test nodes**, which
   shreds graph structure and puts the comparison in the small-sample
   regime where spectral methods are strongest.
3. **Cora and CiteSeer showed no signal for any method** — a red flag for
   the pipeline, not for GNNs: these are the canonical graph benchmarks.
4. **Several verdict labels contradicted the ledger's own decision rule**
   (prose said "dead end" where the interval did not exclude the +0.02
   minimum useful effect).
5. Fixed untuned configs, a single metric, three objectives — thin coverage
   for a claim about "GNNs" as a class.

The honest options were: present a much narrower claim, or run one
decisive study that removes every objection. We chose the second.

## 2. Design of the second study and the reasoning behind each choice

Requirements set in advance: independent graph information, intact large
graphs, multiple GNN families and objectives, additional metrics, more
independent datasets — and three additions the requirement list missed:

- **A graph-independence screen as a gate, not an excuse.** For every
  dataset, measure whether the natural graph is recoverable from a
  feature-kNN graph (edge Jaccard at matched density) and whether it
  carries label-relevant structure (edge homophily + Platonov's adjusted
  homophily). Datasets whose graphs are feature re-derivations cannot test
  the hypothesis and are excluded *before* any model runs.
- **The same-inputs principle.** The fair opponent for a GNN is not
  feature-only MDT but **MDT given the same graphs as views** (adjacency →
  row-stochastic transition → one more view in the trajectory product).
  If spectral consensus with identical inputs matches the GNN, the GNN has
  no case. This single decision shaped everything downstream.
- **Pre-registration with code-enforced verdicts.** Thresholds (+0.02 AMI
  minimum useful effect, 95% t-intervals clustered by dataset, no verdict
  under 5 datasets), label-free model selection, and the rule that verdict
  labels are emitted only by `verdicts.py` — never written by hand. Every
  deviation is a timestamped amendment recorded before the rows it
  affects. This directly fixes flaw 4 of the first study.

Literature came first (zero compute): published multiplex clustering
numbers for ACM/DBLP/IMDB were collected from O2MAC, DMGI, HDMI, MvAGC,
MCGC, DuaLGR, BTGF, InfoMGF, BMGC and cross-checked. Two findings shaped
the study: (a) **two incompatible dataset lineages share the same names**
(DBLP-4057-authors vs DBLP-7907-papers; IMDB-4780 vs IMDB-3550) — copying
numbers across lineages, which several papers do, is meaningless; (b)
cross-paper re-runs of the *same* method diverge by up to 30 NMI points,
so placement targets a published *range*, never a single number.

## 3. What was built

`experiments/graph_mdt/` — self-contained suite, no torch-geometric:

- `datasets.py` — unified loaders (DMGI pickles, multi-view .mat,
  heterophilous npz) → (features, adjacency views, labels), cached npz.
  Nine datasets: acm, dblp, imdb (multiplex metapath graphs), cora,
  citeseer (intact citation graphs), amazon_ratings, minesweeper,
  roman_empire (heterophily controls, Platonov 2023), amazon (held-out
  validation). No subsampling anywhere.
- `independence.py` — the screen (step 2).
- `placement.py` — full-graph protocol runs matching the literature
  (all nodes, KMeans on embedding, ACC/macro-F1/NMI/ARI/AMI, 5 seeds).
- `methods.py` — implicit sparse MDT (LinearOperator SVD, works at 24k
  nodes without dense n²), consensus and trajectory-selection fusion,
  2-layer GCN/SAGE/MLP encoders with hand-rolled sparse message passing,
  GAE and BGRL objectives, GNN→MLP distillation, per-node view gating,
  A² two-hop views, modularity-based label-free selection.
- `grid.py` / `verdicts.py` — resume-safe runner; mechanical verdicts.

## 4. Results, round by round, with causes

### 4.1 Independence screen (13 graph views)

All natural views passed the Jaccard gate (max 0.11): none is a feature
re-derivation — the exact property the first study lacked. The per-view
homophily table later *predicted* method behaviour: ACM PAP (0.73 adjusted
homophily vs 0.56 for feature-kNN) and DBLP PAP/PPrefP carry genuine extra
signal; IMDB MAM sits *below* the feature graph (0.17 vs 0.24); DBLP PATAP
(0.009) is noise; roman_empire's graph is anti-correlated with labels
(−0.047) while its features are strong (0.585).

Two harness bugs surfaced and were fixed here, both relevant to the lab
beyond this study:

- **Gaussian-euclidean kNN collapses on bag-of-words features**: measured
  ACM feature-graph homophily 0.36 (euclidean) vs 0.71 (cosine). This is
  distance concentration in high-dimensional sparse data. It also
  retroactively explains part of the first study's dead Cora/CiteSeer.
- **Silhouette-based trajectory selection prefers collapsed embeddings.**
  On ACM, consensus reached NMI 0.63 while silhouette-selected trajectories
  returned 0.004: a noise-view-dominated diffusion smears all points into
  one blob, KMeans splits the blob into tight fake clusters, and silhouette
  rewards exactly that geometry. Selection now uses Newman modularity of
  the KMeans partition on the union input graph — still label-free, and
  degeneracy-proof because a collapsed partition has no modularity. The
  lab's `select_trajectory` uses the silhouette criterion and inherits
  this failure mode.

### 4.2 Placement against the published record (36 rows)

Full-graph NMI at dim 32, all label-free, zero training:

| dataset | feat only | graph only | all views | published range |
|---|---:|---:|---:|---|
| ACM | 0.510 | 0.003 | **0.631** | 0.69–0.76 |
| DBLP-7907 | 0.062 | 0.400 | **0.590** | 0.41–0.62 (HDMI 0.582) |
| IMDB-3550 | **0.148** | 0.005 | 0.002 | ≈0.196–0.198 |

Interpretation: on DBLP, MDT consensus with graph views **beats HDMI and
every pre-2024 published method** and trails only BTGF (2024); on ACM it
lands 0.06 under O2MAC. Fusion genuinely works where the screen said the
graphs carry signal (+0.19 over the best single arm on DBLP) and fails
exactly where the screen said the graph is weaker than the features
(IMDB). Caveat stated up front: DMGI-lineage papers report test-split NMI
while ours is full-graph; a paper version needs protocol-aligned reruns.

### 4.3 Main grid (440 rows: 11 arms × 8 datasets × 5 seeds)

Arms: MDT consensus / selection / feat-only / graph-only; GCN and SAGE
backbones × GAE and BGRL objectives on the uniformly fused, top-64-capped
adjacency; MLP+SSL and raw-KMeans feature references; GNN→MLP
distillation. Uniform settings frozen in the ledger (dim 32, hidden 128,
150 epochs, modularity checkpointing); every engineering amendment
(BatchNorm after the Cora smoke showed textbook BGRL collapse; ARPACK
fail-fast; the neighbour cap) is timestamped before the rows it touched.

Pivotal verdicts (6 primary datasets):

| comparison | mean AMI effect | 95% CI | wins | verdict |
|---|---:|---|---|---|
| gcn_gae vs mdt_select_all | −0.068 | [−0.152, +0.017] | 1/6 | dead_end |
| sage_gae vs mdt_select_all | −0.072 | [−0.160, +0.015] | 1/6 | dead_end |
| sage_bgrl vs mdt_select_all | −0.286 | [−0.492, −0.080] | 1/6 | dead_end |
| gcn_bgrl vs mdt_select_all | −0.103 | [−0.236, +0.029] | 2/6 | inconclusive |
| (all four) vs mdt_consensus_all | −0.08…−0.30 | upper ≤ +0.03 | ≤2/6 | dead_end / inconcl. |

Why the GNNs lose: their only wins are on the datasets whose graphs the
screen flagged as weak (imdb +0.02–0.04, amazon_ratings +0.005–0.02) — the
regime where MDT's diffusion also has nothing to work with. Where graphs
are informative, spectral consensus extracts the signal at least as well
as message passing, without training. Supporting reads: GNNs beat raw
KMeans-on-features 5/6 (+0.04–0.08, not significant), so they are not
broken — they are simply not better than the spectral route. Distilled
MLPs match their GNN teachers (graph unnecessary at inference —
replicating the first study's OOS finding at full scale). Controls behaved
as designed: on roman_empire (anti-correlated graph) message passing
hurts every GNN, and minesweeper (structure-only signal) is the one
regime with a real GNN edge (+0.05 over everything spectral).

### 4.4 Enhancement round E1/E3 (120 rows) — using GND ideas *inside* MDT

Rationale: the grid located MDT's two failure modes — noise views poison
uniform consensus, and diffusion cannot exploit heterophilous structure.
Before licensing any learned (GNN) component, the pre-registered rule was
heuristic-first: if the cheap version shows no effect, the learned version
is not run.

- **E1, per-node view gating** (gate = cosine between a node's features
  and its view-v neighbour average): failed its primary readout (−0.085 vs
  consensus, 1/6) and failed the IMDB rescue. The failure is diagnostic,
  not random: the signal anchors on *feature agreement*, so across all 9
  datasets it helps exactly where features are at least as informative as
  graphs (ACM: the sharpened variant reaches ACC 0.886 / NMI 0.685 —
  O2MAC-level, exploratory status) and destroys performance where graphs
  dominate (DBLP 0.55 → 0.03). A gate must estimate *view quality*, not
  feature agreement; that reframing is the round's real output.
- **E3, A² two-hop views**: dead as an enhancement (−0.033), and decisive
  in the other direction — it does **not** close the GNN's minesweeper
  edge (−0.048 vs gcn_gae there). So the one graph-specific GNN gain
  survives a spectral counter-attack; it is real, small, and lives on a
  synthetic control.
- **Validation on untouched amazon** (40 rows): the dataset turned out
  uninformative (no method above 0.05 AMI), so the exploratory ACM number
  stays exploratory. The gate's direction behaved exactly as the
  feature-anchoring model predicts even there (gates 0.011–0.014 >
  consensus 0.001, because amazon's graphs are pure noise and its features
  weakly informative).

## 5. Limitations of my own study (things a reviewer will attack)

Stated plainly because the first study earned criticism for not doing so:

1. **CPU-only budget compromises**: 150 epochs, top-64 neighbour cap,
   fixed lr/hidden, no per-method tuning. Defensible (tuning without
   labels is itself unsolved, and the selection criterion was identical
   for every arm), but a GNN advocate will argue tuned GNNs with more
   capacity could close the gap. The intervals cannot exclude a +0.02 GNN
   advantage on most comparisons — only large advantages are excluded.
2. **sage_bgrl collapses on 7/9 datasets** — my hand-rolled BGRL is
   evidently sensitive; gcn_bgrl is the only stable BGRL cell. GAE cells
   are solid. If BGRL matters to a reviewer, a reference implementation
   should replace mine.
3. **The MLP+SSL feature ceiling collapsed** (~0.02 AMI everywhere), so
   the "fusion value" comparisons lean on raw KMeans instead. A proper
   feature-SSL baseline (e.g. SimCLR-style with a validated recipe) is
   missing.
4. **The ledger promised spectral clustering on embeddings as a
   robustness check; it was never run.** KMeans is the single clustering
   protocol in all 600 rows.
5. **Placement protocol mismatch**: full-graph vs test-split NMI for the
   DMGI-lineage anchors; the comparison is honest about direction, not
   exact about magnitude.
6. **Two of the nine datasets (amazon, amazon_ratings) carry no
   recoverable signal for any method**, reducing effective statistical
   power below the nominal count.
7. The trajectory-selection port is lightweight (16 sampled + one-hot
   candidates), not the lab's exact `select_trajectory`.

None of these, in my assessment, threatens the headline direction — every
comparison points the same way and the placement result is strong — but
they bound how far the claims can be pushed in a paper.

## 6. What I consider established

1. Given identical inputs on intact graphs, **no tested GNN beats MDT
   consensus**; the burden of proof has moved to the GNN side.
2. **Zero-training MDT with graph views is competitive with published
   multiplex GNN clustering** (DBLP: above HDMI; ACM: 0.06 under O2MAC).
3. **Three lab tooling fixes** independent of any GNN question:
   modularity-based selection (silhouette is degeneracy-blind), cosine
   kNN for BoW-like features, consensus-over-trajectories preferred to
   single-trajectory selection (+0.05 to +0.10 AMI on acm/dblp).
4. **Per-node view weighting has real leverage** (exploratory ACM ACC
   0.886) but needs a view-quality signal; feature-anchored gating is
   understood and closed.
5. The **only surviving graph-specific GNN gain** is +0.05 on a synthetic
   structure-only control, and a cheap spectral A² view does not replicate
   it.

## 7. Next plans, ranked by information per unit cost

1. **Decisions I need from you** (no compute): publication route — my
   recommendation is a new experiments section in the mdt_ddm paper
   (multiplex benchmarks, placement + grid) rather than a standalone
   negative-results paper; whether the GNN thread stays closed; whether
   the tooling fixes should be ported into the lab's MDT code now.
2. **α-normalized MDT views (~half a day, pre-registered).** The
   diffusion-maps α-family is implemented in `src/diffusionloss.py` but
   has never touched the MDT line: all MDT transitions are plain
   row-stochastic (α=0), and the single-view DDM experiments fix α per
   dataset without any sweep. The noise views that poison consensus (ACM
   PLP, DBLP PATAP/PTP) are precisely hub-dominated, and α=1 exists to
   remove degree/density bias before diffusion. One-line change to view
   construction; targets the measured failure mechanism directly; cheaper
   than any learned component. This is my top experimental candidate.
3. **Cross-view agreement gate (~1 week, new pre-registration).** Gate
   view v at node i by the agreement of its view-v neighbourhood with the
   *other* views' neighbourhoods (no anchoring on any single view). The
   licensed successor to E1; if its heuristic shows an effect, a small
   GNN gate becomes the first justified GNN component in the pipeline.
4. **Paper-hardening reruns (2–3 days)**: protocol-aligned placement
   (test-split NMI for the DMGI lineage), spectral-clustering robustness
   check over the existing embeddings, and a reference BGRL to replace
   mine — items 2, 4, 5 of the limitations list.
5. **A real validation dataset for the sharp-gate result**: the
   O2MAC-lineage DBLP-4057 / IMDB-4780 files would double as a
   cross-lineage check of the placement claims.

## 8. Process note

Everything above is reproducible from the repo: pre-registrations and all
amendments are in the ledger with their evidence; all 600 result rows are
in `results/graph_mdt/*.jsonl`; verdict labels come only from
`experiments/graph_mdt/verdicts.py`. The discipline cost real results —
the sharpened gate cannot claim its best number because the smoke run
touched the rescue dataset — and that is exactly what makes the rest of
the numbers defensible.
