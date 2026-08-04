# MDT × GNN — decision points (2026-07-24)

Two pre-registered rounds, 560+ grid rows, 8 intact graph datasets, verdicts
emitted by code. Full record: `GRAPH_MDT_RESEARCH_LEDGER.pdf`,
`results/graph_mdt/`.

## Headline results

1. **Given identical inputs, no GNN beats MDT consensus.** GCN/SAGE ×
   GAE/BGRL, effects -0.07 to -0.30 AMI vs MDT-with-graph-views, wins ≤2/6
   primary datasets. GNN wins appear only where the pre-run screen showed
   weak graphs — exactly where MDT's diffusion assumption also hurts.
2. **Zero-training MDT reaches published GNN territory** once given the same
   graphs: DBLP-7907 NMI 0.59 (HDMI 0.582, BTGF 0.624), ACM 0.63 (O2MAC
   0.69). Caveat for a paper: protocols differ across lineages; numbers need
   aligned reruns, not table copying.
3. **Per-node view gating has leverage but the heuristic signal is wrong.**
   Feature-anchored gates lift ACM to ACC 0.886 / NMI 0.685 (exploratory,
   O2MAC-level) yet collapse DBLP. Validation on untouched amazon was
   uninformative (no method above 0.05 AMI there), so the ACM number stays
   exploratory; the gate's *direction* behaved exactly as the
   feature-anchoring model predicts across all 9 datasets.
4. **The GNN's only surviving unique gain:** +0.05 AMI on minesweeper
   (structure-only synthetic control); a spectral A² view does not close it.

## Decisions needed

1. **Publication route.** Recommend: new experiments section in the mdt_ddm
   paper (multiplex benchmarks strengthen the lab method) rather than a
   standalone negative-results paper. Needs protocol-aligned baseline reruns.
2. **GNN thread.** Closed under the frozen rules. Only licensed reopening: a
   *new* pre-registration with cross-view neighbourhood agreement as the
   gate signal (~1 week). Worth it or not?
3. **Lab tooling fixes independent of any paper:**
   - silhouette-based trajectory selection prefers collapsed embeddings —
     replace with modularity-on-graph criterion;
   - Gaussian-euclidean kNN collapses on BoW features (homophily 0.36 vs
     0.71 cosine) — affects existing MDT pipelines;
   - random-trajectory consensus beats single-trajectory selection.

## Methodology guarantees (why these numbers can be trusted)

Pre-registered gates and thresholds; label-free model selection everywhere;
every amendment timestamped and evidence-linked before affected rows
existed; exploratory results explicitly denied claim status; verdict labels
computed only by `verdicts.py`, never written by hand.
