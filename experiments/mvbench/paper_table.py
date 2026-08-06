"""Emit the out-of-sample tables for the MDT paper from the mvbench metrics.

    python -m experiments.mvbench.paper_table            # writes paper/tables/, prints markdown

Design decisions that make the table *fair* rather than just favourable, since
either one could be attacked in review:

* **MDT is represented by all five variants of the paper's Tab. 2**, not by a
  convenient one. The baseline for every W-L column is MDT-DIRECT: it is both
  the paper's own headline variant and the best of the six out of sample
  (0.417 vs 0.380 for CVX-RAND), so it is the fair reference in either reading.
  It only became the best once Q_CH was scored the way the reference code
  scores it -- see CH_TARGET in bench.py; under the paper's written formula
  DIRECT falls to 0.361 and CVX-RAND leads.
* **An oracle row bounds MDT from above.** Per cell it takes the max over the
  six variants. It uses test labels, so it is not a method; it is there so a
  reader can separate "this operator family is too weak" (false) from "no
  label-free rule finds its best member" (true).
* **Competitors are not strawmen.** GCCA runs on the raw views with a
  variance-based rank cut, not on the n x n transition matrices the reference
  repo passes it (that scored ~0.03 AMI). ID and p-AD select their diffusion
  time with the reference implementation's own entropy normalisation. SpecRaGE
  gets a retry on the initialisations where it diverges.
* **The two rows that hurt MDT stay in.** The uniform-mean operator and raw
  concatenated features are the controls that decide whether any of this
  machinery pays for itself. A table without them is not worth printing.
* **PRR uses the paper's Eq. 21 baseline (MDT-RAND)** so the column can be read
  directly against Fig. 6, but as a ratio of means -- averaging per-cell ratios
  has a pole wherever a baseline cell lands near zero AMI.
* Std is over the 19 *dataset* means, matching the clustered interval: seeds
  within a dataset are not independent benchmark tasks.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

METRICS = Path("results/mvbench/all_metrics.jsonl")
OUT = Path("paper/tables")
BASELINE = "mdt_direct"
PRR_BASE = "mdt_rand"          # the paper's Eq. 21 reference
MDT_VARIANTS = ["mdt_cvx_rand", "mdt_direct", "mdt_cst", "mdt_selected",
                "mdt_bsc", "mdt_rand"]

# (key, display name, group, MDT special case per paper Sec. 3.5, native OOS)
ROWS = [
    ("mdt_cvx_rand",  r"MDT-\textsc{Cvx-Rand}", "mdt", True,  False),
    ("mdt_direct",    r"MDT-\textsc{Direct}",   "mdt", True,  False),
    ("mdt_cst",       r"MDT-\textsc{Cst}",      "mdt", True,  False),
    ("mdt_bsc",       r"MDT-\textsc{Bsc}",      "mdt", True,  False),
    ("mdt_rand",      r"MDT-\textsc{Rand}",     "mdt", True,  False),
    ("mdt_selected",  r"MDT-\textsc{Sil}~$^\ddagger$", "mdt", True, False),
    ("__oracle__",    r"\emph{MDT best-of-6}~$^\dagger$", "mdt", True, False),
    ("ad",            r"AD \cite{katz2019}",     "lit", True,  False),
    ("p_ad",          r"\textsc{p}-AD \cite{kuchroo2022}", "lit", True, False),
    ("id",            r"ID \cite{kuchroo2022}",  "lit", True,  False),
    ("mvd",           r"MVD \cite{lindenbaum2020}", "lit", False, False),
    ("cr_diff",       r"\textsc{Cr-Diff} \cite{wang2012}", "lit", False, False),
    ("com_diff",      r"\textsc{Com-Diff} \cite{shnitzer2018}", "lit", False, False),
    ("gcca",          r"GCCA",                   "ind", False, True),
    ("dgcca",         r"DGCCA \cite{benton2017}", "ind", False, True),
    ("specrage",      r"SpecRaGE \cite{yacobi2025}", "ind", False, True),
    ("uniform_fused", r"Uniform mean ($t$ steps)", "ctl", True, False),
    ("features",      r"Concat.\ features",      "ctl", False, True),
]
GROUPS = [("mdt", r"\emph{MDT variants (Tab.~2)}"),
          ("lit", r"\emph{Operator-based multi-view diffusion (Tab.~1)}"),
          ("ind", r"\emph{Non-diffusion, inductive by construction}"),
          ("ctl", r"\emph{Controls}")]
APPENDIX = ["mdt_cvx_rand", "mdt_direct", "ad", "id", "mvd", "cr_diff",
            "uniform_fused", "specrage", "gcca"]
APPENDIX_HEAD = [r"MDT-\textsc{Cvx}", r"MDT-\textsc{Dir}", "AD", "ID", "MVD",
                 r"\textsc{Cr-D}", "Unif.", "SpecRaGE", "GCCA"]


def load():
    rows = [json.loads(l) for l in METRICS.open() if l.strip()]
    by = {(r["dataset"], r["seed"], r["method"]): r for r in rows}
    datasets = sorted({r["dataset"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    return rows, by, datasets, seeds


def oracle(by, dataset, seed):
    return max(by[(dataset, seed, m)]["inductive_ami"] for m in MDT_VARIANTS)


def per_dataset(by, datasets, seeds, method, metric="inductive_ami"):
    """Mean over seeds for each dataset; None where the method does not apply."""
    out = {}
    for dataset in datasets:
        if method == "__oracle__":
            values = [oracle(by, dataset, s) for s in seeds]
        else:
            values = [by[(dataset, s, method)][metric] for s in seeds
                      if (dataset, s, method) in by]
        out[dataset] = float(np.mean(values)) if values else None
    return out


def sign_p(effects: np.ndarray) -> float | None:
    wins = int((effects > 0).sum())
    losses = int((effects < 0).sum())
    total = wins + losses
    if total == 0:
        return None
    extreme = min(wins, losses)
    return min(1.0, 2 * sum(comb(total, i) for i in range(extreme + 1)) / 2 ** total)


def build(by, datasets, seeds):
    base = per_dataset(by, datasets, seeds, BASELINE)
    prr_base = per_dataset(by, datasets, seeds, PRR_BASE)
    stats = {}
    for key, *_ in ROWS:
        ind = per_dataset(by, datasets, seeds, key)
        test = per_dataset(by, datasets, seeds, key, "test_ami")
        shared = [d for d in datasets if ind[d] is not None]
        values = np.asarray([ind[d] for d in shared])
        # Oriented as the BASELINE's record: effects[d] > 0 means MDT-Cvx-Rand
        # beat this row on dataset d.  Reported as "MDT W-L" so the direction is
        # in the column name -- a bare "W-L" is ambiguous and gets misread.
        effects = np.asarray([base[d] - ind[d] for d in shared])
        seconds = ([by[(d, s, key)]["train_seconds"] for d in shared for s in seeds
                    if (d, s, key) in by] if key != "__oracle__" else [])
        # PRR denominator restricted to the SAME datasets: Com-Diff only applies
        # to the four two-view sets, and dividing its mean by a 19-dataset
        # baseline mean would compare different populations.
        denominator = float(np.mean([prr_base[d] for d in shared
                                     if prr_base[d] is not None]))
        stats[key] = {
            "ind": values.mean(), "ind_std": values.std(ddof=1),
            "test": np.mean([test[d] for d in shared]),
            "prr": values.mean() / denominator if abs(denominator) > 1e-3 else float("nan"),
            "wins": int((effects > 0).sum()), "losses": int((effects < 0).sum()),
            "p": sign_p(effects), "n_datasets": len(shared),
            "seconds": float(np.mean(seconds)) if seconds else None,
        }
    # Holm over the family of comparisons against the baseline
    family = sorted(((k, v["p"]) for k, v in stats.items()
                     if v["p"] is not None and k != BASELINE), key=lambda x: x[1])
    running = 0.0
    for index, (key, p) in enumerate(family):
        running = max(running, min(1.0, (len(family) - index) * p))
        stats[key]["p_holm"] = running
    stats[BASELINE]["p_holm"] = None
    return stats


def main_table(stats) -> str:
    lines = [r"\begin{table}[H]\centering\footnotesize",
             r"\caption{\textbf{Out-of-sample multi-view clustering.} "
             r"Inductive AMI (mean$\pm$std over the 19 dataset means, 10 seeds each) "
             r"on held-out points that were never in the operator. $^{*}$ marks methods "
             r"the MDT operator space contains (Sec.~3.5): a win over those concerns "
             r"trajectory choice, not the framework. $^{\S}$ marks methods that are "
             r"inductive by construction and therefore do not use the shared Nystr\"om "
             r"rule. PRR is Eq.~\ref{eq:prr} against MDT-\textsc{Rand}, as a ratio of "
             r"means. \emph{MDT W--L} is the number of datasets on which "
             r"MDT-\textsc{Direct} -- the paper's headline variant, and the strongest "
             r"MDT variant out of sample -- beats "
             r"and loses to that row, by exact sign test; $p_{\text{H}}$ "
             r"is Holm-corrected over the 17-comparison family. "
             r"$^\ddagger$ silhouette-ranked pool, ours, not a variant of the paper. "
             r"$^\dagger$ per-cell max over the six MDT variants; it uses test "
             r"labels, so it is an upper bound rather than a method. "
             r"$^\P$ the oracle contains the baseline and so cannot lose to it; "
             r"its sign test is degenerate by construction. "
             r"MDT-\textsc{Cst} is included for completeness although Tab.~2 "
             r"scopes it to manifold learning rather than clustering. "
             r"SpecRaGE diverged to a \texttt{nan} loss on 3 of 190 fits "
             r"(BBCSport, high-dimensional sparse text) and was refit from a new "
             r"initialisation; no MDT variant required this.}",
             r"\label{tab:oos-multiview}",
             r"\begin{tabular}{l c c c c c c}", r"\toprule",
             r"Method & Ind.\ AMI & Test AMI & PRR & MDT W--L & $p_{\text{H}}$ & Time (s)\\"]
    best = max(v["ind"] for k, v in stats.items() if k != "__oracle__")
    for group, title in GROUPS:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{7}}{{l}}{{{title}}}\\")
        for key, name, grp, encompassed, native in ROWS:
            if grp != group:
                continue
            s = stats[key]
            mark = ("$^{*}$" if encompassed else "") + ("$^{\\S}$" if native else "")
            value = f"{s['ind']:.3f}$\\pm${s['ind_std']:.3f}"
            if key != "__oracle__" and abs(s["ind"] - best) < 1e-9:
                value = rf"\textbf{{{value}}}"
            wl = ("--" if key == BASELINE else
                  r"n/a$^{\P}$" if key == "__oracle__" else
                  f"{s['wins']}--{s['losses']}")
            # The oracle contains the baseline, so it can never lose to it: its
            # sign test is 19-0 by construction and a p-value there would read as
            # a finding.  Suppress it.
            ph = ("--" if s.get("p_holm") is None or key == "__oracle__" else
                  (rf"\textbf{{{s['p_holm']:.3f}}}" if s["p_holm"] < .05
                   else f"{s['p_holm']:.3f}"))
            time = "--" if s["seconds"] is None else f"{s['seconds']:.1f}"
            note = r"\,\footnotesize(2 views)" if key == "com_diff" else ""
            lines.append(rf"{name}{mark}{note} & {value} & {s['test']:.3f} & "
                         rf"{s['prr']:.2f} & {wl} & {ph} & {time}\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def appendix_table(by, datasets, seeds) -> str:
    columns = {m: per_dataset(by, datasets, seeds, m) for m in APPENDIX}
    lines = [r"\begin{table}[H]\centering\scriptsize",
             r"\caption{\textbf{Per-dataset inductive AMI}, mean over 10 seeds. "
             r"Best per row in bold. `--' = not applicable (\textsc{Com-Diff} is "
             r"two-view only). Only three datasets (Movies, NUS-WIDE, ProteinFold) are near "
             r"the AMI floor for every method; on 3Sources, BBCSport, Prokaryotic, "
             r"Reuters-1200 and WebKB the MDT variants are below $0.11$ while another "
             r"method clears $0.15$, so those are method failures rather than "
             r"uninformative data.}",
             r"\label{tab:oos-per-dataset}",
             r"\begin{tabular}{l" + " c" * len(APPENDIX) + "}", r"\toprule",
             "Dataset & " + " & ".join(APPENDIX_HEAD) + r"\\", r"\midrule"]
    for dataset in datasets:
        values = [columns[m][dataset] for m in APPENDIX]
        finite = [v for v in values if v is not None]
        top = max(finite) if finite else None
        cells = ["--" if v is None else
                 (rf"\textbf{{{v:.3f}}}" if top is not None and abs(v - top) < 1e-9
                  else f"{v:.3f}") for v in values]
        lines.append(f"{dataset.replace('_', '-')} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def markdown(stats) -> str:
    out = ["| Method | MDT case | native OOS | Ind. AMI | Test AMI | PRR | MDT W-L | p_Holm | s |",
           "|---|---|---|---|---|---|---|---|---|"]
    for group, title in GROUPS:
        out.append(f"| **{title.replace(chr(92)+'emph{','').replace('}','').replace(chr(92)+'textasciitilde',' ')}** | | | | | | | | |")
        for key, name, grp, enc, native in ROWS:
            if grp != group:
                continue
            s = stats[key]
            clean = (name.replace(r"\textsc{", "").replace(r"\emph{", "")
                     .replace("}", "").replace("~$^\\ddagger$", " (ours)")
                     .replace("~$^\\dagger$", " (oracle)"))
            clean = clean.split(r"\cite")[0].replace("\\ ", " ").strip()
            wl = "-" if key == BASELINE else f"{s['wins']}-{s['losses']}"
            ph = "-" if s.get("p_holm") is None else f"{s['p_holm']:.3f}"
            sec = "-" if s["seconds"] is None else f"{s['seconds']:.1f}"
            out.append(f"| {clean} | {'yes' if enc else 'no'} | {'yes' if native else 'no'} | "
                       f"{s['ind']:.3f}±{s['ind_std']:.3f} | {s['test']:.3f} | {s['prr']:.2f} | "
                       f"{wl} | {ph} | {sec} |")
    return "\n".join(out)


def main() -> None:
    rows, by, datasets, seeds = load()
    stats = build(by, datasets, seeds)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mvbench_main.tex").write_text(main_table(stats) + "\n", encoding="utf-8")
    (OUT / "mvbench_per_dataset.tex").write_text(
        appendix_table(by, datasets, seeds) + "\n", encoding="utf-8")
    print(markdown(stats))
    print(f"\nwrote {OUT/'mvbench_main.tex'} and {OUT/'mvbench_per_dataset.tex'}")
    print(f"{len(datasets)} datasets x {len(seeds)} seeds, {len(rows)} rows")


if __name__ == "__main__":
    main()
