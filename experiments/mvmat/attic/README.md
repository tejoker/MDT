# attic — exploratory scripts, not part of the deliverable

One-off probes written while the MDT results in the repo README were being
worked out. They are kept for provenance: the prose claims in the README's
"Multi-view extension (MDT)" section came from runs of these, and the raw
outputs are under `results/mvmat_*`.

They are **not maintained and not reproducible as-is**:

- several hardcode an absolute `sys.path.insert('/home/nicolasbigeard/...')`
  and will not import on another machine,
- they import each other by bare module name, so they only run with this
  directory on `sys.path`,
- most print to stdout rather than writing a result file.

The supported entry points are the files one level up — `experiment.py`,
`path_selection.py`, `oos_compare.py`, `beam_path.py` — documented in the
repo README under "Usage".
