#!/bin/bash
# Deep out-of-sample extension of a Multi-view Diffusion Trajectory (Helix-B, 2 views).
# Requires the MDT package on the path; set `mdt_repo` in config.yml.
python -m experiments.helix_mv.experiment -c experiments/helix_mv/config.yml
