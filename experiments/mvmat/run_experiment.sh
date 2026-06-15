#!/bin/bash
# Deep OOS extension of an MDT on a real multi-view .mat dataset.
# Datasets: github.com/ChuanbinZhang/Multi-view-datasets (set data.path / data.name in config.yml).
python -m experiments.mvmat.experiment -c experiments/mvmat/config.yml
