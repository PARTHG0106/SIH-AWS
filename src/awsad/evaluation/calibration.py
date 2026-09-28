"""Chronological, purged partitions for fitting and calibrating a stacker."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import _segments


def temporal_partitions(timestamps, labels, *, purge_hours=168):
    """Return disjoint fit/tune/calibrate row indices (60/20/20 by time order).

    Gaps are elapsed time, not assumed sample counts. Entire labelled positive
    runs crossing a partition boundary are excluded, so a single known event
    cannot train the classifier and also calibrate its threshold. Unknown
    labels are excluded without removing their positions from the time axis.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    y = np.asarray(labels, dtype=float)
    if len(ts) != len(y) or ts.hasnans or not ts.is_monotonic_increasing:
        raise ValueError("aligned, chronological, nonmissing timestamps required")
    if purge_hours < 0:
        raise ValueError("purge_hours must be nonnegative")
    n = len(ts)
    masks = {key: np.zeros(n, dtype=bool) for key in ("fit", "tune", "calibrate")}
    if n < 5:
        return {key: np.flatnonzero(mask) for key, mask in masks.items()}
    first, second = int(n * .6), int(n * .8)
    gap = pd.Timedelta(hours=purge_hours)
    masks["fit"] = np.asarray(ts < ts[first] - gap)
    masks["tune"] = np.asarray((ts >= ts[first]) & (ts < ts[second] - gap))
    masks["calibrate"] = np.asarray(ts >= ts[second])
    for start, end in _segments(y):
        if sum(mask[start:end + 1].any() for mask in masks.values()) > 1:
            for mask in masks.values():
                mask[start:end + 1] = False
    known = np.isin(y, [0, 1])
    return {key: np.flatnonzero(mask & known) for key, mask in masks.items()}


def balanced_group_class_weights(labels, groups):
    """Equal total weight per group, then per represented class within a group."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    if labels.shape != groups.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("aligned known binary labels and groups required")
    weights = np.zeros(len(labels), dtype=float)
    for group in np.unique(groups):
        rows = groups == group
        classes = np.unique(labels[rows])
        for label in classes:
            selected = rows & (labels == label)
            weights[selected] = 1.0 / (len(classes) * selected.sum())
    return weights * len(weights) / weights.sum() if len(weights) else weights
