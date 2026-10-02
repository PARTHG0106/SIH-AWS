"""Spatial / neighbour consistency check on the synthetic-injection benchmark.

The SIH headline example is a station diverging from network consensus while
neighbours agree — that's exactly what we'd demonstrate to claim this capability.
We can't use real Indian AWS, but on the held-out real SURFRAD network
(bon / fpk / gwn, ~150 km apart in the same region), this is the only available
honest spatial check. Comparison: a station whose value diverges from the
network-mean while both peers agree → large spatial residual.
"""
from __future__ import annotations

import numpy as np

from awsad.minute_detection import CHANNELS

PEERS_BY_GROUP = {
    "bon|noaa_surfrad": ("fpk|noaa_surfrad", "gwn|noaa_surfrad"),
    "fpk|noaa_surfrad": ("bon|noaa_surfrad", "gwn|noaa_surfrad"),
    "gwn|noaa_surfrad": ("bon|noaa_surfrad", "fpk|noaa_surfrad"),
}


def network_residual(value, peer_means):
    """Signed residual: station - median(peers). None if no peers or invalid input."""
    peers = [p for p in peer_means if p is not None and np.isfinite(p)]
    if value is None or not np.isfinite(value) or not peers:
        return None
    return float(value - np.median(peers))


def network_consistency(value, peer_means, TOLERANCES):
    peers = [p for p in peer_means if p is not None and np.isfinite(p)]
    if value is None or not np.isfinite(value) or len(peers) < 1:
        return {"consensus": "unknown", "residual": None, "trigger": None}
    median_peer = float(np.median(peers))
    res = value - median_peer
    tol = TOLERANCES.get("default", 5.0)
    triggered = bool(abs(res) > tol)
    return {"consensus": "diverging" if triggered else "agrees",
            "residual": round(float(res), 3),
            "trigger": (f"station vs network median diverges by {res:.2f}" if triggered else
                        "agrees with peer consensus"),
            "tolerance": tol, "peer_count": len(peers)}