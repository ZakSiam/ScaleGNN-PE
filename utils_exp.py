"""Small JSON helpers shared by experiment runners."""

from __future__ import annotations

import hashlib
import json

import numpy as np


class NumpyArrayEncoder(json.JSONEncoder):
    """JSON encoder that understands common NumPy scalar/array objects."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def stable_hash_dict(payload: dict) -> str:
    """Return a deterministic short hash suitable for result filenames."""

    serialized = json.dumps(payload, sort_keys=True, cls=NumpyArrayEncoder)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]
