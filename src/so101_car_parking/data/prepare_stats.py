import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(
        description="Compute Pi05 state/action quantiles using training frames only."
    )
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    root = args.data / "train"
    keys = ["observation.state", "action"]
    chunks = {key: [] for key in keys}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=keys)
        for key in keys:
            chunks[key].append(np.asarray(table[key].to_pylist(), dtype=np.float32))
    stats_path = root / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    counts = {}
    for key in keys:
        values = np.concatenate(chunks[key])
        if values.shape[1:] != (6,) or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid training values for {key}")
        low, high = np.quantile(values, [0.01, 0.99], axis=0)
        stats[key].update(q01=low.tolist(), q99=high.tolist())
        counts[key] = len(values)
    expected = json.loads((root / "meta/info.json").read_text())["total_frames"]
    if any(number != expected for number in counts.values()):
        raise RuntimeError("Training frame count differs from the metadata")
    temporary = stats_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(stats, indent=2) + "\n")
    temporary.replace(stats_path)
    result = dict(
        rows=counts,
        quantiles="Exact empirical q01/q99 over training frames only",
        stats_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(),
    )
    (args.data / "pi05-quantiles.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
