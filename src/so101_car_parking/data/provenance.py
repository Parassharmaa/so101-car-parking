import hashlib
from pathlib import Path


def simulation_source_hash():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "simulation" / name for name in ("sim.py", "parking.py", "teacher.py")]
    paths += [root / "data/collect.py", Path(__file__)]
    paths += sorted(p for p in (root / "simulation/assets").rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
