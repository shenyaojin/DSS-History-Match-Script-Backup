"""Record software versions and input/mesh hashes of every finished v11_pinv run (results/provenance.json)."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "output" / "v11_pinv"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git(path):
    try:
        return subprocess.run(["git", "-C", str(path), "log", "-1", "--format=%H %cd"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return None


out = {"fibeRIS_commit": git(REPO / "fibeRIS"), "runs": {}}
for d in sorted(p for p in RUNS.iterdir() if p.is_dir() and (p / f"v11_pinv_{p.name}_input.i").exists()):
    deck = (d / f"v11_pinv_{d.name}_input.i").read_text()
    mesh = re.search(r"\[Mesh\].*?\n\[\]", deck, re.S).group(0)        # mesh is generated in-deck
    log = (d / "simulation.log").read_text(errors="ignore") if (d / "simulation.log").exists() else ""
    ver = {k: (m.group(1).strip() if (m := re.search(rf"{k}:\s*(.*)", log)) else None)
           for k in ["MOOSE Version", "PETSc Version", "Executable Timestamp", "Num Processors"]}
    out["runs"][d.name] = {"input_sha256": sha(deck.encode()), "mesh_block_sha256": sha(mesh.encode()),
                           "n_steps": len(re.findall(r"^Time Step \d+", log, re.M)), **ver}
(RUNS / "results" / "provenance.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
