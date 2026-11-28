"""
Write requirements-lock.txt: the exact versions of the app's packages installed RIGHT NOW.

Why: the saved models in models/*.joblib are only guaranteed to load with the scikit-learn
version that trained them. Pinning the exact versions makes the Docker image (and anyone
reproducing the project) use the same ones.

Run this in the same virtual environment that ran notebook 03:
    python src/pin_versions.py
"""
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lines = ["# Exact versions used to train and test the models (made by src/pin_versions.py)"]
for raw in (root / "requirements-app.txt").read_text().splitlines():
    raw = raw.split("#")[0].strip()
    if not raw:
        continue
    name = re.split(r"[<>=!~ ]", raw)[0]
    try:
        lines.append(f"{name}=={version(name)}")
    except PackageNotFoundError:
        sys.exit(f"{name} is not installed in this environment")
(root / "requirements-lock.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
