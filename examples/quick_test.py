"""Lightweight repository smoke test for the XAI-UQ-CGI public repository."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "resource_3_synthetic_demo"

cfg = json.loads((R3 / "config.json").read_text(encoding="utf-8"))
assert int(cfg["seed"]) == 20260729

with (R3 / "insar_obs.csv").open(newline="", encoding="utf-8-sig") as f:
    insar_rows = list(csv.DictReader(f))
with (R3 / "gnss_obs.csv").open(newline="", encoding="utf-8-sig") as f:
    gnss_rows = list(csv.DictReader(f))

assert len(insar_rows) == 427, f"Expected 427 InSAR observations, found {len(insar_rows)}"
assert len(gnss_rows) == 9, f"Expected 9 GNSS stations, found {len(gnss_rows)}"

los = np.asarray(cfg["los_enu_normalized"], dtype=float)
assert np.isclose(np.linalg.norm(los), 1.0, atol=5e-4), f"LOS vector is not normalized: {los}"

# Minimal Mogi forward-model check at the first InSAR coordinate.
nu = float(cfg["nu"])
src = cfg["m_true"]
x = float(insar_rows[0]["x_m"])
y = float(insar_rows[0]["y_m"])
dx = x - float(src["x0"])
dy = y - float(src["y0"])
d = float(src["d"])
dv = float(src["dV"])
r = np.sqrt(dx*dx + dy*dy + d*d)
c = (1.0 - nu) / np.pi * dv / (r**3)
u = np.array([c*dx, c*dy, c*d])
assert np.all(np.isfinite(u))
assert np.isfinite(float(los @ u))

required_json = {
    "diagnostics.json": ["reference_residual", "mcmc", "mismatch"],
    "decision_record.json": ["robust_parameters", "dominant_data_driver", "limits_of_interpretation"],
    "ablation.json": ["ablation", "reduction_vs_InSAR_pct", "reduction_vs_GNSS_pct"],
}
for filename, keys in required_json.items():
    obj = json.loads((R3 / filename).read_text(encoding="utf-8"))
    for key in keys:
        assert key in obj, f"{filename} is missing key: {key}"

print("Seed:", cfg["seed"])
print("InSAR observations:", len(insar_rows))
print("GNSS stations:", len(gnss_rows))
print("Normalized LOS vector:", los)
print("QUICK TEST PASSED")
