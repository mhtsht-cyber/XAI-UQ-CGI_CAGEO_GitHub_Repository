# Repository Resource 3 for Computers & Geosciences
# Article: Explainable and Uncertainty-Aware Computational Geodetic Inversion: A Review and Framework for Multi-Source Earth Observation
# Authors: Mohit Sheode and D. Kishan
# Corresponding author: Mohit Sheode, Department of Civil Engineering, MANIT Bhopal, India; mhtsht@gmail.com; 223111001@stu.manit.ac.in

"""
Trace-plot replay for Repository Resource 3.

This auxiliary script independently replays the seven-parameter posterior using
an explicit Goodman-Weare stretch-move ensemble sampler implemented in NumPy.
It reads the fixed observations and configuration supplied with Repository Resource 3 and writes trace plots, a compressed chain archive, and replay diagnostics.
The primary reference implementation remains or3_demo.py (emcee + ArviZ).
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import arviz as az

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())
SEED = int(CFG["seed"])
NU = float(CFG["nu"])
LB = np.asarray(CFG["bounds_lower"], dtype=float)
UB = np.asarray(CFG["bounds_upper"], dtype=float)
LOS = np.asarray(CFG["los_enu_normalized"], dtype=float)
RT = np.array([CFG["ramp_true"][k] for k in ("c0", "cx", "cy")], dtype=float)
S_I = float(CFG["noise"]["insar"])
S_H = float(CFG["noise"]["gnss_h"])
S_V = float(CFG["noise"]["gnss_v"])
MT = np.array([CFG["m_true"][k] for k in ("x0", "y0", "d", "dV")], dtype=float)

ins = np.genfromtxt(ROOT / "insar_obs.csv", delimiter=",", names=True)
gns = np.genfromtxt(ROOT / "gnss_obs.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
XI, YI, LOS_OBS = ins["x_m"], ins["y_m"], ins["los_m"]
XG, YG = gns["x_m"].astype(float), gns["y_m"].astype(float)
GE_O, GN_O, GU_O = gns["uE_m"].astype(float), gns["uN_m"].astype(float), gns["uU_m"].astype(float)

PNAMES = [r"$x_0$ (m)", r"$y_0$ (m)", r"$d$ (m)", r"$\Delta V$ (m$^3$)", r"$c_0$ (m)", r"$c_x$ (m/m)", r"$c_y$ (m/m)"]


def mogi_batch(theta: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Vectorized Mogi displacement for theta shape (n, >=4)."""
    x0 = theta[:, 0, None]
    y0 = theta[:, 1, None]
    d = theta[:, 2, None]
    dV = theta[:, 3, None]
    dx = x[None, :] - x0
    dy = y[None, :] - y0
    R = np.sqrt(dx * dx + dy * dy + d * d)
    c = (1.0 - NU) / np.pi * dV / (R**3)
    return c * dx, c * dy, c * d


def log_prob_batch(theta: np.ndarray) -> np.ndarray:
    theta = np.atleast_2d(theta).astype(float)
    ok = np.all((theta >= LB) & (theta <= UB), axis=1)
    out = np.full(theta.shape[0], -np.inf, dtype=float)
    if not np.any(ok):
        return out
    t = theta[ok]
    ue, un, uu = mogi_batch(t, XI, YI)
    pred_los = ue * LOS[0] + un * LOS[1] + uu * LOS[2]
    pred_los += t[:, 4, None] + t[:, 5, None] * XI[None, :] + t[:, 6, None] * YI[None, :]
    ss = np.sum(((pred_los - LOS_OBS[None, :]) / S_I) ** 2, axis=1)
    ge, gn, gu = mogi_batch(t, XG, YG)
    ss += np.sum(((ge - GE_O[None, :]) / S_H) ** 2, axis=1)
    ss += np.sum(((gn - GN_O[None, :]) / S_H) ** 2, axis=1)
    ss += np.sum(((gu - GU_O[None, :]) / S_V) ** 2, axis=1)
    out[ok] = -0.5 * ss
    return out


def residual_one(theta: np.ndarray) -> np.ndarray:
    ue, un, uu = mogi_batch(theta[None, :], XI, YI)
    pred_los = ue[0] * LOS[0] + un[0] * LOS[1] + uu[0] * LOS[2]
    pred_los += theta[4] + theta[5] * XI + theta[6] * YI
    ge, gn, gu = mogi_batch(theta[None, :], XG, YG)
    return np.concatenate([
        (pred_los - LOS_OBS) / S_I,
        (ge[0] - GE_O) / S_H,
        (gn[0] - GN_O) / S_H,
        (gu[0] - GU_O) / S_V,
    ])


def fit_ls() -> np.ndarray:
    sol = least_squares(
        residual_one,
        np.asarray(CFG["ls_initial"], dtype=float),
        bounds=(LB, UB),
        method="trf",
        x_scale="jac",
        ftol=1e-12,
        xtol=1e-12,
    )
    return sol.x


def run_ensemble(center: np.ndarray, ensemble_index: int, nwalkers=40, nsteps=9000, burn=2250, thin=5, a=2.0):
    ndim = center.size
    scale = np.asarray(CFG["mcmc_init_scales"], dtype=float)
    rng = np.random.default_rng(SEED + 50000 + ensemble_index)
    pos = np.clip(center + rng.normal(size=(nwalkers, ndim)) * scale, LB + 1e-9, UB - 1e-9)
    lp = log_prob_batch(pos)
    kept = []
    accepted = np.zeros(nwalkers, dtype=int)
    attempted = np.zeros(nwalkers, dtype=int)
    groups = [np.arange(0, nwalkers, 2), np.arange(1, nwalkers, 2)]
    for step in range(nsteps):
        for active, comp in ((groups[0], groups[1]), (groups[1], groups[0])):
            partners = rng.choice(comp, size=active.size, replace=True)
            u = rng.random(active.size)
            z = ((a - 1.0) * u + 1.0) ** 2 / a
            proposal = pos[partners] + z[:, None] * (pos[active] - pos[partners])
            lp_prop = log_prob_batch(proposal)
            log_alpha = (ndim - 1.0) * np.log(z) + lp_prop - lp[active]
            accept = np.log(rng.random(active.size)) < log_alpha
            attempted[active] += 1
            accepted[active] += accept.astype(int)
            if np.any(accept):
                idx = active[accept]
                pos[idx] = proposal[accept]
                lp[idx] = lp_prop[accept]
        if step >= burn and ((step - burn) % thin == 0):
            kept.append(pos.copy())
        if (step + 1) % 1500 == 0:
            print(f"ensemble {ensemble_index+1}: step {step+1}/{nsteps}", flush=True)
    chain = np.stack(kept, axis=0).transpose(1, 0, 2)  # walkers, draws, params
    return chain, accepted.sum() / attempted.sum()


def trace_figure(A: np.ndarray, idxs: list[int], filename: str, title: str, truth: np.ndarray | None = None):
    nrows = len(idxs)
    fig, axes = plt.subplots(nrows, 1, figsize=(11.5, 2.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes)
    draws = np.arange(A.shape[1]) * int(CFG["mcmc"]["thin"])
    # 16 representative chains: 4 walkers from each of four ensembles
    reps = []
    walkers_per_ensemble = int(CFG["mcmc"]["nwalkers"])
    for e in range(int(CFG["mcmc"]["n_ensembles"])):
        base = e * walkers_per_ensemble
        reps.extend([base + 0, base + 10, base + 20, base + 30])
    for ax, j in zip(axes, idxs):
        for c in reps:
            ax.plot(draws, A[c, :, j], lw=0.55, alpha=0.65)
        med = np.median(A[:, :, j])
        ax.axhline(med, color="black", lw=1.1, label="posterior median")
        if truth is not None and j < truth.size:
            ax.axhline(truth[j], color="black", ls="--", lw=1.0, label="truth")
        ax.set_ylabel(PNAMES[j])
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Post-burn-in MCMC step (thinned display)")
    fig.suptitle(title, y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(ROOT / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def running_diagnostics(A: np.ndarray):
    checkpoints = np.array([150, 300, 600, 900, A.shape[1]], dtype=int)
    checkpoints = np.unique(checkpoints[checkpoints <= A.shape[1]])
    rhat_vals, ess_vals = [], []
    for n in checkpoints:
        r = [float(az.rhat(A[:, :n, j], method="rank")) for j in range(4)]
        e = [float(az.ess(A[:, :n, j], method="bulk")) for j in range(4)]
        rhat_vals.append(r)
        ess_vals.append(e)
    rhat_vals = np.asarray(rhat_vals)
    ess_vals = np.asarray(ess_vals)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for j in range(4):
        axs[0].plot(checkpoints, rhat_vals[:, j], marker="o", label=PNAMES[j])
        axs[1].plot(checkpoints, ess_vals[:, j], marker="o", label=PNAMES[j])
    axs[0].axhline(1.01, color="black", ls="--", lw=1, label="1.01 guide")
    axs[0].set_ylabel("Rank-normalized split R-hat")
    axs[0].set_xlabel("Retained draws per walker-chain")
    axs[0].set_title("Running convergence")
    axs[1].set_ylabel("Bulk effective sample size")
    axs[1].set_xlabel("Retained draws per walker-chain")
    axs[1].set_title("Running information content")
    for ax in axs:
        ax.grid(alpha=0.25)
    axs[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(ROOT / "trace_running_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return checkpoints, rhat_vals, ess_vals


def main():
    ls = fit_ls()
    scale = np.asarray(CFG["mcmc_init_scales"], dtype=float)
    disp_rng = np.random.default_rng(SEED + 9999 + 10)
    centers = []
    for _ in range(int(CFG["mcmc"]["n_ensembles"])):
        c = np.clip(ls + disp_rng.normal(size=7) * float(CFG["mcmc"]["dispersion"]) * scale, LB + 1e-6, UB - 1e-6)
        centers.append(c)
    chains, accs = [], []
    for e, c in enumerate(centers):
        ch, acc = run_ensemble(c, e)
        chains.append(ch)
        accs.append(acc)
    A = np.concatenate(chains, axis=0)  # 160 chains x 1350 draws x 7
    np.savez_compressed(ROOT / "posterior_trace_samples_replay.npz", chains=A, parameter_names=np.array(CFG["param_names"]), centers=np.asarray(centers))
    trace_figure(A, [0, 1, 2, 3], "trace_source_parameters.png", "Source-parameter trace plots: independent stretch-move replay", truth=MT)
    trace_figure(A, [4, 5, 6], "trace_ramp_nuisance_parameters.png", "Ramp-nuisance trace plots: independent stretch-move replay")
    checkpoints, run_rhat, run_ess = running_diagnostics(A)
    rhat = [float(az.rhat(A[:, :, j], method="rank")) for j in range(7)]
    ess = [float(az.ess(A[:, :, j], method="bulk")) for j in range(7)]
    q = np.percentile(A.reshape(-1, 7), [16, 50, 84], axis=0)
    diag = {
        "purpose": "independent trace-plot replay using a transparent NumPy Goodman-Weare stretch-move sampler",
        "primary_reference_implementation": "or3_demo.py using emcee and ArviZ",
        "seed": SEED,
        "shape": {"walker_chains": int(A.shape[0]), "retained_draws_per_chain": int(A.shape[1]), "parameters": int(A.shape[2])},
        "mean_acceptance_fraction": round(float(np.mean(accs)), 4),
        "rank_normalized_split_Rhat": {k: round(v, 4) for k, v in zip(CFG["param_names"], rhat)},
        "bulk_ESS": {k: round(v) for k, v in zip(CFG["param_names"], ess)},
        "posterior_percentiles_16_50_84": {k: [round(float(qi), 6) for qi in q[:, i]] for i, k in enumerate(CFG["param_names"])},
        "running_checkpoints_draws_per_chain": checkpoints.tolist(),
        "running_Rhat_source": run_rhat.round(4).tolist(),
        "running_bulk_ESS_source": run_ess.round(1).tolist(),
    }
    (ROOT / "trace_replay_diagnostics.json").write_text(json.dumps(diag, indent=2))
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
