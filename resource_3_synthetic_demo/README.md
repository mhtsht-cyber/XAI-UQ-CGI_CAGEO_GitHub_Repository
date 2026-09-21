# Resource 3 - Synthetic InSAR-GNSS Reproducibility Package

**Article:** Explainable and Uncertainty-Aware Computational Geodetic Inversion: A Review and Framework for Multi-Source Earth Observation  
**Journal:** Computers & Geosciences  
**Authors:** Mohit Sheode and D. Kishan

## Scope

This resource implements the controlled synthetic same-model proof-of-concept described in Section 8.3. It is not a field validation or operational hazard product. Structural-model error is explored only through the included two-source mismatch experiment.

## Quick start

From this directory:

```bash
pip install -r requirements_lock.txt
python or3_demo.py
```

Fixed random seed: **20260729**.

For the independent transparent trace replay:

```bash
python or3_trace_replay.py
```

## Primary files

- `or3_demo.py` - primary SciPy + emcee + ArviZ implementation used for the manuscript reference values.
- `or3_trace_replay.py` - independent NumPy stretch-move replay used for trace visualisation and an additional convergence check.
- `config.json` - fixed geometry, seed, priors, bounds, noise model, solver settings, and MCMC settings.
- `insar_obs.csv` - exact 427 synthetic InSAR LOS observations.
- `gnss_obs.csv` - exact nine synthetic three-component GNSS stations.
- `diagnostics.json` - residual, MCMC, and mismatch statistics.
- `decision_record.json` - machine-readable Layer-5 explainability record.
- `ablation.json` - InSAR-only, GNSS-only, and joint sensor-removal results.
- `posterior_trace_samples_replay.npz` - compressed replay chains.
- `trace_replay_diagnostics.json` - replay R-hat, ESS, acceptance, and percentile summaries.
- `run_log.txt` / `trace_replay_log.txt` - console logs for the reference run and replay.

## Manuscript figure files

- `f6_data.png` - Fig. 6, Layer-1 synthetic inputs.
- `f7_fit.png` - Fig. 7, observed/modelled/residual LOS fields.
- `f8_posterior.png` - Fig. 8, joint posterior.
- `f9_tradeoff.png` - Fig. 9, depth-volume trade-off and fusion benefit.
- `f10_calibration.png` - Fig. 10, 300-realization least-squares calibration.
- `f11_mismatch.png` - Fig. 11, structural-mismatch test.

The original primary `run_log.txt` retains the legacy output names from an earlier manuscript numbering. `FIGURE_FILENAME_MAPPING.txt` documents the mapping; no numerical result was changed by the renumbering.

## Trace diagnostics

- `trace_source_parameters.png`
- `trace_ramp_nuisance_parameters.png`
- `trace_running_diagnostics.png`

See [`DIAGNOSTIC_SUMMARY.md`](DIAGNOSTIC_SUMMARY.md) for the principal values and embedded diagnostic figures.

## Licence

Source code is released under the repository MIT License. Synthetic observations, figures, JSON outputs, and documentation are released under CC BY 4.0.
