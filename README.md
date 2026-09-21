# XAI-UQ-CGI: Explainable and Uncertainty-Aware Computational Geodetic Inversion

Public code, derived data, and reproducibility repository accompanying the manuscript:

**Explainable and Uncertainty-Aware Computational Geodetic Inversion: A Review and Framework for Multi-Source Earth Observation**  
Target journal: **Computers & Geosciences**  
Authors: **Mohit Sheode and D. Kishan**

## Purpose

This repository provides the computational materials required to reproduce the controlled synthetic InSAR-GNSS demonstration and to inspect the workflow and bibliographic audit used in the manuscript. No proprietary software is required for the computational demonstration; the executable components use Python and open-source scientific libraries.

## Repository contents

- [`resource_1_workflow_template/`](resource_1_workflow_template/) - method-agnostic XAI-UQ-CGI workflow scaffold corresponding to Algorithm 1.
- [`resource_2_bibliographic_audit/`](resource_2_bibliographic_audit/) - derived bibliographic audit supporting manuscript Tables 1 and 2.
- [`resource_3_synthetic_demo/`](resource_3_synthetic_demo/) - complete reproducibility package for the synthetic InSAR-GNSS experiment in Section 8.3, including code, fixed configuration, random seed, exact synthetic observations, diagnostics, trace plots, ablation results, and decision-record JSON.
- [`examples/quick_test.py`](examples/quick_test.py) - lightweight repository integrity and forward-model smoke test.

## Quick test

The journal requires at least one simple test/example. After cloning or downloading the repository, run:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python examples/quick_test.py
```

A successful run ends with:

```text
QUICK TEST PASSED
```

The quick test verifies the fixed seed, the 427 InSAR observations, the 9 three-component GNSS stations, the normalized LOS vector, key machine-readable outputs, and a finite Mogi forward-model evaluation. It does not run the full MCMC experiment.

## Full Section 8.3 reproduction

```bash
cd resource_3_synthetic_demo
pip install -r requirements_lock.txt
python or3_demo.py
```

The primary script regenerates the synthetic observations, least-squares and Bayesian inversion outputs, sensor-removal ablation, 300-realization least-squares calibration, structural-mismatch experiment, manuscript Figs. 6-11, configuration file, diagnostics, and decision-record JSON.

To replay the posterior with the transparent auxiliary stretch-move implementation and regenerate the trace diagnostics:

```bash
python or3_trace_replay.py
```

See [`resource_3_synthetic_demo/README.md`](resource_3_synthetic_demo/README.md) and [`resource_3_synthetic_demo/DIAGNOSTIC_SUMMARY.md`](resource_3_synthetic_demo/DIAGNOSTIC_SUMMARY.md) for details.

## Fixed configuration

The principal synthetic experiment uses fixed random seed **20260729**. Exact settings are stored in `resource_3_synthetic_demo/config.json`.

## Scope

The experiment is a controlled same-model synthetic proof-of-concept under known ground truth. It is not a field validation or an operational hazard product. Structural-model error is examined only through the included two-source mismatch test.

## Bibliographic-audit provenance

Repository Resource 2 contains derived reviewer-readable indexes reconstructed from the Scopus and Dimensions literature exports used for the narrative review. The original raw database exports and database-specific proprietary metadata fields are not redistributed; they are retained privately by the authors for editorial verification.

## Licences

- **Source code:** MIT License (`LICENSE`).
- **Synthetic data, derived bibliographic audit, figures, and documentation:** CC BY 4.0 (`DATA_AND_DOCUMENTATION_LICENSE.md`).

Third-party Python packages remain under their respective licences.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The repository release corresponding to the submitted manuscript is **v1.0.0**.

## Contact

Mohit Sheode  
Department of Civil Engineering, Maulana Azad National Institute of Technology (MANIT), Bhopal 462003, India  
Email: mhtsht@gmail.com; 223111001@stu.manit.ac.in
