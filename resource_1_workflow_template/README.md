# Resource 1 - XAI-UQ-CGI Workflow Template

**Article:** Explainable and Uncertainty-Aware Computational Geodetic Inversion: A Review and Framework for Multi-Source Earth Observation  
**Journal:** Computers & Geosciences  
**Authors:** Mohit Sheode and D. Kishan

## Purpose

This resource is the method-agnostic computational scaffold corresponding to Section 8.2 and Algorithm 1. It mirrors the ten workflow steps and the five XAI-UQ-CGI layers. It is intentionally a scaffold: problem-specific import, forward-model, inversion, covariance, and diagnostic components are represented by documented placeholders.

The fully instantiated synthetic Mogi example is provided in [`../resource_3_synthetic_demo/`](../resource_3_synthetic_demo/).

## File

- `or1_workflow_template.py` - Python scaffold mapping the ten algorithmic steps to the five framework layers.

## Use

The file is intended to be copied and completed for a new inversion problem. It compiles as supplied but raises `NotImplementedError` at problem-specific steps until the user provides data loaders, a forward operator, an inversion engine, and diagnostics.

## Layer mapping

| Layer | Algorithm steps | Purpose |
|---|---:|---|
| L1 - Data harmonization | 1-3 | Ingest heterogeneous observation blocks, harmonize frames/units, and declare uncertainty. |
| L2 - Forward-model encoding | 4 | Assemble the forward operator, priors, bounds, covariance, and regularization. |
| L3 - Inversion engine | 5-6 | Select and run a documented estimator or sampler. |
| L4 - Uncertainty propagation | 7-8 | Quantify residual structure, non-uniqueness, sensitivity, and uncertainty. |
| L5 - Explainable decision layer | 9-10 | Produce an auditable interpretation record and archive configuration choices. |

## Licence

Source code is released under the repository MIT License. Documentation is released under CC BY 4.0.
