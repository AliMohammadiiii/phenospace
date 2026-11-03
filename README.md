# PhenotypeSpace

Research software developed by Ali Mohammadi for Cytognosis to explore phenotype graphs and archetype-based models for psychological assessment data. The maintained example is a **synthetic engineering demonstration**, not a reproduction or validation of historical real-data findings. It uses PyTorch and PyTorch Geometric; AnnData provides the matrix container. Historical work and its collaborators remain recorded in the project artifacts and sources; no additional individual attribution is inferred here.

## CPU quickstart

Python 3.10–3.12. For a fresh virtual environment, from a checkout:

```sh
python -m pip install '.[ml,dev]'
phenospace-demo --output /tmp/phenospace-demo-001
pytest
```

The output directory must not already exist. The example generates 48 synthetic independent rows, splits them before fitting, masks observed values, trains for 25 epochs, saves a model and checks checkpoint reload. It reports test masked-node MSE for the full model, training-mean baseline and two focused ablations; the results are illustrative, not evidence of superiority. `manifest.json` records IDs, fit scope, versions, seed, input digest, configuration and metrics. `synthetic_inputs.npz`, `predictions.npy` and `model.pt` support inspection. The default CPU path requires no credentials or real data. PyTorch installation is platform-specific; accelerator configurations are outside the tested contract.

## Code tour and protocol

- [`pipeline.py`](src/phenospace/graph_models/pipeline.py): AnnData adapter, train-only scaling and correlation graph, masked loss, weighted GCN and archetype bottleneck.
- [`demo.py`](src/phenospace/graph_models/demo.py): synthetic generation, row split, training, baseline, ablations, checkpoint and manifest.
- [`test_graph_pipeline.py`](tests/test_graph_pipeline.py): input, masking, gradient, edge-weight and independent-inference contracts.
- [`docs/MAINTAINED_PIPELINE.md`](docs/MAINTAINED_PIPELINE.md): method differences, historical provenance and limitations.

The sample is an **inductive masked reconstruction** experiment on independent synthetic rows. A real dataset may require participant-grouped or temporal splitting, and its preprocessing must be fit only on training records. During training, fixed reference embeddings are recomputed from training graphs. Evaluation and deployment use these saved train-derived references; a prediction is independent of companion graphs. This differs scientifically from the historical batch-hull model, whose predictions can change with batch membership. Correlations are nonnegative thresholded training correlations and isolated nodes remain valid via GCN self-loops. Missing values are not targets. The only supported convolution here is weighted GCN.

## Provenance and scope

This repository is a standalone public source snapshot with a new Git history. It includes the maintained synthetic demonstration, tests and selected historical Python scripts. It excludes participant-level records, research papers, ontology data and bulk historical experiment outputs; an aggregate historical scorecard is included for a repository test. The [data source registry](data/manifests/data_sources.yaml) and [inventory manifests](reports/repository_audit/) are historical metadata: listed paths describe the original research repository and do not imply those files are included here.

The original research baseline was `cc2957eff36d17c38a659ffb50d56769518eeb87`, and the maintained implementation was developed in the original repository at `c8a6d2a7e000083a4d00be464761f62c8f39aa2d` ([historical CI run](https://github.com/AliMohammadiiii/phenospace-research-archive/actions/runs/41592040628)). These identifiers describe the original history, not commits in this standalone snapshot. That full history is being retained privately as `AliMohammadiiii/phenospace-research-archive`; access is not implied. See [historical methods](docs/MAINTAINED_PIPELINE.md) and [license scope and credits](NOTICE.md). The example does not reproduce or validate historical real-data findings.
