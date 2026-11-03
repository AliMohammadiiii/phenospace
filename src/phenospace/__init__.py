"""phenospace — phenotype-space research codebase (canonical `src` package).

Developed for Cytognosis (cytognosis.org): models map real psychological
assessment responses (PHQ-9, GAD-7, ISI, PSS) onto a structured phenotype-
vector space, with archetype models, RT-dynamics analysis, and GNN/AANet
archetype networks.

Submodule map:
    common                — repo-root/path resolution, IO, config & manifests
    questionnaire_mapping — phenotype axes, HPO mapping, projection engine
    synthetic_population  — synthetic cohort generation / balancing
    archetypes            — archetype modeling (v0/v0.1), user assignment
    ontology_embedding    — transbox + metric-finetuned phenotype embeddings
    graph_models          — phenotype graph construction + GNN/AANet training
    disease_mapping       — HP/ontology curation for the disease space
    evaluation            — metric/scoring helpers used across pipelines

The design rule of this package is: **zero hardcoded absolute paths**. All path
resolution goes through :mod:`phenosphere.common.paths`.
"""

__version__ = "0.1.0"
