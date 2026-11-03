"""Bounded, synthetic, CPU-only inductive masked-reconstruction demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from torch_geometric.data import Batch

from . import pipeline
from .pipeline import (
    AANetStyleGraphBottleneckGNN,
    AnnDataPhenotypeGraphDataset,
    fit_preprocessing,
    masked_mse,
)


def run(output, seed=42, epochs=25):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(48, 2))
    values = latent @ np.array([[1, 0.6, -0.3, 0.7, 0.2, -0.8], [0.2, -0.6, 0.9, 0.4, -0.7, 0.3]])
    values += rng.normal(scale=0.15, size=values.shape)
    values[rng.random(values.shape) < 0.08] = np.nan
    adata = ad.AnnData(values.astype(np.float32))
    adata.obs_names = [f"synthetic-{i:03d}" for i in range(len(values))]
    adata.var_names = [f"axis-{i}" for i in range(values.shape[1])]
    order = rng.permutation(len(values))
    train, test = order[:36], order[36:]
    train_ids, test_ids = adata.obs_names[train].tolist(), adata.obs_names[test].tolist()
    assert not set(train_ids) & set(test_ids)
    fitted = fit_preprocessing(adata, train_ids)
    hidden = np.zeros_like(values, dtype=bool)
    for i in range(len(values)):
        observed = np.flatnonzero(np.isfinite(values[i]))
        hidden[i, rng.choice(observed, max(1, len(observed) // 3), replace=False)] = True
    data = AnnDataPhenotypeGraphDataset(adata, fitted, hidden)
    training = Batch.from_data_list([data[i] for i in train])
    testing = Batch.from_data_list([data[i] for i in test])
    results = {
        "train_mean_baseline_mse": float(
            masked_mse(torch.zeros_like(testing.y), testing.y, testing.target_mask)
        )
    }
    checkpoint = None
    predictions = None
    for name, graph, bottleneck in [
        ("full", True, True),
        ("no_graph", False, True),
        ("no_bottleneck", True, False),
    ]:
        torch.manual_seed(seed)
        model = AANetStyleGraphBottleneckGNN(use_graph=graph, use_bottleneck=bottleneck)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(epochs):
            model.train()
            model.set_references(training)
            optimizer.zero_grad()
            loss = masked_mse(model(training), training.y, training.target_mask)
            loss.backward()
            optimizer.step()
        model.eval()
        model.set_references(training)
        with torch.no_grad():
            pred = model(testing)
            results[name] = {
                "train_mse": float(loss),
                "test_mse": float(masked_mse(pred, testing.y, testing.target_mask)),
            }
        if name == "full":
            checkpoint, predictions = model.state_dict(), pred
    torch.save(checkpoint, output / "model.pt")
    restored = AANetStyleGraphBottleneckGNN()
    restored.references = torch.empty_like(checkpoint["references"])
    restored.load_state_dict(torch.load(output / "model.pt", map_location="cpu", weights_only=True))
    restored.eval()
    with torch.no_grad():
        difference = float((restored(testing) - predictions).abs().max())
    if (
        difference > 1e-6
        or not np.isfinite(
            list(
                [results["train_mean_baseline_mse"]]
                + [v["test_mse"] for v in results.values() if isinstance(v, dict)]
            )
        ).all()
    ):
        raise RuntimeError("nonfinite result or checkpoint mismatch")
    np.savez(output / "synthetic_inputs.npz", values=values, hidden=hidden, train=train, test=test)
    np.save(output / "predictions.npy", predictions.detach().numpy())
    input_hash = hashlib.sha256((output / "synthetic_inputs.npz").read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    manifest = {
        "schema": 1,
        "data": "synthetic",
        "task": "inductive masked-node reconstruction",
        "seed": seed,
        "epochs": epochs,
        "unit": "independent synthetic row",
        "split": {"train": train_ids, "test": test_ids},
        "reference_ids_in_order": train_ids,
        "module_sha256": {
            Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in [__file__, pipeline.__file__]
        },
        "fit": fitted,
        "input_sha256": input_hash,
        "code_commit_if_run_in_checkout": commit,
        "checkout_dirty": dirty,
        "python": platform.python_version(),
        "versions": {
            p: version(p) for p in ["phenospace", "torch", "torch-geometric", "anndata", "numpy"]
        },
        "config": {"hidden_dim": 16, "n_archetypes": 3, "learning_rate": 0.01},
        "results": results,
        "reload_max_abs_difference": difference,
        "warnings": ["One synthetic split and seed; no real-data or generalization claim."],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(output), "results": results, "reload_max_abs_difference": difference},
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new output directory; never overwritten")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    run(args.output, args.seed, args.epochs)


if __name__ == "__main__":
    main()
