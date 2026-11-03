import anndata as ad
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from phenospace.graph_models.pipeline import (
    AANetStyleGraphBottleneckGNN,
    AnnDataPhenotypeGraphDataset,
    fit_preprocessing,
    grouped_split,
    masked_mse,
)


def fixture():
    a = ad.AnnData(
        np.array(
            [[1.0, 2.0, np.nan], [2.0, 3.0, 1.0], [4.0, 5.0, 2.0], [8.0, 1.0, 4.0]],
            dtype=np.float32,
        )
    )
    a.obs_names = ["a", "b", "c", "d"]
    a.var_names = ["x", "y", "z"]
    hidden = np.zeros(a.shape, dtype=bool)
    hidden[:, 0] = True
    fitted = fit_preprocessing(a, ["a", "b", "c"])
    return a, hidden, fitted


def test_train_only_and_input_contract():
    a, hidden, fitted = fixture()
    a.X[3] = 999
    assert fit_preprocessing(a, ["a", "b", "c"]) == fitted
    data = AnnDataPhenotypeGraphDataset(a, fitted, hidden)
    assert data[0].x[0, 0] == 0
    assert data[0].x[2, 1] == 0
    bad = hidden.copy()
    bad[0, 2] = True
    with pytest.raises(ValueError, match="unobserved"):
        AnnDataPhenotypeGraphDataset(a, fitted, bad)
    with pytest.raises(ValueError, match="axis order"):
        AnnDataPhenotypeGraphDataset(a[:, ::-1], fitted, hidden)
    fitted["edge_index"] = [[99], [0]]
    fitted["edge_weight"] = [1.0]
    with pytest.raises(ValueError, match="graph"):
        AnnDataPhenotypeGraphDataset(a, fitted, hidden)


def test_gradient_mask_weights_and_batch_independence():
    torch.manual_seed(1)
    a, hidden, fitted = fixture()
    data = AnnDataPhenotypeGraphDataset(a, fitted, hidden)
    train = Batch.from_data_list([data[i] for i in range(3)])
    model = AANetStyleGraphBottleneckGNN()
    model.set_references(train)
    loss = masked_mse(model(train), train.y, train.target_mask)
    loss.backward()
    assert torch.isfinite(loss) and model.conv.lin.weight.grad.abs().sum() > 0
    assert model.queries.grad.abs().sum() > 0
    model.eval()
    one = model(Batch.from_data_list([data[3]]))
    together = model(Batch.from_data_list([data[3], data[0]]))[: a.n_vars]
    reversed_batch = model(Batch.from_data_list([data[0], data[3]]))[a.n_vars :]
    torch.testing.assert_close(one, together, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(one, reversed_batch, atol=1e-6, rtol=1e-6)
    altered = Batch.from_data_list([data[3]])
    if altered.edge_weight.numel():
        altered.edge_weight *= 0
        assert not torch.allclose(one, model(altered))
    with pytest.raises(ValueError, match="no observed"):
        masked_mse(one, one, torch.zeros_like(one, dtype=torch.bool))


def test_reject_bad_ids_and_allow_disconnected_graph():
    a, hidden, fitted = fixture()
    with pytest.raises(ValueError, match="training IDs"):
        fit_preprocessing(a, ["a", "a"])
    with pytest.raises(ValueError, match="training IDs"):
        fit_preprocessing(a, [])
    duplicate = a.copy()
    duplicate.obs_names = ["a", "a", "c", "d"]
    with pytest.raises(ValueError, match="unique"):
        fit_preprocessing(duplicate, ["a"])
    fitted["edge_index"] = [[], []]
    fitted["edge_weight"] = []
    data = AnnDataPhenotypeGraphDataset(a, fitted, hidden)
    graph = Batch.from_data_list([data[0]])
    model = AANetStyleGraphBottleneckGNN()
    model.set_references(graph)
    assert torch.isfinite(model(graph)).all()


def test_grouped_split_keeps_participants_together():
    ids = ["a1", "a2", "b1", "b2", "c1", "c2"]
    groups = ["a", "a", "b", "b", "c", "c"]
    train, test = grouped_split(ids, groups, seed=7)
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(ids)
    assert {groups[ids.index(i)] for i in train}.isdisjoint({groups[ids.index(i)] for i in test})
    assert (train, test) == grouped_split(ids, groups, seed=7)
    with pytest.raises(ValueError, match="two groups"):
        grouped_split(ids, ["a"] * len(ids))
