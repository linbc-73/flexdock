"""Cache-based inference data module.

Loads preprocessed heterograph ``.pt`` files directly from a cache directory,
optionally applying the inference-time transform (pocket reduction, nearby-atom
selection). This is useful when the same cached graphs are used for training
and one wants to avoid re-featurizing from raw files at inference time.
"""

import os

import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from lightning.pytorch import LightningDataModule

from flexdock.data.modules import ComplexData


class CachePredictionDataset(Dataset):
    def __init__(
        self,
        cache_path: str,
        complex_names,
        transform=None,
        pocket_reduction: bool = False,
        pocket_buffer: float = 20.0,
        pocket_min_size: int = 1,
        only_nearby_residues_atomic: bool = False,
    ):
        super().__init__()
        self.cache_path = cache_path
        self.complex_names = complex_names
        self.transform = transform

        self.pocket_reduction = pocket_reduction
        self.pocket_buffer = pocket_buffer
        self.pocket_min_size = pocket_min_size
        self.only_nearby_residues_atomic = only_nearby_residues_atomic

    def get(self, idx):
        name = self.complex_names[idx]
        graph_path = os.path.join(self.cache_path, f"heterograph-{name}.pt")

        try:
            complex_graph = torch.load(graph_path, weights_only=False)
        except Exception as e:
            print(f"Failed to load cache for {name}: {e}")
            complex_graph = ComplexData()
            complex_graph["name"] = name
            complex_graph["success"] = False
            return complex_graph

        if hasattr(complex_graph, "mol"):
            delattr(complex_graph, "mol")
        if "mol" in complex_graph:
            del complex_graph["mol"]

        complex_graph.name = name

        # InferenceDataModule expects these attributes to exist.
        if not hasattr(complex_graph, "amber_subset_mask"):
            complex_graph.amber_subset_mask = ""

        return complex_graph

    def __getitem__(self, idx):
        data = self.get(self.indices()[idx])
        if hasattr(data, "success") and not data.success:
            return data
        if self.transform is None:
            data["success"] = True
            return data
        try:
            data = self.transform(data)
        except Exception as e:
            name = data["name"] if "name" in data else self.complex_names[idx]
            print(f"Failed to apply transform for {name}: {e}")
            data = ComplexData()
            data["name"] = name
            data["success"] = False
            return data
        data["success"] = True
        return data

    def len(self):
        return len(self.complex_names)


class CacheInferenceDataModule(LightningDataModule):
    def __init__(
        self,
        input_csv,
        cache_path,
        transform=None,
        limit_complexes: int = None,
        pocket_reduction: bool = False,
        pocket_buffer: float = 20.0,
        pocket_min_size: int = 1,
        only_nearby_residues_atomic: bool = False,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        super().__init__()
        self.input_csv = input_csv
        self.cache_path = cache_path
        self.transform = transform
        self.limit_complexes = limit_complexes

        self.pocket_reduction = pocket_reduction
        self.pocket_buffer = pocket_buffer
        self.pocket_min_size = pocket_min_size
        self.only_nearby_residues_atomic = only_nearby_residues_atomic
        self.batch_size = batch_size
        self.num_workers = num_workers

    def predict_dataloader(self):
        input_df = pd.read_csv(self.input_csv, index_col=None)
        complex_names = input_df["pdbid"].tolist()
        if self.limit_complexes is not None:
            complex_names = complex_names[: self.limit_complexes]

        dataset = CachePredictionDataset(
            cache_path=self.cache_path,
            complex_names=complex_names,
            transform=self.transform,
            pocket_buffer=self.pocket_buffer,
            pocket_reduction=self.pocket_reduction,
            pocket_min_size=self.pocket_min_size,
            only_nearby_residues_atomic=self.only_nearby_residues_atomic,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
