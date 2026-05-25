import os
import dataclasses
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import pickle
import logging
from tqdm import tqdm

import torch

from flexdock.data.parse.base import read_strings_from_txt
from flexdock.data.parse.parser import ComplexParser
from flexdock.data.feature.featurizer import Featurizer


@dataclasses.dataclass
class TrainingPipelineConfig:
    dataset: str
    complex_file: str
    data_dir: str
    cache_path: str
    apo_protein_file: str
    holo_protein_file: str
    num_workers: int = 1
    esm_embeddings_path: str = None


class TrainingDataPipeline:
    def __init__(
        self,
        config: TrainingPipelineConfig,
        featurizer_cfg,
    ):
        self.config = config
        self.featurizer = Featurizer.from_config(featurizer_cfg)
        self.parser = ComplexParser(esm_embeddings_path=self.config.esm_embeddings_path)
        self.apo_protein_file = config.apo_protein_file
        self.holo_protein_file = config.holo_protein_file
        self.base_dir = config.data_dir

    def process_all_complexes(self):
        logging.info(
            f"Processing complexes from [{self.config.complex_file}]"
            f"and saving it to [{self.config.cache_path}]"
        )
        os.makedirs(self.config.cache_path, exist_ok=True)

        complex_names_all = read_strings_from_txt(self.config.complex_file)
        logging.info(f"Loading {len(complex_names_all)} complexes.")

        max_idx = -1
        for idx, complex_name in enumerate(complex_names_all):
            if os.path.exists(f"{self.config.cache_path}/heterograph-{complex_name}.pt") and \
               os.path.exists(f"{self.config.cache_path}/rdkit_ligand-{complex_name}.pkl"):
                max_idx = max(max_idx, idx)
        
        if max_idx >= 0:
            logging.info(f"Found cached files up to index {max_idx} in complex_names_all. Skipping to {max_idx + 1}.")
            complex_names_all = complex_names_all[max_idx + 1:]

        CHUNK_SIZE = 10

        processed_names = []

        list_indices = list(range(len(complex_names_all) // CHUNK_SIZE + 1))
        # random.shuffle(list_indices)
        for i in tqdm(list_indices, desc="Processing chunks"):
            complex_names = complex_names_all[CHUNK_SIZE * i : CHUNK_SIZE * (i + 1)]

            complex_inputs_shard = []
            for idx, complex_name in tqdm(enumerate(complex_names), total=len(complex_names), desc="Processing complexes"):
                if os.path.exists(f"{self.config.cache_path}/heterograph-{complex_name}.pt") and os.path.exists(f"{self.config.cache_path}/rdkit_ligand-{complex_name}.pkl"):
                    processed_names.append(complex_name)
                    continue
                complex_inputs = self.parser.parse_complex(self.prepare_input_files(complex_name))
                if complex_inputs is not None:
                    complex_inputs_shard.append(complex_inputs)

            logging.info(f"Num workers={self.config.num_workers}")
            
            results = []
            with ProcessPool(max_workers=self.config.num_workers) as pool:
                future_to_name = {
                    pool.schedule(self.featurizer.featurize_complex, args=(complex_inputs,), timeout=300): complex_inputs["name"]
                    for complex_inputs in complex_inputs_shard
                }
                
                for future in future_to_name:
                    name = future_to_name[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except TimeoutError:
                        logging.warning(f"Task for {name} timed out after 60 seconds. Skipping and killing underlying processes.")
                    except Exception as e:
                        logging.error(f"Task for {name} failed with {e}")

            for result in results:
                if result is None:
                    continue

                complex_graph = result["complex_graph"]
                ligand = result["ligand"]
                name = result["name"]

                torch.save(
                    complex_graph,
                    f"{self.config.cache_path}/heterograph-{name}.pt",
                )

                with open(
                    f"{self.config.cache_path}/rdkit_ligand-{name}.pkl", "wb"
                ) as f:
                    pickle.dump((ligand[0]), f)
                processed_names.append(name)

        with open(f"{self.config.cache_path}/complex_names.pkl", "wb") as f:
            pickle.dump(processed_names, f)

    def prepare_input_files(self, complex_name):
        if self.config.dataset == "pdbbind":
            complex_dict = {
                "dataset": self.config.dataset,
                "base_dir": self.base_dir,
                "name": complex_name,
                "ligand_description": "filename",
                "apo_rec_path": f"{self.base_dir}/{complex_name}/{complex_name}_{self.apo_protein_file}.pdb",
                "holo_rec_path": f"{self.base_dir}/{complex_name}/{complex_name}_{self.holo_protein_file}.pdb",
            }

        elif self.config.dataset == "plinder":
            complex_dict = {
                "dataset": self.config.dataset,
                "base_dir": self.base_dir,
                "name": complex_name,
                "ligand_description": "filename",
                "apo_rec_path": f"{self.base_dir}/{complex_name}/{self.apo_protein_file}.pdb",
                "holo_rec_path": f"{self.base_dir}/{complex_name}/{self.holo_protein_file}.pdb",
            }
        return complex_dict
