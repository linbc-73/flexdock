import os
import pandas as pd
import torch
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from flexdock.data.parse.protein import parse_pdb_from_path as parse_pdb_pmd
from flexdock.data.feature.protein import get_nearby_residue_mask
from rdkit import Chem
import argparse
import tqdm

BASE_DIR = "/data/protein/BC_Data/Docking_Data/apo2mol_dataset/data_folder"
CSV_PATH = "/data/protein/BC_Data/induce-fit/figrdock_exp/train_data_filter/apo2mol_final_train_set.csv"

def get_pocket_residue_str(
    holo_pos,
    ligand_pos,
    atom_rec_index,
    pocket_cutoff: float = 5.0,
    pocket_min_size: int = 1,
):
    nearby_residue_mask = get_nearby_residue_mask(
        atom_pos=holo_pos,
        lig_pos=ligand_pos,
        atom_rec_index=atom_rec_index,
        cutoff=pocket_cutoff,
        min_residues=pocket_min_size,
    )

    nearby_residue_idxs = torch.argwhere(nearby_residue_mask).squeeze()
    pocket_residue_str = ",".join(str(idx.item() + 1) for idx in nearby_residue_idxs)
    return pocket_residue_str

def process_complex(complex_id):
    try:
        lig_name = complex_id.split('__')[-1]
        ligand_path = os.path.join(BASE_DIR, complex_id, f"{lig_name}.sdf")
        
        # Use rdkit to read sdf
        suppl = Chem.SDMolSupplier(ligand_path, removeHs=True, sanitize=False)
        lig_mol = next(suppl)
        if lig_mol is None:
            return None
        lig_pos = torch.tensor(lig_mol.GetConformer().GetPositions()).float()
        
        apo_rec_path = os.path.join(BASE_DIR, complex_id, "receptor_apo_prot.pdb")
        holo_rec_path = os.path.join(BASE_DIR, complex_id, "receptor_holo_prot.pdb")

        if not os.path.exists(apo_rec_path) or not os.path.exists(holo_rec_path):
            return None

        holo_rec_struct = parse_pdb_pmd(
            path=holo_rec_path, remove_hs=True, reorder=True
        )
        holo_pos = torch.tensor(holo_rec_struct.get_coordinates(0)).float()

        atom_rec_index = torch.tensor(
            [atom.residue.idx for atom in holo_rec_struct.atoms]
        ).long()

        nearby_residue_mask = get_nearby_residue_mask(
            atom_pos=holo_pos,
            lig_pos=lig_pos,
            atom_rec_index=atom_rec_index,
            cutoff=5.0,
            min_residues=1,
        )

        nearby_residue_idxs = torch.argwhere(nearby_residue_mask).squeeze()
        
        if nearby_residue_idxs.dim() == 0:
            pocket_residue_str = str(nearby_residue_idxs.item() + 1)
        else:
            pocket_residue_str = ",".join(str(idx.item() + 1) for idx in nearby_residue_idxs)

        complex_dict = {
            "pdbid": complex_id,
            "apo_protein_file": apo_rec_path,
            "holo_protein_file": holo_rec_path,
            "base_dir": BASE_DIR,
            "ligand_input": None,
            "ligand_description": "filename",
            "pocket_residues": pocket_residue_str,
        }
        return complex_dict

    except Exception as e:
        logging.error(f"Failed to process {complex_id} due to {e}")
        return None

def run(is_train: bool, output_csv_path: str, num_workers: int = 4):
    logging.getLogger().setLevel("INFO")

    df_list = []
    
    # Read CSV
    df = pd.read_csv(CSV_PATH)
    if is_train:
        complex_ids = df[df['split'].isin(['train', 'valid'])]['id'].tolist()
    else:
        complex_ids = df[df['split'] == 'test']['id'].tolist()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_complex, cid): cid for cid in complex_ids}
        for future in tqdm.tqdm(as_completed(futures), total=len(complex_ids), desc="Processing complexes"):
            result = future.result()
            if result is not None:
                df_list.append(result)

    df_out = pd.DataFrame.from_dict(df_list)
    logging.info(f"Number of examples processed={df_out.shape[0]}")
    df_out.to_csv(output_csv_path, index=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pocket CSV for apo2mol dataset")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for the dataset")
    parser.add_argument("--csv_path", type=str, default=CSV_PATH, help="Path to the input CSV file containing complex IDs and splits")
    parser.add_argument("--is_train", action="store_true", help="Whether to generate training set or test set")
    parser.add_argument("--output_csv_path", type=str, default="/data/protein/BC_Data/flexdock/examples/inference_apo2mol.csv", help="Path to save the generated CSV file")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes")
    args = parser.parse_args()
    BASE_DIR = args.base_dir
    CSV_PATH = args.csv_path
    run(args.is_train, args.output_csv_path, args.num_workers)
