import os
import pandas as pd
import torch
import logging

from flexdock.data.parse.protein import parse_pdb_from_path as parse_pdb_pmd
from flexdock.data.feature.protein import get_nearby_residue_mask
from rdkit import Chem

BASE_DIR = "/data/protein/BC_Data/Docking_Data/apo2mol_dataset/data_folder"
BASE_ALIGN_DIR = "/data/protein/BC_Data/induce-fit/figrdock_exp/comp_apo_generation/esm_test_set_aligned"
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

def run():
    logging.getLogger().setLevel("INFO")

    df_list = []
    
    # Read CSV
    df = pd.read_csv(CSV_PATH)
    test_ids = df[df['split'] == 'test']['id'].tolist()
    
    for complex_id in test_ids:
        try:
            lig_name = complex_id.split('__')[-1]
            ligand_path = os.path.join(BASE_DIR, complex_id, f"{lig_name}.sdf")
            
            # Use rdkit to read sdf
            suppl = Chem.SDMolSupplier(ligand_path, removeHs=True, sanitize=False)
            lig_mol = next(suppl)
            if lig_mol is None:
                continue
            lig_pos = torch.tensor(lig_mol.GetConformer().GetPositions()).float()
            
            apo_rec_path = os.path.join(BASE_ALIGN_DIR, complex_id, "rot-receptor_apo_esm_patched.pdb")
            holo_rec_path = os.path.join(BASE_ALIGN_DIR, complex_id, "rot-receptor_holo_prot.pdb")

            if not os.path.exists(apo_rec_path) or not os.path.exists(holo_rec_path):
                continue

            holo_rec_struct = parse_pdb_pmd(
                path=holo_rec_path, remove_hs=True, reorder=True
            )
            holo_pos = torch.tensor(holo_rec_struct.get_coordinates(0)).float()

            atom_rec_index = torch.tensor(
                [atom.residue.idx for atom in holo_rec_struct.atoms]
            ).long()

            pocket_residue_str = get_pocket_residue_str(
                holo_pos=holo_pos,
                ligand_pos=lig_pos,
                atom_rec_index=atom_rec_index,
                pocket_cutoff=5.0,
                pocket_min_size=1,
            )

            complex_dict = {
                "pdbid": complex_id,
                "apo_protein_file": apo_rec_path,
                "holo_protein_file": holo_rec_path,
                "base_dir": BASE_DIR,
                "ligand_input": None,
                "ligand_description": "filename",
                "pocket_residues": pocket_residue_str,
            }
            df_list.append(complex_dict)

        except Exception as e:
            logging.error(f"Failed to add to inference file due to {e}")
            continue

    df_out = pd.DataFrame.from_dict(df_list)
    logging.info(f"Number of examples processed={df_out.shape[0]}")
    df_out.to_csv("/data/protein/BC_Data/flexdock/examples/inference_apo2mol_esm_input.csv", index=None)

if __name__ == "__main__":
    run()
