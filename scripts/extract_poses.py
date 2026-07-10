import os
import shutil
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from pathlib import Path
import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flexdock.data.parse import molecule

def find_sdf_files(folder, absolute=True):
    if not os.path.isdir(folder):
        raise ValueError(f"指定的路径不是目录: {folder}")

    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if os.path.isfile(path) and fname.lower().endswith('.sdf'):
            return os.path.abspath(path) if absolute else fname
    return None

def write_mol_with_coords(mol, new_coords, path):
    w = Chem.SDWriter(path)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x, y, z = new_coords.astype(np.double)[i]
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))
    w.write(mol)
    w.close()

def extract_poses(args):
    out_dir = Path(args.output_dir)
    save_dir = Path(args.save_dir) if args.save_dir else out_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = args.data_dir
    df = pd.read_csv(args.input_csv, index_col="pdbid")

    for complex_id in df.index:
        res_dir = out_dir / complex_id
        if not res_dir.exists():
            print(f"Skipping {complex_id}, no predictions found.")
            continue

        save_complex_dir = save_dir / complex_id
        save_complex_dir.mkdir(parents=True, exist_ok=True)

        pred_files = []
        if (res_dir / "docking_predictions.pkl").exists():
            pred_files.append((res_dir / "docking_predictions.pkl", "docking"))
        if (res_dir / "relax_predictions.pkl").exists():
            pred_files.append((res_dir / "relax_predictions.pkl", "relax"))

        if not pred_files:
            print(f"Skipping {complex_id}, no predictions found.")
            continue

        print(f"Processing {complex_id}")

        # Ligand processing defaults
        ligs = molecule.read_mols_v2(base_dir=f"{data_dir}/{complex_id}", remove_hs=False)
        mol = None
        if len(ligs) > 0:
            mol = Chem.RemoveAllHs(ligs[0])
            
            # Try copying original holo ligand
            holo_file_1 = Path(data_dir) / complex_id / f"{complex_id}_ligand.sdf"
            holo_file_2 = Path(data_dir) / complex_id / "ligand.sdf"
            for hc in [holo_file_1, holo_file_2]:
                if hc.exists():
                    import shutil
                    shutil.copy(hc, save_complex_dir / "holo_ligand.sdf")
                    break
        else:
            print(f"Failed to load original ligand for {complex_id}")

        # Protein processing defaults
        apo_rec_path = df.loc[complex_id].apo_protein_file
        struct = None
        coords = None
        if __import__("os").path.exists(apo_rec_path):
            import shutil
            shutil.copy(apo_rec_path, save_complex_dir / "apo_protein.pdb")
            from flexdock.data.parse.protein import parse_pdb_from_path
            
            struct = parse_pdb_from_path(apo_rec_path, remove_hs=True, reorder=True)
            coords = struct.get_coordinates(0)
        else:
            print(f"Apo protein path {apo_rec_path} not found.")

        # Extract poses for both docking and relax if available
        for pred_file, prefix in pred_files:
            import pickle
            with open(pred_file, "rb") as f:
                preds = pickle.load(f)
            
            if mol is not None:
                ligand_pos_list = preds["ligand_pos"]
                if type(ligand_pos_list) is not list:
                    if len(ligand_pos_list.shape) == 2:
                        ligand_pos_list = [ligand_pos_list]
                
                for i, pos in enumerate(ligand_pos_list):
                    l_fname = f"pred_ligand_{prefix}_{i}.sdf" if prefix == "docking" else f"pred_ligand_{prefix}.sdf"
                    if mol.GetNumAtoms() == len(pos):
                        write_mol_with_coords(mol, pos, str(save_complex_dir / l_fname))
                    else:
                        print(f"Atom count mismatch for ligand {complex_id}: mol({mol.GetNumAtoms()}) vs pos({len(pos)})")
            
            if struct is not None:
                atom_mask = preds["atom_mask"]
                pocket_mask = preds.get("pocket_atom_mask", atom_mask)
                atom_pos_list = preds["atom_pos"]
                if type(atom_pos_list) is not list:
                    if len(atom_pos_list.shape) == 2:
                        atom_pos_list = [atom_pos_list]
                
                mask_to_use = None
                if len(atom_pos_list) > 0:
                    first_pos = atom_pos_list[0]
                    if len(first_pos) == sum(atom_mask):
                        mask_to_use = atom_mask
                    elif len(first_pos) == sum(pocket_mask):
                        mask_to_use = pocket_mask
                
                if mask_to_use is not None:
                    for i, pos in enumerate(atom_pos_list):
                        p_fname = f"pred_protein_{prefix}_{i}.pdb" if prefix == "docking" else f"pred_protein_{prefix}.pdb"
                        new_coords = coords.copy()
                        new_coords[mask_to_use] = pos
                        struct.coordinates = new_coords
                        struct.save(str(save_complex_dir / p_fname), overwrite=True)
                else:
                    print(f"Could not match atom mask size with predicted positions for {complex_id}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, help="Directory to save extracted files. Defaults to output_dir.")
    args = parser.parse_args()
    extract_poses(args)
