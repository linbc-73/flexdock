import copy
import logging
import math
import pickle
from multiprocessing import Pool
import wandb
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm


from flexdock.data.parse import molecule
from flexdock.data.parse.parser import ComplexParser
from flexdock.metrics.docking import compute_metrics as compute_docking_metrics
from flexdock.metrics.relaxation import (
    compute_ligand_alignment_metrics,
    compute_protein_alignment_metrics,
)
from flexdock.metrics.posebusters import bust
from flexdock.geometry.ops import rigid_transform_kabsch


parser = ComplexParser(esm_embeddings_path=None)


def _check_prediction_shapes(inputs, predictions):
    if predictions is None or inputs is None:
        return

    if inputs.get("true_atom_pos") is None:
        return

    if isinstance(predictions["atom_pos"], list):
        assert inputs["true_atom_pos"].shape == predictions["atom_pos"][0].shape
    else:
        assert inputs["true_atom_pos"].shape == predictions["atom_pos"].shape


class Evaluator:
    def __init__(self, args):
        self.args = args

    def evaluate(self, input_csv, output_dir):
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        df = pd.read_csv(input_csv, index_col="pdbid")
        complexes_all = df.index.tolist()
        complexes_available = [
            complex_name
            for complex_name in complexes_all
            if (output_dir / complex_name).exists()
        ]

        dock_inf_results = []
        relax_inf_results = []

        for complex_id in complexes_available:
            try:
                dock_output_file = output_dir / complex_id / "docking_predictions.pkl"
                relax_output_file = output_dir / complex_id / "relax_predictions.pkl"

                if not dock_output_file.exists() and not relax_output_file.exists():
                    logging.info(f"No predictions found for {complex_id}")
                    continue

                if not self.args.only_relaxation:
                    with dock_output_file.open("rb") as f:
                        dock_predictions = pickle.load(f)

                    inputs = self.prepare_inputs(
                        complex_id=complex_id,
                        apo_rec_path=df.loc[complex_id].apo_protein_file,
                        holo_rec_path=df.loc[complex_id].holo_protein_file,
                        data_dir=self.args.data_dir,
                        load_mol=True,
                        dataset="pdbbind",
                        pocket_atom_mask=dock_predictions["atom_mask"],
                        base_path=df.loc[complex_id].base_dir if "base_dir" in df.columns else None,
                    )
                    
                    if inputs is None:
                        logging.warning(f"Failed to prepare inputs for {complex_id}")
                        continue

                    _check_prediction_shapes(inputs, dock_predictions)

                    apo_holo_pocket_rmsd = np.nan
                    apo_holo_pocket_rmsd_aligned = np.nan
                    apo_pred_pocket_rmsds = []
                    pred_pocket_gains = []
                    if inputs.get("apo_pos") is not None and inputs.get("true_atom_pos") is not None:
                        p_mask = dock_predictions.get("pocket_atom_mask")
                        a_pos = inputs["apo_pos"][p_mask] if p_mask is not None else inputs["apo_pos"]
                        h_pos = inputs["true_atom_pos"][p_mask] if p_mask is not None else inputs["true_atom_pos"]
                        if a_pos.shape == h_pos.shape and len(a_pos) > 0:
                            apo_holo_pocket_rmsd = float(np.sqrt(np.mean(np.sum((a_pos - h_pos)**2, axis=-1))))
                            try:
                                R, tr = rigid_transform_kabsch(a_pos, h_pos, as_numpy=True)
                                a_pos_aligned = a_pos @ R.swapaxes(-1, -2) + tr[None, :]
                                apo_holo_pocket_rmsd_aligned = float(np.sqrt(np.mean(np.sum((a_pos_aligned - h_pos)**2, axis=-1))))
                            except Exception as e:
                                pass
                            
                            try:
                                pred_pos_list = dock_predictions["atom_pos"] if isinstance(dock_predictions["atom_pos"], list) else [dock_predictions["atom_pos"]]
                                for pred_pos in pred_pos_list:
                                    p_pred_pos = pred_pos[p_mask] if p_mask is not None else pred_pos
                                    if p_pred_pos.shape == h_pos.shape:
                                        cur_apo_pred = float(np.sqrt(np.mean(np.sum((a_pos - p_pred_pos)**2, axis=-1))))
                                        apo_pred_pocket_rmsds.append(cur_apo_pred)
                            except Exception as e:
                                pass

                    dock_metrics = self.compute_docking_metrics(
                        complex_id=complex_id,
                        pred_atom_pos_list=dock_predictions["atom_pos"],
                        pred_lig_pos_list=dock_predictions["ligand_pos"],
                        true_atom_pos=inputs["true_atom_pos"],
                        true_lig_pos=inputs["true_lig_pos"],
                        filter_hs=dock_predictions["filterHs"],
                        ca_mask=dock_predictions["ca_mask"],
                        nearby_atom_mask=dock_predictions["pocket_atom_mask"],
                        orig_mol=inputs["mol"],
                    )
                    confidences = dock_predictions["confidence"]
                    if confidences is not None:
                        confidences = confidences[np.argsort(confidences)[::-1]]

                    dock_inf_dict = {
                        "complex_id": dock_predictions["name"],
                        "run_times": dock_predictions["run_time"],
                        "confidences": confidences.tolist()
                        if confidences is not None
                        else None,
                        "apo_holo_pocket_rmsd": apo_holo_pocket_rmsd,
                        "apo_holo_pocket_rmsd_aligned": apo_holo_pocket_rmsd_aligned,
                        "apo_pred_pocket_rmsds": apo_pred_pocket_rmsds,
                        **dock_metrics,
                    }
                    dock_inf_results.append(dock_inf_dict)

                else:
                    with relax_output_file.open("rb") as f:
                        relax_predictions = pickle.load(f)

                if relax_output_file.exists():
                    with relax_output_file.open("rb") as f:
                        relax_predictions = pickle.load(f)

                    inputs = self.prepare_inputs(
                        complex_id=complex_id,
                        apo_rec_path=df.loc[complex_id].apo_protein_file,
                        holo_rec_path=df.loc[complex_id].holo_protein_file,
                        data_dir=self.args.data_dir,
                        load_mol=True,
                        dataset=self.args.dataset,
                        pocket_atom_mask=relax_predictions["atom_mask"],
                        base_path=df.loc[complex_id].base_dir if "base_dir" in df.columns else None,
                    )
                    
                    if inputs is None:
                        logging.warning(f"Failed to prepare inputs for {complex_id} (relaxation)")
                        continue

                    _check_prediction_shapes(
                        inputs=inputs, predictions=relax_predictions
                    )

                    relax_metrics = self.compute_relaxation_metrics(
                        pred_atom_pos=relax_predictions["atom_pos"],
                        pred_lig_pos=relax_predictions["ligand_pos"],
                        orig_apo_pos=inputs["apo_pos"],
                        true_atom_pos=inputs["true_atom_pos"],
                        true_lig_pos=inputs["true_lig_pos"],
                        filter_hs=relax_predictions["filterHs"],
                        ca_mask=relax_predictions["ca_mask"],
                        c_mask=relax_predictions["c_mask"],
                        n_mask=relax_predictions["n_mask"],
                        nearby_atom_mask=relax_predictions["pocket_atom_mask"],
                        orig_mol=inputs["mol"],
                    )

                    relax_inf_dict = {
                        "pdb_id": relax_predictions["name"],
                        "success": relax_predictions["success"],
                        "lig_pred": relax_predictions["ligand_pos"],
                        "atom_pred": relax_predictions["atom_pos"],
                        "lig_true": inputs["true_lig_pos"],
                        "atom_true": inputs["true_atom_pos"],
                        "filterHs": relax_predictions["filterHs"],
                        "apo_rec_path": df.loc[complex_id].apo_protein_file,
                        "pocket_mask": relax_predictions["amber_subset_mask"],
                        "mol": inputs["mol"],
                        "add_hs": True,
                        "time": relax_predictions["run_time"],
                        **relax_metrics,
                    }
                    relax_inf_results.append(relax_inf_dict)

            except Exception as e:
                logging.error(
                    f"{complex_id}: Failed to evaluate due to {e}", exc_info=True
                )
                continue

        if dock_inf_results:
            csv_path = output_dir / f"complex_rmsds_{self.args.align_proteins_by}.csv"
            try:
                with open(csv_path, "w") as f:
                    f.write("complex_id,lig_rmsd_top1,lig_rmsd_top5,lig_rmsd_top10,aa_rmsd_top1,aa_rmsd_top5,aa_rmsd_top10,bb_rmsd_top1,bb_rmsd_top5,bb_rmsd_top10,apo_holo_pocket_rmsd,apo_holo_pocket_rmsd_aligned,apo_pred_pocket_rmsd_top1,apo_pred_pocket_rmsd_top5,apo_pred_pocket_rmsd_top10,pred_pocket_gain_top1,pred_pocket_gain_top5,pred_pocket_gain_top10\n")
                    for res in dock_inf_results:
                        c_id = res.get("complex_id", "unknown")
                        rmsds = res.get("rmsds", [])
                        aa_rmsds = res.get("aa_rmsds", [])
                        bb_rmsds = res.get("bb_rmsds", [])
                        
                        apo_holo_pocket_rmsd = res.get("apo_holo_pocket_rmsd", np.nan)
                        apo_holo_pocket_rmsd_aligned = res.get("apo_holo_pocket_rmsd_aligned", np.nan)
                        apo_pred_pocket_rmsds = res.get("apo_pred_pocket_rmsds", [])
                        
                        if apo_holo_pocket_rmsd is None: apo_holo_pocket_rmsd = np.nan
                        if apo_holo_pocket_rmsd_aligned is None: apo_holo_pocket_rmsd_aligned = np.nan

                        if not rmsds: continue
                        
                        base_apo_holo = apo_holo_pocket_rmsd_aligned if self.args.align_proteins_by != "noalign" and not np.isnan(apo_holo_pocket_rmsd_aligned) else apo_holo_pocket_rmsd
                        pred_pocket_gains = [base_apo_holo - aa_rmsd for aa_rmsd in aa_rmsds] if not np.isnan(base_apo_holo) else []
                        
                        lig_rmsd_1 = rmsds[0] if len(rmsds) > 0 else np.nan
                        lig_rmsd_5 = min(rmsds[:5]) if len(rmsds) >= 1 else np.nan
                        lig_rmsd_10 = min(rmsds[:10]) if len(rmsds) >= 1 else np.nan
                        
                        aa_rmsd_1 = aa_rmsds[0] if len(aa_rmsds) > 0 else np.nan
                        aa_rmsd_5 = min(aa_rmsds[:5]) if len(aa_rmsds) >= 1 else np.nan
                        aa_rmsd_10 = min(aa_rmsds[:10]) if len(aa_rmsds) >= 1 else np.nan

                        bb_rmsd_1 = bb_rmsds[0] if len(bb_rmsds) > 0 else np.nan
                        bb_rmsd_5 = min(bb_rmsds[:5]) if len(bb_rmsds) >= 1 else np.nan
                        bb_rmsd_10 = min(bb_rmsds[:10]) if len(bb_rmsds) >= 1 else np.nan

                        apo_pred_pocket_rmsd_top1 = apo_pred_pocket_rmsds[0] if len(apo_pred_pocket_rmsds) > 0 else np.nan
                        apo_pred_pocket_rmsd_top5 = min(apo_pred_pocket_rmsds[:5]) if len(apo_pred_pocket_rmsds) >= 1 else np.nan
                        apo_pred_pocket_rmsd_top10 = min(apo_pred_pocket_rmsds[:10]) if len(apo_pred_pocket_rmsds) >= 1 else np.nan

                        pred_pocket_gain_top1 = pred_pocket_gains[0] if len(pred_pocket_gains) > 0 else np.nan
                        pred_pocket_gain_top5 = max(pred_pocket_gains[:5]) if len(pred_pocket_gains) >= 1 else np.nan
                        pred_pocket_gain_top10 = max(pred_pocket_gains[:10]) if len(pred_pocket_gains) >= 1 else np.nan

                        f.write(f"{c_id},{lig_rmsd_1:.4f},{lig_rmsd_5:.4f},{lig_rmsd_10:.4f},{aa_rmsd_1:.4f},{aa_rmsd_5:.4f},{aa_rmsd_10:.4f},{bb_rmsd_1:.4f},{bb_rmsd_5:.4f},{bb_rmsd_10:.4f},{apo_holo_pocket_rmsd:.4f},{apo_holo_pocket_rmsd_aligned:.4f},{apo_pred_pocket_rmsd_top1:.4f},{apo_pred_pocket_rmsd_top5:.4f},{apo_pred_pocket_rmsd_top10:.4f},{pred_pocket_gain_top1:.4f},{pred_pocket_gain_top5:.4f},{pred_pocket_gain_top10:.4f}\n")
                print(f"Saved per-complex RMSD detailed metrics to {csv_path}")
            except Exception as e:
                print(f"Could not save CSV due to: {e}")

        docking_metrics = self.aggregate_docking_metrics(dock_inf_results)
        relax_metrics = self.aggregate_relaxation_metrics(relax_inf_results)

        if self.args.wandb:
            if len(docking_metrics):
                wandb.log(docking_metrics)

            if len(relax_metrics):
                wandb.log(relax_metrics)

        print("Relaxation Metrics")
        print(relax_metrics)
        print()

        if not self.args.only_relaxation:
            print("Docking Metrics")
            print(docking_metrics)
            print()

    def prepare_inputs(
        self,
        complex_id,
        holo_rec_path: str,
        apo_rec_path: str,
        data_dir: str,
        load_mol: bool,
        dataset: str = "pdbbind",
        pocket_atom_mask=None,
        base_path=None,
    ):
        if load_mol:
            target_dir = f"{base_path}/{complex_id}" if base_path else f"{data_dir}/{complex_id}"
            print(f"Reading molecule from {target_dir}")
            
            if dataset == "posebusters":
                try:
                    mol = molecule.read_molecule(
                        f"{target_dir}/{complex_id}_ligand.sdf",
                        remove_hs=False,
                        sanitize=True,
                    )
                    mol = Chem.RemoveAllHs(mol)
                except Exception as e:
                    print(f"Could not load mol due to {e}")
                    mol = None
            else:
                try:
                    mols = molecule.read_mols_v2(base_dir=target_dir)
                    if len(mols) > 0:
                        mol = mols[0]
                        mol = Chem.RemoveAllHs(mol)
                    else:
                        print(f"No molecule found in {target_dir}")
                        mol = None
                except Exception as e:
                    print(f"Could not load mol due to {e}")
                    mol = None
        else:
            mol = None

        parsed_protein_inputs = parser.parse_protein(
            complex_dict={
                "name": complex_id,
                "apo_rec_path": apo_rec_path,
                "holo_rec_path": holo_rec_path,
            }
        )
        if parsed_protein_inputs is None:
            return None

        holo_rec_struct = parsed_protein_inputs["holo_rec_struct"]
        apo_rec_struct = parsed_protein_inputs["apo_rec_struct"]

        if holo_rec_struct is not None:
            holo_rec_pos = holo_rec_struct.get_coordinates(0)
            apo_rec_pos = apo_rec_struct.get_coordinates(0)

            if pocket_atom_mask is not None:
                if len(holo_rec_pos) == len(apo_rec_pos):
                    holo_rec_pos = holo_rec_pos[pocket_atom_mask]
                else:
                    logging.warning(f"Shape mismatch: Apo ({len(apo_rec_pos)}) vs Holo ({len(holo_rec_pos)}). Skipping holo atom filtering.")
                    holo_rec_pos = None

                apo_rec_pos = apo_rec_pos[pocket_atom_mask]

            # predictions["atom_mask"] contains pocket + buffer atoms
            inputs = {
                "mol": mol,
                "apo_pos": apo_rec_pos,
                "true_lig_pos": mol.GetConformer().GetPositions(),
                "true_atom_pos": holo_rec_pos,
            }
            return inputs
        else:
            return None

    def compute_docking_metrics(
        self,
        complex_id,
        pred_atom_pos_list,
        pred_lig_pos_list,
        true_atom_pos,
        true_lig_pos,
        filter_hs,
        ca_mask,
        nearby_atom_mask,
        orig_mol,
    ):
        metrics = compute_docking_metrics(
            complex_id=complex_id,
            pred_atom_pos_list=pred_atom_pos_list,
            pred_lig_pos_list=pred_lig_pos_list,
            true_atom_pos=true_atom_pos,
            true_lig_pos=true_lig_pos,
            filterHs=filter_hs,
            ca_mask=ca_mask,
            nearby_atom_mask=nearby_atom_mask,
            orig_mol=copy.deepcopy(orig_mol),
            align_proteins_by=self.args.align_proteins_by,
        )
        return metrics

    def compute_relaxation_metrics(
        self,
        pred_atom_pos,
        pred_lig_pos,
        orig_apo_pos,
        true_atom_pos,
        true_lig_pos,
        filter_hs,
        ca_mask,
        c_mask,
        n_mask,
        nearby_atom_mask,
        orig_mol,
    ):
        mol_noh = copy.deepcopy(orig_mol)
        Chem.RemoveStereochemistry(mol_noh)
        mol_noh = Chem.RemoveHs(mol_noh, sanitize=False)

        apo_R, apo_tr = rigid_transform_kabsch(
            pred_atom_pos[ca_mask],
            orig_apo_pos[ca_mask],
            as_numpy=True,
        )
        lig_pred = pred_lig_pos @ apo_R.swapaxes(-1, -2) + apo_tr[None, :]
        atom_pred = pred_atom_pos @ apo_R.swapaxes(-1, -2) + apo_tr[None, :]

        holo_R, holo_tr = rigid_transform_kabsch(
            true_atom_pos[nearby_atom_mask], atom_pred[nearby_atom_mask], as_numpy=True
        )
        lig_true = true_lig_pos @ holo_R.swapaxes(-1, -2) + holo_tr
        atom_true = true_atom_pos @ holo_R.swapaxes(-1, -2) + holo_tr

        metrics = {}
        metrics.update(
            compute_ligand_alignment_metrics(
                lig_pred[filter_hs], lig_true, mol_noh, return_aligned_pos=False
            )
        )
        metrics.update(
            compute_protein_alignment_metrics(
                pos=atom_pred,
                ref_pos=atom_true,
                nearby_atom_mask=nearby_atom_mask,
                ca_mask=ca_mask,
                c_mask=c_mask,
                n_mask=n_mask,
            )
        )

        return metrics

    def aggregate_docking_metrics(self, docking_inf_results):
        if not len(docking_inf_results):
            return {}
        performance_metrics = {}

        metrics_gathered = {}
        N = len(docking_inf_results[0]["rmsds"])
        for key in [
            "rmsds",
            "rmsds_before_alignment",
            "bb_rmsds",
            "aa_rmsds",
            "centroid_distances",
            "run_times",
            "confidences",
        ]:
            inf_results_tmp = docking_inf_results[0]
            if inf_results_tmp[key] is not None:
                metric_gathered = np.array(
                    [np.array(inf_results[key]) for inf_results in docking_inf_results]
                )
                metrics_gathered[key] = metric_gathered
            else:
                metrics_gathered[key] = None

        for overlap in [""]:
            rmsds = metrics_gathered["rmsds"]
            bb_rmsds = metrics_gathered["bb_rmsds"]
            aa_rmsds = metrics_gathered["aa_rmsds"]
            centroid_distances = metrics_gathered["centroid_distances"]
            run_times = metrics_gathered["run_times"]
            confidences = metrics_gathered["confidences"]

            performance_metrics.update(
                {
                    f"{overlap}run_times_std": run_times.std().__round__(2),
                    f"{overlap}run_times_mean": run_times.mean().__round__(2),
                    f"{overlap}mean_rmsd": rmsds.mean(),
                    f"{overlap}rmsds_below_2": (
                        100 * (rmsds < 2).sum() / len(rmsds) / N
                    ),
                    f"{overlap}rmsds_below_5": (
                        100 * (rmsds < 5).sum() / len(rmsds) / N
                    ),
                    f"{overlap}rmsds_percentile_50": np.percentile(rmsds, 50).round(2),
                    f"{overlap}mean_centroid": centroid_distances.mean().__round__(2),
                    f"{overlap}centroid_below_2": (
                        100
                        * (centroid_distances < 2).sum()
                        / len(centroid_distances)
                        / N
                    ).__round__(2),
                    f"{overlap}centroid_below_5": (
                        100
                        * (centroid_distances < 5).sum()
                        / len(centroid_distances)
                        / N
                    ).__round__(2),
                    f"{overlap}centroid_percentile_50": np.percentile(
                        centroid_distances, 50
                    ).round(2),
                }
            )

            performance_metrics.update(
                {
                    f"{overlap}mean_aa_rmsd": aa_rmsds.mean(),
                    f"{overlap}aa_rmsds_below_0.5": (
                        100 * (aa_rmsds < 0.5).sum() / len(aa_rmsds) / N
                    ),
                    f"{overlap}aa_rmsds_below_1": (
                        100 * (aa_rmsds < 1).sum() / len(aa_rmsds) / N
                    ),
                    f"{overlap}aa_rmsds_below_2": (
                        100 * (aa_rmsds < 2).sum() / len(aa_rmsds) / N
                    ),
                    f"{overlap}aa_rmsds_percentile_50": np.percentile(
                        aa_rmsds, 50
                    ).round(2),
                }
            )

            performance_metrics.update(
                {
                    f"{overlap}mean_bb_rmsd": bb_rmsds.mean(),
                    f"{overlap}bb_rmsds_below_0.5": (
                        100 * (bb_rmsds < 0.5).sum() / len(bb_rmsds) / N
                    ),
                    f"{overlap}bb_rmsds_below_1": (
                        100 * (bb_rmsds < 1).sum() / len(bb_rmsds) / N
                    ),
                    f"{overlap}bb_rmsds_below_2": (
                        100 * (bb_rmsds < 2).sum() / len(bb_rmsds) / N
                    ),
                    f"{overlap}bb_rmsds_percentile_50": np.percentile(
                        bb_rmsds, 50
                    ).round(2),
                }
            )

            if N >= 5:
                top5_rmsds = np.min(rmsds[:, :5], axis=1)
                top5_centroid_distances = centroid_distances[
                    np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)
                ][:, 0]

                performance_metrics.update(
                    {
                        f"{overlap}top5_rmsds_below_2": (
                            100 * (top5_rmsds < 2).sum() / len(top5_rmsds)
                        ).__round__(2),
                        f"{overlap}top5_rmsds_below_5": (
                            100 * (top5_rmsds < 5).sum() / len(top5_rmsds)
                        ).__round__(2),
                        f"{overlap}top5_rmsds_percentile_50": np.percentile(
                            top5_rmsds, 50
                        ).round(2),
                        f"{overlap}top5_centroid_below_2": (
                            100
                            * (top5_centroid_distances < 2).sum()
                            / len(top5_centroid_distances)
                        ).__round__(2),
                        f"{overlap}top5_centroid_below_5": (
                            100
                            * (top5_centroid_distances < 5).sum()
                            / len(top5_centroid_distances)
                        ).__round__(2),
                        f"{overlap}top5_centroid_percentile_50": np.percentile(
                            top5_centroid_distances, 50
                        ).round(2),
                    }
                )

                top5_aa_rmsds = np.min(aa_rmsds[:, :5], axis=1)
                performance_metrics.update(
                    {
                        f"{overlap}top5_aa_rmsds_below_0.5": (
                            100 * (top5_aa_rmsds < 0.5).sum() / len(top5_aa_rmsds)
                        ).__round__(2),
                        f"{overlap}top5_aa_rmsds_below_1": (
                            100 * (top5_aa_rmsds < 1).sum() / len(top5_aa_rmsds)
                        ).__round__(2),
                        f"{overlap}top5_aa_rmsds_below_2": (
                            100 * (top5_aa_rmsds < 2).sum() / len(top5_aa_rmsds)
                        ).__round__(2),
                        f"{overlap}top5_aa_rmsds_percentile_50": np.percentile(
                            top5_aa_rmsds, 50
                        ).round(2),
                    }
                )

                top5_bb_rmsds = np.min(bb_rmsds[:, :5], axis=1)
                performance_metrics.update(
                    {
                        f"{overlap}top5_bb_rmsds_below_0.5": (
                            100 * (top5_bb_rmsds < 0.5).sum() / len(top5_bb_rmsds)
                        ).__round__(2),
                        f"{overlap}top5_bb_rmsds_below_1": (
                            100 * (top5_bb_rmsds < 1).sum() / len(top5_bb_rmsds)
                        ).__round__(2),
                        f"{overlap}top5_bb_rmsds_below_2": (
                            100 * (top5_bb_rmsds < 2).sum() / len(top5_bb_rmsds)
                        ).__round__(2),
                        f"{overlap}top5_bb_rmsds_percentile_50": np.percentile(
                            top5_bb_rmsds, 50
                        ).round(2),
                    }
                )

            if N >= 10:
                top10_rmsds = np.min(rmsds[:, :10], axis=1)
                top10_centroid_distances = centroid_distances[
                    np.arange(rmsds.shape[0])[:, None],
                    np.argsort(rmsds[:, :10], axis=1),
                ][:, 0]
                performance_metrics.update(
                    {
                        f"{overlap}top10_rmsds_below_2": (
                            100 * (top10_rmsds < 2).sum() / len(top10_rmsds)
                        ).__round__(2),
                        f"{overlap}top10_rmsds_below_5": (
                            100 * (top10_rmsds < 5).sum() / len(top10_rmsds)
                        ).__round__(2),
                        f"{overlap}top10_rmsds_percentile_50": np.percentile(
                            top10_rmsds, 50
                        ).round(2),
                        f"{overlap}top10_centroid_below_2": (
                            100
                            * (top10_centroid_distances < 2).sum()
                            / len(top10_centroid_distances)
                        ).__round__(2),
                        f"{overlap}top10_centroid_below_5": (
                            100
                            * (top10_centroid_distances < 5).sum()
                            / len(top10_centroid_distances)
                        ).__round__(2),
                        f"{overlap}top10_centroid_percentile_50": np.percentile(
                            top10_centroid_distances, 50
                        ).round(2),
                    }
                )

                top10_aa_rmsds = np.min(aa_rmsds[:, :10], axis=1)
                performance_metrics.update(
                    {
                        f"{overlap}top10_aa_rmsds_below_0.5": (
                            100 * (top10_aa_rmsds < 0.5).sum() / len(top10_aa_rmsds)
                        ).__round__(2),
                        f"{overlap}top10_aa_rmsds_below_1": (
                            100 * (top10_aa_rmsds < 1).sum() / len(top10_aa_rmsds)
                        ).__round__(2),
                        f"{overlap}top10_aa_rmsds_below_2": (
                            100 * (top10_aa_rmsds < 2).sum() / len(top10_aa_rmsds)
                        ).__round__(2),
                        f"{overlap}top10_aa_rmsds_percentile_50": np.percentile(
                            top10_aa_rmsds, 50
                        ).round(2),
                    }
                )

                top10_bb_rmsds = np.min(bb_rmsds[:, :10], axis=1)
                performance_metrics.update(
                    {
                        f"{overlap}top10_bb_rmsds_below_0.5": (
                            100 * (top10_bb_rmsds < 0.5).sum() / len(top10_bb_rmsds)
                        ).__round__(2),
                        f"{overlap}top10_bb_rmsds_below_1": (
                            100 * (top10_bb_rmsds < 1).sum() / len(top10_bb_rmsds)
                        ).__round__(2),
                        f"{overlap}top10_bb_rmsds_below_2": (
                            100 * (top10_bb_rmsds < 2).sum() / len(top10_bb_rmsds)
                        ).__round__(2),
                        f"{overlap}top10_bb_rmsds_percentile_50": np.percentile(
                            top10_bb_rmsds, 50
                        ).round(2),
                    }
                )

            if confidences is not None:
                confidence_ordering = np.argsort(confidences, axis=1)[:, ::-1]

                filtered_rmsds = rmsds[
                    np.arange(rmsds.shape[0])[:, None], confidence_ordering
                ][:, 0]
                filtered_centroid_distances = centroid_distances[
                    np.arange(rmsds.shape[0])[:, None], confidence_ordering
                ][:, 0]

                performance_metrics.update(
                    {
                        f"{overlap}filtered_rmsds_below_2": (
                            100 * (filtered_rmsds < 2).sum() / len(filtered_rmsds)
                        ).__round__(2),
                        f"{overlap}filtered_rmsds_below_5": (
                            100 * (filtered_rmsds < 5).sum() / len(filtered_rmsds)
                        ).__round__(2),
                        f"{overlap}filtered_rmsds_percentile_50": np.percentile(
                            filtered_rmsds, 50
                        ).round(2),
                        f"{overlap}filtered_centroid_below_2": (
                            100
                            * (filtered_centroid_distances < 2).sum()
                            / len(filtered_centroid_distances)
                        ).__round__(2),
                        f"{overlap}filtered_centroid_below_5": (
                            100
                            * (filtered_centroid_distances < 5).sum()
                            / len(filtered_centroid_distances)
                        ).__round__(2),
                        f"{overlap}filtered_centroid_percentile_50": np.percentile(
                            filtered_centroid_distances, 50
                        ).round(2),
                    }
                )

                filtered_aa_rmsds = np.min(
                    aa_rmsds[
                        np.arange(aa_rmsds.shape[0])[:, None], confidence_ordering
                    ][:, :1],
                    axis=1,
                )

                performance_metrics.update(
                    {
                        f"{overlap}filtered_aa_rmsds_below_0.5": (
                            100
                            * (filtered_aa_rmsds < 0.5).sum()
                            / len(filtered_aa_rmsds)
                        ).__round__(2),
                        f"{overlap}filtered_aa_rmsds_below_1": (
                            100 * (filtered_aa_rmsds < 1).sum() / len(filtered_aa_rmsds)
                        ).__round__(2),
                        f"{overlap}filtered_aa_rmsds_below_2": (
                            100 * (filtered_aa_rmsds < 2).sum() / len(filtered_aa_rmsds)
                        ).__round__(2),
                        f"{overlap}filtered_aa_rmsds_percentile_50": np.percentile(
                            filtered_aa_rmsds, 50
                        ).round(2),
                    }
                )

                filtered_bb_rmsds = np.min(
                    bb_rmsds[
                        np.arange(bb_rmsds.shape[0])[:, None], confidence_ordering
                    ][:, :1],
                    axis=1,
                )

                performance_metrics["combined_metric"] = (
                    performance_metrics["filtered_rmsds_below_2"]
                    + 0.25 * performance_metrics["filtered_aa_rmsds_below_1"]
                )

                performance_metrics.update(
                    {
                        f"{overlap}filtered_bb_rmsds_below_0.5": (
                            100
                            * (filtered_bb_rmsds < 0.5).sum()
                            / len(filtered_bb_rmsds)
                        ).__round__(2),
                        f"{overlap}filtered_bb_rmsds_below_1": (
                            100 * (filtered_bb_rmsds < 1).sum() / len(filtered_bb_rmsds)
                        ).__round__(2),
                        f"{overlap}filtered_bb_rmsds_below_2": (
                            100 * (filtered_bb_rmsds < 2).sum() / len(filtered_bb_rmsds)
                        ).__round__(2),
                        f"{overlap}filtered_bb_rmsds_percentile_50": np.percentile(
                            filtered_bb_rmsds, 50
                        ).round(2),
                    }
                )

                if N >= 5:
                    top5_filtered_rmsds = np.min(
                        rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][
                            :, :5
                        ],
                        axis=1,
                    )
                    top5_filtered_centroid_distances = centroid_distances[
                        np.arange(rmsds.shape[0])[:, None], confidence_ordering
                    ][:, :5][
                        np.arange(rmsds.shape[0])[:, None],
                        np.argsort(
                            rmsds[
                                np.arange(rmsds.shape[0])[:, None], confidence_ordering
                            ][:, :5],
                            axis=1,
                        ),
                    ][
                        :, 0
                    ]

                    performance_metrics.update(
                        {
                            f"{overlap}top5_filtered_rmsds_below_2": (
                                100
                                * (top5_filtered_rmsds < 2).sum()
                                / len(top5_filtered_rmsds)
                            ).__round__(2),
                            f"{overlap}top5_filtered_rmsds_below_5": (
                                100
                                * (top5_filtered_rmsds < 5).sum()
                                / len(top5_filtered_rmsds)
                            ).__round__(2),
                            f"{overlap}top5_filtered_rmsds_percentile_50": np.percentile(
                                top5_filtered_rmsds, 50
                            ).round(
                                2
                            ),
                            f"{overlap}top5_filtered_centroid_below_2": (
                                100
                                * (top5_filtered_centroid_distances < 2).sum()
                                / len(top5_filtered_centroid_distances)
                            ).__round__(2),
                            f"{overlap}top5_filtered_centroid_below_5": (
                                100
                                * (top5_filtered_centroid_distances < 5).sum()
                                / len(top5_filtered_centroid_distances)
                            ).__round__(2),
                            f"{overlap}top5_filtered_centroid_percentile_50": np.percentile(
                                top5_filtered_centroid_distances, 50
                            ).round(
                                2
                            ),
                        }
                    )

                    top5_filtered_aa_rmsds = np.min(
                        aa_rmsds[
                            np.arange(aa_rmsds.shape[0])[:, None], confidence_ordering
                        ][:, :5],
                        axis=1,
                    )

                    performance_metrics.update(
                        {
                            f"{overlap}top5_filtered_aa_rmsds_below_0.5": (
                                100
                                * (top5_filtered_aa_rmsds < 0.5).sum()
                                / len(top5_filtered_aa_rmsds)
                            ).__round__(2),
                            f"{overlap}top5_filtered_aa_rmsds_below_1": (
                                100
                                * (top5_filtered_aa_rmsds < 1).sum()
                                / len(top5_filtered_aa_rmsds)
                            ).__round__(2),
                            f"{overlap}top5_filtered_aa_rmsds_below_2": (
                                100
                                * (top5_filtered_aa_rmsds < 2).sum()
                                / len(top5_filtered_aa_rmsds)
                            ).__round__(2),
                            f"{overlap}top5_filtered_aa_rmsds_percentile_50": np.percentile(
                                top5_filtered_aa_rmsds, 50
                            ).round(
                                2
                            ),
                        }
                    )

                    top5_filtered_bb_rmsds = np.min(
                        bb_rmsds[
                            np.arange(bb_rmsds.shape[0])[:, None], confidence_ordering
                        ][:, :5],
                        axis=1,
                    )

                    performance_metrics.update(
                        {
                            f"{overlap}top5_filtered_bb_rmsds_below_0.5": (
                                100
                                * (top5_filtered_bb_rmsds < 0.5).sum()
                                / len(top5_filtered_bb_rmsds)
                            ).__round__(2),
                            f"{overlap}top5_filtered_bb_rmsds_below_1": (
                                100
                                * (top5_filtered_bb_rmsds < 1).sum()
                                / len(top5_filtered_bb_rmsds)
                            ).__round__(2),
                            f"{overlap}top5_filtered_bb_rmsds_below_2": (
                                100
                                * (top5_filtered_bb_rmsds < 2).sum()
                                / len(top5_filtered_bb_rmsds)
                            ).__round__(2),
                            f"{overlap}top5_filtered_bb_rmsds_percentile_50": np.percentile(
                                top5_filtered_bb_rmsds, 50
                            ).round(
                                2
                            ),
                        }
                    )

                if N >= 10:
                    top10_filtered_rmsds = np.min(
                        rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][
                            :, :10
                        ],
                        axis=1,
                    )
                    top10_filtered_centroid_distances = centroid_distances[
                        np.arange(rmsds.shape[0])[:, None], confidence_ordering
                    ][:, :10][
                        np.arange(rmsds.shape[0])[:, None],
                        np.argsort(
                            rmsds[
                                np.arange(rmsds.shape[0])[:, None], confidence_ordering
                            ][:, :10],
                            axis=1,
                        ),
                    ][
                        :, 0
                    ]

                    performance_metrics.update(
                        {
                            f"{overlap}top10_filtered_rmsds_below_2": (
                                100
                                * (top10_filtered_rmsds < 2).sum()
                                / len(top10_filtered_rmsds)
                            ).__round__(2),
                            f"{overlap}top10_filtered_rmsds_below_5": (
                                100
                                * (top10_filtered_rmsds < 5).sum()
                                / len(top10_filtered_rmsds)
                            ).__round__(2),
                            f"{overlap}top10_filtered_rmsds_percentile_25": np.percentile(
                                top10_filtered_rmsds, 25
                            ).round(
                                2
                            ),
                            f"{overlap}top10_filtered_rmsds_percentile_50": np.percentile(
                                top10_filtered_rmsds, 50
                            ).round(
                                2
                            ),
                            f"{overlap}top10_filtered_rmsds_percentile_75": np.percentile(
                                top10_filtered_rmsds, 75
                            ).round(
                                2
                            ),
                            f"{overlap}top10_filtered_centroid_below_2": (
                                100
                                * (top10_filtered_centroid_distances < 2).sum()
                                / len(top10_filtered_centroid_distances)
                            ).__round__(2),
                            f"{overlap}top10_filtered_centroid_below_5": (
                                100
                                * (top10_filtered_centroid_distances < 5).sum()
                                / len(top10_filtered_centroid_distances)
                            ).__round__(2),
                            f"{overlap}top10_filtered_centroid_percentile_50": np.percentile(
                                top10_filtered_centroid_distances, 50
                            ).round(
                                2
                            ),
                        }
                    )

                    top10_filtered_aa_rmsds = np.min(
                        aa_rmsds[
                            np.arange(aa_rmsds.shape[0])[:, None], confidence_ordering
                        ][:, :10],
                        axis=1,
                    )

                    performance_metrics.update(
                        {
                            f"{overlap}top10_filtered_aa_rmsds_below_0.5": (
                                100
                                * (top10_filtered_aa_rmsds < 0.5).sum()
                                / len(top10_filtered_aa_rmsds)
                            ).__round__(2),
                            f"{overlap}top10_filtered_aa_rmsds_below_1": (
                                100
                                * (top10_filtered_aa_rmsds < 1).sum()
                                / len(top10_filtered_aa_rmsds)
                            ).__round__(2),
                            f"{overlap}top10_filtered_aa_rmsds_below_2": (
                                100
                                * (top10_filtered_aa_rmsds < 2).sum()
                                / len(top10_filtered_aa_rmsds)
                            ).__round__(2),
                            f"{overlap}top10_filtered_aa_rmsds_percentile_50": np.percentile(
                                top10_filtered_aa_rmsds, 50
                            ).round(
                                2
                            ),
                        }
                    )

                    top10_filtered_bb_rmsds = np.min(
                        bb_rmsds[
                            np.arange(bb_rmsds.shape[0])[:, None], confidence_ordering
                        ][:, :10],
                        axis=1,
                    )

                    performance_metrics.update(
                        {
                            f"{overlap}top10_filtered_bb_rmsds_below_0.5": (
                                100
                                * (top10_filtered_bb_rmsds < 0.5).sum()
                                / len(top10_filtered_bb_rmsds)
                            ).__round__(2),
                            f"{overlap}top10_filtered_bb_rmsds_below_1": (
                                100
                                * (top10_filtered_bb_rmsds < 1).sum()
                                / len(top10_filtered_bb_rmsds)
                            ).__round__(2),
                            f"{overlap}top10_filtered_bb_rmsds_below_2": (
                                100
                                * (top10_filtered_bb_rmsds < 2).sum()
                                / len(top10_filtered_bb_rmsds)
                            ).__round__(2),
                            f"{overlap}top10_filtered_bb_rmsds_percentile_50": np.percentile(
                                top10_filtered_bb_rmsds, 50
                            ).round(
                                2
                            ),
                        }
                    )

        return performance_metrics

    def aggregate_relaxation_metrics(
        self, inference_result_dicts, num_workers: int = 1
    ):
        if not len(inference_result_dicts):
            return {}
        bust_dfs = []
        with tqdm(total=len(inference_result_dicts)) as pbar:
            with Pool(num_workers) as p:
                for bust_df in p.imap_unordered(
                    bust, inference_result_dicts, chunksize=10
                ):
                    bust_dfs.append(bust_df)
                    pbar.update(1)
        bust_df = pd.concat(bust_dfs)
        bust_df = bust_df.fillna(False)

        aggregate_metrics = {}
        for key in bust_df.columns:
            if key in [
                "lig_scrmsds",
                "lig_rmsds",
                "lig_centered_rmsds",
                "lig_aligned_rmsds",
                "aa_rmsds",
                "bb_rmsds",
                "lig_tr_mags",
            ]:
                upper_bounds = [0.5, 1.0, 2.0, 5.0]
                lower_bounds = None
                percentiles = [0, 25, 50, 75, 100]
            elif key == "lig_rot_mags":
                upper_bounds = [angle * math.pi / 180 for angle in [90, 45, 30, 15]]
                lower_bounds = None
                percentiles = [0, 25, 50, 75, 100]
            elif key == "lddt_pli":
                upper_bounds = None
                lower_bounds = [0.7, 0.9]
                percentiles = [0, 25, 50, 75, 100]
            else:
                upper_bounds = None
                lower_bounds = None
                percentiles = None
            if upper_bounds is not None:
                for threshold in upper_bounds:
                    aggregate_metrics[f"{key}_lt_{threshold}"] = (
                        (bust_df[key] < threshold).astype(float).mean()
                    )
                    aggregate_metrics[f"pb_valid_and_{key}_lt_{threshold}"] = (
                        (bust_df["pb_valid"] & (bust_df[key] < threshold))
                        .astype(float)
                        .mean()
                    )
                    aggregate_metrics[f"pb_valid_given_{key}_lt_{threshold}"] = (
                        aggregate_metrics[f"pb_valid_and_{key}_lt_{threshold}"]
                        / aggregate_metrics[f"{key}_lt_{threshold}"]
                    )
            if lower_bounds is not None:
                for threshold in lower_bounds:
                    aggregate_metrics[f"{key}_gt_{threshold}"] = (
                        (bust_df[key] > threshold).astype(float).mean()
                    )
                    aggregate_metrics[f"pb_valid_and_{key}_gt_{threshold}"] = (
                        (bust_df["pb_valid"] & (bust_df[key] > threshold))
                        .astype(float)
                        .mean()
                    )
                    aggregate_metrics[f"pb_valid_given_{key}_gt_{threshold}"] = (
                        aggregate_metrics[f"pb_valid_and_{key}_gt_{threshold}"]
                        / aggregate_metrics[f"{key}_gt_{threshold}"]
                    )
            if percentiles is not None:
                percentile_values = np.percentile(
                    bust_df[key].values, percentiles, axis=-1
                )
                for percentile, percentile_value in zip(percentiles, percentile_values):
                    aggregate_metrics[
                        f"{key}_percentile_{percentile}"
                    ] = percentile_value
            if key in [
                "mol_pred_loaded",
                "mol_cond_loaded",
                "sanitization",
                "all_atoms_connected",
                "bond_lengths",
                "bond_angles",
                "internal_steric_clash",
                "aromatic_ring_flatness",
                "double_bond_flatness",
                "internal_energy",
                "protein-ligand_maximum_distance",
                "minimum_distance_to_protein",
                "minimum_distance_to_organic_cofactors",
                "minimum_distance_to_inorganic_cofactors",
                "minimum_distance_to_waters",
                "volume_overlap_with_protein",
                "volume_overlap_with_organic_cofactors",
                "volume_overlap_with_inorganic_cofactors",
                "volume_overlap_with_waters",
                "pb_valid",
                "success",
            ]:
                aggregate_metrics[key] = bust_df[key].astype(float).mean()
            aggregate_metrics["time_avg"] = bust_df["time"].mean()
            aggregate_metrics["time_std"] = bust_df["time"].std()
            aggregate_metrics["time_total"] = bust_df["time"].sum()

        return aggregate_metrics
