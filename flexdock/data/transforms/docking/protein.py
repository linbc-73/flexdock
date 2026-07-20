from typing import Optional

from torch_geometric.transforms import BaseTransform

import numpy as np
from scipy.spatial.transform import Rotation
import torch


from flexdock.data.feature.helpers import (
    to_atom_grid_torch,
    filter_flexible_residues,
    rotate_backbone_numpy,
    rotate_backbone_torch,
)
from flexdock.geometry.ops import axis_angle_to_matrix

# TODO: Phase this out!
from flexdock.data.conformers.modify import (
    modify_conformer_torsion_angles,
    modify_sidechains_old,
)
from flexdock.geometry.manifolds import so3, torus
from torch_scatter import scatter_mean


class NearbyAtomsTransform(BaseTransform):
    def __init__(
        self,
        only_nearby_residues_atomic,
        nearby_residues_atomic_radius,
        nearby_residues_atomic_min,
        fast_updates: bool = False,
    ):
        self.only_nearby_residues_atomic = only_nearby_residues_atomic
        self.nearby_residues_atomic_radius = nearby_residues_atomic_radius
        self.nearby_residues_atomic_min = nearby_residues_atomic_min
        self.fast_updates = fast_updates

    def _compute_nearby_atoms(self, data):
        atom_grid, x, y = to_atom_grid_torch(
            data["atom"].orig_holo_pos, data["receptor"].lens_receptors
        )
        ligand_atoms = data["ligand"].orig_pos

        if isinstance(ligand_atoms, np.ndarray):
            ligand_atoms = torch.from_numpy(ligand_atoms).float()
        ligand_atoms -= data.original_center

        minimum_distance = (
            torch.cdist(atom_grid, ligand_atoms).min(dim=1).values.min(dim=1).values
        )
        nearby_residues = minimum_distance < self.nearby_residues_atomic_radius

        # if there are less than nearby_residues_atomic_min residues nearby, we take the nearby_residues_atomic_min closest residues
        if torch.sum(nearby_residues) < self.nearby_residues_atomic_min:
            # print(f'Found only {nearby_residues.sum()} nearby residues for {complex_graph.name}')
            _, closest_residues = torch.topk(
                minimum_distance, k=self.nearby_residues_atomic_min, largest=False
            )
            nearby_residues = torch.zeros_like(nearby_residues)
            nearby_residues[closest_residues] = True
            # print(f'Found {nearby_residues.sum()} nearby residues for {complex_graph.name} from total of {len(nearby_residues)} residues')

        data["receptor"].nearby_residues = nearby_residues
        nearby_atoms = torch.zeros(
            (atom_grid.shape[0], atom_grid.shape[1]), dtype=torch.bool
        )
        nearby_atoms[nearby_residues] = True
        nearby_atoms = nearby_atoms[x, y]
        return nearby_atoms

    def apply_transform_new_pipeline(self, data):
        nearby_atoms = data["atom"].nearby_atoms
        # This presently captures sidechains and backbone
        atom_edge_index = data["atom", "atom_bond", "atom"].edge_index
        nearby_atom_edges = (
            nearby_atoms[atom_edge_index[0]] & nearby_atoms[atom_edge_index[1]]
        )

        # Update rotatable mask to only edges composed of nearby atoms
        data["atom", "atom_bond", "atom"].edge_mask[~nearby_atom_edges] = False

        return data

    def __call__(self, data):
        if self.only_nearby_residues_atomic:
            # We don't need to compute it if its already there
            if "nearby_atoms" not in data["atom"]:
                nearby_atoms = self._compute_nearby_atoms(data)
                data["atom"].nearby_atoms = nearby_atoms

            if self.fast_updates:
                data = self.apply_transform_new_pipeline(data)
                return data

            filter_flexible_residues(data, data["atom"].nearby_atoms)
        return data


class ProteinTransform:
    def __init__(
        self,
        flexible_backbone: bool = False,
        flexible_sidechains: bool = False,
        sidechain_tor_bridge: bool = False,
        use_bb_orientation_feats: bool = False,
        bb_prior: bool = None,
        fast_updates: bool = False,
        bb_sigma_mode: str = "fixed",
        bb_sigma_ref_rmsd: float = 2.0,
        bb_sigma_power: float = 1.0,
        bb_sigma_min_scale: float = 0.5,
        bb_sigma_max_scale: float = 2.0,
        bb_bridge_drift_clip: Optional[float] = None,
    ):
        self.flexible_backbone = flexible_backbone
        self.flexible_sidechains = flexible_sidechains
        self.sidechain_tor_bridge = sidechain_tor_bridge
        self.use_bb_orientation_feats = use_bb_orientation_feats
        self.bb_prior = bb_prior
        self.fast_updates = fast_updates
        self.bb_sigma_mode = bb_sigma_mode
        self.bb_sigma_ref_rmsd = bb_sigma_ref_rmsd
        self.bb_sigma_power = bb_sigma_power
        self.bb_sigma_min_scale = bb_sigma_min_scale
        self.bb_sigma_max_scale = bb_sigma_max_scale
        self.bb_bridge_drift_clip = bb_bridge_drift_clip

    def _compute_bb_sigma_scale(self, data, calpha_mask, device):
        if self.bb_sigma_mode != "predicted":
            return None

        # Prefer model prediction field if present; fall back to ground-truth
        # residue_rmsd or the rmsd_res field stored in residue_flexibility caches.
        flex_signal = None
        if hasattr(data["receptor"], "residue_rmsd_pred"):
            flex_signal = data["receptor"].residue_rmsd_pred
        elif hasattr(data["receptor"], "residue_rmsd"):
            flex_signal = data["receptor"].residue_rmsd
        elif hasattr(data["receptor"], "rmsd_res"):
            flex_signal = data["receptor"].rmsd_res

        if flex_signal is None:
            return None

        flex_signal = flex_signal.to(device=device).float().reshape(-1)
        if flex_signal.numel() == 0:
            return None

        if hasattr(data["receptor"], "nearby_residues"):
            nearby_mask = data["receptor"].nearby_residues.to(device=device)
            if nearby_mask.dtype != torch.bool:
                nearby_mask = nearby_mask > 0
            nearby_mask = nearby_mask.reshape(-1)
            if nearby_mask.numel() == flex_signal.numel():
                masked = torch.zeros_like(flex_signal)
                masked[nearby_mask] = torch.clamp(flex_signal[nearby_mask], min=0.0)
                flex_signal = masked
            else:
                flex_signal = torch.clamp(flex_signal, min=0.0)
        else:
            flex_signal = torch.clamp(flex_signal, min=0.0)

        if calpha_mask.numel() == flex_signal.numel():
            flex_signal = flex_signal[calpha_mask]

        power = max(float(self.bb_sigma_power), 1e-6)
        ref = max(float(self.bb_sigma_ref_rmsd), 1e-6)
        scale = torch.pow(flex_signal / ref, power)
        scale = torch.clamp(
            scale,
            min=float(self.bb_sigma_min_scale),
            max=float(self.bb_sigma_max_scale),
        )
        return scale

    def __call__(self, data, t_dict, sigma_dict):
        if self.flexible_backbone:
            data = self.apply_backbone_transform(data, t_dict, sigma_dict)

        if self.flexible_sidechains:
            data = self.apply_sidechain_transform(data, t_dict, sigma_dict)

        return data

    def apply_backbone_transform(self, data, t_dict, sigma_dict):
        t_bb_tr, t_bb_rot = t_dict["bb_tr"], t_dict["bb_rot"]
        bb_tr_sigma, bb_rot_sigma = (
            sigma_dict["bb_tr_sigma"],
            sigma_dict["bb_rot_sigma"],
        )

        if self.fast_updates:
            calpha_mask = data["atom"].ca_mask
        else:
            calpha_mask = data["atom"].calpha
        calpha_apo = data["atom"].orig_aligned_apo_pos[calpha_mask]
        calpha_holo = data["atom"].orig_holo_pos[calpha_mask]

        # Prior applies
        if self.bb_prior is not None:
            calpha_apo = self.bb_prior(calpha_apo, calpha_holo)

        bb_rot_delta_holo = data["receptor"].rot_vec

        sigma_scale = self._compute_bb_sigma_scale(
            data=data,
            calpha_mask=calpha_mask,
            device=calpha_holo.device,
        )
        if sigma_scale is None:
            bb_tr_sigma_eff = bb_tr_sigma
            bb_rot_sigma_eff = bb_rot_sigma
        else:
            # sigma_scale is per-CA/per-residue; expand last dim so it
            # broadcasts with the 3D coordinate tensors below.
            sigma_scale = sigma_scale.unsqueeze(-1)
            bb_tr_sigma_eff = bb_tr_sigma * sigma_scale
            bb_rot_sigma_eff = bb_rot_sigma * sigma_scale

        # Save per-residue sigma scale for the model to use as an explicit
        # conditioning feature when predicting backbone drift.
        if self.bb_sigma_mode == "predicted":
            data["receptor"].bb_sigma_scale = sigma_scale.squeeze(-1).detach().clone()
        else:
            data["receptor"].bb_sigma_scale = torch.ones(
                calpha_holo.shape[0], device=calpha_holo.device, dtype=torch.float
            )

        calpha_atoms_mu_t = calpha_apo * (1 - t_bb_tr) + calpha_holo * t_bb_tr
        sigma_t = bb_tr_sigma_eff * np.sqrt(t_bb_tr * (1 - t_bb_tr))
        calpha_atoms_t = calpha_atoms_mu_t + sigma_t * torch.randn_like(
            calpha_atoms_mu_t
        )
        data.bb_tr_drift = (calpha_holo - calpha_atoms_t) / (1 - t_bb_tr)

        # Compute mu_t and sigma_t for bridge
        bb_rot_delta_mu_t = so3.exp_map_at_point(
            tangent_vec=so3.log_map_at_point(
                point=t_bb_rot * bb_rot_delta_holo,
                base_point=torch.zeros_like(bb_rot_delta_holo),
            ),
            base_point=torch.zeros_like(bb_rot_delta_holo),
        )
        sigma_t = bb_rot_sigma_eff * np.sqrt(t_bb_rot * (1 - t_bb_rot))
        # Sample from IGSO(3) for given mu and sigma
        bb_rot_delta_t = so3.sample_from_igso3(mu=bb_rot_delta_mu_t, sigma=sigma_t)

        # Using definition of the drift as provided in Riemannian Flow Matching
        # Our target distribution as at t=1
        data.bb_rot_drift = so3.log_map_at_point(
            point=bb_rot_delta_holo, base_point=bb_rot_delta_t
        ) / (1 - t_bb_rot)

        if not torch.is_tensor(data.bb_rot_drift):
            data.bb_rot_drift = torch.tensor(data.bb_rot_drift)

        # Clip extreme drift targets to prevent numerical instability when
        # t is close to 1 and/or flexible regions use large sigma scales.
        if self.bb_bridge_drift_clip is not None:
            clip_val = float(self.bb_bridge_drift_clip)
            data.bb_tr_drift = torch.clamp(
                data.bb_tr_drift, min=-clip_val, max=clip_val
            )
            data.bb_rot_drift = torch.clamp(
                data.bb_rot_drift, min=-clip_val, max=clip_val
            )

        # Since we apply updates to data['atom'].pos, we need R_holo.T * R_t
        rot_holo_to_t = Rotation.from_rotvec(
            -bb_rot_delta_holo.numpy()
        ) * Rotation.from_rotvec(bb_rot_delta_t.numpy())
        rot_holo_to_t = Rotation.as_rotvec(rot_holo_to_t)

        if self.fast_updates:
            new_pos, _ = rotate_backbone_torch(
                atoms=data["atom"].pos,
                t_vec=(calpha_atoms_t - calpha_holo),
                rot_mat=axis_angle_to_matrix(torch.tensor(rot_holo_to_t).float()),
                lens_receptors=data["receptor"].lens_receptors,
                detach=False,
                total_rot=None,
            )
            data["atom"].pos = new_pos.float()
        else:
            # Updates data['atom'].pos
            new_pos = rotate_backbone_numpy(
                atoms=data["atom"].pos,
                t_vec=(calpha_atoms_t - calpha_holo).numpy(),
                rot_vec=rot_holo_to_t,
                lens_receptors=data["receptor"].lens_receptors,
            )
            data["atom"].pos = torch.from_numpy(new_pos).float()
        data["receptor"].pos = data["atom"].pos[calpha_mask]

        if self.use_bb_orientation_feats:
            atom_grid, x, y = to_atom_grid_torch(
                data["atom"].pos, data["receptor"].lens_receptors
            )
            data["receptor"].bb_orientation = torch.cat(
                [atom_grid[:, 0] - atom_grid[:, 1], atom_grid[:, 2] - atom_grid[:, 1]],
                dim=1,
            )

        return data

    def apply_sidechain_transform(self, data, t_dict, sigma_dict):
        sc_tor_sigma = sigma_dict["sc_tor_sigma"]
        t_sc_tor = t_dict["sc_tor"]

        if sc_tor_sigma is None:
            raise ValueError(
                "sc_tor_sigma cannot be None when flexible_sidechains=True"
            )

        if self.fast_updates:
            sc_tor_delta_holo_all = data[
                "atom", "atom_bond", "atom"
            ].sc_conformer_match_rotations
            sc_tor_delta_holo = sc_tor_delta_holo_all[
                data["atom", "atom_bond", "atom"].edge_mask
            ]
        else:
            sc_tor_delta_holo = np.concatenate(data.sc_conformer_match_rotations)
            sc_tor_delta_holo = torch.from_numpy(sc_tor_delta_holo).float()

        sigma_t = sc_tor_sigma * np.sqrt(t_sc_tor * (1 - t_sc_tor))
        sigma_t = torch.tensor(sigma_t).float()

        sc_tor_delta_mu_t = torus.exp_map_at_point(
            tangent_vec=t_sc_tor
            * torus.log_map_at_point(
                point=sc_tor_delta_holo,
                base_point=torch.zeros_like(sc_tor_delta_holo),
            ),
            base_point=torch.zeros_like(sc_tor_delta_holo),
        )

        sc_tor_delta_t = torus.sample_from_wrapped_normal(
            mu=sc_tor_delta_mu_t, sigma=sigma_t
        )

        data.sidechain_tor_score = torus.log_map_at_point(
            point=sc_tor_delta_holo, base_point=sc_tor_delta_t
        ) / (1 - t_sc_tor)

        # find the right normalization factor, note none is used in the model currently
        # TODO
        data.sidechain_tor_sigma_edge = np.ones(len(data.sidechain_tor_score))
        update_to_t = sc_tor_delta_t - sc_tor_delta_holo

        if self.fast_updates:
            data["atom"].pos = modify_conformer_torsion_angles(
                pos=data["atom"].pos,
                edge_index=data["atom", "atom_bond", "atom"].edge_index,
                mask_rotate=data["atom", "atom_bond", "atom"].edge_mask,
                fragment_index=data["atom_bond", "atom"].atom_fragment_index,
                torsion_updates=update_to_t,
                sidechains=True,
            )
        else:
            data["atom"].pos = modify_sidechains_old(
                data, data["atom"].pos, update_to_t.numpy()
            )

        return data


class UseApoInputTransform(BaseTransform):
    """Replace current atom/receptor positions with backbone-aligned apo coords.

    This ensures the model receives the true apo structure as input for the
    residue-level RMSD prediction task, rather than the conformer-matched
    structure used during docking training.
    """

    def __call__(self, data):
        if not hasattr(data["atom"], "orig_aligned_apo_pos"):
            raise ValueError(
                "UseApoInputTransform requires atom.orig_aligned_apo_pos"
            )
        data["atom"].pos = data["atom"].orig_aligned_apo_pos.clone()
        data["receptor"].pos = data["atom"].pos[data["atom"].ca_mask]
        return data


class ComputeResidueRMSDTransform(BaseTransform):
    """Compute per-residue apo-holo RMSD and attach it to the receptor store.

    The RMSD is computed over all heavy atoms of each residue using the
    backbone-aligned apo positions (``atom.orig_aligned_apo_pos``) and the
    holo positions (``atom.orig_holo_pos``). Unlike
    :class:`ResidueRMSDTargetTransform`, this transform does **not** modify
    ``atom.pos``; it only stores ``receptor.residue_rmsd`` so that downstream
    transforms (e.g. flexible-backbone noise scaling) can use the ground-truth
    flexibility signal.
    """

    def __call__(self, data):
        if not hasattr(data["atom"], "orig_aligned_apo_pos") or not hasattr(
            data["atom"], "orig_holo_pos"
        ):
            raise ValueError(
                "ComputeResidueRMSDTransform requires both "
                "atom.orig_aligned_apo_pos and atom.orig_holo_pos"
            )

        atom_rec_index = data[
            "atom", "atom_rec_contact", "receptor"
        ].edge_index[1]
        apo_pos = data["atom"].orig_aligned_apo_pos
        holo_pos = data["atom"].orig_holo_pos

        num_residues = data["receptor"].x.shape[0]
        per_atom_sq = ((apo_pos - holo_pos) ** 2).sum(dim=-1)
        per_res_sq = scatter_mean(
            per_atom_sq, atom_rec_index, dim=0, dim_size=num_residues
        )
        per_res_rmsd = torch.sqrt(per_res_sq + 1e-8)

        data["receptor"].residue_rmsd = per_res_rmsd
        return data


class ResidueRMSDTargetTransform(BaseTransform):
    """Compute per-residue apo-holo RMSD and attach it as a regression target.

    The RMSD is computed over all heavy atoms of each residue using the
    backbone-aligned apo positions (``atom.orig_aligned_apo_pos``) and the
    holo positions (``atom.orig_holo_pos``). Only residues flagged by
    ``receptor.nearby_residues`` receive a non-NaN target; other residues are
    ignored during training/validation.
    """

    def __call__(self, data):
        if not hasattr(data["atom"], "orig_aligned_apo_pos") or not hasattr(
            data["atom"], "orig_holo_pos"
        ):
            raise ValueError(
                "ResidueRMSDTargetTransform requires both "
                "atom.orig_aligned_apo_pos and atom.orig_holo_pos"
            )

        # Use the final aligned apo structure as the model input.
        data["atom"].pos = data["atom"].orig_aligned_apo_pos.clone()
        data["receptor"].pos = data["atom"].pos[data["atom"].ca_mask]

        atom_rec_index = data["atom", "receptor"].edge_index[1]
        apo_pos = data["atom"].orig_aligned_apo_pos
        holo_pos = data["atom"].orig_holo_pos

        num_residues = data["receptor"].x.shape[0]
        per_atom_sq = ((apo_pos - holo_pos) ** 2).sum(dim=-1)
        per_res_sq = scatter_mean(per_atom_sq, atom_rec_index, dim=0, dim_size=num_residues)
        per_res_rmsd = torch.sqrt(per_res_sq + 1e-8)

        data["receptor"].residue_rmsd = per_res_rmsd
        return data
