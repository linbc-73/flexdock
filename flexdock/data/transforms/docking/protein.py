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
        sc_sigma_mode: str = "fixed",
        class_sigma_scales: Optional[list] = None,
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
        self.sc_sigma_mode = sc_sigma_mode
        self.class_sigma_scales = class_sigma_scales or [0.5, 1.0, 2.0]

    def _compute_bb_sigma_scale(self, data, calpha_mask, device):
        if self.bb_sigma_mode == "class":
            return self._compute_class_sigma_scale(
                data,
                device,
                field="residue_rmsd_class",
                mask=calpha_mask if calpha_mask.numel() == data["receptor"].x.shape[0] else None,
            )

        if self.bb_sigma_mode != "predicted":
            return None

        # Prefer model prediction field if present; fall back to residue_rmsd target.
        flex_signal = None
        if hasattr(data["receptor"], "residue_rmsd_pred"):
            flex_signal = data["receptor"].residue_rmsd_pred
        elif hasattr(data["receptor"], "residue_rmsd"):
            flex_signal = data["receptor"].residue_rmsd

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

    def _compute_class_sigma_scale(self, data, device, field="residue_rmsd_class", mask=None):
        if not hasattr(data["receptor"], field):
            return None

        classes = getattr(data["receptor"], field).to(device=device).long().reshape(-1)
        if classes.numel() == 0:
            return None

        scales = torch.tensor(
            self.class_sigma_scales,
            dtype=torch.float32,
            device=device,
        )
        # Clamp out-of-range classes to the nearest valid class.
        classes = torch.clamp(classes, min=0, max=len(scales) - 1)
        scale = scales[classes]

        if mask is not None:
            scale = scale[mask]

        return scale

    def _sample_bb_rot_delta_t(self, mu, sigma):
        """Sample from IGSO(3) supporting scalar or per-residue sigma.

        The upstream ``so3.sample_from_igso3`` only accepts a scalar sigma.
        When ``sigma`` is a 1-D tensor (one value per residue), we group by
        unique sigma values and sample each group separately.
        """
        if not torch.is_tensor(sigma) or sigma.dim() == 0:
            return so3.sample_from_igso3(mu=mu, sigma=sigma)

        sigma_np = sigma.detach().cpu().numpy()
        mu_np = mu.detach().cpu().numpy()
        unique_sigmas = np.unique(sigma_np)

        if len(unique_sigmas) == 1:
            return so3.sample_from_igso3(mu=mu, sigma=unique_sigmas[0].item())

        out = np.empty_like(mu_np)
        for sig in unique_sigmas:
            mask = sigma_np == sig
            mu_subset = mu_np[mask]
            sample_subset = so3.sample_from_igso3(mu=mu_subset, sigma=sig.item())
            if torch.is_tensor(sample_subset):
                sample_subset = sample_subset.detach().cpu().numpy()
            out[mask] = sample_subset
        return torch.from_numpy(out).float().to(mu.device)

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
            bb_tr_sigma_eff = bb_tr_sigma * sigma_scale
            bb_rot_sigma_eff = bb_rot_sigma * sigma_scale

        calpha_atoms_mu_t = calpha_apo * (1 - t_bb_tr) + calpha_holo * t_bb_tr
        sigma_t = bb_tr_sigma_eff * np.sqrt(t_bb_tr * (1 - t_bb_tr))
        sigma_t = torch.as_tensor(sigma_t).float()
        if sigma_t.dim() == 1:
            # Per-residue sigma scales need an extra dimension to broadcast with [N, 3].
            sigma_t = sigma_t.unsqueeze(-1)
        calpha_atoms_t = calpha_atoms_mu_t + sigma_t * torch.randn_like(
            calpha_atoms_mu_t
        )
        # Clamp the bridge denominator to avoid singular drift targets when
        # t_bb_tr is sampled extremely close to 1.
        bb_tr_denom = max(1 - t_bb_tr, 1e-3)
        data.bb_tr_drift = (calpha_holo - calpha_atoms_t) / bb_tr_denom

        # Compute mu_t and sigma_t for bridge
        bb_rot_delta_mu_t = so3.exp_map_at_point(
            tangent_vec=so3.log_map_at_point(
                point=t_bb_rot * bb_rot_delta_holo,
                base_point=torch.zeros_like(bb_rot_delta_holo),
            ),
            base_point=torch.zeros_like(bb_rot_delta_holo),
        )
        sigma_t = bb_rot_sigma_eff * np.sqrt(t_bb_rot * (1 - t_bb_rot))
        # Sample from IGSO(3) for given mu and sigma.
        # The upstream sampler only supports scalar sigma; handle per-residue
        # sigmas by grouping identical values (at most 3 groups in class mode).
        bb_rot_delta_t = self._sample_bb_rot_delta_t(
            mu=bb_rot_delta_mu_t, sigma=sigma_t
        )

        # Using definition of the drift as provided in Riemannian Flow Matching
        # Our target distribution as at t=1. Clamp the denominator to avoid
        # singular drift targets when t_bb_rot is sampled extremely close to 1.
        bb_rot_denom = max(1 - t_bb_rot, 1e-3)
        data.bb_rot_drift = so3.log_map_at_point(
            point=bb_rot_delta_holo, base_point=bb_rot_delta_t
        ) / bb_rot_denom

        if not torch.is_tensor(data.bb_rot_drift):
            data.bb_rot_drift = torch.tensor(data.bb_rot_drift)

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

    def _compute_sc_sigma_scale(self, data, device):
        if self.sc_sigma_mode != "class":
            return None

        if not hasattr(data["receptor"], "residue_rmsd_class"):
            return None

        classes = data["receptor"].residue_rmsd_class.to(device=device).long().reshape(-1)
        if classes.numel() == 0:
            return None

        scales = torch.tensor(
            self.class_sigma_scales,
            dtype=torch.float32,
            device=device,
        )
        classes = torch.clamp(classes, min=0, max=len(scales) - 1)
        residue_scales = scales[classes]

        atom_rec_index = data["atom", "receptor"].edge_index[1]
        edge_index = data["atom", "atom_bond", "atom"].edge_index
        edge_mask = data["atom", "atom_bond", "atom"].edge_mask
        rot_edges = edge_index[:, edge_mask]
        src_res = atom_rec_index[rot_edges[0].long()]
        return residue_scales[src_res]

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

        sc_sigma_scale = self._compute_sc_sigma_scale(
            data, device=sc_tor_delta_holo.device
        )
        if sc_sigma_scale is None:
            sc_tor_sigma_eff = sc_tor_sigma
        else:
            sc_tor_sigma_eff = sc_tor_sigma * sc_sigma_scale

        sigma_t = sc_tor_sigma_eff * np.sqrt(t_sc_tor * (1 - t_sc_tor))
        sigma_t = torch.as_tensor(sigma_t).float()

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

        # Clamp the denominator to avoid singular score targets when t_sc_tor
        # is sampled extremely close to 1.
        sc_tor_denom = max(1 - t_sc_tor, 1e-3)
        data.sidechain_tor_score = torus.log_map_at_point(
            point=sc_tor_delta_holo, base_point=sc_tor_delta_t
        ) / sc_tor_denom

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


class ResidueRMSDTargetTransform(BaseTransform):
    """Compute per-residue flexibility metric and attach it as a target.

    The metric can be either the RMSD over all heavy atoms of each residue
    (``residue_rmsd``) or the RMSD of the C-alpha atom only (``calpha_rmsd``).
    Both are computed between the backbone-aligned apo positions
    (``atom.orig_aligned_apo_pos``) and the holo positions
    (``atom.orig_holo_pos``).

    When ``classification`` is True, the continuous value is discretized into
    ordered bins and stored as ``residue_rmsd_class``.
    """

    def __init__(
        self,
        classification: bool = False,
        bins: Optional[list] = None,
        set_positions: bool = True,
        metric: str = "residue_rmsd",
    ):
        self.classification = classification
        self.bins = bins
        self.set_positions = set_positions
        self.metric = metric
        assert metric in {"residue_rmsd", "calpha_rmsd"}, (
            f"Unsupported flexibility metric: {metric}"
        )
        if classification:
            assert bins is not None and len(bins) > 0, "bins must be provided for classification"

    def __call__(self, data):
        if not hasattr(data["atom"], "orig_aligned_apo_pos") or not hasattr(
            data["atom"], "orig_holo_pos"
        ):
            raise ValueError(
                "ResidueRMSDTargetTransform requires both "
                "atom.orig_aligned_apo_pos and atom.orig_holo_pos"
            )

        if self.set_positions:
            # Use the final aligned apo structure as the model input.
            data["atom"].pos = data["atom"].orig_aligned_apo_pos.clone()
            data["receptor"].pos = data["atom"].pos[data["atom"].ca_mask]

        atom_rec_index = data["atom", "receptor"].edge_index[1]
        apo_pos = data["atom"].orig_aligned_apo_pos
        holo_pos = data["atom"].orig_holo_pos

        num_residues = data["receptor"].x.shape[0]
        if self.metric == "residue_rmsd":
            per_atom_sq = ((apo_pos - holo_pos) ** 2).sum(dim=-1)
            per_res_sq = scatter_mean(
                per_atom_sq, atom_rec_index, dim=0, dim_size=num_residues
            )
        elif self.metric == "calpha_rmsd":
            ca_mask = data["atom"].ca_mask
            ca_atom_rec_index = atom_rec_index[ca_mask]
            per_atom_sq = ((apo_pos[ca_mask] - holo_pos[ca_mask]) ** 2).sum(dim=-1)
            per_res_sq = scatter_mean(
                per_atom_sq,
                ca_atom_rec_index,
                dim=0,
                dim_size=num_residues,
            )
        else:
            raise ValueError(f"Unsupported flexibility metric: {self.metric}")
        per_res_rmsd = torch.sqrt(per_res_sq + 1e-8)

        data["receptor"].residue_rmsd = per_res_rmsd

        if self.classification:
            bins = torch.tensor(
                self.bins, device=per_res_rmsd.device, dtype=per_res_rmsd.dtype
            )
            data["receptor"].residue_rmsd_class = torch.bucketize(per_res_rmsd, bins)
        return data
