import torch
from torch import nn
from torch_geometric.data import Data
import numpy as np
from ase import Atoms
# from jarvis.core.atoms import Atoms as JAtoms


def perturb_structure(atoms: Atoms, max_frac: float = 0.5, rng: np.random.Generator | None = None) -> Atoms:
    """
    Displace each atom by up to max_frac * (minimum interatomic distance).
    If no valid interatomic distance exists (e.g., single atom or overlapping atoms),
    return the structure unchanged.
    """
    n = len(atoms)
    if n < 2:
        return atoms

    atoms_new = atoms.copy()
    # Pairwise distances (mic=True handles periodic cells with minimum image)
    D = atoms.get_all_distances(mic=True)

    # Take only off-diagonal entries (unique pairs)
    iu = np.triu_indices(n, k=1)
    pair_dists = D[iu]

    # Keep finite, strictly positive distances
    mask = np.isfinite(pair_dists) & (pair_dists > 0.0)
    if not np.any(mask):
        # Nothing sensible to use as a scale; skip perturbation
        return atoms

    min_dist = pair_dists[mask].min()
    max_disp = max_frac * float(min_dist)

    if max_disp <= 0.0:
        return atoms

    if rng is None:
        rng = np.random.default_rng()

    displacements = rng.uniform(-max_disp, max_disp, size=(n, 3))
    atoms_new.positions += displacements
    return atoms_new


def apply_strain(atoms: Atoms, max_strain: float = 0.05) -> Atoms:
    """
    Apply an anisotropic strain: scale each cell vector by a factor in [1.0, 1.0+max_strain].
    """
    atoms_new = atoms.copy()
    factors = 1.0 + np.random.uniform(0, max_strain, size=3)
    strain_matrix = np.diag(factors)
    new_cell = atoms.cell.dot(strain_matrix)
    atoms_new.set_cell(new_cell, scale_atoms=True)
    return atoms_new


# def _cart_to_frac(cart, lattice):
#     return np.asarray(cart) @ np.linalg.inv(np.asarray(lattice))

# def _min_interatomic_distance_mic(atoms: JAtoms) -> float:
#     lattice = np.asarray(atoms.lattice_mat)
#     f = np.asarray(atoms.frac_coords)
#     n = len(f)
#     if n < 2:
#         return 0.0
#     min_d = np.inf
#     for i in range(n):
#         df = f[i] - f[i+1:]
#         df -= np.round(df)                # wrap to [-0.5, 0.5)
#         dc = df @ lattice
#         if dc.size:
#             md = np.min(np.linalg.norm(dc, axis=1))
#             if md < min_d:
#                 min_d = md
#     return float(min_d if np.isfinite(min_d) else 0.0)

# def perturb_structure_jarvis(atoms: JAtoms, max_frac: float = 0.05) -> JAtoms:
#     """
#     Displace each atom by a random distance in [0, max_frac * min_dist]
#     along a random 3D direction.

#     Parameters
#     ----------
#     atoms : jarvis.core.atoms.JAtoms
#         Input structure.
#     max_frac : float
#         Maximum displacement fraction of the minimum interatomic distance.

#     Returns
#     -------
#     jarvis.core.atoms.JAtoms
#         Perturbed structure.
#     """
#     if atoms.num_atoms == 0:
#         return atoms

#     min_dist = _min_interatomic_distance_mic(atoms)
#     if min_dist <= 0 or not np.isfinite(min_dist):
#         return atoms

#     lat = np.asarray(atoms.lattice_mat)
#     cart = np.asarray(atoms.cart_coords)

#     # Sample magnitudes uniformly in [0, max_frac * min_dist]
#     mags = np.random.uniform(0.0, max_frac * min_dist, size=atoms.num_atoms)

#     # Sample random directions (normalized)
#     vecs = np.random.normal(size=(atoms.num_atoms, 3))
#     norms = np.linalg.norm(vecs, axis=1).reshape(-1, 1)
#     dirs = vecs / norms

#     # Apply displacement
#     disp = mags.reshape(-1, 1) * dirs
#     cart_new = cart + disp

#     # Wrap back into cell
#     frac_new = _cart_to_frac(cart_new, lat) % 1.0

#     return JAtoms(lattice_mat=lat, elements=list(atoms.elements),
#                  coords=frac_new, cartesian=False)

# def apply_strain_jarvis(atoms: JAtoms, max_strain: float = 0.05) -> JAtoms:
#     factors = 1.0 + np.random.uniform(0.0, max_strain, size=3)
#     strain_matrix = np.diag(factors)
#     lat = np.asarray(atoms.lattice_mat)
#     new_lat = lat @ strain_matrix
#     frac = np.asarray(atoms.frac_coords) % 1.0
#     return JAtoms(lattice_mat=new_lat, elements=list(atoms.elements), coords=frac, cartesian=False)


class GraphPerturbationAugmentation(nn.Module):
    def __init__(self, perturbation):
        super().__init__()
        self.perturbation = perturbation

    @torch.no_grad()
    def forward(self, graph: Data) -> Data:
        graph = graph.clone()
        graph.edge_attr = graph.edge_attr + torch.randn_like(graph.edge_attr) * self.perturbation  # mean 0, std sfg.perturbation
        return graph
    

class GraphAttrMaskingAugmentation(nn.Module):
    def __init__(self, mask_prob):
        super().__init__()
        self.mask_prob = mask_prob

    @torch.no_grad()
    def forward(self, graph: Data) -> Data:
        graph = graph.clone()
        # mask node and edge as 0 with probability cfg.mask_prob
        node_mask = torch.rand(graph.num_nodes) < self.mask_prob
        graph.x[node_mask] = 0.0
        edge_mask = torch.rand(graph.num_edges) < self.mask_prob
        graph.edge_attr[edge_mask] = 0.0
        return graph


class TokenRandomMaskingAugmentation(nn.Module):
    """
    Standard BERT-style MLM augmentation following the 80/10/10 rule:
    - 80% of masked tokens → [MASK]
    - 10% of masked tokens → random token
    - 10% of masked tokens → keep original
    """
    def __init__(self, mask_prob, mask_token, vocab_size=None, special_token_ids=None):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.vocab_size = vocab_size
        self.special_token_ids = special_token_ids or []

    @torch.no_grad()
    def forward(self, texts: dict) -> dict:
        input_ids = texts['input_ids'].clone()
        labels = texts["input_ids"].clone()

        # Create probability matrix for masking
        # IMPORTANT: Must be on the same device as input_ids
        probability_matrix = torch.full(input_ids.shape, self.mask_prob, device=input_ids.device)

        # Don't mask special tokens (PAD, CLS, SEP, etc.)
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_token_id in self.special_token_ids:
            special_tokens_mask |= (input_ids == special_token_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Select tokens to mask (default 15%)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # Set labels: only compute loss on masked tokens
        labels[~masked_indices] = -100

        # 80% of the time: replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=input_ids.device)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token

        # 10% of the time: replace with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=input_ids.device)).bool() & masked_indices & ~indices_replaced
        if self.vocab_size is not None:
            random_words = torch.randint(
                self.vocab_size,
                input_ids.shape,
                dtype=input_ids.dtype,
                device=input_ids.device
            )
            input_ids[indices_random] = random_words[indices_random]

        # Remaining 10% of the time: keep original token (no change needed)

        # update texts
        texts['labels'] = labels
        texts['input_ids'] = input_ids
        return texts
    

