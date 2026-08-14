# -*- coding: utf-8 -*-
"""
dmpnn_core.py — Shared engine for the D-MPNN multi-task model
================================================================
This file has NO command-line interface and does nothing on its own — it is
imported by other scripts (e.g. dmpnn_predict_and_explain.py). Keeping it
separate means the featurisation and architecture live in exactly ONE place,
so predictions and explanations can never silently drift out of sync with
each other or with how the model was trained.

Contents
--------
  1. Featurisation       — atom_features(), bond_features(), canonicalise(),
                            smiles_to_graph()   (byte-for-byte identical to
                            the training script's featurisation)
  2. Dataset              — MoleculeDataset, for building a PyG dataset from
                            a DataFrame of SMILES (no labels required)
  3. Model architecture   — DMPNNEncoder, DMPNN
  4. Model loading        — load_trained_model(): reads model_config.json +
                            best_model.pt from a directory and rebuilds the
                            exact network the weights were trained with
  5. Prediction           — predict_all_tasks()
  6. Integrated Gradients — per-molecule + dataset-level ("global") feature
                            attribution, plus the highlighted-molecule PNG
                            renderer

Do not edit the featurisation or architecture in this file without also
retraining the model — the two must always match exactly.
"""

import os
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

from torch_geometric.data import Data, Dataset
from torch_geometric.nn import global_mean_pool, global_add_pool


# ══════════════════════════════════════════════════════════════════════════
#  1. FEATURISATION  (must exactly match the training script)
# ══════════════════════════════════════════════════════════════════════════

ATOM_FEATURES = {
    "atomic_num":    list(range(1, 119)),
    "degree":        list(range(0, 11)),
    "formal_charge": [-2, -1, 0, 1, 2, 3],
    "valence":       list(range(0, 9)),
    "hybridization": [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    "num_hs":      list(range(0, 9)),
    "is_in_ring":  [False, True],
    "is_aromatic": [False, True],
    "chiral_tag": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
}

BOND_FEATURES = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "is_conjugated": [False, True],
    "is_in_ring":    [False, True],
    "stereo": [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ],
}


def one_hot(value, choices):
    enc = [0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else len(choices)
    enc[idx] = 1
    return enc


def atom_features(atom):
    feats  = one_hot(atom.GetAtomicNum(),     ATOM_FEATURES["atomic_num"])
    feats += one_hot(atom.GetDegree(),        ATOM_FEATURES["degree"])
    feats += one_hot(atom.GetFormalCharge(),  ATOM_FEATURES["formal_charge"])
    feats += one_hot(atom.GetTotalValence(),  ATOM_FEATURES["valence"])
    feats += one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
    feats += one_hot(atom.GetTotalNumHs(),    ATOM_FEATURES["num_hs"])
    feats += one_hot(atom.IsInRing(),         ATOM_FEATURES["is_in_ring"])
    feats += one_hot(atom.GetIsAromatic(),    ATOM_FEATURES["is_aromatic"])
    feats += [atom.GetMass() / 100.0, 0.0]
    feats += one_hot(atom.GetChiralTag(), ATOM_FEATURES["chiral_tag"])
    cip    = atom.GetPropsAsDict().get("_CIPCode", "")
    feats += [int(cip == "R"), int(cip == "S")]
    return feats


def bond_features(bond):
    feats  = one_hot(bond.GetBondType(),     BOND_FEATURES["bond_type"])
    feats += one_hot(bond.GetIsConjugated(), BOND_FEATURES["is_conjugated"])
    feats += one_hot(bond.IsInRing(),        BOND_FEATURES["is_in_ring"])
    feats += one_hot(bond.GetStereo(),       BOND_FEATURES["stereo"])
    return feats


_SALT_REMOVER = SaltRemover()  # default salt definitions — same as training


def desalt_mol(mol):
    """
    Strip salts / counterions and keep only the largest remaining fragment.

    Matches the exact method used to build the training data's
    "SMILES_desalted" column:
      1. SaltRemover().StripMol(mol)  — removes fragments matching RDKit's
         default salt-definition list (chlorides, sodium, TFA, etc.)
      2. If more than one fragment remains afterwards (StripMol doesn't
         catch every possible counterion / multi-component mixture), fall
         back to keeping the fragment with the most heavy atoms.

    Returns None if desalting fails for any reason.
    """
    try:
        mol = _SALT_REMOVER.StripMol(mol)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        return mol
    except Exception:
        return None


def canonicalise(smi: str, remove_salts: bool = True):
    """
    Returns RDKit canonical SMILES with stereo preserved, or None if the
    input can't be parsed.

    If remove_salts=True (the default), salts/counterions are stripped first
    via desalt_mol() — the same SaltRemover + largest-fragment method used
    to prepare the original training data's "SMILES_desalted" column — so
    predictions stay consistent with training. Set remove_salts=False if
    your input is already desalted and you want the SMILES used exactly as
    given.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if remove_salts:
        mol = desalt_mol(mol)
        if mol is None:
            return None
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def smiles_to_graph(smiles: str, labels=None):
    """
    Build a directed-bond PyG Data object for one molecule.

    `labels` is accepted for compatibility with the training script's
    signature (multi-task label list) but is NOT used anywhere in this file
    — inference and explanation only ever need x / edge_index / edge_attr /
    reverse_edge_index. Pass labels=None (or any placeholder) when scoring
    new molecules.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    src_list, dst_list, edge_attr_list, rev_list = [], [], [], []
    for b_idx, bond in enumerate(mol.GetBonds()):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf   = bond_features(bond)
        fwd_idx, rev_idx = 2 * b_idx, 2 * b_idx + 1

        src_list.append(i); dst_list.append(j)
        edge_attr_list.append(bf); rev_list.append(rev_idx)

        src_list.append(j); dst_list.append(i)
        edge_attr_list.append(bf); rev_list.append(fwd_idx)

    n_bond_feats = len(BOND_FEATURES["bond_type"]) + 1 + 2 + 2 + len(BOND_FEATURES["stereo"]) + 1
    if len(src_list) == 0:
        edge_index         = torch.zeros((2, 0), dtype=torch.long)
        edge_attr          = torch.zeros((0, n_bond_feats), dtype=torch.float)
        reverse_edge_index = torch.zeros((0,), dtype=torch.long)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float)
        reverse_edge_index = torch.tensor(rev_list, dtype=torch.long)

    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        reverse_edge_index=reverse_edge_index, n_atoms=x.shape[0],
        smiles=smiles,
    )


# ══════════════════════════════════════════════════════════════════════════
#  2. DATASET  (inference — no labels required)
# ══════════════════════════════════════════════════════════════════════════

class MoleculeDataset(Dataset):
    """
    Builds graphs from a DataFrame's SMILES column. No label columns needed.

    `.invalid`     holds (row_index, raw_smiles) pairs that failed to parse.
    `.raw_smiles`  holds the original, pre-desalting SMILES string for each
                   graph in `.graphs`, in the same order — useful for
                   reporting predictions against exactly what the user typed
                   even though the model itself sees the desalted/canonical
                   form (graphs[i].smiles).
    """

    def __init__(self, df, smiles_col, remove_salts: bool = True):
        super().__init__()
        self.graphs     = []
        self.invalid    = []
        self.raw_smiles = []

        for idx, row in df.iterrows():
            raw_smi = str(row[smiles_col])
            can_smi = canonicalise(raw_smi, remove_salts=remove_salts)
            if can_smi is None:
                self.invalid.append((idx, raw_smi))
                continue
            g = smiles_to_graph(can_smi)
            if g is None:
                self.invalid.append((idx, raw_smi))
                continue
            self.graphs.append(g)
            self.raw_smiles.append(raw_smi)

    def len(self):      return len(self.graphs)
    def get(self, idx): return self.graphs[idx]


# ══════════════════════════════════════════════════════════════════════════
#  3. MODEL ARCHITECTURE  (must exactly match the training script)
# ══════════════════════════════════════════════════════════════════════════

class DMPNNEncoder(nn.Module):
    def __init__(self, node_in, edge_in, hidden=300, n_layers=3, dropout=0.0):
        super().__init__()
        self.hidden   = hidden
        self.n_layers = n_layers

        self.W_i = nn.Sequential(
            nn.Linear(node_in + edge_in, hidden, bias=False),
            nn.BatchNorm1d(hidden),
        )
        self.W_m      = nn.Linear(hidden, hidden, bias=False)
        self.W_m_norm = nn.LayerNorm(hidden)
        self.W_o = nn.Sequential(
            nn.Linear(node_in + hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.SiLU()

    def forward(self, data):
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr
        rev_idx    = data.reverse_edge_index
        batch      = data.batch

        src, dst = edge_index[0], edge_index[1]
        n_atoms  = x.shape[0]

        h = self.act(self.W_i(torch.cat([x[src], edge_attr], dim=-1)))

        for _ in range(self.n_layers):
            agg = torch.zeros(n_atoms, self.hidden, device=x.device)
            agg = agg.scatter_add(0, dst.unsqueeze(1).expand(-1, self.hidden), h)

            m = agg[src] - h[rev_idx]

            h_init = h.clone()
            h = self.act(self.W_m_norm(self.W_m(m)) + h_init)
            h = self.dropout(h)

        atom_msg = torch.zeros(n_atoms, self.hidden, device=x.device)
        atom_msg = atom_msg.scatter_add(
            0, src.unsqueeze(1).expand(-1, self.hidden), h)

        h_atoms = self.W_o(torch.cat([x, atom_msg], dim=-1))

        g = torch.cat([
            global_mean_pool(h_atoms, batch),
            global_add_pool(h_atoms, batch),
        ], dim=-1)
        return g


class DMPNN(nn.Module):
    def __init__(self, node_in, edge_in, hidden=300, n_layers=3,
                 dropout=0.0, n_tasks=3, n_classes=2):
        super().__init__()
        self.n_tasks = n_tasks
        self.encoder = DMPNNEncoder(
            node_in=node_in, edge_in=edge_in,
            hidden=hidden, n_layers=n_layers, dropout=dropout,
        )
        in_dim = hidden * 2
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Dropout(dropout),

                nn.Linear(hidden, hidden // 2),
                nn.LayerNorm(hidden // 2),
                nn.SiLU(),
                nn.Dropout(dropout),

                nn.Linear(hidden // 2, hidden // 4),
                nn.SiLU(),

                nn.Linear(hidden // 4, n_classes),
            )
            for _ in range(n_tasks)
        ])

    def forward(self, data):
        g = self.encoder(data)
        logits = torch.stack([head(g) for head in self.task_heads], dim=1)
        return logits


# ══════════════════════════════════════════════════════════════════════════
#  4. MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_trained_model(model_dir, device, n_layers=None, hp_config_path=None):
    """
    Loads model_config.json + best_model.pt from `model_dir` and rebuilds
    the exact DMPNN architecture the weights were trained with.

    node_dim / edge_dim / n_tasks / task_names come from model_config.json.
    `hidden` is recovered directly from the checkpoint's tensor shapes, so it
    never needs to be supplied manually.

    `n_layers` CANNOT be recovered from the checkpoint: DMPNNEncoder reuses
    one tied weight matrix across every message-passing round, so no tensor
    shape reveals how many rounds were used at training time. Supply it via:
      - the `n_layers` argument directly, OR
      - `hp_config_path` pointing at best_hyperparameters.json (has "n_layers"), OR
      - a "n_layers" key already present in model_config.json
    Get this wrong and the model will load without error but produce silently
    incorrect predictions and explanations.
    """
    with open(os.path.join(model_dir, "model_config.json")) as f:
        config = json.load(f)

    state = torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device)
    hidden = state["encoder.W_i.0.weight"].shape[0]

    if n_layers is None and hp_config_path:
        with open(hp_config_path) as f:
            hp = json.load(f)
        n_layers = hp.get("n_layers")

    if n_layers is None:
        n_layers = config.get("n_layers")

    if n_layers is None:
        n_layers = 3
        print(f"  WARNING: n_layers not supplied and not found in any config "
              f"file. Defaulting to n_layers={n_layers}. If this does not "
              f"match how the model was trained, predictions and "
              f"explanations will be silently WRONG — pass the correct "
              f"value via --n_layers or --hp_config.")

    print(f"  Inferred hidden={hidden} from checkpoint; using n_layers={n_layers}")

    model = DMPNN(
        node_in=config["node_dim"],
        edge_in=config["edge_dim"],
        hidden=hidden,
        n_layers=n_layers,
        dropout=0.0,              # no-op at eval() time
        n_tasks=config["n_tasks"],
    ).to(device)

    model.load_state_dict(state)
    model.eval()
    return model, config


# ══════════════════════════════════════════════════════════════════════════
#  5. PREDICTION
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_all_tasks(model, loader, device, n_tasks):
    """Returns (smiles_list, probs array of shape (N, n_tasks))."""
    model.eval()
    all_smiles, all_probs = [], []
    for data in loader:
        data   = data.to(device)
        logits = model(data)
        probs  = F.softmax(logits, dim=-1)[:, :, 1]  # prob of "active"
        all_probs.append(probs.cpu().numpy())
        if hasattr(data, "smiles"):
            all_smiles.extend(data.smiles)
    all_probs = np.vstack(all_probs) if all_probs else np.zeros((0, n_tasks))
    return all_smiles, all_probs


# ══════════════════════════════════════════════════════════════════════════
#  6. INTEGRATED GRADIENTS  (per-molecule + dataset-level attribution)
# ══════════════════════════════════════════════════════════════════════════

def get_atom_feature_blocks():
    """Ordered (block_name, start_col, end_col) matching atom_features()."""
    blocks = []
    cur = 0

    def add_onehot(name, choices_key):
        nonlocal cur
        width = len(ATOM_FEATURES[choices_key]) + 1
        blocks.append((name, cur, cur + width))
        cur += width

    add_onehot("atomic_num",    "atomic_num")
    add_onehot("degree",        "degree")
    add_onehot("formal_charge", "formal_charge")
    add_onehot("valence",       "valence")
    add_onehot("hybridization", "hybridization")
    add_onehot("num_hs",        "num_hs")
    add_onehot("is_in_ring",    "is_in_ring")
    add_onehot("is_aromatic",   "is_aromatic")

    blocks.append(("mass_raw", cur, cur + 2)); cur += 2

    add_onehot("chiral_tag", "chiral_tag")

    blocks.append(("CIP_R_S", cur, cur + 2)); cur += 2
    return blocks


def get_bond_feature_blocks():
    """Ordered (block_name, start_col, end_col) matching bond_features()."""
    blocks = []
    cur = 0

    def add_onehot(name, choices_key):
        nonlocal cur
        width = len(BOND_FEATURES[choices_key]) + 1
        blocks.append((name, cur, cur + width))
        cur += width

    add_onehot("bond_type",     "bond_type")
    add_onehot("is_conjugated", "is_conjugated")
    add_onehot("is_in_ring",    "is_in_ring")
    add_onehot("stereo",        "stereo")
    return blocks


class IGResult:
    def __init__(self):
        self.atom_attr = None          # (N_atoms,)  signed, per-atom total
        self.atom_block_attr = None    # dict block_name -> float
        self.bond_attr = None          # (N_bonds,)  signed, per-bond total
        self.bond_block_attr = None    # dict block_name -> float
        self.pred_orig = None
        self.pred_baseline = None
        self.completeness_gap = None   # |sum(attr) - (pred_orig-pred_baseline)|


def _forward_scalar(model, x, edge_attr, edge_index, reverse_edge_index, batch,
                     task_idx, target):
    fake_data = SimpleNamespace(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        reverse_edge_index=reverse_edge_index, batch=batch,
    )
    logits = model(fake_data)                      # (1, n_tasks, 2)
    task_logits = logits[0, task_idx]               # (2,)
    if target == "logit_diff":
        return task_logits[1] - task_logits[0]      # unbounded, no saturation
    elif target == "prob_active":
        return F.softmax(task_logits, dim=-1)[1]
    elif target == "logit_active":
        return task_logits[1]
    else:
        raise ValueError(f"Unknown target '{target}'")


def integrated_gradients_single(model, smiles, task_idx, device,
                                 baseline="zero", steps=50, target="logit_diff"):
    """Runs IG for one molecule / one task. Returns IGResult, or None if the
    SMILES can't be parsed."""
    data = smiles_to_graph(smiles)
    if data is None:
        return None

    x = data.x.to(device)
    edge_attr = data.edge_attr.to(device)
    edge_index = data.edge_index.to(device)
    reverse_edge_index = data.reverse_edge_index.to(device)
    batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)

    if baseline == "zero":
        baseline_x = torch.zeros_like(x)
        baseline_edge_attr = torch.zeros_like(edge_attr)
    else:
        raise ValueError(f"Unknown baseline '{baseline}'")

    # Midpoint Riemann sum: avoids degenerate alpha=0/alpha=1 endpoints while
    # converging to the same integral as steps -> inf.
    alphas = (torch.arange(steps, device=device, dtype=torch.float32) + 0.5) / steps

    grad_x_accum = torch.zeros_like(x)
    grad_edge_accum = torch.zeros_like(edge_attr)

    for alpha in alphas:
        x_interp = (baseline_x + alpha * (x - baseline_x)).clone().requires_grad_(True)
        edge_interp = (baseline_edge_attr + alpha * (edge_attr - baseline_edge_attr)).clone().requires_grad_(True)

        out = _forward_scalar(model, x_interp, edge_interp, edge_index,
                               reverse_edge_index, batch, task_idx, target)

        grad_x, grad_edge = torch.autograd.grad(out, [x_interp, edge_interp])
        grad_x_accum += grad_x.detach()
        grad_edge_accum += grad_edge.detach()

    avg_grad_x = grad_x_accum / steps
    avg_grad_edge = grad_edge_accum / steps

    IG_x = (x - baseline_x) * avg_grad_x
    IG_edge = (edge_attr - baseline_edge_attr) * avg_grad_edge

    with torch.no_grad():
        pred_orig = _forward_scalar(model, x, edge_attr, edge_index,
                                     reverse_edge_index, batch, task_idx, target).item()
        pred_baseline = _forward_scalar(model, baseline_x, baseline_edge_attr,
                                         edge_index, reverse_edge_index, batch,
                                         task_idx, target).item()

    result = IGResult()
    result.atom_attr = IG_x.sum(dim=1).cpu().numpy()
    result.pred_orig = pred_orig
    result.pred_baseline = pred_baseline

    n_bonds = IG_edge.shape[0] // 2
    bond_totals = IG_edge.sum(dim=1).view(n_bonds, 2).sum(dim=1)
    result.bond_attr = bond_totals.cpu().numpy()

    atom_blocks = get_atom_feature_blocks()
    result.atom_block_attr = {
        name: IG_x[:, s:e].sum().item() for name, s, e in atom_blocks
    }
    bond_blocks = get_bond_feature_blocks()
    edge_summed = IG_edge.view(n_bonds, 2, -1).sum(dim=1)
    result.bond_block_attr = {
        name: edge_summed[:, s:e].sum().item() for name, s, e in bond_blocks
    }

    total_attr = result.atom_attr.sum() + result.bond_attr.sum()
    result.completeness_gap = abs(total_attr - (pred_orig - pred_baseline))
    return result


def draw_molecule_with_attributions(smiles, atom_attr, bond_attr, out_path,
                                     title=None):
    """Saves a PNG (or SVG fallback) of the molecule with atoms/bonds
    colour-highlighted: green = pushes prediction toward 'active',
    red = pushes it toward 'inactive'. Returns the actual path written."""
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    max_abs = max(np.abs(atom_attr).max() if len(atom_attr) else 0, 1e-8)

    def color_for(v):
        t = float(np.clip(abs(v) / max_abs, 0, 1))
        if v >= 0:
            return (1 - t, 1.0, 1 - t)
        else:
            return (1.0, 1 - t, 1 - t)

    atom_colors = {i: color_for(v) for i, v in enumerate(atom_attr)}
    atom_radii = {i: 0.3 + 0.35 * float(np.clip(abs(v) / max_abs, 0, 1))
                  for i, v in enumerate(atom_attr)}

    bond_colors = {}
    if bond_attr is not None and len(bond_attr):
        max_abs_b = max(np.abs(bond_attr).max(), 1e-8)
        for b_idx, bond in enumerate(mol.GetBonds()):
            v = bond_attr[b_idx]
            t = float(np.clip(abs(v) / max_abs_b, 0, 1))
            bond_colors[bond.GetIdx()] = (1 - t, 1.0, 1 - t) if v >= 0 else (1.0, 1 - t, 1 - t)

    used_svg = False
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(600, 500)
    except AttributeError:
        # This RDKit build wasn't compiled with Cairo support. SVG rendering
        # is always available, so fall back to it.
        used_svg = True
        drawer = rdMolDraw2D.MolDraw2DSVG(600, 500)
        root, _ext = os.path.splitext(out_path)
        out_path = root + ".svg"

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        highlightBonds=list(bond_colors.keys()),
        highlightBondColors=bond_colors,
        legend=title or "",
    )
    drawer.FinishDrawing()

    if used_svg:
        with open(out_path, "w") as f:
            f.write(drawer.GetDrawingText())
    else:
        with open(out_path, "wb") as f:
            f.write(drawer.GetDrawingText())

    return out_path


def aggregate_global(model, smiles_list, task_idx, device,
                      baseline="zero", steps=50, target="logit_diff",
                      progress_every=25, progress_cb=None):
    """
    Runs IG over every molecule in `smiles_list` for one task and aggregates:
      - atom-feature-block driving forces (mean signed + mean |attr|)
      - bond-feature-block driving forces
      - per-element (atom symbol) driving forces
    Returns (atom_block_summary, bond_block_summary, element_summary) as
    DataFrames.
    """
    atom_block_records = []
    bond_block_records = []
    element_signed = {}
    n_ok, n_fail = 0, 0
    gaps = []

    for i, smi in enumerate(smiles_list):
        res = integrated_gradients_single(
            model, smi, task_idx, device,
            baseline=baseline, steps=steps, target=target)
        if res is None:
            n_fail += 1
            continue
        n_ok += 1
        gaps.append(res.completeness_gap)
        atom_block_records.append(res.atom_block_attr)
        bond_block_records.append(res.bond_block_attr)

        mol = Chem.MolFromSmiles(smi)
        for atom_idx, v in enumerate(res.atom_attr):
            sym = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            element_signed.setdefault(sym, []).append(v)

        if progress_every and (i + 1) % progress_every == 0:
            msg = f"  IG progress: {i+1}/{len(smiles_list)} molecules ({n_fail} unparsable so far)"
            print(msg)
            if progress_cb:
                progress_cb(i + 1, len(smiles_list))

    atom_block_df = pd.DataFrame(atom_block_records)
    bond_block_df = pd.DataFrame(bond_block_records)

    atom_block_summary = pd.DataFrame({
        "mean_signed_attr": atom_block_df.mean(),
        "mean_abs_attr":    atom_block_df.abs().mean(),
    }).sort_values("mean_abs_attr", ascending=False)

    bond_block_summary = pd.DataFrame({
        "mean_signed_attr": bond_block_df.mean(),
        "mean_abs_attr":    bond_block_df.abs().mean(),
    }).sort_values("mean_abs_attr", ascending=False)

    element_summary = pd.DataFrame({
        "element": list(element_signed.keys()),
        "n_atoms": [len(v) for v in element_signed.values()],
        "mean_signed_attr": [np.mean(v) for v in element_signed.values()],
        "mean_abs_attr":    [np.mean(np.abs(v)) for v in element_signed.values()],
    }).sort_values("mean_abs_attr", ascending=False)

    print(f"\n  Molecules processed: {n_ok}  |  unparsable/skipped: {n_fail}")
    if gaps:
        print(f"  Completeness gap (|sum(IG) - (f(x)-f(baseline))|): "
              f"mean={np.mean(gaps):.4f}  max={np.max(gaps):.4f}")
        print("  (If this is large relative to typical logit-diff magnitudes, "
              "increase --steps.)")

    return atom_block_summary, bond_block_summary, element_summary


def plot_block_importance(summary_df, title, out_path):
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(summary_df))))
    order = summary_df.index
    colors = ["#2ca02c" if v >= 0 else "#d62728"
              for v in summary_df.loc[order, "mean_signed_attr"]]
    ax.barh(order, summary_df.loc[order, "mean_signed_attr"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean signed IG attribution (+ pushes toward 'active')")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
