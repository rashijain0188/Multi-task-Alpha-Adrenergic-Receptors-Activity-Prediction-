# -*- coding: utf-8 -*-
"""
run_predict_explain_spyder.py
================================
Spyder-friendly driver for dmpnn_predict_and_explain.py.

Why this file exists
---------------------
dmpnn_predict_and_explain.py is written as a CLI tool (argparse), which is
awkward to hit "Run" (F5) on in Spyder — it'll error on missing required
arguments. This script sets everything as a simple `Args`-style config block
(same pattern as run_ig_spyder.py) and calls the underlying functions
directly, so you can just edit the paths below and press F5 (or run cell by
cell with Ctrl+Enter, since it's split into #%% cells).

Put this file in the SAME FOLDER as:
  dmpnn_core.py
  dmpnn_predict_and_explain.py
(or adjust the sys.path.append below to point at that folder).
"""

import os
import sys

# If this script lives somewhere else, point this at the folder containing
# dmpnn_core.py and dmpnn_predict_and_explain.py:
sys.path.append(r"folder_name")

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

import dmpnn_core as core
import dmpnn_predict_and_explain as dpe   # reuses explain_one_molecule() etc.

#%% ---------------------------------------------------------------------
# 1. CONFIG -- edit these paths/values, then run this cell
# -------------------------------------------------------------------------

class Args:
    # -- Input molecules --------------------------------------------------
    csv          = r"Molecules_to_check.csv"
    smiles_col   = "SMILES"
    remove_salts = True

    # -- Trained model ------------------------------------------------------
    model_dir   = r"folder_name"
    n_layers    = 3
    hp_config   = r"best_hyperparameters.json"

    # -- Output ---------------------------------------------------------
    out_dir     = r"predict_out"
    threshold   = 0.5
    batch_size  = 64
    device      = "cuda"        # set to "cpu" if you don't have a GPU handy

    # -- Which tasks to report on (None = all tasks in the model) ---------
    tasks       = None          # e.g. ["ADRA1A", "ADRA1D"]

    # -- Automatic explanations of top predictions -------------------------
    explain_top_n    = 3        # per task, explain the N most-confidently-active
                                 # molecules. Set to 0 to skip explanations.
    explain_smiles   = None     # e.g. ["CC(=O)Oc1ccccc1C(=O)O"] to always explain
    ig_steps         = 50
    ig_target         = "logit_diff"   # or "prob_active" / "logit_active"
    ig_baseline       = "zero"

    # -- Optional dataset-wide feature-importance report --------------------
    global_importance  = False   # True = also run IG over the whole file (slow)
    max_mols_global     = 500

args = Args()

#%% ---------------------------------------------------------------------
# 2. LOAD MODEL -- run this cell once per Spyder session
# -------------------------------------------------------------------------

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
if args.device == "cuda" and device.type == "cpu":
    print("CUDA not available -- falling back to CPU.")

print("Loading trained model ...")
model, config = core.load_trained_model(
    args.model_dir, device, n_layers=args.n_layers, hp_config_path=args.hp_config)

all_task_names = config["task_names"]
if args.tasks:
    for t in args.tasks:
        assert t in all_task_names, f"'{t}' not in {all_task_names}"
    task_names = args.tasks
else:
    task_names = all_task_names
print(f"Tasks: {task_names}")

#%% ---------------------------------------------------------------------
# 3. LOAD & FEATURISE INPUT CSV -- run once per new input file
# -------------------------------------------------------------------------

print(f"Loading {args.csv} ...")
df = pd.read_csv(args.csv)
assert args.smiles_col in df.columns, (
    f"Column '{args.smiles_col}' not found. Available: {list(df.columns)}")
print(f"Loaded {len(df)} rows.")

print(f"Desalting: {'ON (keeping largest organic fragment)' if args.remove_salts else 'OFF (using SMILES as given)'}")
print("Turning molecules into graphs ...")
dataset = core.MoleculeDataset(df, args.smiles_col, remove_salts=args.remove_salts)
if dataset.invalid:
    print(f"[warn] Skipped {len(dataset.invalid)} unparseable SMILES "
          f"(rows: {[i for i, _ in dataset.invalid][:15]}"
          f"{' ...' if len(dataset.invalid) > 15 else ''})")
print(f"{len(dataset)} valid molecules ready.")

loader = DataLoader(dataset, batch_size=args.batch_size)

#%% ---------------------------------------------------------------------
# 4. PREDICT -- run to score every molecule on every task
# -------------------------------------------------------------------------

print("Running predictions ...")
smiles_out, probs = core.predict_all_tasks(
    model, loader, device, n_tasks=len(all_task_names))

out_df = pd.DataFrame({
    "Input_SMILES": dataset.raw_smiles,
    "Desalted_SMILES": smiles_out,
})
for tname in task_names:
    t_idx = all_task_names.index(tname)
    out_df[f"{tname}_prob_active"] = probs[:, t_idx].round(4)
    out_df[f"{tname}_prediction"] = np.where(
        probs[:, t_idx] >= args.threshold, "Active", "Inactive")

predictions_path = os.path.join(args.out_dir, "predictions.csv")
out_df.to_csv(predictions_path, index=False)
print(f"Predicted {len(out_df)} / {len(df)} input molecules.")
print(f"Saved -> {predictions_path}")
print(out_df.head(10).to_string(index=False))

#%% ---------------------------------------------------------------------
# 5. EXPLAIN TOP PREDICTIONS -- Integrated Gradients on the most
#    confident hits per task (+ any SMILES you explicitly listed)
# -------------------------------------------------------------------------

if args.explain_top_n > 0 or args.explain_smiles:
    explain_dir_root = os.path.join(args.out_dir, "explanations")

    for tname in task_names:
        t_idx = all_task_names.index(tname)
        task_dir = os.path.join(explain_dir_root, tname)

        smiles_to_explain = []

        if args.explain_top_n > 0:
            task_probs = probs[:, t_idx]
            order = np.argsort(-task_probs)[:args.explain_top_n]
            for rank, i in enumerate(order, start=1):
                smiles_to_explain.append(
                    (f"rank{rank:02d}_p{task_probs[i]:.2f}", smiles_out[i]))

        if args.explain_smiles:
            for smi in args.explain_smiles:
                can = core.canonicalise(smi, remove_salts=args.remove_salts)
                if can is None:
                    print(f"[warn] could not parse requested explain_smiles entry: {smi}")
                    continue
                smiles_to_explain.append(("requested", can))

        if not smiles_to_explain:
            continue

        print(f"\nTask '{tname}' — explaining {len(smiles_to_explain)} molecule(s):")
        for label, smi in smiles_to_explain:
            tag = f"{label}_{dpe._safe_filename(smi)}"
            print(f"  - {smi}")
            dpe.explain_one_molecule(
                model, smi, t_idx, tname, device, task_dir, tag,
                steps=args.ig_steps, target=args.ig_target,
                baseline=args.ig_baseline)

    print(f"\nExplanations saved under: {explain_dir_root}")
    # To view a picture inline in Spyder's Plots pane / IPython console:
    # from IPython.display import Image, display
    # display(Image(r"<path to a .png from explanations/ above>"))

#%% ---------------------------------------------------------------------
# 6. (OPTIONAL) DATASET-WIDE FEATURE IMPORTANCE -- slow, off by default
#    Set args.global_importance = True in the CONFIG cell to enable.
# -------------------------------------------------------------------------

if args.global_importance:
    global_dir = os.path.join(args.out_dir, "global_importance")
    os.makedirs(global_dir, exist_ok=True)

    smiles_pool = list(dict.fromkeys(smiles_out))  # dedup, keep order
    if len(smiles_pool) > args.max_mols_global:
        print(f"Capping to first {args.max_mols_global} of "
              f"{len(smiles_pool)} molecules (raise args.max_mols_global "
              f"for a fuller picture).")
        smiles_pool = smiles_pool[:args.max_mols_global]

    for tname in task_names:
        t_idx = all_task_names.index(tname)
        print(f"\nTask '{tname}':")
        atom_summary, bond_summary, element_summary = core.aggregate_global(
            model, smiles_pool, t_idx, device,
            baseline=args.ig_baseline, steps=args.ig_steps, target=args.ig_target)

        atom_csv = os.path.join(global_dir, f"global_atom_feature_importance_{tname}.csv")
        bond_csv = os.path.join(global_dir, f"global_bond_feature_importance_{tname}.csv")
        elem_csv = os.path.join(global_dir, f"global_element_importance_{tname}.csv")
        atom_summary.to_csv(atom_csv)
        bond_summary.to_csv(bond_csv)
        element_summary.to_csv(elem_csv, index=False)

        core.plot_block_importance(
            atom_summary, f"Atom-feature driving forces — {tname}",
            os.path.join(global_dir, f"global_atom_feature_importance_{tname}.png"))
        core.plot_block_importance(
            bond_summary, f"Bond-feature driving forces — {tname}",
            os.path.join(global_dir, f"global_bond_feature_importance_{tname}.png"))

        print(f"Saved -> {atom_csv}\n         {bond_csv}\n         {elem_csv}")

    print(f"\nGlobal importance report saved under: {global_dir}")

#%% ---------------------------------------------------------------------
print(f"\nAll done! Everything is in: {os.path.abspath(args.out_dir)}")
