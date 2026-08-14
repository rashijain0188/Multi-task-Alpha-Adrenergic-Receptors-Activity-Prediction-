# -*- coding: utf-8 -*-
"""
dmpnn_predict_and_explain.py — Predict + explain, from a CSV of SMILES
=========================================================================
This is the ONLY script most people need to run. Give it a CSV of molecules
and a trained model folder, and it will:

  1. Read your CSV and turn every molecule into a graph — automatically
     desalting each SMILES first (stripping counterions/salts and keeping
     only the largest organic fragment, e.g. "CC(=O)O.[Na]" -> "CC(=O)O"),
     matching how the training data was prepared. Pass --keep_salts to
     disable this and use the SMILES exactly as given.
  2. Load the trained D-MPNN model.
  3. Predict an "active" probability for every molecule, for every task.
  4. Save a clean, easy-to-read predictions.csv.
  5. Automatically explain the model's most confident predictions using
     Integrated Gradients: a highlighted picture of each molecule (green =
     pushed the model toward "active", red = pushed it toward "inactive")
     plus a CSV of the exact per-atom / per-bond numbers behind it.
  6. Optionally (with --global_importance) produce a report of which
     molecular features matter most across the WHOLE input file, not just
     one molecule at a time.

It relies on dmpnn_core.py — keep both files in the same folder.

--------------------------------------------------------------------------
Requirements
--------------------------------------------------------------------------
  pip install torch torch-geometric rdkit pandas numpy matplotlib

--------------------------------------------------------------------------
Files you need before running this script
--------------------------------------------------------------------------
  A model folder containing:
    best_model.pt        (the trained weights)
    model_config.json     (saved automatically by the training script)
  Plus, since the number of D-MPNN message-passing layers can't be read
  back out of the checkpoint, ONE of:
    --n_layers 3                              (just tell it directly), or
    --hp_config path/to/best_hyperparameters.json   (also saved by training)

--------------------------------------------------------------------------
Quick start
--------------------------------------------------------------------------
  python dmpnn_predict_and_explain.py \\
      --csv my_molecules.csv \\
      --smiles_col SMILES \\
      --model_dir  model \\
      --hp_config  hpo/best_hyperparameters.json \\
      --out_dir    results

That's it — predictions.csv and a folder of explanation pictures will be
written to `results/`.

--------------------------------------------------------------------------
What you get in --out_dir
--------------------------------------------------------------------------
  results/
    predictions.csv                      <- every molecule, every task
    explanations/
      <task_name>/
        rank01_<smiles-ish>.png          <- highlighted molecule picture
        rank01_<smiles-ish>_atoms.csv    <- per-atom attribution numbers
        rank01_<smiles-ish>_bonds.csv    <- per-bond attribution numbers
        ...
    global_importance/                   <- only if --global_importance is set
      global_atom_feature_importance_<task>.csv / .png
      global_bond_feature_importance_<task>.csv / .png
      global_element_importance_<task>.csv
"""

import os
import sys
import re
import argparse

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

import dmpnn_core as core


# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════

def _safe_filename(smiles: str, max_len: int = 40) -> str:
    """Turn a SMILES string into something safe to use in a filename."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", smiles).strip("_")
    return s[:max_len] if s else "molecule"


def explain_one_molecule(model, smiles, task_idx, task_name, device,
                          out_dir, tag, steps, target, baseline):
    """Runs IG on one molecule/task, saves a PNG + atom/bond attribution
    CSVs. Returns the IGResult, or None if the SMILES couldn't be parsed."""
    res = core.integrated_gradients_single(
        model, smiles, task_idx, device,
        baseline=baseline, steps=steps, target=target)
    if res is None:
        print(f"    [skip] could not parse SMILES for explanation: {smiles}")
        return None

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, tag)

    img_path = core.draw_molecule_with_attributions(
        smiles, res.atom_attr, res.bond_attr, base + ".png",
        title=f"{task_name}: P(active)={_sigmoid_of_logit_diff(res.pred_orig):.2f}"
        if target == "logit_diff" else f"{task_name}: score={res.pred_orig:.2f}",
    )

    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    atom_rows = [
        {"atom_idx": i, "element": mol.GetAtomWithIdx(i).GetSymbol(),
         "attribution": float(v)}
        for i, v in enumerate(res.atom_attr)
    ]
    pd.DataFrame(atom_rows).sort_values(
        "attribution", key=abs, ascending=False
    ).to_csv(base + "_atoms.csv", index=False)

    bond_rows = []
    for b_idx, bond in enumerate(mol.GetBonds()):
        bond_rows.append({
            "bond_idx": b_idx,
            "begin_atom": bond.GetBeginAtomIdx(),
            "end_atom": bond.GetEndAtomIdx(),
            "bond_type": str(bond.GetBondType()),
            "attribution": float(res.bond_attr[b_idx]),
        })
    pd.DataFrame(bond_rows).sort_values(
        "attribution", key=abs, ascending=False
    ).to_csv(base + "_bonds.csv", index=False)

    print(f"    saved -> {img_path}")
    print(f"    f(x)={res.pred_orig:.4f}  f(baseline)={res.pred_baseline:.4f}  "
          f"completeness_gap={res.completeness_gap:.4f}")
    return res


def _sigmoid_of_logit_diff(logit_diff):
    return 1.0 / (1.0 + np.exp(-logit_diff))


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Predict activity for a CSV of molecules with a trained "
                    "D-MPNN model, and automatically explain the most "
                    "confident predictions with Integrated Gradients.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Required inputs -----------------------------------------------
    parser.add_argument("--csv", required=True,
                        help="Input CSV containing a column of SMILES strings.")
    parser.add_argument("--smiles_col", default="SMILES",
                        help="Name of the SMILES column in --csv.")
    parser.add_argument("--model_dir", required=True,
                        help="Folder containing best_model.pt and model_config.json.")
    parser.add_argument("--keep_salts", action="store_true",
                        help="Skip desalting: use SMILES exactly as given. "
                            "By default, salts/counterions are stripped and "
                            "only the largest organic fragment is kept "
                            "(matches how the training data was prepared).")

    # -- Architecture detail that can't be read from the checkpoint -----
    parser.add_argument("--n_layers", type=int, default=None,
                        help="Number of D-MPNN message-passing rounds used "
                            "at training time. Cannot be inferred from the "
                            "checkpoint — provide this OR --hp_config.")
    parser.add_argument("--hp_config", default=None,
                        help="Path to best_hyperparameters.json saved during "
                            "training (contains n_layers, among others).")

    # -- Output -----------------------------------------------------------
    parser.add_argument("--out_dir", default="results",
                        help="Folder to write predictions and explanations into.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for the Active/Inactive call.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # -- Tasks to report on ------------------------------------------------
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Restrict to specific task names (default: all "
                            "tasks in the model).")

    # -- Automatic explanations of top predictions -------------------------
    parser.add_argument("--explain_top_n", type=int, default=5,
                        help="For each task, automatically generate IG "
                            "explanations for the N most-confidently-active "
                            "molecules. Set to 0 to skip explanations entirely.")
    parser.add_argument("--explain_smiles", nargs="+", default=None,
                        help="Specific SMILES (must appear in --csv) to "
                            "always explain, regardless of ranking.")
    parser.add_argument("--ig_steps", type=int, default=50,
                        help="Integrated Gradients interpolation steps. "
                            "Higher = more accurate but slower.")
    parser.add_argument("--ig_target", default="logit_diff",
                        choices=["logit_diff", "prob_active", "logit_active"],
                        help="Scalar model output IG explains. logit_diff "
                            "(recommended) avoids softmax saturation.")
    parser.add_argument("--ig_baseline", default="zero", choices=["zero"])

    # -- Optional dataset-wide feature-importance report --------------------
    parser.add_argument("--global_importance", action="store_true",
                        help="Also compute a dataset-wide report of which "
                            "atom/bond features matter most overall (slower "
                            "— runs IG on every molecule, capped by "
                            "--max_mols_global).")
    parser.add_argument("--max_mols_global", type=int, default=500,
                        help="Cap on molecules used for --global_importance "
                            "(cost scales linearly with molecule count).")

    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*64}")
    print("  D-MPNN — Predict & Explain")
    print(f"  input csv   : {args.csv}")
    print(f"  model dir   : {args.model_dir}")
    print(f"  output dir  : {args.out_dir}")
    print(f"  device      : {device}")
    print(f"{'='*64}\n")

    # -- Load input CSV --------------------------------------------------
    if not os.path.exists(args.csv):
        print(f"ERROR: input CSV not found: {args.csv}")
        sys.exit(1)
    df = pd.read_csv(args.csv)
    if args.smiles_col not in df.columns:
        print(f"ERROR: column '{args.smiles_col}' not found in {args.csv}.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    print(f"  Loaded {len(df)} rows from {args.csv}")

    # -- Featurise ---------------------------------------------------------
    remove_salts = not args.keep_salts
    print(f"  Desalting: {'ON (keeping largest organic fragment)' if remove_salts else 'OFF (using SMILES as given)'}")
    print("  Turning molecules into graphs …")
    dataset = core.MoleculeDataset(df, args.smiles_col, remove_salts=remove_salts)
    if dataset.invalid:
        print(f"  [warn] Skipped {len(dataset.invalid)} SMILES that could "
              f"not be parsed by RDKit (rows: "
              f"{[i for i, _ in dataset.invalid][:15]}"
              f"{' …' if len(dataset.invalid) > 15 else ''})")
    if len(dataset) == 0:
        print("  No valid molecules found — nothing to predict. Exiting.")
        sys.exit(1)
    print(f"  {len(dataset)} valid molecules ready.")

    loader = DataLoader(dataset, batch_size=args.batch_size)

    # -- Load model --------------------------------------------------------
    print("\n  Loading trained model …")
    model, config = core.load_trained_model(
        args.model_dir, device, n_layers=args.n_layers,
        hp_config_path=args.hp_config)

    all_task_names = config["task_names"]
    if args.tasks:
        for t in args.tasks:
            if t not in all_task_names:
                print(f"ERROR: task '{t}' not in model's task_names {all_task_names}")
                sys.exit(1)
        task_names = args.tasks
    else:
        task_names = all_task_names
    print(f"  Tasks: {task_names}")

    # -- Predict -----------------------------------------------------------
    print("\n  Running predictions …")
    smiles_out, probs = core.predict_all_tasks(model, loader, device,
                                                n_tasks=len(all_task_names))

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
    print(f"  Predicted {len(out_df)} / {len(df)} input molecules.")
    print(f"  Saved -> {predictions_path}")
    print("\n  Preview:")
    print(out_df.head(10).to_string(index=False))

    # -- Automatic explanations of top hits ---------------------------------
    if args.explain_top_n > 0 or args.explain_smiles:
        print(f"\n{'='*64}")
        print("  Explaining predictions with Integrated Gradients")
        print(f"{'='*64}")

        explain_dir_root = os.path.join(args.out_dir, "explanations")

        for tname in task_names:
            t_idx = all_task_names.index(tname)
            task_dir = os.path.join(explain_dir_root, tname)

            smiles_to_explain = []  # list of (label, smiles)

            if args.explain_top_n > 0:
                task_probs = probs[:, t_idx]
                order = np.argsort(-task_probs)[:args.explain_top_n]
                for rank, i in enumerate(order, start=1):
                    smiles_to_explain.append(
                        (f"rank{rank:02d}_p{task_probs[i]:.2f}", smiles_out[i]))

            if args.explain_smiles:
                for smi in args.explain_smiles:
                    can = core.canonicalise(smi, remove_salts=remove_salts)
                    if can is None:
                        print(f"    [warn] could not parse requested "
                              f"--explain_smiles entry: {smi}")
                        continue
                    smiles_to_explain.append(("requested", can))

            if not smiles_to_explain:
                continue

            print(f"\n  Task '{tname}' — explaining {len(smiles_to_explain)} molecule(s):")
            for label, smi in smiles_to_explain:
                tag = f"{label}_{_safe_filename(smi)}"
                print(f"  - {smi}")
                explain_one_molecule(
                    model, smi, t_idx, tname, device, task_dir, tag,
                    steps=args.ig_steps, target=args.ig_target,
                    baseline=args.ig_baseline)

    # -- Optional dataset-wide feature importance ----------------------------
    if args.global_importance:
        print(f"\n{'='*64}")
        print("  Dataset-wide feature importance (this can take a while)")
        print(f"{'='*64}")

        global_dir = os.path.join(args.out_dir, "global_importance")
        os.makedirs(global_dir, exist_ok=True)

        smiles_pool = list(dict.fromkeys(smiles_out))  # dedup, keep order
        if len(smiles_pool) > args.max_mols_global:
            print(f"  Capping to first {args.max_mols_global} of "
                  f"{len(smiles_pool)} molecules (raise --max_mols_global "
                  f"for a fuller picture).")
            smiles_pool = smiles_pool[:args.max_mols_global]

        for tname in task_names:
            t_idx = all_task_names.index(tname)
            print(f"\n  Task '{tname}':")
            atom_summary, bond_summary, element_summary = core.aggregate_global(
                model, smiles_pool, t_idx, device,
                baseline=args.ig_baseline, steps=args.ig_steps,
                target=args.ig_target)

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

            print(f"  Saved -> {atom_csv}")
            print(f"  Saved -> {bond_csv}")
            print(f"  Saved -> {elem_csv}")

    print(f"\n{'='*64}")
    print(f"  All done! Everything is in: {os.path.abspath(args.out_dir)}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
