"""Multi-seed retraining of the final neural models (MaSTNeT, MLP, ReMasker)
and MissForest on the fixed test_train_split, to quantify how much the
final test metrics vary.

Usage
-----
    python experiments/seed_variance_study.py --models mastnet,mlp,remasker --seeds 42,43,44,45,46
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import config
from experiments.pytorch_study import train_pytorch_single_split
from experiments.sklearn_study import train_sklearn_single_split
from experiments.train_final import assemble_hyps, build_pytorch_cfg
from models.registry import get_model_class
from utils.data import load_dataset
from utils.tuning import load_optuna_study, set_seed

NEURAL_MODELS = ["mastnet", "mlp", "remasker"]


def _test_metrics(results):
    """Normalise the two possible return shapes (dict from cache, TuningResult when freshly computed)."""
    return results["test_metrics"] if isinstance(results, dict) else results.test_metrics


def run_one(model_name, seed, df, train_idx, test_idx, device, n_inferences, output_root):
    """Train one model with one seed and return its test metrics."""
    hyps = assemble_hyps(model_name, load_optuna_study(model_name).best_trial.params)
    model_class = get_model_class(model_name)
    output_dir = output_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    if isinstance(model_class, type) and issubclass(model_class, torch.nn.Module):
        # MaSTNeT and MLP go through the PyTorch training loop
        cfg = build_pytorch_cfg(model_name, hyps)

        results, _, _, _, _ = train_pytorch_single_split(
            coords_raw=torch.tensor(df[config.coordinates].astype(float).to_numpy()),
            values_raw=torch.tensor(df[config.parameters].astype(float).to_numpy()),
            model_class=model_class,
            hyps=hyps,
            train_idx=train_idx,
            val_idx=test_idx,  # matches train_final.py: test set used for early stopping
            test_idx=test_idx,
            model_name=model_name,
            split_path="final",
            trial_id=seed,
            seed=seed,
            tuning_mode=False,
            device=device,
            do_dropout=True,
            n_inferences=n_inferences,
            save_model=False,
            cfg=cfg,
            output_dir=str(output_dir),
        )
    else:
        # Everything else (ReMasker, MissForest, ...) goes through the sklearn-style path
        results, _, _, _ = train_sklearn_single_split(
            df=df,
            model_class=model_class,
            hyps=hyps,
            test_idx=test_idx,
            train_idx=train_idx,
            val_idx=test_idx,
            model_name=model_name,
            split_path="final",
            trial_id=seed,
            seed=seed,
            tuning_mode=False,
            save_model=False,
            output_dir=str(output_dir),
        )

    return _test_metrics(results)


def flatten_metrics(metrics, model_name, seed):
    """Turn the nested metrics dict into flat rows for a DataFrame."""
    rows = []
    for metric_name, value in metrics.get("Global", {}).items():
        rows.append({"model": model_name, "seed": seed, "variable": "Global", "metric": metric_name, "value": value})
    for var_name, var_metrics in metrics.get("PerVariable", {}).items():
        for metric_name, value in var_metrics.items():
            rows.append({"model": model_name, "seed": seed, "variable": var_name, "metric": metric_name, "value": value})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed variance study for the final neural models.")
    parser.add_argument("--models", type=str, default=",".join(NEURAL_MODELS))
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--n_inferences", type=int, default=50, help="MC dropout passes per run")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_aggregate", action="store_true",
                        help="Only train/cache results, skip writing the combined CSVs. "
                             "Use this for per-(model,seed) SLURM array tasks running in parallel, "
                             "since they'd otherwise race on the same CSV files; do one final run "
                             "without this flag afterwards to aggregate everything from the cache.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    models = args.models.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data and splits
    df = load_dataset()
    split = json.load(open(Path(config.output_dir_splits) / "test_train_split.json"))
    test_idx = np.array(split["test_idx"], dtype=int)
    train_idx = np.array(split["train_idx"], dtype=int)

    # Output dir
    output_root = Path(config.output_dir) / "testing/seed_variance"
    output_root.mkdir(parents=True, exist_ok=True)

    # Train and evaluate every model at every seed
    all_rows = []
    for model_name in models:
        for seed in seeds:
            logging.info(f"=== {model_name} | seed {seed} ===")
            metrics = run_one(model_name, seed, df, train_idx, test_idx, device, args.n_inferences, output_root)
            all_rows.extend(flatten_metrics(metrics, model_name, seed))

    if args.skip_aggregate:
        logging.info("Skipping CSV aggregation (--skip_aggregate); result caches are already written.")
    else:
        # Per-(model, seed, variable, metric) rows across all runs
        df_raw = pd.DataFrame(all_rows)
        df_raw.to_csv(output_root / "seed_variance_raw.csv", index=False)

        # Mean/std/count across seeds, per model/variable/metric
        df_summary = (
            df_raw.groupby(["model", "variable", "metric"])["value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        df_summary.to_csv(output_root / "seed_variance_summary.csv", index=False)

        logging.info("\n" + df_summary[df_summary.variable == "Global"].to_string(index=False))
        logging.info(f"Saved raw per-seed results to {output_root / 'seed_variance_raw.csv'}")
        logging.info(f"Saved mean/std summary to {output_root / 'seed_variance_summary.csv'}")
