"""Train the final model using best Optuna hyperparameters on the fixed train/test split."""

import argparse
import json
import logging
import types
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import config
from experiments.modelconfig import ablation_study
from experiments.pytorch_study import train_pytorch_single_split
from experiments.sklearn_study import train_sklearn_single_split
from models.registry import get_model_class
from utils.data import load_dataset
from utils.plotting import generate_animation
from utils.tuning import load_optuna_study, set_seed


def assemble_hyps(model_name: str, raw: dict) -> dict:
    """Re-structure flat Optuna params into the nested dict expected by train functions."""
    if model_name == "mastnet":
        strategy = raw.get("strategy", "random")
        radius = raw.get("sphere_mask_radius", 0.0) if strategy == "sphere" else 0.0
        width = raw.get("transect_mask_width", 0.0) if strategy == "transect" else 0.0
        masking_strategies = (
            ["random", "sphere"] if strategy == "sphere"
            else ["random", "transect"] if strategy == "transect"
            else ["random"]
        )
        return {
            "train": {
                "n_neighbours": raw["n_neighbours"],
                "batch_size": raw["batch_size"],
                "learning_rate": raw["lr"],
                "patience": 15,
                "n_epochs": 200,
                "mask_ratio": raw["mask_ratio"],
                "loss": "hetero",
                "optimizer": torch.optim.Adam,
                "masking": {
                    "sphere_mask_radius": radius,
                    "transect_mask_width": width,
                    "masking_strategies": masking_strategies,
                },
            },
            "model": {
                "d_model": raw["d_model"],
                "nhead": raw["nhead"],
                "nlayers": raw["nlayers"],
                "dim_feedforward": raw["dim_feedforward"],
                "dropout": raw["dropout"],
            },
        }
    elif model_name == "mlp":
        return {
            "train": {
                "batch_size": raw["batch_size"],
                "learning_rate": raw["lr"],
                "patience": 15,
                "n_epochs": 200,
                "mask_ratio": raw["mask_ratio"],
                "loss": "hetero",
                "optimizer": torch.optim.Adam,
            },
            "model": {
                "hidden_dims": raw["hidden_dims"],
                "dropout": raw["dropout"],
            },
        }
    elif model_name == "remasker":
        return {**raw, "max_epochs": 200}
    else:
        # sklearn models: params are passed directly as kwargs to model_class(**hyps)
        return raw


def build_pytorch_cfg(model_name, hyps, exp_id=247):
    """Build the base config for MaSTNeT and MLP, applying Optuna overrides."""
    cfg = ablation_study[f"exp{exp_id}"]["config"]
    if model_name != "mastnet":
        # MLP does not use graph setup
        return cfg

    cfg.dropout = hyps["model"]["dropout"]
    cfg.n_neighbours = hyps["train"]["n_neighbours"]
    cfg.mask_ratio = hyps["train"]["mask_ratio"]
    masking = hyps["train"].get("masking", {})
    cfg.masking_strategies = masking.get("masking_strategies", ["random"])
    cfg.sphere_mask_radius = masking.get("sphere_mask_radius", 0.0)
    cfg.sphere_mask_p = 1.0
    cfg.transect_mask_width = masking.get("transect_mask_width", 0.0)
    cfg.transect_mask_p = 1.0
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final model training with best Optuna params.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name: mastnet, mlp, knn, mice, missforest, remasker, ...")
    parser.add_argument("--n_inferences", type=int, default=50,
                        help="MC dropout passes for uncertainty estimation (PyTorch only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda",
                        help="PyTorch device (cuda or cpu)")
    parser.add_argument("--no_animations", action="store_true",
                        help="Skip generating MP4 animations (faster, saves disk space)")
    args = parser.parse_args()

    # Logging
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Setup
    model_name = args.model_name
    seed = args.seed
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path("output/testing") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset and fixed test-train split
    df = load_dataset()
    set_seed(seed)
    split = json.load(open(Path(config.output_dir_splits) / "test_train_split.json"))
    test_idx = np.array(split["test_idx"], dtype=int)
    train_idx = np.array(split["train_idx"], dtype=int)

    # Load best Optuna hyperparameters
    study = load_optuna_study(model_name)
    raw_params = study.best_trial.params
    logging.info(f"Best trial #{study.best_trial.number}: {raw_params}")
    hyps = assemble_hyps(model_name, raw_params)

    model_class = get_model_class(model_name)

    if isinstance(model_class, type) and issubclass(model_class, torch.nn.Module):
        logging.info("PyTorch model...")

        # Load MaSTNeT architecture config and apply Optuna overrides
        cfg = build_pytorch_cfg(model_name, hyps, exp_id=73)

        results, y_true, y_pred, uncertainty_pair, scaler_dict = train_pytorch_single_split(
            coords_raw=torch.tensor(df[config.coordinates].astype(float).to_numpy()),
            values_raw=torch.tensor(df[config.parameters].astype(float).to_numpy()),
            model_class=model_class,
            hyps=hyps,
            train_idx=train_idx,
            val_idx=test_idx,   # final run: use test set as early-stopping criterion
            test_idx=test_idx,
            model_name=model_name,
            split_path="final",
            trial_id=0,
            seed=seed,
            tuning_mode=False,
            device=device,
            do_dropout=True,
            n_inferences=args.n_inferences,
            save_model=True,
            cfg=cfg,
            output_dir=str(output_dir),
        )
        variance_aleatoric = uncertainty_pair[0] if uncertainty_pair is not None else None
        variance_epistemic = uncertainty_pair[1] if uncertainty_pair is not None else None

    else:
        logging.info("Sklearn model...")

        # sklearn / function-based / remasker path
        save_model = not isinstance(model_class, types.FunctionType)  # e.g. MissForest cannot be stored
        results, y_true, y_pred, scaler_dict = train_sklearn_single_split(
            df=df,
            model_class=model_class,
            hyps=hyps,
            test_idx=test_idx,
            train_idx=train_idx,
            val_idx=test_idx,   # final run: val == test
            model_name=model_name,
            split_path="final",
            trial_id=0,
            seed=seed,
            tuning_mode=False,
            save_model=save_model,
            output_dir=str(output_dir),
        )
        variance_aleatoric, variance_epistemic = None, None

    if y_pred is None or y_true is None:
        # Do not overwrite existing runs
        logging.info(f"Results for model{model_name} seed={seed} were loaded from cache; "
                     f"skipping CSV/animation regeneration (delete the cached JSON to force a rerun).")
    else:
        # Strip prepended coordinate columns if present (some sklearn paths return full X)
        if y_pred.shape[1] != len(config.parameters):
            coord_dim = y_pred.shape[1] - len(config.parameters)
            y_pred = y_pred[:, coord_dim:]

        # Save ground truth and predictions
        pd.DataFrame(y_pred, columns=config.parameters).to_csv(
            output_dir / f"model{model_name}_y_pred.csv", index=False
        )
        pd.DataFrame(y_true, columns=config.parameters).to_csv(
            output_dir / f"model{model_name}_y_true.csv", index=False
        )
        logging.info(f"Predictions saved to {output_dir}")

        # Plot animations of imputation and uncertainties if given
        if args.no_animations:
            logging.info("Skipping animations (--no_animations)")
        else:
            # Create imputed df
            df_imputed = pd.DataFrame(y_true, columns=config.parameters)
            df_imputed = df_imputed.combine_first(pd.DataFrame(y_pred, columns=config.parameters))
            df_imputed[config.coordinates] = df[config.coordinates].values

            # Per-parameter animations
            for param in config.parameters:
                generate_animation(
                    df_imputed, scaler_dict=scaler_dict, parameter=param,
                    save_as=str(output_dir / f"model{model_name}_{param}_imputation.mp4"),
                )

                if variance_aleatoric is not None:
                    df_var = pd.DataFrame(variance_aleatoric, columns=config.parameters)
                    df_var[config.coordinates] = df[config.coordinates].values
                    generate_animation(
                        df_var, scaler_dict=None, parameter=param,
                        save_as=str(output_dir / f"model{model_name}_{param}_aleatoric.mp4"),
                    )

                    df_var = pd.DataFrame(variance_epistemic, columns=config.parameters)
                    df_var[config.coordinates] = df[config.coordinates].values
                    generate_animation(
                        df_var, scaler_dict=None, parameter=param,
                        save_as=str(output_dir / f"model{model_name}_{param}_epistemic.mp4"),
                    )

    logging.info("Done.")
