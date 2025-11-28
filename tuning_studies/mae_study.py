import glob
from pathlib import Path, PurePath
import optuna
import pandas as pd
import torch
import numpy as np
import json
from time import time
import os
import argparse
import logging

from oceanmae.losses import build_loss, name_to_loss_spec
from oceanmae.trainer import Trainer
from oceanmae.early_stopping import EarlyStopping
from oceanmae.dataset import prepare_mae_loaders, load_dataset
from models.mae import OceanMAE
from utils.tuning import make_optuna_callback

import config
from utils.plotting import plot_loss, plot_simple_reconstruction_error
from utils.tuning import set_seed, TuningResult


# --- Logging ------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def suggest_hyperparameters(trial):
    return {
            "train": {
                "batch_size": trial.suggest_categorical("batch_size", [128, 512, 1024]),
                "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                "patience": trial.suggest_int("patience", 3, 12),
                "n_epochs": trial.suggest_int("epochs", 20, 80),
                "mask_ratio": trial.suggest_float("mask_ratio", 0.0, 0.9),
                "loss": trial.suggest_categorical("loss", ["mse", "hetero"]),
                "optimizer": torch.optim.Adam
            },
            "model": {
                "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
                "nhead": trial.suggest_categorical("nhead", [2, 4]),
                "nlayers": trial.suggest_int("nlayers", 2, 6),
                "dim_feedforward": trial.suggest_categorical("dim_feedforward", [128, 256, 512]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            }
        }


def train_mae_single_split(df, model_class, hyps, train_idx, val_idx, test_idx, model_name,
                           split_path, trial_id, optuna_callback=None, seed=42, device=torch.device("cpu")):
    # Create output subdir
    model_outdir = Path(config.output_dir_tuning) / model_name
    os.makedirs(model_outdir, exist_ok=True)

    # Output file
    split_fname = PurePath(split_path).stem
    base_name = f"model{model_name}_split{split_fname}_trial{trial_id}"
    main_path = Path(model_outdir) / base_name
    csv_fname = main_path.with_name(main_path.name + ".csv")

    # Check if file already exists
    if csv_fname.exists():
        logging.info(f"Results already exist for {split_fname}. Skipping.")
        results = pd.read_csv(csv_fname)
        return results, None, None

    # Set deterministic seeds
    generator = set_seed(seed)

    # Get hyperparameters
    batch_size = hyps["train"]["batch_size"]
    model_hyps = hyps["model"]
    optimizer_class = hyps["train"]["optimizer"]
    learning_rate = hyps["train"]["learning_rate"]
    patience = hyps["train"]["patience"]
    n_epochs = hyps["train"]["n_epochs"]
    mask_ratio = hyps["train"]["mask_ratio"]

    # Get Loss specification
    loss_spec = name_to_loss_spec(loss_name=hyps["train"]["loss"])

    # Init results object
    results = TuningResult(split=split_fname, seed=seed, model=model_name, hyp_combo_id=trial_id, hyps=hyps)

    # Prepare data
    full_loader, train_loader, val_loader, test_loader, scaler_dict, coord_dim, value_dim = prepare_mae_loaders(
        coords=torch.tensor(df[config.coordinates].astype(float).to_numpy()),
        values=torch.tensor(df[config.parameters].astype(float).to_numpy()),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=batch_size,
        generator=generator
    )

    # Initialize model and trainer
    st = time()
    model = model_class(**model_hyps).to(device)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    loss_fn = build_loss(loss_spec)
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device)
    early_stopper = EarlyStopping(patience=patience)
    def_time = time() - st

    # Train model
    strain = time()
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=n_epochs,
        early_stopping=early_stopper,
        mask_ratio=mask_ratio,
        optuna_callback=optuna_callback)
    train_time = time() - strain

    # Loss plot
    plot_loss(history, close_plot=True, save_as=main_path.with_name(main_path.name + "_lossplot.png"))

    # Full prediction/reconstruction
    sval = time()
    full_coords, full_pred, full_var = trainer.reconstruct_full_dataset(loader=full_loader, mc_dropout=False)
    df_imputed = pd.DataFrame(full_pred.cpu().detach().numpy().copy(), columns=config.parameters)  # Prediction
    df_imputed[config.coordinates] = df[config.coordinates]  # Add coordinates
    pred_time = time() - sval

    # Transform to numpy
    all_values = torch.cat([values for _, values, _ in full_loader], dim=0)
    y_true = all_values.detach().cpu().numpy()
    y_pred = full_pred.detach().cpu().numpy()

    # Reconstruction plot
    recplot_fname = main_path.with_name(main_path.name + "_reconstruction_error.png")
    plot_simple_reconstruction_error(y_true, y_pred, save_as=recplot_fname, close=True)

    # Evaluate on validation and test set separately
    val_rmse = np.sqrt(np.nanmean((y_true[val_idx] - y_pred[val_idx]) ** 2))  # For model tuning
    test_rmse = np.sqrt(
        np.nanmean((y_true[test_idx] - y_pred[test_idx]) ** 2))  # Not needed here, final model performance
    reconstruction_rmse = np.sqrt(np.nanmean((y_true - y_pred) ** 2))  # For representation quality

    # Write results to results class
    results.val_rmse = val_rmse
    results.test_rmse = test_rmse
    results.train_time = train_time + def_time
    results.pred_time = pred_time
    results.mean_aleatoric_uncertainty = float(full_var.mean())
    results.std_aleatoric_uncertainty = float(full_var.std())
    results.stop_epoch = early_stopper.epoch
    results.reconstruction_rmse = reconstruction_rmse
    results.metrics_all = history["metrics"]
    results.metrics_last = history["metrics"][max(history["metrics"].keys())] if "metrics" in history and history[
        "metrics"] else {}

    # Store results on disc
    results.save(csv_fname)

    logging.info(f"Split {split_fname} finished, val_rmse={val_rmse:.8f}")

    return results, y_true, y_pred


def optuna_objective(trial):
    """ Optuna objective function: One training run on one split. """
    # Init torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test indexes
    test_idx = np.array(json.load(open(f"{config.output_dir_splits}/test_train_split.json"))["test_idx"], dtype=int)

    # Load dataset
    df = load_dataset()

    # Load all split paths
    split_paths = sorted(glob.glob(os.path.join(config.output_dir_splits, "selected_splits/fold_*.json")))

    # Priors
    hyp_dict = suggest_hyperparameters(trial)
    n_epochs = hyp_dict["train"]["n_epochs"]

    # Training (fixed seed)
    val_losses = []
    for split_i, split_path in enumerate(split_paths):
        logging.info(f"Training trial {trial.number} on split {split_path}")

        # Load split
        split = json.load(open(split_path))
        train_idx = np.array(split["train_idx"])
        val_idx = np.array(split["val_idx"])

        # Train
        results, _, _ = train_mae_single_split(
            df=df,
            model_class=OceanMAE,
            hyps=hyp_dict,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            model_name="mae_optuna",
            split_path=split_path,
            trial_id=trial.number,
            optuna_callback=make_optuna_callback(trial, split_i, n_epochs),
            seed=42 + int(trial.number) + split_i,
            device=device
        )

        logging.info(f"Validation RMSE: {results.val_rmse:.8f}")
        val_losses.append(results.val_rmse)

    # Validation RMSE across all splits
    mean_val_rmse = np.mean(val_losses)
    logging.info(f"Trial {trial.number} finished, mean_val_rmse={mean_val_rmse:.4f}")
    return mean_val_rmse


if __name__ == "__main__":
    model_name = "mae"

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials to run")
    parser.add_argument("--db", type=str, default=f"sqlite:///{config.output_dir_tuning}/tuning.db", help="Database path")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Set up Optuna study
    sampler = optuna.samplers.TPESampler(n_startup_trials=20,  # More initial random exploration
                                         multivariate=True)  # Learn joint distributions
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(study_name=f"{model_name}_tuning",
                                direction="minimize",
                                sampler=sampler,
                                pruner=None,
                                storage=args.db,
                                load_if_exists=True)

    logging.info("Starting OPTUNA study...")
    study.optimize(optuna_objective, n_trials=args.n_trials)  # , n_jobs=1)

    # Save results
    logging.info("Storing OPTUNA study results...")
    df_trials = study.trials_dataframe()
    df_trials.to_csv("optuna_trials_mae.csv", index=False)

    logging.info("Best trial:", study.best_trial.params)
