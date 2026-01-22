import glob
import platform
from functools import partial
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

from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from nn_utils.losses import build_loss, name_to_loss_spec
from nn_utils.trainer import Trainer
from nn_utils.early_stopping import EarlyStopping
from nn_utils.dataset import prepare_mae_loaders, load_dataset
from utils.tuning import make_optuna_callback, get_model_class

import config
from utils.plotting import plot_loss, plot_simple_reconstruction_error
from utils.tuning import set_seed, TuningResult


# Load data globally
DATA = load_dataset()


def suggest_hyperparameters(trial, model_name="mae"):
    """ Suggest hyperparameters for the masked auto encoder. """
    if model_name == "unet":
        return {
                "train": {
                    "batch_size": 128,  # trial.suggest_categorical("batch_size", [128, 512, 1024]),
                    "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                    "patience": 5,  # trial.suggest_int("patience", 3, 12),
                    "n_epochs": 20,  # trial.suggest_int("epochs", 20, 80),
                    "mask_ratio": trial.suggest_float("mask_ratio", 0.0, 0.9),
                    "loss": trial.suggest_categorical("loss", ["mse", "hetero"]),
                    "optimizer": torch.optim.Adam
                },
                "model": {
                    "base_channels": trial.suggest_categorical("base_channels", [32, 64, 128, 256]),
                    "num_blocks": trial.suggest_int("num_blocks", 2, 5),
                    "dropout": trial.suggest_float("dropout", 0.0, 0.4)
                }
            }
    elif model_name == "mae":
        return {
            "train": {
                "batch_size": trial.suggest_categorical("batch_size", [128, 512, 1024]),
                "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                "patience": 5,  # trial.suggest_int("patience", 3, 12),
                "n_epochs": 20,  # trial.suggest_int("epochs", 20, 80),
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
    elif model_name == "mae_finetune":
        n_epochs = 20

        # Test decreasing mask_ratio as well as higher mask ratios
        use_decreasing = trial.suggest_categorical("use_decreasing", [True, False])
        if use_decreasing:
            start_ratio = trial.suggest_float("mask_start", 0.90, 1.0)
            end_ratio = trial.suggest_float("mask_end", 0.80, 0.95)
            mask_ratio = lambda epoch: start_ratio - (epoch / n_epochs) * (start_ratio - end_ratio)
        else:
            mask_ratio = trial.suggest_float("mask_ratio", 0.85, 1.0)

        return {
            "train": {
                "batch_size": 128,
                "learning_rate": 9.570854918884402e-05,
                "patience": 5,
                "n_epochs": n_epochs,
                "mask_ratio": mask_ratio,
                "loss": "hetero",
                "optimizer": torch.optim.Adam
            },
            "model": {
                "d_model": 128,
                "nhead": 4,
                "nlayers": 6,
                "dim_feedforward": 128,
                "dropout": 0.007883109330264194,
            }
        }
    else:
        raise NotImplementedError(f"Could not find model {model_name}.")


def train_mae_single_split(df, model_class, hyps, train_idx, val_idx, test_idx, model_name, split_path, trial_id,
                           tuning_mode=True, optuna_callback=None, seed=42, device=torch.device("cpu")):
    """ Run model on one split and store results. """
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

    if not tuning_mode:
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

        # Loss plot
        plot_loss(history, close_plot=True, save_as=main_path.with_name(main_path.name + "_lossplot.png"))

        # Reconstruction plot
        recplot_fname = main_path.with_name(main_path.name + "_reconstruction_error.png")
        plot_simple_reconstruction_error(y_true, y_pred, save_as=recplot_fname, close=True)

        # Evaluate on validation and test set separately
        val_rmse = np.sqrt(np.nanmean((y_true[val_idx] - y_pred[val_idx]) ** 2))  # For model tuning
        test_rmse = np.sqrt(
            np.nanmean((y_true[test_idx] - y_pred[test_idx]) ** 2))  # Not needed here, final model performance
        reconstruction_rmse = np.sqrt(np.nanmean((y_true - y_pred) ** 2))  # For representation quality

        # Write results to results class
        results.test_rmse = test_rmse
        results.pred_time = pred_time
        results.mean_aleatoric_uncertainty = float(full_var.mean())
        results.std_aleatoric_uncertainty = float(full_var.std())
        results.reconstruction_rmse = reconstruction_rmse
    else:
        val_rmse = trainer.best_val_loss
        y_true = None
        y_pred = None

    results.val_rmse = val_rmse
    results.train_time = train_time + def_time
    results.stop_epoch = early_stopper.epoch
    results.metrics_all = history["metrics"]
    results.metrics_last = history["metrics"][max(history["metrics"].keys())] if "metrics" in history and history[
        "metrics"] else {}

    # Store results on disc
    results.save(csv_fname)

    logging.info(f"Split {split_fname} finished, val_rmse={val_rmse:.8f}")

    return results, y_true, y_pred


def optuna_objective(trial, model_name):
    """ Optuna objective function: One trial runs on all 5 splits. """
    # Init torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test indexes
    test_idx = np.array(json.load(open(f"{config.output_dir_splits}/test_train_split.json"))["test_idx"], dtype=int)

    # Load all split paths
    split_paths = sorted(glob.glob(os.path.join(config.output_dir_splits, "selected_splits/fold_*.json")))

    # Priors
    hyp_dict = suggest_hyperparameters(trial, model_name=model_name)
    n_epochs = hyp_dict["train"]["n_epochs"]

    # Training
    val_losses = []
    for split_i, split_path in enumerate(split_paths):
        logging.info(f"Training trial {trial.number} on split {split_path}")

        # Load split
        split = json.load(open(split_path))
        train_idx = np.array(split["train_idx"])
        val_idx = np.array(split["val_idx"])

        # Train
        results, _, _ = train_mae_single_split(
            df=DATA,
            model_class=get_model_class(model_name),
            hyps=hyp_dict,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            model_name=model_name,
            split_path=split_path,
            trial_id=trial.number,
            optuna_callback=None,  # make_optuna_callback(trial, split_i, n_epochs),
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
    model_name = "mae"  # @todo not needed anymore

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials to run")
    parser.add_argument("--model_name", type=str, default="mae", help="Model name")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Set up Optuna study
    if platform.system() == "Windows":
        storage = f"sqlite:///{config.output_dir_tuning}/{args.model_name}/{args.model_name}_tuning.db"
    else:
        journal_path = f"{config.output_dir_tuning}/{args.model_name}/{args.model_name}_tuning.log"
        storage = JournalStorage(JournalFileBackend(journal_path))

    sampler = optuna.samplers.TPESampler(n_startup_trials=20,  # More initial random exploration
                                         multivariate=True)  # Learn joint distributions
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(study_name=f"{args.model_name}_tuning",
                                direction="minimize",
                                sampler=sampler,
                                pruner=None,
                                storage=storage,
                                load_if_exists=True)

    logging.info("Starting OPTUNA study...")
    study.optimize(partial(optuna_objective, model_name=args.model_name), n_trials=args.n_trials)

    # # Save results
    # logging.info("Storing OPTUNA study results...")
    # df_trials = study.trials_dataframe()
    # df_trials.to_csv(f"optuna_trials_{model_name}.csv", index=False)

    # logging.info("Best trial:", study.best_trial.params)

