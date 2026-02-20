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
import gc
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from tqdm import tqdm

from nn_utils.losses import build_loss, name_to_loss_spec
from nn_utils.trainer import Trainer, NeighbourAdapter, PointwiseAdapter
from nn_utils.early_stopping import EarlyStopping
from nn_utils.dataset import prepare_neighbourhood_loaders, load_dataset, prepare_pointwise_loaders
from utils.metrics import compute_metrics
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
    elif model_name == "mastnet":
        return {
            "train": {
                "n_neighbours": trial.suggest_int("n_neighbours", 2, 60),
                "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
                "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                "patience": 5,  # trial.suggest_int("patience", 3, 12),
                "n_epochs": 20,  # trial.suggest_int("epochs", 20, 80),
                "mask_ratio": trial.suggest_float("mask_ratio", 0.0, 0.99),
                "loss": trial.suggest_categorical("loss", ["mse", "hetero"]),
                "optimizer": torch.optim.Adam
            },
            "model": {
                "d_model": trial.suggest_categorical("d_model", [32, 64, 128, 256, 512]),
                "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
                "nlayers": trial.suggest_int("nlayers", 2, 8),
                "dim_feedforward": trial.suggest_categorical("dim_feedforward", [128, 256, 512, 1024]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            }
        }
    elif model_name == "mastnet_finetune":
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
    elif model_name == "mlp":
        return {"train": {
                    "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
                    "learning_rate": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                    "patience": 10,  # trial.suggest_int("patience", 3, 12),
                    "n_epochs": 80,  # trial.suggest_int("epochs", 20, 80),
                    "mask_ratio": trial.suggest_float("mask_ratio", 0.0, 0.99),
                    "loss": trial.suggest_categorical("loss", ["mse", "hetero"]),
                    "optimizer": torch.optim.Adam
                },
                "model": {
                    "hidden_dims": trial.suggest_categorical("hidden_dims", [
                        [64, 64],  # Narrow and shallow
                        [128, 64],  # Moderate
                        [256, 128],  # Wider
                        [128, 64, 32],  # Deeper and narrow
                        [256, 128, 64]  # Deeper and wider
                    ]),
                    "dropout": trial.suggest_float("dropout", 0.0, 0.4)
            }
                }
    else:
        raise NotImplementedError(f"Could not find model {model_name}.")


def get_hyp(hyps, *keys, default=None):
    """Safely retrieve nested hyperparameters from dict."""
    val = hyps
    for k in keys:
        val = val.get(k, default) if isinstance(val, dict) else default
    return val


def train_pytorch_single_split(coords_raw, values_raw, model_class, hyps, train_idx, val_idx, test_idx, model_name,
                               split_path, trial_id,
                               tuning_mode=True, optuna_callback=None, seed=42, device=torch.device("cpu"),
                               save_model=False, output_dir=None,
                               coords_only=False, do_dropout=False, n_inferences=1):
    """ Run model on one split and store results. """
    # Output dir and file
    model_outdir = Path(output_dir) if output_dir else Path(config.output_dir_tuning) / model_name
    os.makedirs(model_outdir, exist_ok=True)
    split_fname = PurePath(split_path).stem
    base_name = f"model{model_name}_split{split_fname}_trial{trial_id}"
    main_path = Path(model_outdir) / base_name
    json_fname = main_path.with_name(main_path.name + ".json")

    # Skip if file already exists
    if json_fname.exists():
        logging.info(f"Results already exist for {split_fname}. Skipping.")
        with open(json_fname) as f:
            results = json.load(f)
        if tuning_mode:
            return results, None, None, None
        else:
            return results, None, None, None, None

    # Set seeds
    generator = set_seed(seed)

    # Hyperparameters
    n_neighbours = get_hyp(hyps, "train", "n_neighbours", default=5)
    batch_size = get_hyp(hyps, "train", "batch_size", default=64)
    optimizer_class = get_hyp(hyps, "train", "optimizer", default=torch.optim.Adam)
    learning_rate = get_hyp(hyps, "train", "learning_rate", default=1e-3)
    patience = get_hyp(hyps, "train", "patience", default=5)
    n_epochs = get_hyp(hyps, "train", "n_epochs", default=20)
    mask_ratio = get_hyp(hyps, "train", "mask_ratio", default=0.8)
    model_hyps = get_hyp(hyps, "model", default={})
    loss_spec = name_to_loss_spec(get_hyp(hyps, "train", "loss", default="mse"))

    # Init results object
    results = TuningResult(split=split_fname, seed=seed, model=model_name, hyp_combo_id=trial_id, hyps=hyps)

    # Adapter and data loaders
    model_adapters = {"mastnet": NeighbourAdapter, "mlp": PointwiseAdapter}
    loader_funcs = {"mastnet": prepare_neighbourhood_loaders, "mlp": prepare_pointwise_loaders}

    if model_name not in model_adapters.keys():
        raise ValueError(f"Unknown model_name {model_name}")

    # Hyps for data loader
    loader_kwargs = dict(coords=coords_raw,
            values=values_raw,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            batch_size=batch_size,
            generator=generator)
    if model_name == "mastnet":
        loader_kwargs["n_neighbours"] = n_neighbours

    # Adapter and loader
    adapter = model_adapters[model_name]()
    full_loader, train_loader, val_loader, test_loader, scaler_dict, coord_dim, value_dim = (
        loader_funcs[model_name](**loader_kwargs))

    # Initialize model optimizer, loss and trainer
    st = time()
    model = model_class(**model_hyps).to(device)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    loss_fn = build_loss(loss_spec)
    trainer = Trainer(model=model, adapter=adapter, optimizer=optimizer, loss_fn=loss_fn, device=device, coords_only=coords_only)
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
        optuna_callback=optuna_callback,
        do_dropout=do_dropout
    )
    train_time = time() - strain

    # Full prediction/reconstruction
    if not tuning_mode:

        # Store predictions for all MC passes
        all_preds = []
        all_vars = []
        all_times = []

        for i in range(n_inferences):
            # Update seed
            set_seed(seed + i)

            # Prediction
            sval = time()
            full_pred, full_var = trainer.reconstruct_full_dataset(loader=full_loader, do_dropout=do_dropout, show_progress=(i==0))
            pred_time = time() - sval

            all_preds.append(full_pred.unsqueeze(0))
            all_vars.append(full_var.unsqueeze(0))
            all_times.append(pred_time)

        # Stack predictions (n_inferences, N, output_dim)
        all_preds = torch.cat(all_preds, dim=0)
        all_vars = torch.cat(all_vars, dim=0)

        # Compute uncertainties
        y_pred_mean = all_preds.mean(dim=0)  # Mean prediction
        epistemic_uncertainty = all_preds.var(dim=0, unbiased=False)
        aleatoric_uncertainty = all_vars.mean(dim=0)
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        total_uncertainty = total_uncertainty.detach().cpu().numpy()

        # Prediction time
        time_mean = np.mean(all_times)
        time_std = np.std(all_times)

        # True values
        if model_name == "mastnet":
            all_values = torch.cat([batch["query_features"] for batch in full_loader], dim=0)
        elif model_name == "mlp":
            all_values = torch.cat([batch["features"] for batch in full_loader], dim=0)
        else:
            raise ValueError(f"Unknown model_name {model_name}")

        # To CPU and numpy
        y_true = all_values.detach().cpu().numpy()
        y_pred = y_pred_mean.detach().cpu().numpy()
        epistemic_uncertainty = epistemic_uncertainty.detach().cpu().numpy()
        aleatoric_uncertainty = aleatoric_uncertainty.detach().cpu().numpy()

        # Loss plot
        plot_loss(history, close_plot=True, save_as=main_path.with_name(main_path.name + "_lossplot.png"))

        # Reconstruction  RMSE and plot
        reconstruction_rmse = np.sqrt(np.nanmean((y_pred - y_true) ** 2))
        recplot_fname = main_path.with_name(main_path.name + "_reconstruction_error.png")
        plot_simple_reconstruction_error(y_true, y_pred, save_as=recplot_fname, close=True)

        # Evaluate on validation and test set separately
        val_metrics = compute_metrics(y_true=y_true[val_idx], y_pred=y_pred[val_idx], var_names=config.parameters)
        test_metrics = compute_metrics(y_true=y_true[test_idx], y_pred=y_pred[test_idx], var_names=config.parameters)

        # Write results to results class
        results.test_metrics = test_metrics
        results.val_metrics = val_metrics

        results.pred_time = float(time_mean)
        results.pred_time_std = float(time_std)

        # results.reconstruction_rmse = reconstruction_rmse

        results.mean_epistemic_uncertainty = float(epistemic_uncertainty.mean())
        results.std_epistemic_uncertainty = float(epistemic_uncertainty.std())
        results.mean_aleatoric_uncertainty = float(aleatoric_uncertainty.mean())
        results.std_aleatoric_uncertainty = float(aleatoric_uncertainty.std())
        results.mean_total_uncertainty = float(total_uncertainty.mean())
        results.std_total_uncertainty = float(total_uncertainty.std())
    else:
        y_true, y_pred, full_var = None, None, None

    results.val_loss = trainer.best_val_loss
    results.train_time = train_time + def_time
    results.stop_epoch = early_stopper.epoch
    results.metrics_all = history["metrics"]
    results.metrics_last = history["metrics"][max(history["metrics"].keys())] if "metrics" in history and history["metrics"] else {}

    # Store results on disc
    results.save(json_fname, model=model if save_model else None)

    logging.info(f"Split {split_fname} finished, val_loss={trainer.best_val_loss:.8f}")

    # Clean up
    del full_loader, train_loader, val_loader, test_loader
    del model, trainer, optimizer, loss_fn

    if tuning_mode:
        return results, y_true, y_pred, scaler_dict
    else:
        return results, y_true, y_pred, [aleatoric_uncertainty, epistemic_uncertainty], scaler_dict


def optuna_objective(trial, model_name, output_dir):
    """ Optuna objective function: One trial runs on all 5 splits. """
    # Init torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load test indexes
    test_idx = np.array(json.load(open(f"{config.output_dir_splits}/test_train_split.json"))["test_idx"], dtype=int)

    # Load all split paths
    split_paths = sorted(glob.glob(os.path.join(config.output_dir_splits, "selected_splits/fold_*.json")))

    # Priors
    hyp_dict = suggest_hyperparameters(trial, model_name=model_name)
    n_epochs = hyp_dict["train"]["n_epochs"]

    coords_raw = torch.tensor(DATA[config.coordinates].astype(float).to_numpy())
    values_raw = torch.tensor(DATA[config.parameters].astype(float).to_numpy())

    # Training
    val_losses = []
    for split_i, split_path in enumerate(split_paths):
        logging.info(f"Training trial {trial.number} on split {split_path}")

        # Load split
        split = json.load(open(split_path))
        train_idx = np.array(split["train_idx"])
        val_idx = np.array(split["val_idx"])

        # Train
        results, _, _, _ = train_pytorch_single_split(
            coords_raw=coords_raw,
            values_raw=values_raw,
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
            device=device,
            output_dir=output_dir,
            coords_only=False
        )

        logging.info(f"Validation loss: {results.val_loss:.8f}")
        val_losses.append(results.val_loss)

        # Clean up
        del results
        torch.cuda.empty_cache()
        gc.collect()

    # Validation loss across all splits
    mean_val_loss = np.mean(val_losses)
    logging.info(f"Trial {trial.number} finished, mean_val_loss={mean_val_loss:.4f}")
    return mean_val_loss


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials to run")
    parser.add_argument("--model_name", type=str, default="mae", help="Model name")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory")
    args = parser.parse_args()

    # Setup logging
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Create output directory
    if args.output_dir == "":
        output_dir = Path(f"{config.output_dir_tuning}/{args.model_name}/")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up Optuna study
    if platform.system() == "Windows":
        storage = f"sqlite:///{str(output_dir)}/{args.model_name}_tuning.db"
    else:
        journal_path = f"{str(output_dir)}/{args.model_name}_tuning.log"
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
    study.optimize(partial(optuna_objective, model_name=args.model_name, output_dir=output_dir), n_trials=args.n_trials)

    # # Save results
    # logging.info("Storing OPTUNA study results...")
    # df_trials = study.trials_dataframe()
    # df_trials.to_csv(f"optuna_trials_{model_name}.csv", index=False)

    # logging.info("Best trial:", study.best_trial.params)

