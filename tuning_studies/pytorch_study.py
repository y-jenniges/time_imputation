import glob
import platform
from functools import partial
from itertools import chain
from pathlib import Path, PurePath
import optuna
import torch
import numpy as np
import json
from time import time
import argparse
import logging
import gc
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import psutil
import os

from nn_utils.losses import build_loss, name_to_loss_spec
from nn_utils.trainer import Trainer, NeighbourAdapter, PointwiseAdapter, GraphProvider
from nn_utils.early_stopping import EarlyStopping
from nn_utils.dataset import prepare_neighbourhood_loaders, load_dataset, prepare_pointwise_loaders, \
    prepare_learned_neighbourhood_loaders, LearnedNeighbourDataset
from utils.metrics import compute_metrics
from utils.tuning import get_model_class

import config
from utils.plotting import plot_loss, plot_simple_reconstruction_error
from utils.tuning import set_seed, TuningResult


# Load data globally
DATA = load_dataset()


def ram():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3

print("RAM after loading DATA:", ram())

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
        loss = trial.suggest_categorical("loss", ["mse", "hetero", "physics_hetero"])
        lambda_smooth = trial.suggest_float("lambda_smooth", 1e-4, 1e-3, log=True) if loss == "physics_hetero" else None
        return {
            "train": {
                "n_neighbours": trial.suggest_int("n_neighbours", 2, 60),
                "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
                "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                "patience": 5,  # trial.suggest_int("patience", 3, 12),
                "n_epochs": 1,  #20,  # trial.suggest_int("epochs", 20, 80),
                "mask_ratio": trial.suggest_float("mask_ratio", 0.0, 0.99),
                "loss": loss,
                "lambda_smooth": lambda_smooth,
                "optimizer": torch.optim.Adam
            },
            "model": {
                "d_model": trial.suggest_categorical("d_model", [32, 64, 128, 256, 512]),
                "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
                "nlayers": trial.suggest_int("nlayers", 2, 8),
                "dim_feedforward": trial.suggest_categorical("dim_feedforward", [128, 256, 512, 1024]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.4),
                # "pos_hidden_dim": trial.suggest_categorical("pos_hidden_dim", [32, 64, 128, 256, 512]),
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
        loss = trial.suggest_categorical("loss", ["mse", "hetero"])
        lambda_smooth = trial.suggest_float("lambda_smooth", 1e-4, 1e-3, log=True) if loss == "physics_hetero" else None
        
        return {"train": {
                    "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
                    "learning_rate": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                    "patience": 10,  # trial.suggest_int("patience", 3, 12),
                    "n_epochs": 80,  # trial.suggest_int("epochs", 20, 80),
                    "mask_ratio": trial.suggest_float("mask_ratio", 0.0, 0.99),
                    "loss":loss,
                    "lambda_smooth": lambda_smooth,
                    "optimizer": torch.optim.Adam
                },
                "model": {
                    "hidden_dims": trial.suggest_categorical("hidden_dims", [
                        [64, 64],  # Narrow and shallow
                        [128, 64],  # Moderate
                        [256, 128],  # Wider
                        [128, 64, 32],  # Deeper and narrow
                        [256, 128, 64],  # Deeper and wider

                        # Slightly larger / deeper options
                        [256, 128, 64, 32],  # Deeper
                        [512, 256, 128],  # Wider
                        [512, 256, 128, 64],  # Deep + wide
                        [256, 256, 128, 64],  # Wide first layers
                        [128, 128, 64, 32, 16],  # Very deep narrow
                    ]),
                    "dropout": trial.suggest_float("dropout", 0.0, 0.4)
            }
                }
    elif model_name == "ann-att":
        return {
            "train": {
                "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
                "learning_rate": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                "patience": 10,  # trial.suggest_int("patience", 3, 12),
                "n_epochs": 80,  # trial.suggest_int("epochs", 20, 80),
                "mask_ratio": trial.suggest_float("mask_ratio", 0.0, 0.99),
                "loss": trial.suggest_categorical("loss", ["mse", "hetero"]),
                "optimizer": torch.optim.Adam
            },
            "model": {
                "in_dim": 17,
                "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64, 128, 256]),
                "out_dim": 6
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
                               do_dropout=False, n_inferences=1,
                               cfg=None):
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
    lambda_smooth = get_hyp(hyps, "train", "lambda_smooth", default=None)
    loss_spec = name_to_loss_spec(get_hyp(hyps, "train", "loss", default="mse"))

    # Init results object
    results = TuningResult(split=split_fname, seed=seed, model=model_name, hyp_combo_id=trial_id, hyps=hyps)

    # Adapter and data loaders
    model_adapters = {"mastnet": NeighbourAdapter, "mlp": PointwiseAdapter, "ann_att": PointwiseAdapter}
    loader_funcs = {"mastnet": prepare_learned_neighbourhood_loaders, "mlp": prepare_pointwise_loaders, "ann_att": prepare_pointwise_loaders}

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
        graph_provider = GraphProvider(n_neighbours=n_neighbours, update_every=5,
                                       graph_mode= cfg.graph_mode,
                                       graph_space=cfg.graph_space,
                                       graph_metric=cfg.graph_metric,
                                       fill_strategy=cfg.fill_strategy,
                                       test_idx=test_idx, val_idx=val_idx)
        loader_kwargs["n_neighbours"] = n_neighbours
        loader_kwargs["graph_provider"] = graph_provider
    else:
        graph_provider = None

    # Adapter and loader
    adapter = model_adapters[model_name]()
    full_loader, train_loader, val_loader, test_loader, scaler_dict, coord_dim, value_dim, dists, full_coords, full_values, full_mask = (
        loader_funcs[model_name](**loader_kwargs))

    # Adding global mean data and cfg to mastnet
    if model_name == "mastnet":
        total_sum, total_count = 0.0, 0.0
        for batch in train_loader:
            feats = batch["query_features"]
            mask = batch["query_mask"].float()

            total_sum += feats.nansum(dim=0)
            total_count += mask.sum(dim=0)

        global_means = total_sum / total_count
        model_hyps["global_means"] = global_means
        model_hyps["cfg"] = cfg
        logging.info(f"Global means: {global_means}")
    else:
        global_means = torch.ones(value_dim)

    if dists is not None and lambda_smooth is not None:
        sigma = np.median(dists)
        loss_spec["kwargs"]["sigma"] = sigma
        loss_spec["kwargs"]["lambda_smooth"] = lambda_smooth

    logging.info(f"RAM after loader init: {ram()}")

    # Initialize model optimizer, loss and trainer
    st = time()
    model = model_class(**model_hyps).to(device)
    loss_fn = build_loss(loss_spec)
    optimizer = optimizer_class(chain(model.parameters(), loss_fn.parameters()), lr=learning_rate)
    trainer = Trainer(model=model, adapter=adapter, optimizer=optimizer, loss_fn=loss_fn, device=device,
                      graph_provider=graph_provider, full_coords=full_coords, full_values=full_values,
                      full_mask=full_mask, global_means=global_means, cfg=cfg)
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
        full_metrics=True, # not tuning_mode   @todo maybe remove arg?
    )
    train_time = time() - strain

    logging.info(f"RAM after fit: {ram()}")

    # Save model and history
    path = model_outdir / f"{model_name}_pytorch.pt"
    torch.save(model.state_dict(), path)
    json.dump(history, open(model_outdir / f"{model_name}_history.json", "w"))

    # Full prediction/reconstruction
    aleatoric_uncertainty, epistemic_uncertainty = np.nan, np.nan
    if not tuning_mode:
        # Loss plot
        plot_loss(history, close_plot=True, save_as=main_path.with_name(main_path.name + "_lossplot.png"))

        # Store predictions for all MC passes
        all_preds = []
        all_vars = []
        all_times = []

        for i in range(n_inferences):
            logging.info(f"Inference: {i:3}")

            # Update seed
            set_seed(seed + i)

            # Prediction
            sval = time()
            full_pred, full_var = trainer.reconstruct_full_dataset(loader=full_loader, do_dropout=do_dropout, show_progress=(i==0))
            pred_time = time() - sval

            all_preds.append(full_pred.unsqueeze(0))
            all_times.append(pred_time)

            if full_var is not None:
                all_vars.append(full_var.unsqueeze(0))

        # Stack predictions (n_inferences, N, output_dim)
        all_preds = torch.cat(all_preds, dim=0)
        all_vars = torch.cat(all_vars, dim=0) if all_vars != [] else None

        # Compute uncertainties
        y_pred_mean = all_preds.mean(dim=0)  # Mean prediction
        epistemic_uncertainty = all_preds.var(dim=0, unbiased=False)
        aleatoric_uncertainty = all_vars.mean(dim=0) if all_vars != [] else None
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty if aleatoric_uncertainty is not None else epistemic_uncertainty
        total_uncertainty = total_uncertainty.detach().cpu().numpy()

        # Prediction time
        time_mean = np.mean(all_times)
        time_std = np.std(all_times)

        # True values
        if model_name == "mastnet":
            all_values = torch.cat([batch["query_features"] for batch in full_loader], dim=0)
        elif model_name == "mlp" or model_name == "ann_att":
            all_values = torch.cat([batch["features"] for batch in full_loader], dim=0)
        else:
            raise ValueError(f"Unknown model_name {model_name}")

        # To CPU and numpy
        logging.info("cpu to np")
        y_true = all_values.detach().cpu().numpy()
        y_pred = y_pred_mean.detach().cpu().numpy()
        epistemic_uncertainty = epistemic_uncertainty.detach().cpu().numpy()
        aleatoric_uncertainty = aleatoric_uncertainty.detach().cpu().numpy() if aleatoric_uncertainty is not None else None
        all_preds_np = all_preds.cpu().numpy()
        all_vars_np = all_vars.cpu().numpy() if all_vars != [] else None

        # Store inferences
        np.save(main_path.with_name(main_path.name + "_all_preds.npy"), all_preds_np)

        if all_vars_np is not None:
            np.save(main_path.with_name(main_path.name + "_all_vars.npy"), all_vars_np)

        # Reconstruction  RMSE and plot
        logging.info("plotting")
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

        results.reconstruction_rmse = reconstruction_rmse

        results.mean_epistemic_uncertainty = float(epistemic_uncertainty.mean())
        results.std_epistemic_uncertainty = float(epistemic_uncertainty.std())
        results.mean_aleatoric_uncertainty = float(aleatoric_uncertainty.mean()) if aleatoric_uncertainty is not None else None
        results.std_aleatoric_uncertainty = float(aleatoric_uncertainty.std()) if aleatoric_uncertainty is not None else None
        results.mean_total_uncertainty = float(total_uncertainty.mean())
        results.std_total_uncertainty = float(total_uncertainty.std())

        results.scalers = scaler_dict
    else:
        y_true, y_pred, full_var = None, None, None

    results.val_rmse = history["metrics"][early_stopper.best_epoch]["Global"]["RMSE"]
    results.train_time = train_time + def_time
    results.stop_epoch = early_stopper.epoch
    results.metrics_all = history["metrics"]
    results.metrics_last = history["metrics"][max(history["metrics"].keys())] if "metrics" in history and history["metrics"] else {}

    if graph_provider is not None:
        results.graph_history = graph_provider.history

    # Store results on disc
    results.save(json_fname, model=model if save_model else None)

    logging.info(f"Split {split_fname} finished, val_loss={trainer.best_val_loss:.8f}, val_rmse={results.val_rmse:.8f}")

    # Clean up
    del full_loader, train_loader, val_loader, test_loader
    del model, trainer, optimizer, loss_fn

    logging.info(f"RAM after cleanup: {ram()}")

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

    coords_raw = torch.from_numpy(DATA[config.coordinates].astype(float).to_numpy())
    values_raw = torch.from_numpy(DATA[config.parameters].astype(float).to_numpy())

    # Training
    val_rmses = []
    for split_i, split_path in enumerate(split_paths):
        logging.info(f"RAM before split: {ram()}")
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
            output_dir=output_dir
        )

        logging.info(f"Validation loss: {results.val_rmse:.8f}")
        val_rmses.append(results.val_rmse)

        # Clean up
        del results
        torch.cuda.empty_cache()
        gc.collect()

        logging.info(f"RAM after split: {ram()}")

    # Validation loss across all splits
    mean_val_rmse = np.mean(val_rmses)
    logging.info(f"Trial {trial.number} finished, mean_val_rmse={mean_val_rmse:.4f}")
    return mean_val_rmse


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

