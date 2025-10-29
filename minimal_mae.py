import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import sklearn
from time import time
import os
from pathlib import Path, PurePath
import joblib
import glob

import matplotlib
matplotlib.use("Qt5Agg")  # explicitly use Qt5
import matplotlib.pyplot as plt

import config
from oceanmae.dataset import random_feature_mask, prepare_mae_loaders, OceanMAEDataset, load_dataset, preprocess, \
    prepare_sklearn_data
from oceanmae.early_stopping import EarlyStopping
from oceanmae.losses import PhysicsLoss, MaskedMSELoss, HeteroscedasticLoss, build_loss
from oceanmae.model import OceanMAE
from oceanmae.trainer import Trainer
from utils.plotting import plot_loss, generate_animation, plot_simple_reconstruction_error
from utils.tuning import TuningResult, set_seed, get_hyperparameter_combinations, combine_csvs


# if __name__ == "__main__":
#     # Init torch device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Load dataset
#     df = load_dataset()
#
#     # Load split
#     split_path = "output/splits/fold_block_0_0.1_15.json"
#     model_name = "mae_rough"
#
#     with open(split_path, "r") as f:
#         fold = json.load(f)
#     train_idx = np.array(fold["train_idx"])
#     val_idx   = np.array(fold["val_idx"])
#     test_idx  = np.array(fold["test_idx"])
#
#     # Load model specs
#     model_specs = config.models[model_name]
#     model_class = model_specs["model"]
#     hyp_combos = get_hyperparameter_combinations(model_specs["hyps"])
#     hyp_combo = hyp_combos[0]
#
#     # Prepare loaders
#     generator = set_seed(0)
#     batch_size = 128
#     full_loader, train_loader, val_loader, test_loader, scaler_dict, coord_dim, value_dim = prepare_mae_loaders(
#         coords=torch.tensor(df[config.coordinates].astype(float).to_numpy()),
#         values=torch.tensor(df[config.parameters].astype(float).to_numpy()),
#         train_idx=train_idx,
#         val_idx=val_idx,
#         test_idx=test_idx,
#         batch_size=batch_size,
#         generator=generator
#     )
#
#     # Init model & loss
#     model = OceanMAE(coord_dim=coord_dim, value_dim=value_dim, d_model=128, nhead=8, nlayers=3, dim_feedforward=256, dropout=0.15).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
#     loss_fn =  MaskedMSELoss()
#     early_stopper = EarlyStopping(patience=10)
#
#     # Training
#     trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device)
#     history = trainer.fit(train_loader=train_loader, val_loader=val_loader, max_epochs=100, early_stopping=early_stopper, mask_ratio=0.5)
#     plot_loss(history)
#
#     # loss_per_epoch = {"train": {}, "val": {}}
#     # for epoch in range(1, 21):
#     #     train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, mask_ratio=0)
#     #     val_loss = evaluate(model, val_loader, loss_fn, device)
#     #     loss_per_epoch["train"][epoch] = train_loss
#     #     loss_per_epoch["val"][epoch] = val_loss
#     #     print(f"Epoch {epoch:03d}: Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f}")
#     # plot_loss(loss_per_epoch, close_plot=True)  # , save_as=loss_fname)
#
#     # Full reconstruction
#     coords_full, reconstruction, rvar = trainer.reconstruct_full_dataset(loader=full_loader, mc_dropout=False)
#     # coords_full, reconstruction, rvar = reconstruct_full_dataset(model=model, dataloader=full_loader, device=device, mc_dropout=False)
#     df_imputed = reconstruction.cpu().detach().numpy().copy()  # Prediction
#     df_imputed = pd.DataFrame(df_imputed, columns=config.parameters)  # Convert to dataframe
#     df_imputed[config.coordinates] = df[config.coordinates]  # Add coordinates
#     generate_animation(df_imputed, scaler_dict, parameter="P_TEMPERATURE", save_as="FULL_seed0_mse.mp4")
#

def run_tuning(model_name, split_path):
    # Init torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    df = load_dataset()

    # Load split
    # split_path = "output/splits/fold_block_0_0.1_15.json"
    # model_name = "mae_rough"
    with open(split_path, "r") as f:
        fold = json.load(f)
    train_idx = np.array(fold["train_idx"])
    val_idx   = np.array(fold["val_idx"])
    test_idx  = np.array(fold["test_idx"])

    # Load model specs
    model_specs = config.models[model_name]
    model_class = model_specs["model"]
    hyp_combos = get_hyperparameter_combinations(model_specs["hyps"])

    # Create output subdir
    model_outdir = Path(config.output_dir_tuning) / model_name
    os.makedirs(model_outdir, exist_ok=True)

    # Determine number of cores for joblib
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    use_joblib = not issubclass(model_class, torch.nn.Module)  # Only CPU models

    # Iterate over seeds and hyps
    def run_seed_hyps(seed, hyps, combo_id):
        # Output file
        split_fname = PurePath(split_path).stem
        base_name = f"split{split_fname}_model{model_name}_seed{seed}_hyps{combo_id}"
        main_path = Path(model_outdir) / base_name
        print(main_path)

        # Check if file already exists
        if os.path.isfile(main_path.with_suffix(".csv")):
            return

        # Set deterministic seeds
        generator = set_seed(seed)

        # Init results
        df_imputed = None
        res = TuningResult(split=split_fname, seed=seed, model=model_name, hyp_combo_id=combo_id, hyps=hyps)

        if issubclass(model_class, OceanMAE):
            # Get hyperparameters
            batch_size = hyps["train"]["batch_size"]
            model_hyps = hyps["model"]
            optimizer_class = hyps["train"]["optimizer"]
            learning_rate = hyps["train"]["learning_rate"]
            loss_class = hyps["train"]["loss"]
            patience = hyps["train"]["patience"]
            n_epochs = hyps["train"]["n_epochs"]
            mask_ratio = hyps["train"]["mask_ratio"]

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
            loss_fn = build_loss(loss_class)
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
                mask_ratio=mask_ratio)
            train_time = time() - strain

            # Plot loss
            loss_fname = main_path.with_name(main_path.name + "_lossplot.png")
            plot_loss(history, close_plot=True, save_as=loss_fname)

            # Full prediction/reconstruction
            sval = time()
            full_coords, full_pred, full_var = trainer.reconstruct_full_dataset(loader=full_loader, mc_dropout=False)
            df_imputed = full_pred.cpu().detach().numpy().copy()  # Prediction
            df_imputed = pd.DataFrame(df_imputed, columns=config.parameters)  # Convert to dataframe
            df_imputed[config.coordinates] = df[config.coordinates]  # Add coordinates
            pred_time = time() - sval

            # Visualize RECONSTRUCTION
            rec_fname = main_path.with_name(main_path.name + "_reconstruction_temperature.mp4")
            generate_animation(df_imputed, scaler_dict, parameter="P_TEMPERATURE", save_as=rec_fname)

            # Transform to numpy
            all_values = torch.cat([values for _, values, _ in full_loader], dim=0)
            y_true = all_values.detach().cpu().numpy()
            y_pred = full_pred.detach().cpu().numpy()

            # Reconstruction plot
            recplot_fname = main_path.with_name(main_path.name + "_reconstruction_error.png")
            plot_simple_reconstruction_error(y_true, y_pred, save_as=recplot_fname)

            # Evaluate on validation and test set separately
            val_rmse = np.sqrt(np.nanmean((y_true[val_idx] - y_pred[val_idx])**2))  # For model tuning
            test_rmse = np.sqrt(np.nanmean((y_true[test_idx] - y_pred[test_idx])**2))  # Not needed here, final model performance
            reconstruction_rmse = np.sqrt(np.nanmean((y_true - y_pred)**2))  # For representation quality

            # Write results to results class
            res.val_rmse = val_rmse
            res.test_rmse = test_rmse
            res.train_time = train_time + def_time
            res.pred_time = pred_time
            res.mean_aleatoric_uncertainty = float(full_var.mean())
            res.std_aleatoric_uncertainty = float(full_var.std())
            res.stop_epoch = early_stopper.epoch
            res.reconstruction_rmse = reconstruction_rmse

            # Final imputation: keep original where present, use prediction where NaN and where test/validation
            final_imputed = y_true.copy()
            final_imputed[test_idx] = y_pred[test_idx]
            final_imputed[val_idx] = y_pred[val_idx]
            final_imputed[np.isnan(final_imputed)] = y_pred[np.isnan(final_imputed)]

            # Prepare df for plotting
            df_imputed = pd.DataFrame(final_imputed, columns=config.parameters)
            df_imputed[config.coordinates] = df[config.coordinates]

            # Compute absolute error per cell
            ae_fname = main_path.with_name(main_path.name + "_imputation_ae_temperature.mp4")
            ae = np.abs(y_true - final_imputed)
            dfi = pd.DataFrame(ae, columns=config.parameters)
            dfi[config.coordinates] = df[config.coordinates]
            print(dfi.P_TEMPERATURE.mean(), dfi.P_TEMPERATURE.std())
            generate_animation(dfi, scaler_dict, parameter="P_TEMPERATURE", save_as=ae_fname)

        elif issubclass(model_class, sklearn.base.BaseEstimator):
            # Prepare data
            x_train, y_true, scaler_dict, coord_dim, value_dim = prepare_sklearn_data(df=df, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

            # Fit model
            st = time()
            imputer = model_class(**hyps)
            imputer.fit(x_train)
            train_time = time() - st

            # Predict
            st = time()
            pred = imputer.transform(x_train)
            pred_time = time() - st

            # Evaluate on validation set
            val_rmse = np.sqrt(np.nanmean((y_true[val_idx] - pred[:, coord_dim :][val_idx])**2))

            # Evaluate on test set (optional, not relevant in tuning)
            test_rmse = np.sqrt(np.nanmean((y_true[test_idx] - pred[:, coord_dim :][test_idx])**2))

            # Store results
            res.train_time = train_time
            res.pred_time = pred_time
            res.val_rmse = val_rmse
            res.test_rmse = test_rmse

            # Full prediction
            df_imputed = pred[:, coord_dim :].copy()  # Prediction
            df_imputed = pd.DataFrame(df_imputed, columns=config.parameters)  # Convert to dataframe
            df_imputed[config.coordinates] = df[config.coordinates]  # Add coordinates

            # AE per cell
            ae_fname = main_path.with_name(main_path.name + "_imputation_ae_temperature.mp4")
            ae = np.abs(y_true - pred[:, coord_dim:])
            dfi = pd.DataFrame(ae, columns=config.parameters)
            dfi[config.coordinates] = df[config.coordinates]
            print(dfi.P_TEMPERATURE.mean(), dfi.P_TEMPERATURE.std())
            generate_animation(dfi, scaler_dict, parameter="P_TEMPERATURE", save_as=ae_fname)

        # Store results on disc
        res.save(main_path.with_suffix(".csv"))

        # Animated plot
        if df_imputed is not None:
            imputed_fname = main_path.with_name(main_path.name + "_imputation_temperature.mp4")
            generate_animation(df_imputed, scaler_dict, parameter="P_TEMPERATURE", save_as=imputed_fname)

    # Parallelize seeds × hyperparameters
    tasks = [(seed, hyps, combo_id) for combo_id, hyps in enumerate(hyp_combos) for seed in config.tuning_seeds]

    if use_joblib:
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(run_seed_hyps)(seed, hyps, combo_id) for seed, hyps, combo_id in tasks)
    else:
        # GPU model: Run sequentially (one process per GPU)
        for seed, hyps, combo_id in tasks:
            run_seed_hyps(seed, hyps, combo_id)


# if __name__ == '__main__':
#     for fname in glob.glob("output/splits/*.json"):
#         print(fname)
#         run_tuning("mean", fname)
#
#         d = combine_csvs("output/tuning/mean/", out_name="tuning_mean.csv", remove_files=True)
#
#     # run_tuning("mae_rough", "C:/Users/yvjennig/PycharmProjects/github/time_imputation/output/splits/fold_block_0_0.1_15.json")
