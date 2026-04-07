from pathlib import Path
import torch
import json
import numpy as np
import config
import pandas as pd
import logging
import argparse
from permetrics import RegressionMetric

import config
from nn_utils.dataset import load_dataset, preprocess
from prepare_and_split.generate_splits import generate_test_set
from tuning_studies.pytorch_study import train_pytorch_single_split
from utils.plotting import generate_animation
from utils.tuning import get_model_class, load_optuna_study, set_seed
from tuning_studies.modelconfig import ablation_study


if __name__ == "__main__":
    GLOBAL = False

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=int, default=0, help="experiment id")
    args = parser.parse_args()

    if GLOBAL:
        df = load_dataset("output/gridding/df_20y_global.csv")

        if Path.is_file(Path(f"{config.output_dir_splits}/GLOBAL_test_train_split.json")):
            split = json.load(open(f"{config.output_dir_splits}/GLOBAL_test_train_split.json"))
            test_idx = np.array(split["test_idx"], dtype=int)
            train_idx = np.array(split["train_idx"], dtype=int)
        else:
            # Global block test dataset
            test_idx, train_idx, df["BLOCK_ID"], test_coverages, global_coverages = (
                generate_test_set(df,
                                  latlon_step=15,
                                  depths_steps=[0, 200, 1000, 12000],
                                  test_fraction=config.test_fraction,
                                  parameters=config.parameters,
                                  tolerance=0.05,
                                  max_iter=100, seed=42,
                                  save_as=Path(config.output_dir_splits) / "GLOBAL_test_train_split.json"))
    else:
        # Load test-train split
        # split = json.load(open(f"{config.output_dir_splits}/test_train_split.json"))  # Test-train
        # test_idx = np.array(split["test_idx"], dtype=int)

        # Load train-val split (15° blocks)
        split = json.load(open(f"{config.output_dir_splits}/selected_splits/fold_block_15.json"))
        test_idx = np.array(split["val_idx"], dtype=int)
        train_idx = np.array(split["train_idx"], dtype=int)

        # Load dataset
        df = load_dataset()

    # Load study
    model_name = "mastnet"
    model_class = get_model_class(model_name)
    device = torch.device("cuda")

    exp_id = f"exp{args.exp_id}"
    cfg = ablation_study[exp_id]["config"]
    output_dir = f"{config.output_dir_tuning}/{model_name}/architecture_ablation/{exp_id}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(f"{output_dir}/{model_name}_{exp_id}.log")])


    hyps = {"train": {
                "n_neighbours": cfg.n_neighbours,
                "batch_size": 128,
                "learning_rate": 1e-4,
                "patience": 10,
                "n_epochs": 100,
                "mask_ratio": cfg.mask_ratio,
                "loss": cfg.loss_name,
                "optimizer": torch.optim.Adam
            },
            "model": {
                "d_model": 64,
                "nhead": 4,
                "nlayers": 3,
                "dim_feedforward": 256,
                "dropout": 0.05
            }
        }

    logging.info(f"device: {device}")
    logging.info(f"hyps: {hyps}")

    # Set seed
    seed = 42
    set_seed(seed=seed)
    i = 0

    results, y_true, y_pred, uncertainties, scaler_dict = train_pytorch_single_split(
        coords_raw=torch.tensor(df[config.coordinates].astype(float).to_numpy()),
        values_raw=torch.tensor(df[config.parameters].astype(float).to_numpy()),
        model_class=model_class,
        hyps=hyps,
        train_idx=train_idx,
        val_idx=test_idx,
        test_idx=test_idx,
        model_name=model_name,
        split_path="final",
        trial_id=i,
        optuna_callback=None,
        seed=seed + i,
        tuning_mode=False,
        device=device,
        do_dropout=True,
        n_inferences=1,
        save_model=True,
        output_dir=output_dir,
        cfg=cfg
    )

    logging.info(f"results: {results}")

    # Store y_true?
    df_true = pd.DataFrame(y_true, columns=config.parameters)
    df_true[config.coordinates] = df[config.coordinates]
    df_true.to_csv(f"{output_dir}/{model_name}_y_true.csv", index=False)

    logging.info(f"Stored y_true in {output_dir}/{model_name}_y_true.csv")

    df_pred = pd.DataFrame(y_pred, columns=config.parameters)
    df_pred[config.coordinates] = df[config.coordinates]
    df_pred.to_csv(f"{output_dir}/{model_name}_y_pred.csv", index=False)

    logging.info(f"Stored y_pred in {output_dir}/{model_name}_y_pred.csv")

    # Error
    val_rmse = np.sqrt(np.nanmean((y_true[test_idx] - y_pred[test_idx]) ** 2))
    logging.info(f"val_rmse: {val_rmse}")

    valid = ~np.isnan(y_true)
    y_true_valid = y_true[valid]
    y_pred_valid = y_pred[valid]
    evaluator = RegressionMetric(y_true_valid[test_idx], y_pred_valid[test_idx])
    logging.info(evaluator.RMSE(multi_output="raw_values"))
    logging.info(evaluator.MAE(multi_output="raw_values"))
    logging.info(evaluator.KGE(multi_output="raw_values"))
    logging.info(evaluator.R2(multi_output="raw_values"))
    logging.info(evaluator.NRMSE(multi_output="raw_values"))
    logging.info(evaluator.OI(multi_output="raw_values"))

    # Set up dataframes
    logging.info("Setting up dataframes...")
    df_rec = pd.DataFrame(y_pred, columns=config.parameters)
    df_rec[config.coordinates] = df[config.coordinates]

    df_aleatoric = pd.DataFrame(uncertainties[0], columns=config.parameters)
    df_aleatoric[config.coordinates] = df[config.coordinates]

    df_epistemic = pd.DataFrame(uncertainties[1], columns=config.parameters)
    df_epistemic[config.coordinates] = df[config.coordinates]

    logging.info("Generating animations...")

    for param in config.parameters:
        # Reconstruction animations
        generate_animation(df_rec, scaler_dict=scaler_dict, parameter=param,
                           save_as=f"{output_dir}/{model_name}_test_model{model_name}_{param}_reconstruction_{model_name}.mp4")

        # # Aleatoric uncertainty animations
        # generate_animation(df_aleatoric, scaler_dict=None, parameter=param,
        #                    save_as=f"{output_dir}/{model_name}_test_model{model_name}_{param}_aleatoric_{model_name}.mp4")
        #
        # # Epistemic uncertainty animations
        # generate_animation(df_epistemic, scaler_dict=None, parameter=param,
        #                    save_as=f"{output_dir}/{model_name}_test_model{model_name}_{param}_epistemic_{model_name}.mp4")


    #
    #
    # # Inference on global data --------------------------------------
    # # Load model
    # model = model_class(**hyps["model"])
    # model.load_state_dict(torch.load(f"output/tuning/{model_name}/pytorch.pt", map_location=device, weights_only=True))
    # model.to(device)
    # model.eval()
    # # Load global train/test idxs (to compute metrics on)
    # split_global = json.load(open(f"{config.output_dir_splits}/GLOBAL_test_train_split.json"))
    # test_idx_global = np.array(split_global["test_idx"], dtype=int)
    # train_idx_global = np.array(split_global["train_idx"], dtype=int)
    # # Load global data
    # df_global = load_dataset("output/gridding/df_20y_global.csv")
    # coords_raw = torch.from_numpy(df_global[config.coordinates].astype(float).to_numpy())
    # values_raw = torch.from_numpy(df_global[config.parameters].astype(float).to_numpy())
    # # Scale global data
    # coords_full, values_full, _ = preprocess(
    #     coords=coords_raw,
    #     values=values_raw,
    #     coord_names=config.coordinates,
    #     parameter_names=config.parameters,
    #     cyclic_time=False,
    #     scaler_dict=scaler_dict
    # )
    # x_scaled = coords_full
    # y_scaled = values_full
    # yglob, yglob_var = model.predict(x=x_scaled,y=y_scaled, n_neighbours=hyps["train"]["n_neighbours"],
    #                                  batch_size=hyps["train"]["batch_size"], device=device)
    #
    # np.nanmean((values_full[test_idx_global, :] - yglob[test_idx_global, :]) ** 2)
    # np.nanstd((values_full[test_idx_global, :] - yglob[test_idx_global, :]) ** 2)
    #
    # param = "P_TEMPERATURE"
    # df_rec_global = pd.DataFrame(yglob, columns=config.parameters)
    # df_rec_global[config.coordinates] = df_global[config.coordinates]
    # generate_animation(df_rec_global, scaler_dict=scaler_dict, parameter=param,
    #                    save_as=f"output/tuning/{model_name}/GLOBAL_model{model_name}_{param}_reconstruction_{model_name}.mp4")
    #
    #
    #
    #
    # # Map NA indices to global ones ----------------------------------------
    # df_global = load_dataset("output/gridding/df_20y_global.csv")
    # split_global = json.load(open(f"{config.output_dir_splits}/GLOBAL_test_train_split.json"))
    # test_idx_global = np.array(split_global["test_idx"], dtype=int)
    # train_idx_global = np.array(split_global["train_idx"], dtype=int)
    #
    # df_na = load_dataset()
    # split_na = json.load(open(f"{config.output_dir_splits}/test_train_split.json"))
    # test_idx_na = np.array(split_na["test_idx"], dtype=int)
    # train_idx_na = np.array(split_na["train_idx"], dtype=int)
    #
    # # Evaluate global prediction on NA test data
    # df_na_test_coords = df_na.iloc[test_idx_na][config.coordinates].reset_index(drop=True)
    # matching_indexes = df_global.index[df_global[config.coordinates].apply(tuple, axis=1).isin(df_na_test_coords[config.coordinates].apply(tuple, axis=1))].tolist()
    # print(np.nanmean((values_full[matching_indexes, :] - yglob[matching_indexes, :]) ** 2))
    # print(np.nanstd((values_full[matching_indexes, :] - yglob[matching_indexes, :]) ** 2))
    #
    # print(np.nanmean((y_true[matching_indexes, :] - y_pred[matching_indexes, :]) ** 2))
    # print(np.nanstd((y_true[matching_indexes, :] - y_pred[matching_indexes, :]) ** 2))

