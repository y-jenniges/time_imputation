import argparse
import logging
from functools import partial
from pathlib import Path, PurePath
from time import time
import optuna
import json
import numpy as np
import glob
import os
import platform

import pandas as pd
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.linear_model import BayesianRidge

import config
from nn_utils.dataset import load_dataset, prepare_sklearn_data
from nn_utils.trainer import compute_metrics
from utils.tuning import set_seed, TuningResult, get_model_class


def suggest_hyperparameters(trial, model_name):
    """ Define hyperparameter search space for each sklearn model. """
    if model_name == "mean":
        return {"strategy": trial.suggest_categorical("strategy", ["mean", "median"])}
    elif model_name == "knn":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 1, 50),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"])}
    elif model_name == "mice":
        return {"estimator": BayesianRidge(),
                "initial_strategy": trial.suggest_categorical("initial_strategy", ["mean", "median"]),
                "imputation_order": trial.suggest_categorical("imputation_order", ["ascending", "descending", "random"]),
                "sample_posterior": True}
    elif model_name == "missforest":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_features": trial.suggest_categorical("max_features", [None, "sqrt", 0.5])}
    elif model_name == "remasker":
        # Ensure valid embed_dim and num_heads combinations
        embed_dim = trial.suggest_categorical("embed_dim", [16, 32, 64, 128, 256, 512])
        valid_heads = [h for h in [2, 4, 8] if embed_dim % h == 0]
        num_heads = trial.suggest_categorical("num_heads", valid_heads)

        return {"batch_size": trial.suggest_categorical("batch_size", [64, 128, 512, 1024]),
                "mask_ratio": trial.suggest_float("mask_ratio", 0.0, 0.99),
                "embed_dim": embed_dim,
                "depth": trial.suggest_int("depth", 2, 8),
                "decoder_depth": trial.suggest_int("decoder_depth", 1, 8),
                "num_heads": num_heads,
                "encode_func": trial.suggest_categorical("encode_func", ["linear", "active"]),
                "max_epochs": 20,
                }

    elif model_name == "hyperimpute":
        return {
            "optimizer": trial.suggest_categorical("optimizer", ["simple", "bayesian", "hyperband"]),
            "n_inner_iter": trial.suggest_categorical("n_inner_iter", [1, 3, 5, 10]),
            "baseline_imputer": trial.suggest_categorical("baseline_imputer", [0, 1, 2]),
            "class_threshold": trial.suggest_categorical("class_threshold", [10, 20, 50, 100]),
            "optimize_thresh": trial.suggest_categorical("optimize_thresh", [1000, 5000]),
            "select_model_by_column": trial.suggest_categorical("select_model_by_column", [True, False]),
            "random_state": 42
        }
    elif model_name == "gain_hyperimpute":
        return {
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "n_epochs": 100, # trial.suggest_int("n_epochs", 50, 200),
            "hint_rate": trial.suggest_float("hint_rate", 0.5, 0.95),
            "loss_alpha": trial.suggest_float("loss_alpha", 1, 100),
            "random_state": 42
        }
    elif model_name == "miracle_hyperimpute":
        return {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "n_hidden": trial.suggest_categorical("n_hidden", [32, 64, 128, 256]),
            "max_steps": 100,
            "random_state": 42
        }
    elif model_name == "miwae_hyperimpute":
        return {
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "n_hidden": trial.suggest_categorical("n_hidden", [32, 64, 128, 256]),
            "latent_size": trial.suggest_int("latent_size", 5, 50),
            "K": trial.suggest_int("K", 1, 5),
            "n_epochs": 100,
            "random_state": 42,
        }
    elif model_name == "sklearn_ice_hyperimpute":
        return {
            "max_iter": trial.suggest_categorical("max_iter", [100, 300, 1000]),
            "tol": trial.suggest_categorical("tol", [1e-4, 1e-3, 1e-2]),
            "initial_strategy": trial.suggest_int("initial_strategy", 0, 3), # ["mean", "median", "most_frequent", "constant"]),
            "imputation_order": trial.suggest_int("imputation_order", 0, 4), #["ascending", "descending", "roman", "arabic", "random"]),
            "random_state": 42,
        }
    elif model_name == "nop_hyperimpute":
        return {}
    elif model_name == "em_hyperimpute":
        return {
            "maxit": trial.suggest_categorical("maxit", [50, 100, 200, 500]),
            "convergence_threshold": trial.suggest_categorical("convergence_threshold", [1e-6, 1e-7, 1e-8]),
            "random_state": 42,
        }
    elif model_name == "mice_hyperimpute":
        return {
            "max_iter": trial.suggest_categorical("max_iter", [100, 300, 1000]),
            "tol": trial.suggest_categorical("tol", [1e-4, 1e-3, 1e-2]),
            "n_imputations": trial.suggest_categorical("n_imputations", [1, 3, 5]),
            "initial_strategy": trial.suggest_int("initial_strategy", 0, 3), # ["mean", "median", "most_frequent", "constant"]),
            "imputation_order": trial.suggest_int("imputation_order", 0, 4), #["ascending", "descending", "roman", "arabic", "random"]),
            "random_state": 42,
        }
    elif model_name == "softimpute_hyperimpute":
        return {
            "max_rank": trial.suggest_categorical("max_rank", [2, 5, 10, 20]),
            "shrink_lambda": trial.suggest_categorical("shrink_lambda", [0.0, 0.01, 0.1, 1.0]),
            "maxit": trial.suggest_categorical("maxit", [100, 300, 1000]),
            "convergence_threshold": trial.suggest_categorical("convergence_threshold", [1e-4, 1e-5, 1e-6]),
            "cv_len": trial.suggest_categorical("cv_len", [2, 3, 5]),
            "random_state": 42,
        }
    elif model_name == "ice_hyperimpute":
        return {
            "max_iter": trial.suggest_categorical("max_iter", [100, 300, 1000]),
            "initial_strategy": trial.suggest_categorical("initial_strategy", ["mean", "median", "most_frequent", "constant"]),
            "imputation_order": trial.suggest_categorical("imputation_order", ["ascending", "descending", "roman", "arabic", "random"]),
            "random_state": 42,
        }
    elif model_name == "missforest_hyperimpute":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "initial_strategy": trial.suggest_categorical(
            "initial_strategy", ["mean", "median", "most_frequent"]),
            "imputation_order": trial.suggest_categorical("imputation_order", ["ascending", "descending", "random"]),
            "max_iter": 100,
            "random_state": 42,
        }
    elif model_name == "sklearn_missforest_hyperimpute":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 10, 20]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "max_iter": 100,
            "random_state": 42,
        }
    elif model_name == "sinkhorn_hyperimpute":
        return {
            "eps": trial.suggest_float("eps", 0.001, 0.1, log=True),
            "lr": trial.suggest_float("lr", 0.1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024]),
            "n_pairs": trial.suggest_int("n_pairs", 1, 5),
            "noise": trial.suggest_float("noise", 0.0, 1e-2),
            "scaling": trial.suggest_float("scaling", 0.5, 0.99),
            "n_epochs": 100,
            "random_state": 42,
        }
    else:
        raise NotImplementedError(f"Could not find model {model_name}.")


def train_sklearn_single_split(df, model_class, hyps, test_idx, train_idx, val_idx, model_name, split_path, trial_id,
                               optuna_callback=None, seed=42, save_model=False, tuning_mode=True):
    # Create output subdir
    model_outdir = Path(config.output_dir_tuning) / model_name
    os.makedirs(model_outdir, exist_ok=True)

    # Output file
    split_fname = PurePath(split_path).stem
    base_name = f"model{model_name}_split{split_fname}_trial{trial_id}"
    main_path = Path(model_outdir) / base_name
    json_fname = main_path.with_name(main_path.name + ".json")

    # Check if file already exists
    if json_fname.exists():
        with open(json_fname) as f:
            results = json.load(f)
        return results, None, None, None

    # Set deterministic seeds
    set_seed(seed)

    # Init results object
    results = TuningResult(split=split_fname, seed=seed, model=model_name, hyp_combo_id=trial_id, hyps=hyps)

    # Prepare data
    x_train, y_true, scaler_dict, coord_dim, value_dim = prepare_sklearn_data(df=df, train_idx=train_idx, val_idx=val_idx,
                                                                          test_idx=test_idx)
    # Fit model
    st = time()
    imputer = model_class(**hyps)
    imputer.fit(x_train)
    train_time = time() - st

    # Predict
    st = time()
    y_pred = imputer.transform(x_train)
    pred_time = time() - st

    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()

    # Evaluate on validation set
    val_rmse = np.sqrt(np.nanmean((y_true[val_idx] - y_pred[:, coord_dim:][val_idx]) ** 2))

    # Compute other metrics
    metrics = compute_metrics(y_true=y_true[val_idx], y_pred=y_pred[:, coord_dim:][val_idx], var_names=config.parameters)

    # Store results
    results.train_time = train_time
    results.pred_time = pred_time
    results.val_rmse = val_rmse
    results.metrics_all = metrics

    if not tuning_mode:
        # Store scalers only for full training (not in tuning mode)
        results.scalers = scaler_dict

    # Store results on disc
    results.save(json_fname, model=imputer if save_model else None)

    return results.make_json_safe(), y_true, y_pred, scaler_dict


def optuna_objective(trial, model_name):
    # Load test indexes
    test_idx = np.array(json.load(open(f"{config.output_dir_splits}/test_train_split.json"))["test_idx"], dtype=int)

    # Load dataset
    df = load_dataset()

    # Load all split paths
    split_paths = sorted(glob.glob(os.path.join(config.output_dir_splits, "selected_splits/fold_*.json")))

    # Priors and model class
    model_class = get_model_class(model_name)
    hyp_dict = suggest_hyperparameters(trial=trial, model_name=model_name)

    # Train
    val_rmses = []
    for split_i, split_path in enumerate(split_paths):
        logging.info(f"Training model {model_name} in trial {trial.number} on split {split_path}")

        # Load split
        split = json.load(open(split_path))
        train_idx = np.array(split["train_idx"])
        val_idx = np.array(split["val_idx"])

        results, _, _, _ = train_sklearn_single_split(
            df=df,
            model_class=model_class,
            hyps=hyp_dict,
            test_idx=test_idx,
            train_idx=train_idx,
            val_idx=val_idx,
            model_name=model_name,
            split_path=split_path,
            trial_id=trial.number,
            optuna_callback=None,
            tuning_mode=True,
            seed=42+trial.number)

        val_rmses.append(results["val_rmse"])

        # # Pruning
        # trial.report(results.val_rmse, step=split_i)
        # if trial.should_prune():
        #     raise optuna.TrialPruned()

    # Return mean RMSE across splits
    return np.mean(val_rmses)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials to run")
    parser.add_argument("--model_name", type=str, default="mean", help="Name of the model to tune")
    args = parser.parse_args()

    # Setup logging
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Set up Optuna study
    if platform.system() == "Windows":
        storage = f"sqlite:///{config.output_dir_tuning}/{args.model_name}/{args.model_name}_tuning.db"
    else:
        journal_path = f"{config.output_dir_tuning}/{args.model_name}/{args.model_name}_tuning.log"
        storage = JournalStorage(JournalFileBackend(journal_path))

    sampler = optuna.samplers.TPESampler(multivariate=True)  # Learn joint distributions
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(study_name=f"{args.model_name}_tuning",
                                direction="minimize",
                                sampler=sampler,
                                pruner=None,
                                storage=storage,
                                load_if_exists=True)
    study.optimize(partial(optuna_objective, model_name=args.model_name), n_trials=args.n_trials, n_jobs=1)

    # Store results
    df_trials = study.trials_dataframe()
    df_trials.to_csv(f"optuna_trials_{args.model_name}.csv", index=False)

    print("Best trial:", study.best_trial.params)
