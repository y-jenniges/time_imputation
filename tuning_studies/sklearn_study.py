import argparse
import logging
from functools import partial
from pathlib import Path, PurePath
from time import time
import optuna
import json
import numpy as np
import pandas as pd
import glob
import os
from missingpy import MissForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge

import config
from nn_utils.dataset import load_dataset, prepare_sklearn_data
from nn_utils.trainer import compute_metrics
from utils.tuning import set_seed, TuningResult


def get_model_class(model_name):
    """ Get model class by name. """
    name_class_map = {
        "mean": SimpleImputer,
        "knn": KNNImputer,
        "missforest": MissForest,
        "mice": IterativeImputer,
    }

    if model_name not in name_class_map.keys():
        raise ValueError(f"Unknown model name: {model_name}")

    return name_class_map[model_name]


def suggest_hyperparameters(trial, model_name):
    """ Define hyperparameter search space for each sklearn model. """
    if model_name == "mean":
        return {"strategy": trial.suggest_categorical("strategy", ["mean", "median"])}
    elif model_name == "knn":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 1, 50),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"])}
    elif model_name == "mice":
        est_name = trial.suggest_categorical("est_name", ["RandomForestRegressor", "BayesianRidge"])
        estimator = RandomForestRegressor() if est_name == "RandomForestRegressor" else BayesianRidge()

        return {"estimator": estimator,
                "max_iter": trial.suggest_int("max_iter", 10, 50),
                "initial_strategy": trial.suggest_categorical("initial_strategy", ["mean", "median"]),
                "imputation_order": trial.suggest_categorical("imputation_order", ["ascending", "descending", "random"])}
    elif model_name == "missforest":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_features": trial.suggest_categorical("max_features", [None, "sqrt", 0.5])}
    else:
        raise NotImplementedError(f"Could not find model {model_name}.")


def train_sklearn_single_split(df, model_class, hyps, test_idx, train_idx, val_idx, model_name, split_path, trial_id, optuna_callback=None, seed=42):
    # Create output subdir
    model_outdir = Path(config.output_dir_tuning) / model_name
    os.makedirs(model_outdir, exist_ok=True)

    # Output file
    split_fname = PurePath(split_path).stem
    base_name = f"split{split_fname}_model{model_name}_seed{seed}_hyps{trial_id}"
    main_path = Path(model_outdir) / base_name
    csv_fname = main_path.with_name(main_path.name + ".csv")

    # Check if file already exists
    if csv_fname.exists():
        results = pd.read_csv(csv_fname)
        return results

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

    # Evaluate on validation set
    val_rmse = np.sqrt(np.nanmean((y_true[val_idx] - y_pred[:, coord_dim:][val_idx]) ** 2))

    # Evaluate on test set (optional, not relevant in tuning)
    test_rmse = np.sqrt(np.nanmean((y_true[test_idx] - y_pred[:, coord_dim:][test_idx]) ** 2))

    # Compute other metrics
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)

    # Store results
    results.train_time = train_time
    results.pred_time = pred_time
    results.val_rmse = val_rmse
    results.test_rmse = test_rmse
    results.metrics_all = metrics

    # Store results on disc
    results.save(csv_fname)

    return results, y_true, y_pred


def optuna_objective(trial, model_name):
    # Load test indexes
    test_idx = np.array(json.load(open(f"{config.output_dir_splits}/test_train_split.json"))["test_idx"], dtype=int)

    # Load dataset
    df = load_dataset()

    # Load all split paths
    split_paths = sorted(glob.glob(os.path.join(config.output_dir_splits, "fold_*.json")))

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

        results, _, _ = train_sklearn_single_split(
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
            seed=42+trial.number)

        val_rmses.append(results.val_rmse)

        # Pruning
        trial.report(results.val_rmse, step=split_i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Return mean RMSE across splits
    return np.mean(val_rmses)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials to run")
    parser.add_argument("--model_name", type=str, default="mean", help="Name of the model to tune")
    parser.add_argument("--db", type=str, default=f"sqlite:///{config.output_dir_tuning}/tuning.db", help="Database path")
    args = parser.parse_args()

    # Set up Optuna study
    sampler = optuna.samplers.TPESampler(multivariate=True)  # Learn joint distributions
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(study_name=f"{args.model_name}_tuning",
                                direction="minimize",
                                sampler=sampler,
                                pruner=None,
                                storage=args.db,
                                load_if_exists=True)
    study.optimize(partial(optuna_objective, model_name=args.model_name), n_trials=args.n_trials, n_jobs=4)

    # Store results
    df_trials = study.trials_dataframe()
    df_trials.to_csv(f"optuna_trials_{args.model_name}.csv", index=False)

    print("Best trial:", study.best_trial.params)
