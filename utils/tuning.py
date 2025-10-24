import pandas as pd
import numpy as np
import torch
from itertools import product
from pathlib import Path
from dataclasses import dataclass, asdict
import os
import glob


@dataclass
class TuningResult:
    split: str
    seed: int
    model: str
    hyp_combo_id: int
    hyps: dict
    test_rmse: float = np.nan
    val_rmse: float = np.nan
    train_time: float = np.nan
    pred_time: float = np.nan
    mean_aleatoric_uncertainty: float = np.nan
    std_aleatoric_uncertainty: float = np.nan
    stop_epoch: int = np.nan
    reconstruction_rmse: float = np.nan

    def save(self, fname: Path):
        df = pd.DataFrame([asdict(self)])
        df.to_csv(fname, index=False)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # current GPU
    torch.cuda.manual_seed_all(seed)  # all GPUs, if multiple
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_hyperparameter_combinations(hyps):
    combinations = []

    # PyTorch-style nested dict
    if isinstance(hyps, dict) and "model" in hyps and "train" in hyps:
        model_hyps = hyps["model"]
        train_hyps = hyps["train"]

        # Get keys and values lists
        model_keys, model_values = zip(*model_hyps.items()) if model_hyps else ([], [])
        train_keys, train_values = zip(*train_hyps.items()) if train_hyps else ([], [])

        # Iterate over model hyperparameter combinations
        for model_combo in product(*model_values) if model_values else [()]:
            model_dict = dict(zip(model_keys, model_combo))

            # Iterate over train hyperparameter combinations
            for train_combo in product(*train_values) if train_values else [()]:
                train_dict = dict(zip(train_keys, train_combo))
                combinations.append({"model": model_dict, "train": train_dict})

    # Sklearn-style flat dict
    else:
        keys, values = zip(*hyps.items()) if hyps else ([], [])
        for combo in product(*values) if values else [()]:
            combinations.append(dict(zip(keys, combo)))

    return combinations


def combine_csvs(dir_path, out_name="combined.csv", remove_files=False):
    # Load all result files as df
    res_files = glob.glob(f"{dir_path}/*.csv")

    # Load all files as df
    all_dfs = []
    for f in res_files:
        df = pd.read_csv(f)
        all_dfs.append(df)

    # Concat all result files
    df_final = pd.concat(all_dfs)
    df_final.to_csv(f"{dir_path}/{out_name}", index=False)
    print("Stored file at ", f"{dir_path}/{out_name}")

    # Remove files
    if remove_files:
        for f in res_files:
            os.remove(f)

    return df_final

