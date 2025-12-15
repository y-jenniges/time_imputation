import optuna
import pandas as pd
import numpy as np
import torch
from itertools import product
from pathlib import Path
from dataclasses import dataclass, asdict, field
import os
import glob

from missingpy import MissForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from models.mae import OceanMAE
from models.unet import OceanUNet


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

    metrics_last: dict = field(default_factory=dict)
    metrics_all: dict = field(default_factory=dict)

    def save(self, fname: Path):
        # Serialize nested dicts as JSON
        data = asdict(self)
        data["hyps"] = self.make_json_safe(self.hyps)
        data["metrics_last"] = self.make_json_safe(self.metrics_last)
        data["metrics_all"] = self.make_json_safe(self.metrics_all)

        # Store as dataframe
        df = pd.DataFrame([asdict(self)])
        df.to_csv(fname, index=False)

    def make_json_safe(self, obj):
        if isinstance(obj, dict):  # Dict type
            return {k: self.make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):  # List or tuple
            return [self.make_json_safe(v) for v in obj]
        elif isinstance(obj, type) or callable(obj):  # Classes, functions, losses
            return obj.__name__
        elif isinstance(obj, (np.integer, np.floating)):  # Numbers
            return obj.item()
        else:
            return obj


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


def combine_csvs(dir_path, out_name="combined.csv", remove_files=False):
    # Load all result files as df
    res_files = glob.glob(f"{dir_path}/*.csv")

    if len(res_files) > 1:
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
    elif len(res_files) == 1:
        print("Only one file found")
        df_final = pd.read_csv(res_files[0])
    else:
        print("No files found")
        df_final = pd.DataFrame()

    return df_final


def make_optuna_callback(trial, split_i, n_epochs):
    def callback(epoch, val_losses):
        running_mean = np.nanmean(val_losses)
        global_step = split_i * n_epochs + epoch
        trial.report(running_mean, step=global_step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return callback


def get_model_class(model_name):
    """ Get model class by name. """
    name_class_map = {
        "mean": SimpleImputer,
        "knn": KNNImputer,
        "missforest": MissForest,
        "mice": IterativeImputer,
        "mae": OceanMAE,
        "unet": OceanUNet
    }

    if model_name not in name_class_map.keys():
        raise ValueError(f"Unknown model name: {model_name}")

    return name_class_map[model_name]
