import joblib
import optuna
import pandas as pd
import numpy as np
import torch
from itertools import product
from pathlib import Path
from dataclasses import dataclass, asdict, field
import os
import glob
import json
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from missingpy import MissForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from torch import nn

import config
from models.mastnet import MaSTNeT
from models.unet import OceanUNet


@dataclass
class TuningResult:
    # Meta data
    split: str
    seed: int
    model: str
    hyp_combo_id: int
    hyps: dict

    # Metrics
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

    # Optional model info
    model_framework: str | None = None
    model_class: str | None = None
    model_path: str | None = None

    def save(self, fname: Path, model=None):
        # Store model if given
        if model is not None:
            self.save_model(model=model, model_dir=fname.parent)

        # Make json safe
        data = self.make_json_safe()

        # Store as json
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)

    def save_model(self, model, model_dir:Path):
        model_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(model, BaseEstimator):
            path = model_dir / "sklearn.joblib"
            joblib.dump(model, path)
            self.model_framework = "sklearn"

        elif isinstance(model, nn.Module):
            path = model_dir / "pytorch.pt"
            torch.save(model.state_dict(), path)
            self.model_framework = "pytorch"

        else:
            raise TypeError("Unsupported model type")

        self.model_class = model.__class__.__name__
        self.model_path = str(path)


    def make_json_safe(self):
        # Serialize nested dicts as JSON
        data = asdict(self)
        data = self.make_obj_json_safe(data)
        return data

    def make_obj_json_safe(self, obj):
        if isinstance(obj, dict):  # Dict type
            return {k: self.make_obj_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):  # List or tuple
            return [self.make_obj_json_safe(v) for v in obj]
        elif isinstance(obj, type) or callable(obj):  # Classes, functions, losses
            return obj.__name__
        elif isinstance(obj, np.generic):  # Numbers
            return obj.item()
        elif isinstance(obj, torch.dtype):  # Torch dtypes
            return str(obj)
        elif isinstance(obj, BaseEstimator):
            return obj.__class__.__name__
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
    # Remove file if exists
    if Path.exists(Path(dir_path + out_name)):
        Path(dir_path + out_name).unlink()

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
        "mastnet": MaSTNeT,
        "unet": OceanUNet,
        # "mae": MaSTNeT,
        # "mae_finetune": MaSTNeT,
    }

    if model_name not in name_class_map.keys():
        raise ValueError(f"Unknown model name: {model_name}")

    return name_class_map[model_name]


def load_optuna_study(model_name):
    # Load Optuna study
    base_name = f"{config.output_dir_tuning}{model_name}/{model_name}_tuning"

    if Path.exists(Path(base_name + ".db")):
        study = optuna.load_study(study_name=f"{model_name}_tuning", storage=f"sqlite:///{base_name}.db")
        print("Study loaded as db")
    else:
        storage = JournalStorage(JournalFileBackend(f"{base_name}.log"))
        study = optuna.load_study(study_name=f"{model_name}_tuning", storage=storage)
        print("Study loaded as log")
    return study
