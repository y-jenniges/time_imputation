import torch.nn as nn
import torch
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from missingpy import MissForest  # https://github.com/dnowacki-usgs/missingpy.git + adapted

from oceanmae.model import OceanMAE
from oceanmae.losses import HeteroscedasticLoss, PhysicsLoss, MaskedMSELoss


# Output directories
output_dir = "output/"
output_dir_gridding = output_dir + "gridding/"
output_dir_splits = output_dir + "splits/"
output_dir_tuning = output_dir + "tuning/"
output_dir_correlations = output_dir + "correlations/"
output_dir_high_res_plots = output_dir + "high_res_plots/"
output_dir_plots = output_dir + "plots/"

output_dir_preliminary = output_dir + "preliminary/"

# Parameters to impute
parameters = ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"]

# Coordinate/time columns
coordinates = ["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"]

# --- Gridding --------------------------------------------------------------------------------------------- #
# Quality flags to filter for
quality_flags = [["pqf1", ">0"], ["pqf2", ">2"], ["sqf", ">=-1"]]

# Original COMFORT database
source_db_path = "../../data/comfort.sqlite"

# Name of new database that will be created
dest_db_path = "../ocean_clustering_and_validation/output_global/custom_global.db"

# Specification of the grid
depth_levels = [0, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 3000, 4000, 5000]

grid_configs = {
    "20y_global": {
        "param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
        "lat_min": -90,
        "lat_max": 90,
        "dlat": 1,
        "lon_min": -180,
        "lon_max": 180,
        "dlon": 1,
        "z_min": None,
        "z_max": None,
        "dz": None,
        "z_array": np.array(depth_levels),
        "time_min": "1772-01-01 00:00:00",
        "time_max": "2020-07-08 04:45:00",
        "mode": "Y",
        "selection": None,
        "dtime": 20,
        "note": "Northern Atlantic, all times, 13 depth steps, 6 params"
    },
    "20y_na": {
        "param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
        "lat_min": 0,
        "lat_max": 70,
        "dlat": 1,
        "lon_min": -77,
        "lon_max": 30,
        "dlon": 1,
        "z_min": None,
        "z_max": None,
        "dz": None,
        "z_array": np.array(depth_levels),
        "time_min": "1772-01-01 00:00:00",
        "time_max": "2020-07-08 04:45:00",
        "mode": "Y",
        "selection": None,
        "dtime": 20,
        "note": "Northern Atlantic, all times, 13 depth steps, 6 params"
    },
    "avg": {"param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
            "lat_min": 20,
            "lat_max": 40,
            "dlat": 1,
            "lon_min": -40,
            "lon_max": -10,
            "dlon": 1,
            "z_min": None,
            "z_max": None,
            "dz": None,
            "z_array": np.array(depth_levels),
            "time_min": "1772-01-01 00:00:00",
            "time_max": "2020-07-08 04:45:00",
            "mode": "Y",
            "selection": None,
            "dtime": 300,
            "note": "Subtropical gyre, time average"},

    "100y": {"param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
             "lat_min": 20,
             "lat_max": 40,
             "dlat": 1,
             "lon_min": -40,
             "lon_max": -10,
             "dlon": 1,
             "z_min": None,
             "z_max": None,
             "dz": None,
             "z_array": np.array(depth_levels),
             "time_min": "1772-01-01 00:00:00",
             "time_max": "2020-07-08 04:45:00",
             "mode": "Y",
             "selection": None,
             "dtime": 100,
             "note": "Subtropical gyre, 100 year step"},

    "50y": {"param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
            "lat_min": 20,
            "lat_max": 40,
            "dlat": 1,
            "lon_min": -40,
            "lon_max": -10,
            "dlon": 1,
            "z_min": None,
            "z_max": None,
            "dz": None,
            "z_array": np.array(depth_levels),
            "time_min": "1772-01-01 00:00:00",
            "time_max": "2020-07-08 04:45:00",
            "mode": "Y",
            "selection": None,
            "dtime": 50,
            "note": "Subtropical gyre, 50 year step"},

    "20y": {"param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
            "lat_min": 20,
            "lat_max": 40,
            "dlat": 1,
            "lon_min": -40,
            "lon_max": -10,
            "dlon": 1,
            "z_min": None,
            "z_max": None,
            "dz": None,
            "z_array": np.array(depth_levels),
            "time_min": "1772-01-01 00:00:00",
            "time_max": "2020-07-08 04:45:00",
            "mode": "Y",
            "selection": None,
            "dtime": 20,
            "note": "Subtropical gyre, 20 year step"},

    "10y": {"param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
            "lat_min": 20,
            "lat_max": 40,
            "dlat": 1,
            "lon_min": -40,
            "lon_max": -10,
            "dlon": 1,
            "z_min": None,
            "z_max": None,
            "dz": None,
            "z_array": np.array(depth_levels),
            "time_min": "1772-01-01 00:00:00",
            "time_max": "2020-07-08 04:45:00",
            "mode": "Y",
            "selection": None,
            "dtime": 10,
            "note": "Subtropical gyre, 10 year step"},

    "5y": {"param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
           "lat_min": 20,
           "lat_max": 40,
           "dlat": 1,
           "lon_min": -40,
           "lon_max": -10,
           "dlon": 1,
           "z_min": None,
           "z_max": None,
           "dz": None,
           "z_array": np.array(depth_levels),
           "time_min": "1772-01-01 00:00:00",
           "time_max": "2020-07-08 04:45:00",
           "mode": "Y",
           "selection": None,
           "dtime": 5,
           "note": "Subtropical gyre, 10 year step"},

    "y": {"param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
          "lat_min": 20,
          "lat_max": 40,
          "dlat": 1,
          "lon_min": -40,
          "lon_max": -10,
          "dlon": 1,
          "z_min": None,
          "z_max": None,
          "dz": None,
          "z_array": np.array(depth_levels),
          "time_min": "1772-01-01 00:00:00",
          "time_max": "2020-07-08 04:45:00",
          "mode": "Y",
          "selection": None,
          "dtime": 1,
          "note": "Subtropical gyre, 10 year step"},

    "m": {"param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
          "lat_min": 20,
          "lat_max": 40,
          "dlat": 1,
          "lon_min": -40,
          "lon_max": -10,
          "dlon": 1,
          "z_min": None,
          "z_max": None,
          "dz": None,
          "z_array": np.array(depth_levels),
          "time_min": "1772-01-01 00:00:00",
          "time_max": "2020-07-08 04:45:00",
          "mode": "M",
          "selection": None,
          "dtime": 1,
          "note": "Subtropical gyre, 12 months step"},
}

bathymetry_path = "../../data/bathymetry/gebco_2022_sub_ice_topo/GEBCO_2022_sub_ice_topo.nc"

# --- Splitting --------------------------------------------------------------------------------------------- #
n_splits_per_scheme = 5
data_path = "output/gridding/df_20y_na.csv"
val_fractions = [0.1, 0.2, 0.5]
test_fraction = 0.15

# --- Tuning --------------------------------------------------------------------------------------------- #
tuning_seeds = range(10)
models = {
    "mean": {"model": SimpleImputer, "hyps": {"strategy": ["mean"]}},
    "knn": {"model": KNNImputer, "hyps": {"n_neighbors": [5, 10, 20]}, "weights": ["uniform", "distance"]},
    "mice": {"model": IterativeImputer, "hyps": {"estimator": [BayesianRidge(), RandomForestRegressor()],
                                                 "max_iter": [20, 50],
                                                 "initial_strategy": ["mean", "median"],
                                                 "imputation_order": ["ascending", "descending", "random"],
                                                 }},
    "missforest": {"model": MissForest, "hyps": {"n_estimators": [50, 100],
                                                   "max_depth": [None, 10],
                                                   "min_samples_split": [2, 5],
                                                   "max_features": ["sqrt", 0.5, None]}},  # Required since MissForest uses old sklearn version
    "mae_rough": {"model": OceanMAE, "hyps": {"model": {
        "d_model": [32, 64],  # small, medium, large embedding
        "nlayers": [2, 3],  # typical transformer depth
        "nhead": [2, 4],  # variety to test attention width
        "dim_feedforward": [128],  # FFN capacity
        "dropout": [0.05],
        "coord_dim": [5],
        "value_dim": [len(parameters)]
    },

        "train": {
            "learning_rate": [2e-4, 5e-4],
            "batch_size": [512],
            "n_epochs": [50],  # , 100, 200],
            "optimizer": [torch.optim.Adam],
            "patience": [5, 10],
            "mask_ratio": [0.0, 0.5],
            "loss": [
                {"class": MaskedMSELoss, "kwargs": {}},
                {"class": HeteroscedasticLoss, "kwargs": {}},
            ],
            "mc_dropout": [True],
        }
    },
                  }
    # "kriging": {Kriging},
    # "gain": {GAIN}
}

# models = {
#     "mae": {"model": OceanMAE, "hyps": {"model": {
#                                                 "d_model": [32, 64, 128],  # small, medium, large embedding
#                                                 "nlayers": [2, 3],  # typical transformer depth
#                                                 "nhead": [2, 4],  # variety to test attention width
#                                                 "dim_feedforward": [128, 256],  # FFN capacity
#                                                 "dropout": [0.03, 0.05, 0.07, 0.1],
#                                                 "coord_dim": [5],
#                                                 "value_dim": [len(parameters)]
#                                                 },  # typical for MC dropout
#
#                                               "train": {
# "patience": [10],
#                                                   "learning_rate": [1e-4, 2e-4, 5e-4, 8e-4, 1e-3],
#                                                   "batch_size": [512],
#                                                   "n_epochs": [50],  # , 100, 200],
#                                                   "optimizer": [torch.optim.Adam],
#                                                   "loss": [                # {"class": PhysicsLoss, "kwargs": {"base_loss": HeteroscedasticLoss(), "lambda_phys": 0.05}},
#                 # {"class": PhysicsLoss, "kwargs": {"base_loss": HeteroscedasticLoss(), "lambda_phys": 0.1}},
#                                                            ],
#                                                   "mc_dropout": [True],
#                                               }
#                                               },
# }}

# --- Training --------------------------------------------------------------------------------------------- #
