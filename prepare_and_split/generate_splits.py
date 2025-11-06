import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import json
import os
import glob

import config
from utils.gridding import df_to_gridded_da
from utils.plotting import animate_depth_panels


def assign_blocks(df, latlon_step, depth_edges):
    temp = df.copy()

    # Assign blocks to samples
    temp["LAT_BIN"] = np.floor((temp["LATITUDE"] + 90)/ latlon_step).astype(int)
    temp["LON_BIN"] = np.floor((temp["LONGITUDE"] + 180)/ latlon_step).astype(int)
    temp["DEPTH_BIN"] = pd.cut(temp["LEV_M"], bins=depth_edges, labels=range(len(depth_edges) - 1), right=False)
    temp["TIME_BIN"] = temp["DATEANDTIME"]

    # Generate block IDs
    temp["BLOCK_ID"] = (temp["LAT_BIN"].astype(str) + "_" + temp["LON_BIN"].astype(str) + "_" +
                        temp["DEPTH_BIN"].astype(str) + "_" + temp["TIME_BIN"].astype(str))

    return temp["BLOCK_ID"]


def generate_block_folds(df, latlon_steps, depths, val_fractions, test_fraction, n_splits, out_dir):
    scheme = "block"

    block_folds = []
    for ll_step in latlon_steps:
        # Assign blocks to samples
        id_col = "BLOCK_ID_" + str(ll_step)
        df[id_col] = assign_blocks(df, latlon_step=ll_step, depth_edges=depths)

        # Count number of samples per block
        block_counts = df[id_col].value_counts()
        min_sample_count = np.quantile(block_counts.values, 0.25)
        small_blocks = block_counts[block_counts < min_sample_count].index
        large_blocks = block_counts[block_counts >= min_sample_count].index
        small_block_idx = df[df[id_col].isin(small_blocks)].index.to_numpy()

        # Only use large blocks for CV
        df_large_blocks = df[df[id_col].isin(large_blocks)].copy()
        groups = df_large_blocks[id_col].values

        # CV
        for val_size in val_fractions:
            folds = []
            gss_outer = GroupShuffleSplit(n_splits=n_splits, test_size=test_fraction)

            folds = []  #  = np.empty(len(df_large_blocks), dtype=object)
            for fold_idx, (train_val_idx, test_idx) in enumerate(gss_outer.split(df_large_blocks, groups=groups)):
                # Training data (will be further split into train and validation)
                df_train_val = df_large_blocks.iloc[train_val_idx]
                train_val_groups = df_train_val[id_col].values

                # Inner split (train/validation)
                gss_inner = GroupShuffleSplit(n_splits=1, test_size=val_size)
                train_idx, val_idx = next(gss_inner.split(df_train_val, groups=train_val_groups))

                # Map to original df indices
                train_idx = df_train_val.index[train_idx].to_numpy()
                val_idx = df_train_val.index[val_idx].to_numpy().tolist()
                test_idx = df_large_blocks.index[test_idx].to_numpy().tolist()

                # Add the small block to train set
                train_idx = np.concatenate([train_idx, small_block_idx]).tolist()

                fold = {
                    "fold_id": fold_idx,
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                    "test_idx": test_idx,
                }

                # Save
                fname = os.path.join(out_dir, f"fold_{scheme}_{fold_idx}_{val_size}_{ll_step}.json")
                with open(fname, "w") as f:
                    json.dump(fold, f, indent=2)

                # Store folds by lat/lon step
                block_folds.append(pd.DataFrame({"scheme": [scheme], "split_id": [fold_idx], "val_size": [val_size], "n_splits": [n_splits], "latlon_step": [ll_step], "filepath": [fname]}))

    return pd.concat(block_folds)


def generate_random_folds(df, n_splits, val_fractions, test_fraction, out_dir):
    scheme = "random"

    # Random splits
    random_splits = []
    for split_id in range(n_splits):
        folds = []
        for val_size in val_fractions:
            # Sample test set
            df_test = df.sample(frac=test_fraction)
            test_idx = df_test.index.tolist()
            df_train_val = df.drop(df_test.index)

            # Sample validation set from remaining data
            df_val = df_train_val.sample(frac=val_size)
            val_idx = df_val.index.tolist()
            train_idx = df_train_val.drop(df_val.index).index.tolist()

            fold = {
                "fold_id": split_id,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
            }

            # Save
            fname = os.path.join(out_dir, f"fold_{scheme}_{split_id}_{val_size}.json")
            with open(fname, "w") as f:
                json.dump(fold, f, indent=2)

            random_splits.append(pd.DataFrame({"scheme": [scheme], "split_id": [split_id], "n_splits": [n_splits], "validation_fraction": [val_size], "filepath": [fname]}))

    return pd.concat(random_splits).reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv(config.data_path)
    df["DATEANDTIME"] = pd.to_datetime(df["DATEANDTIME"]).dt.year

    # Oceanographically relevant subdivisions
    latlons = [5, 15, 40]
    depths = [0, 200, 1000, 12000]  # epi meso bathy pelagic
    seed = 42

    # Summarise folds
    df_random = generate_random_folds(df=df,
                                      n_splits=config.n_splits_per_scheme,
                                      val_fractions=config.val_fractions,
                                      test_fraction=config.test_fraction,
                                      out_dir=config.output_dir_splits)
    df_block = generate_block_folds(df=df,
                                    latlon_steps=latlons,
                                    depths=depths,
                                    val_fractions=config.val_fractions,
                                    test_fraction=config.test_fraction,
                                    n_splits=config.n_splits_per_scheme,
                                    out_dir=config.output_dir_splits)

    # Save metadata
    pd.concat([df_random, df_block]).to_csv(os.path.join(config.output_dir_splits, "split_metadata.csv"), index=False)


    # Plotting
    # Color categories
    CATEGORY_COLORS = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:red", "unobserved": "tab:green"}
    categories = list(CATEGORY_COLORS.keys())
    cat_to_int = {cat: i for i, cat in enumerate(categories)}
    cmap = ListedColormap([CATEGORY_COLORS[cat] for cat in categories])

    for fname in glob.glob(os.path.join(config.output_dir_splits, "*.json")):
        print(fname)
        temp = df.copy()

        # Load split
        with open(fname, "r") as f:
            fold = json.load(f)
        train_idx = np.array(fold["train_idx"])
        val_idx   = np.array(fold["val_idx"])
        test_idx  = np.array(fold["test_idx"])

        # Assign test/train/validation categories to dataframe
        temp["SPLIT"] = "unassigned"
        temp.loc[train_idx, "SPLIT"] = "train"
        temp.loc[val_idx, "SPLIT"]   = "validation"
        temp.loc[test_idx, "SPLIT"]  = "test"
        temp.loc[temp.P_TEMPERATURE.isna(), "SPLIT"] = "unobserved"
        print(temp["SPLIT"].value_counts())

        # Map category names to numeric
        temp["CAT"] = pd.Categorical(temp["SPLIT"], categories=categories, ordered=True).codes

        # Plotting
        da = df_to_gridded_da(df=temp, value_col="CAT")  # Convert to xarray
        animate_depth_panels(da, depth_dim="depth", cmap=cmap, save_as=fname.rstrip(".json") + ".mp4")

        print()
