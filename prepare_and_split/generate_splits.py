from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import json
import os
import glob

import config
from utils.gridding import df_to_gridded_da
from utils.plotting import animate_depth_panels
from oceanmae.dataset import load_dataset
from utils.plotting import plot_geo


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


def generate_block_folds(df, parameters, latlon_steps, depths, val_fractions, n_splits, train_idx, seed, out_dir, tolerance=0.05, max_iter=100):
    scheme = "block"

    # Use only training subset
    df_train = df.loc[train_idx].copy()

    fold_records = []
    fold_id = 0
    for ll_step in latlon_steps:
        # Assign blocks
        id_col = "BLOCK_ID_" + str(ll_step)
        df_train[id_col] = assign_blocks(df_train, latlon_step=ll_step, depth_edges=depths)
        block_groups = df_train.groupby(id_col)
        block_sizes = block_groups.size()
        block_ids = block_sizes.index.to_numpy()

        # Compute global coverage ratios per parameter
        coverage_ratios = df_train[parameters].notna().mean()

        for val_size in val_fractions:
            target_val_n = int(len(df_train) * val_size)

            for split_id in range(n_splits):
                # Set seed
                rng = np.random.default_rng(seed + 1000 * ll_step + 100 * split_id + 10 * int(val_size*100))

                found = False
                for i in range(max_iter):
                    # Shuffle blocks
                    shuffled_blocks = rng.permutation(block_ids)
                    selected_blocks = []
                    selected_rows = 0

                    # Pick blocks
                    for b in shuffled_blocks:
                        selected_blocks.append(b)
                        selected_rows += block_sizes[b]
                        if selected_rows >= target_val_n:
                            break

                    # Create validation df
                    df_val = df_train[df_train[id_col].isin(selected_blocks)]
                    val_coverage = df_val[parameters].notna().mean()

                    # Check coverage
                    if (val_coverage - coverage_ratios).abs().max() <= tolerance:
                        print(f"Found a validation set in iteration {i}")
                        val_fold_idx = df_val.index.to_numpy()
                        train_fold_idx = df_train.index.difference(val_fold_idx).to_numpy()

                        # Save
                        fold = {
                            "fold_id": fold_id,
                            "train_idx": train_fold_idx.tolist(),
                            "val_idx": val_fold_idx.tolist(),
                        }

                        # Save
                        fname = os.path.join(out_dir, f"fold_{scheme}_{fold_id}.json")
                        with open(fname, "w") as f:
                            json.dump(fold, f)

                        # Store folds by lat/lon step
                        fold_records.append(
                            {"scheme": scheme, "fold_id": fold_id, "split_id": split_id, "val_size": val_size, "n_splits": n_splits,
                             "latlon_step": ll_step, "filepath": fname})

                        fold_id += 1
                        found = True
                        break

                # Raise error, if no valid split was found after max_iter
                if not found:
                    raise RuntimeError(f"Could not generate fold {fold_id} for val_frac={val_size} at ll_step={ll_step}")

    return pd.DataFrame(fold_records)


def generate_random_folds(df, train_idx, n_splits, val_fractions, seed, out_dir):
    scheme = "random"
    df_train_val = df.loc[train_idx].copy()

    # Random splits
    random_splits = []
    fold_id = 0
    for split_id in range(n_splits):
        for val_size in val_fractions:
            # Set seed
            rng = seed + 100 * split_id + int(1000 * val_size)

            # Sample validation set from remaining data
            df_val = df_train_val.sample(frac=val_size, random_state=rng)
            va_idx = df_val.index.tolist()
            tr_idx = df_train_val.drop(df_val.index).index.tolist()

            fold = {
                "fold_id": fold_id,
                "train_idx": tr_idx,
                "val_idx": va_idx,
            }

            # Save
            fname = os.path.join(out_dir, f"fold_{scheme}_{fold_id}.json")
            with open(fname, "w") as f:
                json.dump(fold, f, indent=2)

            random_splits.append({"scheme": scheme, "fold_id": fold_id, "split_id": split_id, "n_splits": n_splits, "val_size": val_size, "filepath": fname})

            fold_id += 1

    return pd.DataFrame(random_splits)


def generate_test_set(df, latlon_step, depths, test_fraction, parameters, tolerance=0.05, max_iter=100, seed=42, save_as=None):
    """ Generate a test-train split so that the test set contains at least test_fraction rows and the
    parameter coverages resemble those of the original data set with a tolerance. """
    df_tmp = df.copy()

    # Set seed
    rng = np.random.default_rng(seed)

    # Assign blocks
    block_col = "BLOCK_ID"
    df_tmp[block_col] = assign_blocks(df_tmp, latlon_step=latlon_step, depth_edges=depths)

    # Compute global coverage ratios (per parameter)
    coverage_ratios = df_tmp[parameters].notna().mean()

    # Compute block sizes
    block_groups = df_tmp.groupby(block_col)
    block_sizes = block_groups.size()

    total_n = len(df_tmp)
    target_n = int(total_n * test_fraction)

    eligible_blocks = block_sizes.index.to_numpy()

    for i in range(max_iter):
        # Shuffle blocks
        blocks_shuffled = rng.permutation(eligible_blocks)

        # Init variables
        selected_blocks = []
        selected_rows = 0

        # Pick blocks until desired number of samples is reached
        for b in blocks_shuffled:
            selected_blocks.append(b)
            selected_rows += block_sizes[b]
            if selected_rows >= target_n:
                break

        # Create test set df
        df_test = df_tmp[df_tmp[block_col].isin(selected_blocks)]

        # Check coverage
        test_coverage_ratios = df_test[parameters].notna().mean()
        if (test_coverage_ratios - coverage_ratios).abs().max() <= tolerance:
            print(f"Found a test set in iteration {i}")
            test_idx = df_test.index.to_numpy()
            train_idx = df_tmp.index.difference(test_idx).to_numpy()

            # Save
            if save_as is not None:
                with open(save_as, "w") as f:
                    json.dump({"test_idx": test_idx.tolist(), "train_idx": train_idx.tolist()}, f)

            return test_idx, train_idx, df_tmp["BLOCK_ID"], test_coverage_ratios, coverage_ratios

    raise RuntimeError(f"Could not generate a test set")


if __name__ == "__main__":
    # Load data
    df = load_dataset()

    # Oceanographically relevant subdivisions
    latlons = [5, 15, 40]
    depths = [0, 200, 1000, 12000]  # epi meso bathy-pelagic
    seed = 42

    # Generate test set
    test_idx, train_idx, df["BLOCK_ID"], test_coverages, global_coverages = generate_test_set(df, latlon_step=latlons[1], depths=depths, test_fraction=config.test_fraction, parameters=config.parameters, tolerance=0.05, max_iter=100, seed=seed, save_as=Path(config.output_dir_splits) / "test_train_split.json")
    df_train = df.iloc[train_idx].copy().drop(columns=["BLOCK_ID"], axis=1)

    # Visualize test set

    # Generate train-validation folds
    df_random = generate_random_folds(df=df_train,
                                      train_idx=train_idx,
                                      n_splits=config.n_splits_per_scheme,
                                      val_fractions=config.val_fractions,
                                      seed=seed,
                                      out_dir=config.output_dir_splits)
    df_block = generate_block_folds(df=df_train,
                                      parameters=config.parameters,
                                      latlon_steps=latlons,
                                      depths=depths,
                                      val_fractions=config.val_fractions,
                                      n_splits=config.n_splits_per_scheme,
                                      train_idx=train_idx,
                                      seed=seed,
                                      out_dir=config.output_dir_splits,
                                      tolerance=0.05,
                                      max_iter=100)

    # Save metadata
    pd.concat([df_random, df_block]).to_csv(os.path.join(config.output_dir_splits, "split_metadata.csv"), index=False)

    # Plotting
    # Color categories
    CATEGORY_COLORS = {"train": "tab:gray", "validation": "tab:blue", "test": "tab:red", "unobserved": "white"}
    categories = list(CATEGORY_COLORS.keys())
    cat_to_int = {cat: i for i, cat in enumerate(categories)}
    cmap = ListedColormap([CATEGORY_COLORS[cat] for cat in categories])

    # Load global test split
    test_split_path = os.path.join(config.output_dir_splits, "test_train_split.json")
    with open(test_split_path, "r") as f:
        test_split = json.load(f)

    test_idx_global = np.array(test_split["test_idx"])
    train_idx_global = np.array(test_split["train_idx"])

    # Plot test-train split
    tmp = df.copy()
    tmp["color"] = CATEGORY_COLORS["unobserved"]
    tmp.loc[train_idx_global, "color"] = CATEGORY_COLORS["train"]
    tmp.loc[test_idx_global, "color"] = CATEGORY_COLORS["test"]
    for year in np.sort(tmp.DATEANDTIME.unique()):
        plot_geo(tmp[tmp.DATEANDTIME == year], color_label="color", save_as=None)
        plt.title(f"{year}")
        plt.show(block=True)

    # Plot train-validation splits
    for fname in glob.glob(os.path.join(config.output_dir_splits, "fold_*.json")):
        print(fname)
        temp = df.copy()

        # Load split
        with open(fname, "r") as f:
            fold = json.load(f)
        train_idx = np.array(fold["train_idx"])
        val_idx = np.array(fold["val_idx"])

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
