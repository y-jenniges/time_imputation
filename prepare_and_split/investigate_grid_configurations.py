from pathlib import Path
from time import time
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import os
import sqlite3
from kneed import KneeLocator
from scipy.interpolate import UnivariateSpline

import config
from utils.preparation import prepare_database, grid_data_as_df


# # Quick global imputation
# from sklearn.impute import SimpleImputer
# from utils.plotting import plot_each_depth_level, animate_depth_panels
# from utils.gridding import df_to_gridded_da
# df = pd.read_csv("output/gridding/df_20y_global.csv")
# df["DATEANDTIME"] = pd.to_datetime(df["DATEANDTIME"]).dt.year
# x, y, scaler_dicts = preprocess(df, coord_names=config.coordinates, parameter_names=config.parameters)
# cols = ["LATITUDE", "cLONGITUDE", "sLONGITUDE", "LEV_M", "DATEANDTIME"] + config.parameters
# imputer = SimpleImputer()
# df_imputed = pd.DataFrame(imputer.fit_transform(np.concat([x, y], axis=1)), columns=cols)
# df_imputed[config.coordinates] = df[config.coordinates]
#
# param = "P_TEMPERATURE"
# da = df_to_gridded_da(df=df_imputed, value_col=param)
# animate_depth_panels(da, save_as=f"output/{param}_knn_imputation.mp4")


if __name__ == "__main__":
    # --- Prepare database ---------------------------------------------------------------------------- #
    if not Path(config.dest_db_path).exists():
        prepare_database(parameters=config.parameters,
                         quality_flags=config.quality_flags,
                         temperature_to_potential=True,
                         source_db_path=config.source_db_path,
                         dest_db_path=config.dest_db_path)

    # --- Create various grids ------------------------------------------------------------------------ #
    # Load base grid config
    grid_config = config.grid_configs["avg_na"]

    # Iterate over potential time steps
    for dt in config.time_steps:
        grid_config["dtime"] = dt
        grid_id = str(dt) + "y_na"
        print(grid_id)
        if Path("output/gridding", "df_" + grid_id + ".csv").exists():
            continue
        df, table_name = grid_data_as_df(db_path=config.dest_db_path, grid_config=grid_config,
                                         bathymetry_path=config.bathymetry_path,
                                         parameters=config.parameters,
                                         output_dir=config.output_dir)

        # Store
        df.to_csv(f"{config.output_dir_gridding}df_{grid_id}.csv", index=False)

        # Plot missingness per time step
        total_grid_cells_per_time = df.groupby("DATEANDTIME")["LATITUDE"].count()
        missing_counts = df.groupby("DATEANDTIME")[config.parameters].apply(lambda x: x.isna().sum())
        missing_fraction = missing_counts.div(total_grid_cells_per_time, axis=0) * 100
        missing_fraction["Year"] = pd.to_datetime(missing_fraction.index).year
        missing_fraction["overall"] = missing_fraction[config.parameters].mean(axis=1)
        missing_fraction["grid_id"] = grid_id

        plt.figure()
        for feat in config.parameters:
            plt.plot(missing_fraction["Year"], missing_fraction[feat], label=config.parameter_name_map[feat], marker="o", markersize=3)

        plt.ylabel("Missigness [%]")
        plt.xlabel("Year")
        plt.legend()

        ax = plt.gca()
        ax.set_xticks(missing_fraction["Year"])
        ax.set_xticklabels(missing_fraction["Year"].astype(str), rotation=90)

        plt.grid(True, which="major", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(config.output_dir_gridding + f"/missingness_{grid_id}.png", dpi=1000)
        plt.show()

    # --- Analyse grids ------------------------------------------------------------------------------- #
    # Analyse trade-off between coverage and resolution
    results = []
    for dt in config.time_steps:
        fname = "df_" + str(dt) + "y_na.csv"
        print(fname)
        df = pd.read_csv(os.path.join("output/gridding", fname))

        dt = fname.split("_")[1].rstrip("y")
        if dt == "avg":
            dt = 300
        else:
            dt = int(dt)

        # Total grid cells per timestep (will be constant)
        total_grid_cells_per_time = df.groupby("DATEANDTIME")["LATITUDE"].count()

        # Missing counts per variable
        missing_counts = df.groupby("DATEANDTIME")[config.parameters].apply(lambda x: x.isna().sum())

        # Fraction of missing values [%]
        missing_fraction = missing_counts.div(total_grid_cells_per_time, axis=0) * 100

        # Overall coverage per timestep
        missing_fraction["coverage"] = 100 - missing_fraction.mean(axis=1)

        # Mean coverage over all timesteps
        mean_coverage = missing_fraction["coverage"].mean()

        results.append({
            "grid_id": fname.replace(".csv", ""),
            "dt": dt,
            "mean_coverage": mean_coverage,
        })

    df_scores = pd.DataFrame(results).sort_values("dt", ascending=True)
    print(df_scores)

    # Plotting
    x = df_scores["dt"].values
    y = df_scores["mean_coverage"].values

    # Smoothing
    spline = UnivariateSpline(x, y, s=2)
    y_smooth = spline(x)

    # KneeLocator automatically finds the "elbow"
    knee = KneeLocator(x, y_smooth, curve='convex', direction='increasing')
    print("Elbow detected at dt =", knee.knee)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, marker='o')
    plt.plot(x, y_smooth, '-', label='Smoothed', color="orange")
    plt.axvline(knee.knee, color='r', linestyle='--', label=f'Elbow at dt={knee.knee}')
    mask_dense = (x >= 10) & (x <= 40)
    xticks = np.concatenate([x[~mask_dense], x[mask_dense][::10]])
    plt.xticks(sorted(xticks))
    plt.xlabel("Time resolution [years]")
    plt.ylabel("Mean coverage")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config.output_dir_gridding + "/time_resolution_vs_coverage.png", dpi=1000)
    plt.show()
