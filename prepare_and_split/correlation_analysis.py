import seaborn as sns
import matplotlib.pyplot as plt
from pyproj import Transformer
from skgstat import Variogram
from statsmodels.tsa.stattools import acf
import os
import pandas as pd

import matplotlib
matplotlib.use("Qt5Agg")

import config
from nn_utils.dataset import load_dataset


if __name__ == "__main__":
    # Load data
    df = load_dataset()

    # Result containers
    decorrelation_ranges = []
    temporal_autocorrelations = []

    for param in config.parameters:
        print(param)
        df_sub = df[["LONGITUDE", "LATITUDE", "LEV_M", "DATEANDTIME", param]].dropna().sample(n=10000, random_state=42)

        # Transform latitude and longitude
        transformer = Transformer.from_crs(
            "EPSG:4326",
            f"+proj=laea +lat_0={df_sub['LATITUDE'].mean()} +lon_0={df_sub['LONGITUDE'].mean()} +datum=WGS84 +units=m +no_defs",
            # Lambert Azimuthal Equal Area (LAEA) centered dynamically on mean lat/lon using WGS84
            always_xy=True
        )
        x, y = transformer.transform(df_sub["LONGITUDE"], df_sub["LATITUDE"])
        df_sub["LONGITUDE"] = x / 1000
        df_sub["LATITUDE"] = y / 1000

        # Variograms
        spatial_v = Variogram(coordinates=df_sub[["LONGITUDE", "LATITUDE"]].values / 1000.0, values=df_sub[param].values, use_nugget=True)
        depth_v = Variogram(coordinates=df_sub[["LEV_M"]].values, values=df_sub[param].values, use_nugget=True)
        time_v = Variogram(coordinates=df_sub[["DATEANDTIME"]].values, values=df_sub[param].values, use_nugget=True)

        # Variogram plotting
        fig_s = spatial_v.plot()
        fig_s.axes[1].set_ylabel("N samples")
        fig_s.axes[0].set_xlabel("Distance [km]")
        fig_s.axes[0].set_ylabel("Semivariance")
        plt.tight_layout()
        plt.savefig(config.output_dir_high_res_plots + f"/spatial_variogram_{param}.png", dpi=1000)
        plt.show()
        plt.close()

        fig_d = depth_v.plot()
        fig_d.axes[1].set_ylabel("N samples")
        fig_d.axes[0].set_xlabel("Distance [m]")
        fig_d.axes[0].set_ylabel("Semivariance")
        plt.tight_layout()
        plt.savefig(config.output_dir_high_res_plots + f"/depth_variogram_{param}.png", dpi=1000)
        plt.show()
        plt.close()

        fig_t = time_v.plot()
        fig_t.axes[1].set_ylabel("N samples")
        fig_t.axes[0].set_xlabel("Distance [years]")
        fig_t.axes[0].set_ylabel("Semivariance")
        plt.tight_layout()
        plt.savefig(config.output_dir_high_res_plots + f"/time_variogram_{param}.png", dpi=1000)
        plt.show()
        plt.close()

        # Variogram results
        spatial_range, spatial_sill, spatial_nugget = spatial_v.parameters
        depth_range, depth_sill, depth_nugget = depth_v.parameters
        time_range, time_sill, time_nugget = time_v.parameters

        temp_decorr = pd.DataFrame({ "parameter": [param],
            "spatial_range": [spatial_range], "spatial_sill": [spatial_sill], "spatial_nugget": [spatial_nugget],
            "depth_range": [depth_range], "depth_sill": [depth_sill], "depth_nugget": [depth_nugget],
            "time_range": [time_range], "time_sill": [time_sill], "time_nugget": [time_nugget],
        })
        decorrelation_ranges.append(temp_decorr)
        print(temp_decorr)

        # Temporal autocorrelation
        time_means = df_sub.groupby("DATEANDTIME")[param].mean()  # Group by time
        acf_vals = acf(time_means.values, fft=True)
        temp_ta = pd.DataFrame({"parameter": [param]*len(acf_vals), "acf_vals": acf_vals, "lag": range(len(acf_vals))})
        temporal_autocorrelations.append(temp_ta)
        print(temp_ta)

    df_decorrelation_ranges = pd.concat(decorrelation_ranges, ignore_index=True)
    df_temporal_autocorrelations = pd.concat(temporal_autocorrelations, ignore_index=True)

    df_decorrelation_ranges.to_csv(os.path.join(config.output_dir_correlations, "variogram_data.csv"), index=False)
    df_temporal_autocorrelations.to_csv(os.path.join(config.output_dir_correlations, "temporal_autocorrelations.csv"), index=False)

    # Plot combined temporal autocorrelation ranges
    temp = df_temporal_autocorrelations.copy().reset_index(drop=True)
    temp["parameter"] = temp["parameter"].map(config.parameter_name_map)
    temp_pivot = temp.groupby(["parameter", "lag"])["acf_vals"].mean().unstack()

    plt.figure(figsize=(8, 4))
    sns.heatmap(temp_pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.xlabel("Lag (20 years)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(config.output_dir_high_res_plots + "temporal_autocorrelation.png", dpi=1000)
    plt.show()
    plt.close()
