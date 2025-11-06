import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skgstat import Variogram
from statsmodels.tsa.stattools import acf
from libpysal.weights import DistanceBand
from esda.moran import Moran
import os

import matplotlib
matplotlib.use('Qt5Agg')

pd.options.display.width= None
pd.options.display.max_columns= None
pd.options.display.max_rows = None

import config
from utils.gridding import df_to_gridded_da



def moran_correlogram(x, y, values, max_dist, n_bins=10):
    coords_m = np.column_stack((x, y))

    bins = np.linspace(0, max_dist, n_bins+1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    I_values, p_values = [], []
    for i in range(n_bins):
        w = DistanceBand(coords_m, threshold=bins[i+1], binary=True)
        mi = Moran(values, w)
        I_values.append(mi.I)
        p_values.append(mi.p_sim)

    return bin_centers, np.array(I_values), np.array(p_values)


if __name__ == "__main__":
    df = pd.read_csv(config.data_path)
    df["DATEANDTIME"] = pd.to_datetime(df["DATEANDTIME"]).dt.year

    # Convert degrees to km
    R = 6371000  # Earth radius in meters
    lat = np.deg2rad(df["LATITUDE"])
    lon = np.deg2rad(df["LONGITUDE"])
    df["x"] = R * lon * np.cos(lat)
    df["y"] = R * lat

    # Result containers
    decorrelation_ranges = []
    spatial_autocorrelations = []
    temporal_autocorrelations = []

    for param in config.parameters:
        print(param)
        df_sub = df[['x', 'y', 'LEV_M', 'DATEANDTIME', param]].dropna().sample(n=10000, random_state=42)

        # Variograms
        spatial_v = Variogram(coordinates=df_sub[["x", "y"]].values / 1000.0, values=df_sub[param].values)
        depth_v = Variogram(coordinates=df_sub[["LEV_M"]].values, values=df_sub[param].values)
        time_v = Variogram(coordinates=df_sub[["DATEANDTIME"]].values, values=df_sub[param].values)

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

        # Spatial correlogram using Moran's I
        bin_centers, morans_Is, morans_ps = moran_correlogram(x=df_sub["x"].values,
                                                y=df_sub["y"].values,
                                                values=df_sub[param].values,
                                                max_dist=600000, n_bins=15)

        temp_sa = pd.DataFrame({
            "parameter": [param]*len(bin_centers), "bin_centers": bin_centers, "morans_I": morans_Is, "morans_ps": morans_ps})
        spatial_autocorrelations.append(temp_sa)
        print(temp_sa)

        # Temporal autocorrelation
        time_means = df_sub.groupby("DATEANDTIME")[param].mean()  # Group by time
        acf_vals = acf(time_means.values, fft=True)
        temp_ta = pd.DataFrame({"parameter": [param]*len(acf_vals), "acf_vals": acf_vals, "lag": range(len(acf_vals))})
        temporal_autocorrelations.append(temp_ta)
        print(temp_ta)

    df_decorrelation_ranges = pd.concat(decorrelation_ranges, ignore_index=True)
    df_spatial_autocorrelations = pd.concat(spatial_autocorrelations, ignore_index=True)
    df_temporal_autocorrelations = pd.concat(temporal_autocorrelations, ignore_index=True)

    df_decorrelation_ranges.to_csv(os.path.join(config.output_dir_correlations, "variogram_data.csv"), index=False)
    df_spatial_autocorrelations.to_csv(os.path.join(config.output_dir_correlations, "spatial_autocorrelations.csv"), index=False)
    df_temporal_autocorrelations.to_csv(os.path.join(config.output_dir_correlations, "temporal_autocorrelations.csv"), index=False)


    # Plot combined spatial correlogram
    parameter_name_map = {"P_TEMPERATURE": "Potential temperature [°C]",
                          "P_SALINITY": "Salinity [PSU]",
                          "P_OXYGEN": "Oxygen [µmol/kg]",
                          "P_NITRATE": "Nitrate [µmol/kg]",
                          "P_SILICATE": "Silicate [µmol/kg]",
                          "P_PHOSPHATE": "Phosphate [µmol/kg]"}
    temp = df_spatial_autocorrelations.copy()
    temp["bin_centers"] = temp["bin_centers"] / 1000
    temp["parameter"] = temp["parameter"].map(parameter_name_map)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(temp, x="bin_centers", y="morans_I", hue="parameter", marker="o")
    # Overlay gray markers for bins with p > 0.05 (not significant)
    for param in temp["parameter"].unique():
        sub = temp[temp["parameter"] == param]
        non_sig = sub["morans_ps"] > 0.05
        ax.scatter(
            sub["bin_centers"][non_sig],
            sub["morans_I"][non_sig],
            color="black",
            edgecolor="white",
            zorder=5,
            label="_nolegend_"
        )
    plt.xlabel("Distance [km]")
    plt.ylabel("Moran's I")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Parameter", fontsize=10, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(config.output_dir_high_res_plots + f"spatial_correlogram.png", dpi=1000)
    plt.show()
    plt.close()

    # Plot combined temporal autocorrelation ranges
    temp = df_temporal_autocorrelations.copy().reset_index(drop=True)
    temp["parameter"] = temp["parameter"].map(parameter_name_map)
    temp_pivot = temp.groupby(["parameter", "lag"])["acf_vals"].mean().unstack()

    plt.figure(figsize=(8, 4))
    sns.heatmap(temp_pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.xlabel("Lag (20 years)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(config.output_dir_high_res_plots + "temporal_autocorrelation.png", dpi=1000)
    plt.show()
    plt.close()

    # Some profile fun
    for param in config.parameters:
        df_temp = df[['LEV_M', param]].dropna().copy()

        # Compute mean and std per depth level
        mean_profile = df_temp.groupby('LEV_M')[param].mean()
        std_profile = df_temp.groupby('LEV_M')[param].std()
        depths = mean_profile.index.values

        # Plot mean ± SD
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.plot(mean_profile, depths, color='steelblue', label='Mean profile')
        # Scatter points for individual measurements
        ax.scatter(df_temp[param], df_temp['LEV_M'], color='gray', alpha=0.3, s=1, label='Measurements')
        # Profile plot
        ax.fill_betweenx(depths,
                         mean_profile - std_profile,
                         mean_profile + std_profile,
                         color='steelblue', alpha=0.3, label='±1 SD')
        ax.invert_yaxis()
        ax.set_xlabel(parameter_name_map[param])
        ax.set_ylabel("Depth [m]")
        ax.set_title(f"{parameter_name_map[param]} profile (mean ± std)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(config.output_dir_plots +  "/profile_" + param + ".png")
        plt.show()
