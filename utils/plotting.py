import logging
import gsw
import glasbey
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import pearsonr
from shapely.geometry import box
from cartopy.io import shapereader
import matplotlib.animation as animation
import seaborn as sns
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

import config
from utils.gridding import df_to_gridded_da


# def init_logger(logger_name, log_file, level=logging.INFO):
#     """ Set up a logger with output to commandline and a file. """
#     formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
#
#     # File handler
#     fh = logging.FileHandler(log_file)
#     fh.setFormatter(formatter)
#
#     # Command line handler
#     ch = logging.StreamHandler()
#     ch.setFormatter(formatter)
#
#     # Set logging level
#     logger = logging.getLogger(logger_name)
#     logger.setLevel(level)
#
#     # Avoid duplicate handlers if reinitializing
#     if not logger.handlers:
#         logger.addHandler(fh)
#         logger.addHandler(ch)
#
#     # Prevent duplicate output if there is a root logger
#     logger.propagate = False
#
#     return logger
#
#
# def geo_to_cartesian(lat, lon, depth, R=6371000.0):
#     """
#     Convert geographic coordinates to 3D Cartesian.
#     lat, lon in degrees
#     depth in meters (positive down)
#     R = Earth radius in meters
#     """
#     lat_rad = np.radians(lat)
#     lon_rad = np.radians(lon)
#     r = 1.0 - depth / R
#     x = r * np.cos(lat_rad) * np.cos(lon_rad)
#     y = r * np.cos(lat_rad) * np.sin(lon_rad)
#     z = r * np.sin(lat_rad)
#     # Scale from [-1, 1] to [0, 1]
#     x = (x + 1) / 2
#     y = (y + 1) / 2
#     z = (z + 1) / 2
#     return np.array([x, y, z])
#
#
# def geo_distance(lat1, lon1, depth1, lat2, lon2, depth2):
#     p1 = geo_to_cartesian(lat1, lon1, depth1)
#     p2 = geo_to_cartesian(lat2, lon2, depth2)
#     return np.linalg.norm(p1 - p2)


def plot_time_series_for_location(df, y, y_imputed, var_pred, observed_mask, var_names,
                                  lat=None, lon=None, lev=None, var_idx=0, auto_pick=True):
    """
    var_idx: index of variable to plot (0-based)
    If auto_pick=True, selects the location with most months present.
    """
    # convert month column to int if needed
    months_col = df['DATEANDTIME'].astype(int).values

    # find candidate location index set
    if auto_pick:
        # group by location and count non-NaN months
        grp = df.groupby(['LATITUDE','LONGITUDE','LEV_M']).size().reset_index(name='count')
        best = grp.sort_values('count', ascending=False).iloc[0]
        lat0, lon0, lev0 = best['LATITUDE'], best['LONGITUDE'], best['LEV_M']
    else:
        if lat is None or lon is None or lev is None:
            raise ValueError("Specify lat,lon,lev if auto_pick=False")
        lat0, lon0, lev0 = lat, lon, lev

    sel = (df['LATITUDE'] == lat0) & (df['LONGITUDE'] == lon0) & (df['LEV_M'] == lev0)
    idxs = np.where(sel.values)[0]
    if len(idxs) == 0:
        raise ValueError("No rows found for that location.")

    # sort by month
    months = months_col[idxs]
    order = np.argsort(months)
    idxs = idxs[order]
    months = months[order]

    obs = y[idxs, var_idx]
    imp = y_imputed[idxs, var_idx]
    std = np.sqrt(var_pred[idxs, var_idx])
    obs_mask_loc = observed_mask[idxs, var_idx].astype(bool)

    plt.figure(figsize=(8,4))
    # plot imputed mean line
    plt.plot(months, imp, label='Imputed mean', color='C0', marker='o')

    # shading ±1σ
    plt.fill_between(months, imp - std, imp + std, color='C0', alpha=0.25, label='±1σ')

    # show observed points
    plt.scatter(months[obs_mask_loc], obs[obs_mask_loc], color='k', label='Observed', zorder=3)

    # mark imputed-only points
    imputed_only_mask = ~obs_mask_loc
    if imputed_only_mask.any():
        plt.scatter(months[imputed_only_mask], imp[imputed_only_mask], marker='x', color='red', label='Imputed only')

    plt.xlabel('Month')
    plt.ylabel(var_names[var_idx])
    plt.title(f"Location time series at lat={lat0}, lon={lon0}, lev={lev0} — var={var_names[var_idx]}")
    plt.xticks(np.arange(1,13))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)


def plot_regional_monthly_mean(df, y_imputed, var_pred, var_names, var_idx=0):
    """
    Aggregates by month across all rows in df (assumes df rows correspond to entries in y_imputed).
    Plots monthly mean of imputed values and mean predicted std.
    """
    months = df['DATEANDTIME'].astype(int).values
    month_order = np.arange(1,13)

    mean_imps = []
    mean_stds = []
    for m in month_order:
        idxs = np.where(months == m)[0]
        if len(idxs) == 0:
            mean_imps.append(np.nan)
            mean_stds.append(np.nan)
            continue
        mean_imps.append(np.nanmean(y_imputed[idxs, var_idx]))
        mean_stds.append(np.nanmean(np.sqrt(var_pred[idxs, var_idx])))

    mean_imps = np.array(mean_imps)
    mean_stds = np.array(mean_stds)

    plt.figure(figsize=(8,4))
    plt.plot(month_order, mean_imps, marker='o', label='Regional mean (imputed)')
    plt.fill_between(month_order, mean_imps - mean_stds, mean_imps + mean_stds, alpha=0.25, label='Mean ± mean std')
    plt.xlabel('Month')
    plt.ylabel(var_names[var_idx])
    plt.title(f"Regional monthly mean ± mean std — var={var_names[var_idx]}")
    plt.xticks(month_order)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)


def animate_depth_feature(df, feature, interval=500, grid_res=100, cmap="inferno", save_path=None):
    """
    Animate spatial evolution of a feature across depth levels over time
    with filled surfaces, 'inferno' colormap, and coastlines.
    """
    time_col = "DATEANDTIME"  # YEAR
    months = sorted(df[time_col].unique())
    # years = sorted(df["YEAR"].unique())
    depth_levels = sorted(df["LEV_M"].unique())
    n_depths = len(depth_levels)

    # Grid layout
    ncols = 4
    nrows = int(np.ceil(n_depths / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(15, 8),
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True
    )
    axes = axes.flatten()

    # Global limits for axes and color
    vmin, vmax = df[feature].min(), df[feature].max()
    lon_min, lon_max = df["LONGITUDE"].min(), df["LONGITUDE"].max()
    lat_min, lat_max = df["LATITUDE"].min(), df["LATITUDE"].max()

    # Create regular interpolation grid
    lon_grid = np.linspace(lon_min, lon_max, grid_res)
    lat_grid = np.linspace(lat_min, lat_max, grid_res)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Initialize pcolormesh for each depth
    surfaces = []
    for ax, depth in zip(axes, depth_levels):
        ax.set_title(f"Depth {depth} m")
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.gridlines(draw_labels=True)

        # initialize with NaNs
        init_grid = np.full_like(lon_mesh, np.nan)
        surf = ax.pcolormesh(
            lon_mesh, lat_mesh, init_grid, cmap=cmap,
            vmin=vmin, vmax=vmax, shading='auto',
            transform=ccrs.PlateCarree()
        )
        surfaces.append(surf)

    # Hide unused subplots
    for ax in axes[n_depths:]:
        ax.set_visible(False)

    # Shared colorbar
    cbar = fig.colorbar(surfaces[0], ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label(feature)

    # Update function
    def update(frame):
        year = months[frame]
        fig.suptitle(f"{feature} - {time_col} {year}", fontsize=16)

        for surf, depth in zip(surfaces, depth_levels):
            df_slice = df[(df[time_col] == year) & (df["LEV_M"] == depth)]
            if len(df_slice) > 0:
                grid_vals = griddata(
                    (df_slice["LONGITUDE"], df_slice["LATITUDE"]),
                    df_slice[feature],
                    (lon_mesh, lat_mesh),
                    method='linear'
                )
                surf.set_array(grid_vals.ravel())
            else:
                surf.set_array(np.full_like(lon_mesh.ravel(), np.nan))

        return surfaces

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(months),
        interval=interval,
        blit=False,
        repeat=True
    )

    if save_path:
        ani.save(save_path, fps=1000 // interval, dpi=150)

    plt.show()
    return ani


def plot_loss(loss_dict, close_plot=False, save_as=None):
    plt.figure(figsize=(6, 4), dpi=150)

    if isinstance(loss_dict, dict) and 'train' in loss_dict:
        # Train and validation loss
        train_epochs = sorted(loss_dict['train'].keys())
        train_losses = [loss_dict['train'][e] for e in train_epochs]
        plt.plot(train_epochs, train_losses, '-o', label='Train', color='#1f77b4', markersize=4, linewidth=1.5)

        if 'val' in loss_dict:
            val_epochs = sorted(loss_dict['val'].keys())
            val_losses = [loss_dict['val'][e] for e in val_epochs]
            plt.plot(val_epochs, val_losses, '-s', label='Validation', color='#ff7f0e', markersize=4, linewidth=1.5)
    else:
        # Train loss only
        epochs = sorted(loss_dict.keys())
        losses = [loss_dict[e] for e in epochs]
        plt.plot(epochs, losses, '-o', color='#1f77b4', markersize=4, linewidth=1.5, label='Training')

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    # plt.title("Training Loss per Epoch", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    if close_plot:
        plt.close()
    else:
        plt.show()


def plot_reconstruction_error(pred_all, values, test_mask, total_unc, parameter_list, cbar_label="Aleatoric uncertainty"):
    """
    3x2 grid of true vs reconstructed values with Pearson correlation and uncertainty as color.
    Colorbar placed outside the plots on the right.
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    n_features = len(parameter_list)
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.4)

    # Convert to float64 to avoid weird matplotlib top-tick issues
    total_unc = total_unc.cpu().numpy().astype(np.float64)

    # Determine global min/max for color scale
    unc_min = total_unc.min()
    unc_max = total_unc.max()

    sc = None  # scatter reference for colorbar

    for i, pname in enumerate(parameter_list):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])

        true_vals = values[:, i].cpu().numpy()
        pred_vals = pred_all[:, i].cpu().numpy()
        test_idx = test_mask[:, i].bool().cpu().numpy()

        # Scatter colored by uncertainty
        sc = ax.scatter(
            true_vals[test_idx],
            pred_vals[test_idx],
            c=total_unc[test_idx, i],
            cmap='viridis',
            vmin=unc_min,
            vmax=unc_max,
            alpha=0.6,
            edgecolor='none',
            s=12
        )

        # Diagonal
        min_val = np.min(true_vals[test_idx])
        max_val = np.max(true_vals[test_idx])
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')

        # Pearson correlation
        corr = pearsonr(true_vals[test_idx], pred_vals[test_idx])[0] if np.sum(test_idx) > 1 else np.nan

        ax.set_xlabel(f"True {pname}")
        ax.set_ylabel(f"Pred {pname}")
        ax.set_title(f"{pname} (r={corr:.2f})")
        ax.grid(True, alpha=0.3)
        ax.axis('square')
        ax.legend()

    # Add shared colorbar outside
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=15)

    plt.show()


def plot_simple_reconstruction_error(y_true, y_pred, save_as=None, close=False):
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
    axes = axes.flatten()  # flatten to index easily

    for i, feature_name in enumerate(config.parameters):
        a = y_true[:, i]
        b = y_pred[:, i]

        # Remove NaNs
        mask = ~np.isnan(a) & ~np.isnan(b)
        a = a[mask]
        b = b[mask]

        ax = axes[i]
        hb = ax.hexbin(a, b, gridsize=40, cmap="viridis", mincnt=1, bins="log")

        # Correlation
        r, _ = pearsonr(a, b)
        ax.text(
            0.05, 0.95, f"r = {r:.2f}",
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )

        # 1:1 reference line
        min_val = min(a.min(), b.min())
        max_val = max(a.max(), b.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(config.parameter_name_map[feature_name])
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    cax = fig.add_axes((0.90, 0.15, 0.02, 0.7))
    fig.colorbar(hb, cax=cax, label='Point count')

    if save_as is not None:
        plt.savefig(save_as)

    if close:
        plt.close()
    else:
        plt.show()


def plot_depth_panels(
    data,
    depth_dim="depth",
    cmap="inferno",
    vmin=None,
    vmax=None,
    norm=None,
    ncols=4,
    interval=400,
    save_as=None,
    coast_color="lightgrey",
    dpi=150
    ):
    """
    Plot multiple depth-level panels for a 3D DataArray (depth × lat × lon).
    Each subplot corresponds to one depth level.
    """

    # --- Validate input
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D DataArray with dims (depth, lat, lon). Got ndim = {data.ndim}")

    data = data.transpose(depth_dim, "lat", "lon")
    depths = data[depth_dim].values
    ndepths = len(depths)
    nrows = int(np.ceil(ndepths / ncols))

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 3.2, nrows * 2.6),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    axs = axs.flatten()

    if vmin is None:
        vmin = float(data.min())
    if vmax is None:
        vmax = float(data.max())

    pcm_list = []

    for i, d in enumerate(depths):
        ax = axs[i]
        ax.set_extent([float(data.lon.min()), float(data.lon.max()),
                       float(data.lat.min()), float(data.lat.max())])
        ax.add_feature(cfeature.LAND, facecolor=coast_color, zorder=0)
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}

        da2d = data.sel({depth_dim: d})

        if norm is not None:
            pcm = ax.pcolormesh(
                data.lon, data.lat, da2d,
                cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree(),
            )
        else:
            pcm = ax.pcolormesh(
                data.lon, data.lat, da2d,
                cmap=cmap, vmin=vmin, vmax=vmax,
                transform=ccrs.PlateCarree(),
            )
        ax.set_title(f"{d} m", fontsize=9)
        pcm_list.append(pcm)

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    cbar = fig.colorbar(pcm_list[0], ax=axs, orientation="vertical",
                        shrink=0.9, aspect=30, pad=0.02)
    cbar.set_label(data.name)

    # Save
    if save_as:
        plt.savefig(save_as, dpi=dpi)



def animate_depth_panels(
    data,
    depth_dim="depth",
    cmap="inferno",
    vmin=None,
    vmax=None,
    norm=None,
    ncols=4,
    interval=400,
    save_as=None,
    coast_color="lightgrey"):
    """
    Animate multiple depth-level panels for a 4D DataArray (time × depth × lat × lon).
    Each subplot corresponds to one depth level, and all update together per frame.
    Assumes 'time' coordinate contains integers (e.g., years) or datetimes.
    """

    # --- Validate input
    if data.ndim != 4:
        raise ValueError("Expected a 4D DataArray with dims (time, depth, lat, lon).")

    data = data.transpose("time", depth_dim, "lat", "lon")
    times = np.sort(np.unique(data["time"]))
    depths = data[depth_dim].values
    nframes = len(times)
    ndepths = len(depths)
    nrows = int(np.ceil(ndepths / ncols))

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 3.2, nrows * 2.6),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    axs = axs.flatten()

    if vmin is None:
        vmin = float(data.min())
    if vmax is None:
        vmax = float(data.max())

    pcm_list = []

    for i, d in enumerate(depths):
        ax = axs[i]
        ax.set_extent([float(data.lon.min()), float(data.lon.max()),
                       float(data.lat.min()), float(data.lat.max())])
        ax.add_feature(cfeature.LAND, facecolor=coast_color, zorder=0)
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}

        da2d = data.isel(time=0).sel({depth_dim: d})

        if norm is not None:
            pcm = ax.pcolormesh(
                data.lon, data.lat, da2d,
                cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree(),
            )
        else:
            pcm = ax.pcolormesh(
                data.lon, data.lat, da2d,
                cmap=cmap, vmin=vmin, vmax=vmax,
                transform=ccrs.PlateCarree(),
            )
        ax.set_title(f"{d} m", fontsize=9)
        pcm_list.append(pcm)

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    cbar = fig.colorbar(pcm_list[0], ax=axs, orientation="vertical",
                        shrink=0.9, aspect=30, pad=0.02)
    cbar.set_label(data.name)

    # Initialize title
    time_label = fig.suptitle(f"Year: {times[0]}", fontsize=12)

    # Update function
    def update(frame_idx, pcm_list, time_label):
        frame_data = data.isel(time=frame_idx)
        for k, d in enumerate(depths):
            arr = frame_data.sel({depth_dim: d}).values
            pcm_list[k].set_array(arr.ravel())
        # Update suptitle
        time_label.set_text(f"Year: {times[frame_idx]}")
        return pcm_list + [time_label]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=nframes,
        fargs=(pcm_list, time_label),
        interval=interval,
        blit=False
    )

    # --- Save or show
    if save_as:
        save_as = str(save_as)
        if save_as.endswith(".mp4"):
            writer = animation.FFMpegWriter(fps=1000 / interval)
        elif save_as.endswith(".gif"):
            writer = animation.PillowWriter(fps=1000 / interval)
        else:
            raise ValueError("save_as must end with .mp4 or .gif")

        anim.save(save_as, writer=writer, dpi=150)
        print(f"Animation saved as {save_as}")
        plt.close(fig)
    else:
        plt.show()


def generate_animation(df_imputed, scaler_dict=None, parameter="P_TEMPERATURE", save_as=None, cmap="inferno", norm=None):
    temp = df_imputed.copy()

    # Undo scaling
    if scaler_dict is not None:
        for param, scaler in scaler_dict.items():
            # Only unscale parameters
            if param in config.parameters:
                temp[param] = scaler.inverse_transform(temp[param].values.reshape(-1, 1))

    # Transform to xarray
    ds = df_to_gridded_da(temp, value_col=parameter)

    # Animate and save
    if "DATEANDTIME" in df_imputed.columns and len(df_imputed["DATEANDTIME"].unique()) > 1:
        animate_depth_panels(data=ds, save_as=save_as, cmap=cmap, norm=norm)
    else:
        if "time" in ds.dims:
            ds = ds.squeeze(dim="time")
        plot_depth_panels(data=ds, save_as=save_as, cmap=cmap, norm=norm)


def plot_profile(df, param, figsize=(6, 8), save_as=None, dpi=300):
    # Depth profile
    df_temp = df[["LEV_M", param]].dropna().copy()

    # Compute mean and std per depth level
    mean_profile = df_temp.groupby("LEV_M")[param].mean()
    std_profile = df_temp.groupby("LEV_M")[param].std()
    depths = mean_profile.index.values

    # Plot mean ± SD
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("none")
    ax.plot(mean_profile, depths, color="steelblue", label="Mean profile")

    # Scatter points for individual measurements
    ax.scatter(df_temp[param], df_temp["LEV_M"], color="gray", alpha=0.3, s=1, label="Measurements")

    # Profile plot
    ax.fill_betweenx(depths,
                     mean_profile - std_profile,
                     mean_profile + std_profile,
                     color="steelblue", alpha=0.3, label="±1 SD")
    ax.invert_yaxis()
    ax.set_xlabel(config.parameter_name_unit_map[param])
    ax.set_ylabel("Depth [m]")
    ax.set_title(f"{config.parameter_name_map[param]} profile (mean ± std)")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend()

    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=dpi, transparent=True)


def plot_boxplots(df, config, figsize=(2 * 6, 10), save_as=None, dpi=300):
    # Box plots of parameters
    fig, axs = plt.subplots(nrows=len(config.parameters), ncols=1, figsize=figsize)

    for i, param in enumerate(config.parameters):
        sns.boxplot(ax=axs[i], x=df[param])
        axs[i].set_title(config.parameter_name_unit_map[param])
        axs[i].set_xlabel("")

    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=dpi, transparent=True)


def plot_total_missingness(df, config, save_as=None, dpi=300):
    df_miss = pd.DataFrame(df[config.parameters].isna().sum(axis=0) / len(df) * 100).reset_index().rename(
        columns={"index": "parameter", 0: "missingness"})
    df_miss["parameter"] = df_miss["parameter"].map(config.parameter_name_map)

    ax = sns.barplot(df_miss, x="parameter", y="missingness")
    for i in ax.containers:
        ax.bar_label(i, fmt="%.1f")

    plt.xticks(rotation=90)
    plt.xlabel("")
    plt.ylabel("Missingness [%]")

    if save_as:
        plt.savefig(save_as, dpi=dpi)


def plot_missingness_over_column(df, config, col="DATEANDTIME", figsize=(8, 6), save_as=None, dpi=300):
    # Infer missingness information
    temp = df.set_index(col)[config.parameters]
    missingness = temp.isna().groupby(col).mean().reset_index()
    missingness = missingness.melt(id_vars=[col], value_vars=config.parameters, var_name="Parameter",
                                   value_name="missingness")
    missingness["Parameter"] = missingness["Parameter"].map(config.parameter_name_map)

    # Plot
    fig = plt.figure(figsize=figsize)
    sns.lineplot(data=missingness, x=col, y="missingness", hue="Parameter", marker="o")

    plt.xlabel("")
    plt.ylabel("Missingness [%]")

    plt.gca().set_xticks(ticks=np.sort(missingness[col].unique()))
    plt.xticks(rotation=90)
    plt.grid(True, color="black", alpha=0.1, linestyle="--")
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=dpi)


"""
Following function (adapted) from:
Yvonne Jenniges. (2025). y-jenniges/ocean_clustering_and_validation: Biogeochemical Ocean Regions - Code Base (v1.0.1). 
Zenodo. https://doi.org/10.5281/zenodo.15827777
"""


def plot_histograms(df, config, figsize=(12, 6), save_as=None, dpi=300):
    temp = df.rename(columns={col: config.parameter_name_unit_map[col] for col in config.parameters})
    param_names = list(config.parameter_name_unit_map.values())

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        sns.histplot(data=temp, x=param_names[i], ax=ax, kde=True)
        ax.set_title(param_names[i])
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=dpi)


def plot_correlations(df, config, save_as=None, dpi=300):
    # Compute correlations
    temp = df[config.parameters].rename(columns=config.parameter_name_map)
    corr = temp.corr().reset_index()
    corr = pd.melt(corr, id_vars='index')  # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']

    # Indexes for display
    idxs = [0, 6, 12, 18, 24, 30,
            10, 16, 22, 28, 34,
            7, 13, 19, 31,
            17, 23, 35,
            14, 20,
            21]

    # Adapted from https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
    def heatmap(x, y, size, color_values, figsize=(5, 4), idxs_to_show=None):
        fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [15, 1]})
        ax = axs[0]
        # Mapping from column names to integer coordinates
        x_labels = [v for v in sorted(x.unique())]
        y_labels = [v for v in sorted(y.unique())]
        x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
        y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

        def value_to_color(val):
            val_position = float((val - color_min)) / (
                        color_max - color_min)  # Position of value in the input range, relative to the length of the input range
            ind = int(val_position * (n_colors - 1))  # Target index in the color palette
            return palette[ind]

        size_scale = 500
        n_colors = 256  # Use 256 colors for the diverging color palette
        palette = sns.diverging_palette(20, 220, n=n_colors)  # Create the palette
        color_min, color_max = [-1,
                                1]  # Range of values that will be mapped to the palette, i.e. min and max possible correlation
        color = color_values.apply(value_to_color)

        if idxs_to_show:
            size = np.array(size)
            for i in range(len(size)):
                if i not in idxs_to_show:
                    size[i] = np.nan

        sc = ax.scatter(
            x=x.map(x_to_num),  # Use mapping for x
            y=y.map(y_to_num),  # Use mapping for y
            s=size * size_scale,  # Vector of square sizes, proportional to size parameter
            c=color,  # Vector of square color values, mapped to color palette
            marker='s',  # Use square as scatterplot marker
            label=color_values
        )

        # adding annotations to each entry
        for i in range(len(color)):
            if idxs_to_show:
                if i in idxs_to_show:
                    ax.annotate(str(round(color_values[i], 2)), (x_to_num[x[i]], y_to_num[y[i]]), ha='center',
                                va='center', fontsize=8)
            else:
                ax.annotate(str(round(color_values[i], 2)), (x_to_num[x[i]], y_to_num[y[i]]), ha='center', va='center',
                            fontsize=8)

        # Show column labels on the axes
        ax.set_xticks([x_to_num[v] for v in x_labels])
        ax.set_xticklabels(x_labels, rotation=90, horizontalalignment='center')
        ax.set_yticks([y_to_num[v] for v in y_labels])
        ax.set_yticklabels(y_labels)

        # center points in grid cells
        ax.grid(False, 'major')
        ax.grid(True, 'minor')
        ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
        ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
        ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
        ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

        # Add color legend on the right side of the plot
        ax = axs[1]

        col_x = [0] * len(palette)  # Fixed x coordinate for the bars
        bar_y = np.linspace(color_min, color_max, n_colors)  # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5] * len(palette),  # Make bars 5 units wide
            left=col_x,  # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_xlim(1, 2)  # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False)  # Hide grid
        ax.set_facecolor('white')  # Make background white
        ax.set_xticks([])  # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))  # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right()  # Show vertical ticks on the right
        plt.box(False)

    heatmap(
        x=corr['x'],
        y=corr['y'],
        size=corr['value'].abs(),
        color_values=corr['value'],
        idxs_to_show=idxs,
        figsize=(5, 4)
    )

    plt.tight_layout()

    if save_as:
     plt.savefig(save_as, dpi=dpi)


def plot_ts(df, figsize=(4, 4), dpi=None, ncols=5, xlim=None, ylim=None,
            save_as=None, fontsize=None,
            adjust_left=0, adjust_right=1, adjust_top=1, adjust_bottom=0, legend_loc="center right",
            anchor=(0.0, -0.15)):
    """ Plot TS diagram. """
    temp = df.copy()

    # Compute necessary parameters
    temp["pressure"] = gsw.p_from_z(-1 * temp["LEV_M"], temp["LATITUDE"])
    temp["abs_salinity"] = gsw.SA_from_SP(temp["P_SALINITY"], temp["pressure"], temp["LONGITUDE"], temp["LATITUDE"])
    temp["cons_temperature"] = gsw.CT_from_pt(temp["abs_salinity"], temp["P_TEMPERATURE"])
    temp["rho"] = gsw.rho(temp["abs_salinity"], temp["cons_temperature"], temp["pressure"])

    # Plot limits
    smin = temp["abs_salinity"].min() - (0.01 * temp["abs_salinity"].min())
    smax = temp["abs_salinity"].max() + (0.01 * temp["abs_salinity"].max())
    tmin = temp["cons_temperature"].min() - (0.1 * temp["cons_temperature"].max())
    tmax = temp["cons_temperature"].max() + (0.1 * temp["cons_temperature"].max())

    if xlim:
        smin = xlim[0] - (0.01 * xlim[0])
        smax = xlim[1] + (0.01 * xlim[1])

    if ylim:
        tmin = ylim[0] - (0.01 * ylim[0])
        tmax = ylim[1] + (0.01 * ylim[1])

    # Number of gridcells in the x and y dimensions
    xdim = int(round((smax - smin) / 0.1 + 1, 0))
    ydim = int(round((tmax - tmin) / 0.1 + 1, 0))

    # Empty grid
    dens = np.zeros((ydim, xdim))

    # Temperature and salinity vectors
    si = np.linspace(1, xdim - 1, xdim) * 0.1 + smin
    ti = np.linspace(1, ydim - 1, ydim) * 0.1 + tmin

    # Fill grid with densities
    for j in range(0, int(ydim)):
        for i in range(0, int(xdim)):
            dens[j, i] = gsw.rho(si[i], ti[j], 0)

    # Convert to sigma-t
    dens = dens - 1000

    # Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    contours = plt.contour(si, ti, dens, linestyles='dashed', colors='k')
    plt.clabel(contours, fontsize=fontsize, inline=1, fmt='%1.1f')  # label every second level

    hb = plt.hexbin(
        temp["abs_salinity"],
        temp["cons_temperature"],
        cmap="viridis",
        gridsize=100,
        norm=colors.LogNorm(),
        mincnt=1,
    )
    plt.colorbar(hb, label="Log count")

    ax.set_xlabel('Absolute salinity [g/kg]', fontsize=fontsize)
    ax.set_ylabel('Conservative temperature [°C]', fontsize=fontsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=dpi, transparent=True)
    plt.show()


def plot_interactive_geo(df, column="label", color_label="color", color_scale="Inferno", scatter_size=3, margin=5, save_as=None):
    """ Interactive 3d geographic scatter plot. """
    df_display = df.copy()

    longitude_min = df["LONGITUDE"].min()
    longitude_max = df["LONGITUDE"].max()
    latitude_min = df["LATITUDE"].min()
    latitude_max = df["LATITUDE"].max()
    depth_min = df["LEV_M"].min()
    depth_max = df["LEV_M"].max()

    # Define figure
    figure_geo = go.Figure(data=go.Scatter3d(x=df_display.LONGITUDE, y=df_display.LATITUDE, z=df_display.LEV_M,
                                             mode='markers',
                                             marker=dict(size=scatter_size, color=df[color_label], opacity=1,
                                                         colorbar=dict(thickness=20), colorscale=color_scale,
                                                         cmin=df[color_label].min(), cmax=df[color_label].max()
                                                         ),
                                             # hovertemplate='Longitude: %{x}<br>' +
                                             #               'Latitude: %{y}<br>' +
                                             #               'Depth: %{z} m<br>' +
                                             #               'Temperature: %{text[0]:.2f} °C<br>' +
                                             #               'Salinity: %{text[1]:.2f} psu<br>' +
                                             #               'Oxygen: %{text[2]:.2f} µmol/kg<br>' +
                                             #               'Nitrate: %{text[3]:.2f} µmol/kg<br>' +
                                             #               'Silicate: %{text[4]:.2f} µmol/kg<br>' +
                                             #               'Phosphate: %{text[5]:.2f} µmol/kg<br>' +
                                             #               'Label: %{text[6]}<extra></extra>',
                                             # text=df_display[
                                             #     ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE",
                                             #      "P_SILICATE",
                                             #      "P_PHOSPHATE", column]]
                                             ))

    # Update figure layout
    figure_geo.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin),
                             scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Depth [m]",
                                        xaxis=dict(range=[longitude_min, longitude_max]),
                                        yaxis=dict(range=[latitude_min, latitude_max]),
                                        zaxis=dict(range=[depth_max, depth_min])
                                        ),
                             uirevision=True)

    # Save
    if save_as:
        figure_geo.write_html(save_as)

    # Show the plot
    figure_geo.show()


def plot_interactive_embedding(df, color_label=None, color_scale="Inferno", scatter_size=1, save_as=None):
    # Define plot parameters
    if color_label:
        plot_params = {"x": df["e0"], "y": df["e1"], "z": df["e2"], "mode": "markers",
                       "marker": dict(size=scatter_size, colorscale=color_scale, opacity=0.8, color=df[color_label])}

    else:
        plot_params = {"x": df["e0"], "y": df["e1"], "z": df["e2"], "mode": "markers",
                       "marker": dict(size=scatter_size, opacity=0.8)}

    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(**plot_params)])

    # Customize layout
    fig.update_layout(
        title=f"Embedding",
        scene=dict(
            xaxis_title="X-Axis",
            yaxis_title="Y-Axis",
            zaxis_title="Z-Axis"
        )
    )

    # Save
    if save_as:
        fig.write_html(save_as)

    # Show in browser
    fig.show()


def plot_each_depth_level(df, color_label="P_TEMPERATURE", cmap="inferno", vmin=None, vmax=None, save_as=None):
    """
    Panel plot of maps for each depth level with smooth tricontourf coloring.
    Shared vertical colorbar on the right, layout: 3 rows × 4 cols.
    """
    depths = np.sort(df["LEV_M"].unique())
    lamin = df["LATITUDE"].min() - 5
    lamax = df["LATITUDE"].max() + 5
    lomin = df["LONGITUDE"].min() - 5
    lomax = df["LONGITUDE"].max() + 5

    nrows, ncols = 3, 4
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(ncols * 3, nrows * 2.5),
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True
    )
    axs = axs.flatten()

    if vmin is None:
        vmin = df[color_label].min()
    if vmax is None:
        vmax = df[color_label].max()

    levels = np.linspace(vmin, vmax, 21)  # evenly spaced, ensures full colormap use
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cf = None

    for i, d in enumerate(depths):
        sel = df[df["LEV_M"] == d]

        ax = axs[i]
        ax.set_extent([lomin, lomax, lamin, lamax])
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        cf = ax.tricontourf(
            sel["LONGITUDE"], sel["LATITUDE"], sel[color_label],
            levels=levels, cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree()
        )

        ax.set_title(f"{d} m", fontsize=10)

    # Hide empty panels if fewer depths than subplots
    for j in range(i+1, len(axs)):
        axs[j].axis("off")

    # Shared vertical colorbar on the right
    cbar = fig.colorbar(cf, ax=axs, orientation="vertical", shrink=0.9, aspect=30, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
    cbar.set_label(color_label)

    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches="tight")
    plt.show()


def color_code_labels(df, color_noise_black=False, drop_noise=False, column_name="label"):
    """ Add a color for each label in the clustering using the Glasbey library. """
    temp = df.copy()

    # define colors
    unique_labels = np.sort(np.unique(temp[column_name]))
    colors = glasbey.create_palette(palette_size=len(unique_labels))
    color_map = {label: color for label, color in zip(unique_labels, colors)}
    temp["color"] = temp[column_name].map(lambda x: color_map[x])

    # how to deal with -1 labels (which is noise in DBSCAN)
    if color_noise_black:
        temp.loc[temp[column_name] == -1, "color"] = "#000000"
    if drop_noise:
        temp = temp[temp[column_name] != -1]

    return temp


def plot_geo(df, color_label="color", save_as=None, figsize=(6, 6),
             adjust_left=0, adjust_right=0.92, adjust_top=1.1, adjust_bottom=-0.05, pointsize=0.5, dpi=600,
             xlabelpad=20, ylabelpad=0, zlabelpad=0):
    """ 3d scatter plot of a cluster set with given colors. """

    # Define figure and plot grid cells as scatter points
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df["LONGITUDE"], df["LATITUDE"], df["LEV_M"], c=df[color_label], s=pointsize, alpha=1, zorder=1)

    # Get coastlines from Cartopy feature
    # Create a bounding box for the data region
    bbox = box(df["LONGITUDE"].min(), df["LATITUDE"].min(), df["LONGITUDE"].max(), df["LATITUDE"].max())

    # Get cartopy coastline
    shpfilename = shapereader.natural_earth(resolution="110m", category="physical", name="coastline")
    reader = shapereader.Reader(shpfilename)

    # Loop through geometries and clip to desired range
    for record in reader.records():
        # Clip geometry to desited range
        geom = record.geometry.intersection(bbox)

        # Skip geometry, if no intersection with desired range
        if geom.is_empty:
            continue

        # Convert geometry to a list of lines
        if geom.geom_type == "MultiLineString":
            lines = geom.geoms
        elif geom.geom_type == "LineString":
            lines = [geom]
        else:
            continue  # skip if it's not a line

        # Add each line to the plot
        for line in lines:
            x, y = line.xy
            z = np.full_like(x, 0.0)  # Place at surface
            ax.plot(x, y, z, color='black', linewidth=1.5, zorder=10)

    # Set axis limits
    ax.set_xlim(df["LONGITUDE"].min(), df["LONGITUDE"].max())
    ax.set_ylim(df["LATITUDE"].min(), df["LATITUDE"].max())

    ax.set_box_aspect((np.ptp(df["LONGITUDE"]), np.ptp(df["LATITUDE"]), np.ptp(df["LEV_M"]) / 50))

    # Add axis labels
    ax.set_xlabel('Longitude', labelpad=xlabelpad)
    ax.set_ylabel('Latitude', labelpad=ylabelpad)
    ax.set_zlabel('Depth [m]', labelpad=zlabelpad)

    # Invert the Z-axis for depth representation
    plt.gca().invert_zaxis()

    # Define coarse ticks for longitude and latitude
    lon_ticks = np.linspace(df["LONGITUDE"].min(), df["LONGITUDE"].max(), num=5)  # Adjust num for desired spacing
    lat_ticks = np.linspace(df["LATITUDE"].min(), df["LATITUDE"].max(), num=5)

    # Set ticks and labels
    ax.set_xticks(lon_ticks)  # Longitude ticks
    ax.set_xticklabels([f"{tick:.1f}°" for tick in lon_ticks], rotation=45, ha="right")  # Format as degrees

    ax.set_yticks(lat_ticks)  # Latitude ticks
    ax.set_yticklabels([f"{tick:.1f}°" for tick in lat_ticks], rotation=0, ha="left")  # Format as degrees

    # Increase padding between ticks and their labels
    ax.tick_params(axis='x', pad=-5)  # Horizontal ticks
    ax.tick_params(axis='y', pad=-5)  # Vertical ticks
    ax.tick_params(axis='z', pad=10)  # Depth ticks

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=adjust_left, right=adjust_right, top=adjust_top, bottom=adjust_bottom)

    # Save figure
    if save_as:
        plt.savefig(save_as, dpi=dpi)
    # plt.show()

