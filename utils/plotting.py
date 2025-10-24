import logging
import glasbey
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib.animation as animation
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


def plot_histograms(df, parameter_names):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.hist(df[parameter_names[i]].dropna(), bins=100, color='steelblue', alpha=0.7)
        ax.set_title(parameter_names[i])
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


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


def plot_simple_reconstruction_error(y_true, y_pred):
    import matplotlib.pyplot as plt
    import numpy as np

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
    axes = axes.flatten()  # flatten to index easily

    for i, feature_name in enumerate(config.parameters[:6]):  # first 6 features
        a = y_true[:, i]
        b = y_pred[:, i]

        # Remove NaNs
        mask = ~np.isnan(a) & ~np.isnan(b)
        a = a[mask]
        b = b[mask]

        ax = axes[i]
        ax.scatter(a, b, alpha=0.05, s=5)

        # 1:1 reference line
        min_val = min(a.min(), b.min())
        max_val = max(a.max(), b.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(feature_name)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

    plt.tight_layout()
    plt.show()


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


def generate_animation(df_imputed, scaler_dict, parameter="P_TEMPERATURE", save_as=None):
    temp = df_imputed.copy()

    # Undo scaling
    for param, scaler in scaler_dict.items():
        # Only unscale parameters
        if param in config.parameters:
            temp[param] = scaler.inverse_transform(temp[param].values.reshape(-1, 1))

    # Transform to xarray
    ds = df_to_gridded_da(temp, value_col=parameter)

    # Animate and save
    animate_depth_panels(data=ds, save_as=save_as)


# Function from previous paper


def plot_interactive_geo(df, column="label", color_label="color", color_scale="Inferno", scatter_size=3, margin=5, save_as=None):
    """ Interactive 3d geographic scatter plot.
    Taken from: Yvonne Jenniges. (2025). y-jenniges/ocean_clustering_and_validation: Biogeochemical Ocean Regions
    - Code Base (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.15827777"""
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
