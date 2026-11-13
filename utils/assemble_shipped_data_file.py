import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import netCDF4
import xarray as xr

import config


if __name__ == "__main__":
    out_path = "sonati_na_imputed_20y_v1.nc"

    # Load the input data and extract unique coordinates
    df = pd.read_csv("output/gridding/df_20y_na.csv")
    times = pd.DatetimeIndex(np.sort(pd.to_datetime(df["DATEANDTIME"]).unique()))
    depths = np.sort(df["LEV_M"].unique())
    lats = np.sort(df["LATITUDE"].unique())
    lons = np.sort(df["LONGITUDE"].unique())

    # Missingness information
    missingness_mask = df[config.parameters].isna().to_numpy()
    observed = df[config.parameters].to_numpy()

    # Load the final imputation
    final_imputation_path = Path("output/temp/na_final")
    imputed = np.load(final_imputation_path / "x_imputed.npy")
    aleatoric_std = np.load(final_imputation_path / "aleatoric_std.npy")
    epistemic_std = np.load(final_imputation_path / "epistemic_std.npy")
    assert imputed.shape == observed.shape == aleatoric_std.shape == epistemic_std.shape

    # Compute total std
    total_std = np.sqrt(aleatoric_std ** 2 + epistemic_std ** 2)

    # Final values: Keep the measurement where there is one, model prediction elsewhere
    values = np.where(missingness_mask, imputed, observed)

    # Metadata per parameter (same order as config.parameters)
    meta = {
        "potential_temperature": dict(standard_name="sea_water_potential_temperature",
                                      long_name="Potential temperature", units="degree_Celsius"),
        "salinity": dict(standard_name="sea_water_practical_salinity",
                         long_name="Practical salinity (PSS-78, psu)", units="1"),
        "oxygen": dict(standard_name="moles_of_oxygen_per_unit_mass_in_sea_water",
                       long_name="Dissolved oxygen", units="umol kg-1"),
        "nitrate": dict(standard_name="moles_of_nitrate_per_unit_mass_in_sea_water",
                        long_name="Nitrate", units="umol kg-1"),
        "silicate": dict(standard_name="moles_of_silicate_per_unit_mass_in_sea_water",
                         long_name="Silicate", units="umol kg-1"),
        "phosphate": dict(standard_name="moles_of_phosphate_per_unit_mass_in_sea_water",
                          long_name="Phosphate", units="umol kg-1"),
    }
    assert len(meta) == len(config.parameters)

    # Map every row of the input table to its cell in the (time, depth, lat, lon) grid
    it = np.searchsorted(times, pd.to_datetime(df["DATEANDTIME"]))
    iz = np.searchsorted(depths, df["LEV_M"])
    iy = np.searchsorted(lats, df["LATITUDE"])
    ix = np.searchsorted(lons, df["LONGITUDE"])
    assert len(set(zip(it, iz, iy, ix))) == len(df)  # no two rows land in the same cell
    shape = (len(times), len(depths), len(lats), len(lons))
    print(f"Grid shape {shape}: {len(df)} of {np.prod(shape)} cells occupied ({100 * len(df) / np.prod(shape):.2f} %)")

    def to_grid(col, dtype="f4", fill=np.nan):
        """Scatter a flat per-row column onto the 4D grid. Empty cells get the fill value."""
        grid = np.full(shape, fill, dtype=dtype)
        grid[it, iz, iy, ix] = col
        return grid

    # Assemble the netcdf file
    ds = netCDF4.Dataset(out_path, "w", format="NETCDF4")

    # Global attributes
    # Attribute names follow CF (Climate Forecast) and ACDD (Attribute Convention for Data Discovery).
    ds.Conventions = "CF-1.8, ACDD-1.3"
    ds.title = "SONATI-imputed North Atlantic ocean physical and biogeochemical fields"
    ds.summary = ("Gap-filled fields of potential temperature, practical salinity, oxygen, nitrate, silicate and "
                  "phosphate for the North Atlantic on a 1x1° grid, 12 depth levels (0-5000m) and 13 time steps "
                  "(1772-2020). Gaps in the COMFORT database subset were imputed with the SONATI model (mean over "
                  "MC-dropout passes); observed values are kept as measured. Aleatoric, epistemic and total "
                  "(sqrt(aleatoric^2 + epistemic^2)) uncertainties are standard deviations in the units of the "
                  "variable, given for every grid cell (at observed cells they describe the model's reconstruction of "
                  "the measurement). Grid cells that are land or ocean floor are NaN.")
    ds.keywords = "ocean data, North Atlantic, imputation, uncertainty, machine learning, biogeochemistry, gap filling"
    ds.product_version = "v1"
    ds.standard_name_vocabulary = "CF Standard Name Table v95"
    ds.history = f"{datetime.date.today().isoformat()}: Created with utils/assemble_shipped_data_file.py"

    # Provenance
    ds.source = ("SONATI imputation (https://doi.org/10.5281/zenodo.23799887) of a subset of the COMFORT "
                 "database (https://doi.org/10.11582/2022.00039)")
    ds.references = ("Jenniges, Y., Koch, B. P., Sonnewald, M., Maneth, S. (in prep). "
                     "SONATI: Spatio-temporal ocean neighbourhood asymmetric transformer "
                     "for imputation of sparse, tabular data with uncertainty.\n"
                     "Source data: Korablev, A., Olsen, A., Geophysical Institute, University of Bergen, "
                     "Bjerknes Centre for Climate Research (2022). COMFORT Dataset [Data set]. NIRD RDA. "
                     "https://doi.org/10.11582/2022.00039")
    ds.id = "https://doi.org/10.5281/zenodo.24194611"

    # Authorship and licence
    ds.creator_name = "Yvonne Jenniges"
    ds.contributor_name = "Boris Peter Koch, Maike Sonnewald, Sebastian Maneth"
    ds.institution = "University of Bremen"
    ds.license = "CC-BY-4.0, https://creativecommons.org/licenses/by/4.0/"
    ds.acknowledgement = ("YJ has been funded through the University of Bremen, the Alfred-Wegener-Institut, "
                          "Helmholtz-Zentrum für Polar- und Meeresforschung and the Helmholtz School for "
                          "Marine Data Science (MarDATA). BPK, MS and SM were supported by their employing institutions.")

    # Grid and coverage (computed from the data)
    ds.geospatial_lat_resolution = "1 degree_north"
    ds.geospatial_lon_resolution = "1 degree_east"
    ds.geospatial_lat_min = float(lats.min() - 0.5)  # lat/lon give cell centres
    ds.geospatial_lat_max = float(lats.max() + 0.5)
    ds.geospatial_lon_min = float(lons.min() - 0.5)
    ds.geospatial_lon_max = float(lons.max() + 0.5)
    ds.geospatial_vertical_min = float(depths.min())
    ds.geospatial_vertical_max = float(depths.max() + 1000)  # the last depth level stretches from 4000 to 5000m
    # Time stamps are bin starts, so the last bin is shorter
    ds.time_coverage_start = times.min().strftime("%Y-%m-%dT%H:%M:%S")
    ds.time_coverage_end = "2020-07-08T04:45:00"

    # Create dimensions
    ds.createDimension("time", len(times))
    ds.createDimension("depth", len(depths))
    ds.createDimension("lat", len(lats))
    ds.createDimension("lon", len(lons))

    # Create time variable
    t = ds.createVariable("time", "f8", ("time",))
    t.units = "days since 1772-01-01 00:00:00"
    t.calendar = "standard"
    t.standard_name = "time"
    t.long_name = "Start of 20-year time bin"
    t.comment = ("Each time step represents the half-open 20-year period [time, time + 20 years). The last step "
                 "ends at 2020-07-08.")
    t.axis = "T"
    t[:] = netCDF4.date2num(times.to_pydatetime(), t.units, t.calendar)

    # Create depth variable
    d = ds.createVariable("depth", "f8", ("depth",))
    d.units = "m"
    d.standard_name = "depth"
    d.long_name = "Depth below sea surface"
    d.positive = "down"  # COMFORT convention: Positive z points downwards
    d.axis = "Z"
    d[:] = depths

    # Create longitude variable
    lo = ds.createVariable("lon", "f8", ("lon",))
    lo.units = "degrees_east"
    lo.standard_name = "longitude"
    lo.long_name = "Longitude"
    lo.axis = "X"
    lo[:] = lons

    # Create latitude variable
    la = ds.createVariable("lat", "f8", ("lat",))
    la.units = "degrees_north"
    la.standard_name = "latitude"
    la.long_name = "Latitude"
    la.axis = "Y"
    la[:] = lats

    # Create data variables: Value, three uncertainties and the imputed flag per parameter
    dims = ("time", "depth", "lat", "lon")
    for j, (name, m) in enumerate(meta.items()):
        extras = {
            "aleatoric_std": ("aleatoric", aleatoric_std),
            "epistemic_std": ("epistemic", epistemic_std),
            "total_std": ("total", total_std),
        }

        # The parameter itself: Measurement where observed, model prediction where imputed
        v = ds.createVariable(name, "f4", dims, zlib=True, complevel=4, fill_value=np.nan)
        v.standard_name = m["standard_name"]
        v.long_name = m["long_name"]
        v.units = m["units"]
        v.ancillary_variables = " ".join([f"{name}_{s}" for s in extras] + [f"{name}_imputed"])
        v[:] = to_grid(values[:, j])

        # Three uncertainty variables (same grid and units as the parameter)
        for suffix, (kind, arr) in extras.items():
            u = ds.createVariable(f"{name}_{suffix}", "f4", dims, zlib=True, complevel=4, fill_value=np.nan)
            u.long_name = f"{m['long_name']}: {kind} uncertainty (standard deviation)"
            u.units = m["units"]
            u[:] = to_grid(arr[:, j])

        # Flag indicating which values are observed (0) and which are imputed (1); -1 marks cells outside the domain
        f = ds.createVariable(f"{name}_imputed", "i1", dims, zlib=True, complevel=4, fill_value=-1)
        f.long_name = f"{m['long_name']}: imputation flag"
        f.flag_values = np.array([0, 1], dtype="i1")
        f.flag_meanings = "observed imputed"
        f[:] = to_grid(missingness_mask[:, j].astype("i1"), dtype="i1", fill=-1)

    # Close and write netcdf file
    ds.close()

    # Verify: Reopen and compare against the input table
    chk = xr.open_dataset(out_path)
    print(chk)
    for j, name in enumerate(meta):
        # Number of non-NaN cells equals the number of rows in the input table
        assert chk[name].count().item() == len(df), name

        # Values at each row's grid cell equal the values that were meant to be written
        cell = chk[name].isel(time=xr.DataArray(it), depth=xr.DataArray(iz), lat=xr.DataArray(iy), lon=xr.DataArray(ix)).to_numpy()
        np.testing.assert_allclose(cell, values[:, j], rtol=1e-5, atol=1e-6)

        # The imputed flag matches the missingness mask of the input table
        assert (chk[f"{name}_imputed"].isel(time=xr.DataArray(it), depth=xr.DataArray(iz), lat=xr.DataArray(iy), lon=xr.DataArray(ix)).to_numpy()
                == missingness_mask[:, j]).all(), name

        # Uncertainties: Non-NaN in every cell that has a value
        assert chk[f"{name}_total_std"].count().item() == len(df), name
    chk.close()  # release the file handle
    print("Verification passed.")
