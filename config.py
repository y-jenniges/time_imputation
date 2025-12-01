import numpy as np


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

time_steps = [1, 5, 10] + list(range(11, 41)) + [50, 100, 300]
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
        "note": "Global, 6 params"
    },
    "avg_na": {"param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
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
               "dtime": 300,
               "note": "NA, 6 params"},

    "100y_na": {"param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
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
                "dtime": 100,
                "note": "NA, 6 params"},

    "50y_na": {"param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
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
               "dtime": 50,
               "note": "NA, 6 params"},

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
        "note": "NA, 6 params"
    },

    "10y_na": {"param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
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
               "dtime": 10,
               "note": "NA, 6 params"},

    "5y_na": {"param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
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
              "dtime": 5,
              "note": "NA, 6 params"},

    "1y_na": {"param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
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
              "dtime": 1,
              "note": "NA, 6 params"},

    # "m_na": {"param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
    #       "lat_min": 0,
    #       "lat_max": 70,
    #       "dlat": 1,
    #       "lon_min": -77,
    #       "lon_max": 30,
    #       "dlon": 1,
    #       "z_min": None,
    #       "z_max": None,
    #       "dz": None,
    #       "z_array": np.array(depth_levels),
    #       "time_min": "1772-01-01 00:00:00",
    #       "time_max": "2020-07-08 04:45:00",
    #       "mode": "M",
    #       "selection": None,
    #       "dtime": 1,
    #       "note": "NA, 6 params"},
}

bathymetry_path = "../../data/bathymetry/gebco_2022_sub_ice_topo/GEBCO_2022_sub_ice_topo.nc"

# --- Splitting --------------------------------------------------------------------------------------------- #
n_splits_per_scheme = 5
data_path = "output/gridding/df_20y_na.csv"
val_fractions = [0.1, 0.2, 0.5]
test_fraction = 0.15

# --- Tuning --------------------------------------------------------------------------------------------- #

# --- Training --------------------------------------------------------------------------------------------- #

# --- Plotting --------------------------------------------------------------------------------------------- #
parameter_name_map = {"P_TEMPERATURE": "Potential temperature", "P_SALINITY": "Salinity",
                      "P_OXYGEN": "Oxygen",
                      "P_NITRATE": "Nitrate",
                      "P_SILICATE": "Silicate",
                      "P_PHOSPHATE": "Phosphate"}

parameter_name_unit_map = {"P_TEMPERATURE": "Potential temperature [°C]", "P_SALINITY": "Salinity [psu]",
                           "P_OXYGEN": "Oxygen [μmol / kg]",
                           "P_NITRATE": "Nitrate [μmol / kg]",
                           "P_SILICATE": "Silicate [μmol / kg]",
                           "P_PHOSPHATE": "Phosphate [μmol / kg]"}
