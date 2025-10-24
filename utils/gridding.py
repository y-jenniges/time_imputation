""" Classes and functions to grid parameter values. ADAPTED FROM PREVIOUS PAPER. """
import logging
import os
import time
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import xarray as xr
import json
from datetimerange import DateTimeRange
from dateutil.relativedelta import relativedelta

from utils import sqlite_util
from utils.database import get_table_as_df, does_table_exist, remove_tables_like, get_names_of_all_parameter_tables


class GridManager:
    """ Class to manage grids in the database. """
    def __init__(self, connection, grid_info_table="grid_info"):
        self.connection = connection
        self.grid_info_table = grid_info_table

        # Create grid information table if it does not yet exist
        if not does_table_exist(self.connection, self.grid_info_table, "table"):
            q = f"CREATE TABLE {self.grid_info_table}(" \
                f"  grid_id primary key, " \
                f"  lat_min real, lat_max real, dlat real, " \
                f"  lon_min real, lon_max real, dlon real, " \
                f"  z_min text, z_max text, dz text, z_array text, " \
                f"  time_min text, time_max text, mode text, dtime integer, selection text);"
            self.connection.execute(q)

    def does_grid_exist(self, grid_id):
        """ Check if a grid exists. """
        q = f"SELECT * FROM {self.grid_info_table} WHERE grid_id={grid_id};"
        cur = self.connection.execute(q)
        res = cur.fetchall()
        grid_exists = len(res) == 1
        return grid_exists

    def add_grid(self, grid):
        """ Add grid to grid information table. """
        if grid.z_array is not None:
            z_input = f"'None', 'None', 'None', '{json.dumps(grid.z_array.tolist())}'"
        else:
            z_input = f"{grid.z_min}, {grid.z_max}, {grid.dz}, 'None'"

        if grid.selection is not None:
            sel_input = f"'{json.dumps(grid.selection.tolist())}'"
        else:
            sel_input = "'None'"

        # Insert into existing grid info table
        q = f"INSERT INTO {self.grid_info_table}" \
            f"(grid_id, lat_min, lat_max, dlat, lon_min, lon_max, dlon, z_min, z_max, dz, z_array, " \
            f"time_min, time_max, mode, dtime, selection) " \
            f"VALUES({grid.grid_id}, {grid.lat_min}, {grid.lat_max}, {grid.dlat}, " \
            f"{grid.lon_min}, {grid.lon_max}, {grid.dlon}, " \
            f"{z_input}, " \
            f"'{grid.time_min}', '{grid.time_max}', '{grid.mode}', {grid.dtime}, " \
            f"{sel_input});"
        self.connection.execute(q)
        self.connection.commit()

        # Create grid table itself
        grid.grid.to_sql(grid.grid_name, self.connection, if_exists="replace", index=True, index_label="idx")

    def remove_grid(self, grid_id):
        """ Remove grid from database. """
        # Check if grid_id exists
        if does_table_exist(self.connection, "grid_" + str(grid_id)):
            # Load all grids and drop the one to remove from grid information table
            grids = get_table_as_df(self.connection, "grid_info")
            grids = grids.drop(grids[grids.grid_id.astype(str) == str(grid_id)].index)

            # Drop all mapped parameter tables and the grid_table
            remove_tables_like(self.connection, like_pattern=f"%|_{grid_id}", escape_char="|", table_type="table")
            remove_tables_like(self.connection, like_pattern=f"%|_{grid_id}|_%", escape_char="|", table_type="table")

            # Write new grid information table
            grids.to_sql("grid_info", self.connection, if_exists="replace", index=False)

    def remove_all_grids(self):
        """ Remove all grids from database. """
        # Get all grid_ids
        df_grid_info = get_table_as_df(self.connection, "grid_info")

        # Remove all grids
        for grid_id in df_grid_info["grid_id"].values:
            self.remove_grid(grid_id)

    def load_grid(self, grid_id,
                  bathymetry_grid_path="../../data/bathymetry/gebco_2022_sub_ice_topo/gebco_2022_sub_ice_topo.nc",
                  lat_variable="lat", lon_variable="lon", depth_variable="elevation"):
        """
        Load existing spatio-temporal grid.

        Args:
            grid_id (int): ID of the grid to load.
            bathymetry_grid_path (str): Path to bathymetry information.
            lat_variable (str): Name of latitude variable in bathymetry data.
            lon_variable (str): Name of longitude variable in bathymetry data.
            depth_variable (str): Name of depth variable in bathymetry data.
        """
        # Check if grid exists
        grid_exists = self.does_grid_exist(grid_id)

        # If it exists, load it
        if grid_exists:
            # Get grid info
            q = f"SELECT * FROM grid_info WHERE grid_id={grid_id};"
            cur = self.connection.execute(q)
            grid_info = pd.DataFrame(cur.fetchall(), columns=[x[0] for x in cur.description])

            # Init grid parameters
            lat_min = grid_info["lat_min"].values[0] if grid_info["lat_min"].values[0] != "None" else None
            lat_max = grid_info["lat_max"].values[0] if grid_info["lat_max"].values[0] != "None" else None
            dlat = grid_info["dlat"].values[0] if grid_info["dlat"].values[0] != "None" else None
            lon_min = grid_info["lon_min"].values[0] if grid_info["lon_min"].values[0] != "None" else None
            lon_max = grid_info["lon_max"].values[0] if grid_info["lon_max"].values[0] != "None" else None
            dlon = grid_info["dlon"].values[0] if grid_info["dlon"].values[0] != "None" else None
            z_min = float(grid_info["z_min"].values[0]) if grid_info["z_min"].values[0] != "None" else None
            z_max = float(grid_info["z_max"].values[0]) if grid_info["z_max"].values[0] != "None" else None
            dz = float(grid_info["dz"].values[0]) if grid_info["dz"].values[0] != "None" else None
            z_array = json.loads(grid_info["z_array"].values[0]) \
                if grid_info["z_array"].values[0] != "None" else None
            time_min = grid_info["time_min"].values[0] if grid_info["time_min"].values[0] != "None" else None
            time_max = grid_info["time_max"].values[0] if grid_info["time_max"].values[0] != "None" else None
            mode = grid_info["mode"].values[0] if grid_info["mode"].values[0] != "None" else None
            dtime = grid_info["dtime"].values[0] if grid_info["dtime"].values[0] != "None" else None
            selection = json.loads(grid_info["selection"].values[0]) \
                if grid_info["selection"].values[0] != "None" else None

            grid = Grid(lat_min=lat_min, lat_max=lat_max, dlat=dlat,
                        lon_min=lon_min, lon_max=lon_max, dlon=dlon,
                        z_min=z_min, z_max=z_max, dz=dz, z_array=z_array,
                        time_min=time_min, time_max=time_max, mode=mode, dtime=dtime, selection=selection,
                        bathymetry_grid_path=bathymetry_grid_path,
                        lat_variable=lat_variable, lon_variable=lon_variable, depth_variable=depth_variable,
                        grid_id=grid_id)
            return grid
        else:
            print(f"GridManager: Grid with ID {grid_id} does not exists.")
            return

    def create_grid(self, lat_min, lat_max, dlat, lon_min, lon_max, dlon,
                    z_min=None, z_max=None, dz=None, z_array=None,
                    time_min=None, time_max=None, mode=None, dtime=None, selection=None,
                    bathymetry_grid_path=None,
                    lat_variable="lat", lon_variable="lon", depth_variable="elevation"):
        """
        Create a new grid (or load it if it already exists).

        Args:
            lat_min (float): Minimum latitude.
            lat_max (float): Maximum latitude.
            dlat (float): Latitude step size.
            lon_min (float): Minimum longitude.
            lon_max(float): Maximum longitude.
            dlon (float): Longitude step size.
            z_min (float): Minimum depth.
            z_max (float): Maximum depth.
            dz (float): Depth step size.
            z_array (numpy.array<float>): Array specifying depth levels (if depth steps are irregular, use this instead
            of z_min, z_max and dz).
            time_min (str): Minimum time.
            time_max (str): Maximum time.
            mode (str): Time mode/general time frame of interest (either "Y" (year), "YM" (year-month), "YD" (year-day),
            "M" (month), "MD" (month-day) or "D" (day)).
            dtime (str): Time step size (in units specified by mode).
            selection (numpy.ndarray<boolean>): Boolean array as mask for the time array to (de)select specific times
            from the time grid.
            bathymetry_grid_path (str): Path to bathymetry data.
            lat_variable (str): Name of latitude variable in bathymetry data.
            lon_variable (str): Name of longitude variable in bathymetry data.
            depth_variable (str): Name of depth variable in bathymetry data.
        """
        # Determine grid_id
        grid_id = self.get_grid_id_from_params(lat_min, lat_max, dlat, lon_min, lon_max, dlon,
                                               z_min, z_max, dz, z_array,
                                               time_min, time_max, mode, dtime, selection)

        # If the grid already exists, only load it
        if not grid_id:
            # Create grid
            grid = Grid(lat_min=lat_min, lat_max=lat_max, dlat=dlat,
                        lon_min=lon_min, lon_max=lon_max, dlon=dlon,
                        z_min=z_min, z_max=z_max, dz=dz, z_array=z_array,
                        time_min=time_min, time_max=time_max, mode=mode, dtime=dtime, selection=selection,
                        bathymetry_grid_path=bathymetry_grid_path,
                        lat_variable=lat_variable, lon_variable=lon_variable, depth_variable=depth_variable)
            self.add_grid(grid)
        else:
            # Load grid
            logging.info(f"GridManager.create_grid: Grid already exists with ID {grid_id}. Loading grid...")
            grid = self.load_grid(grid_id, bathymetry_grid_path=bathymetry_grid_path)

        return grid

    def get_grid_id_from_params(self, lat_min, lat_max, dlat, lon_min, lon_max, dlon,
                                z_min=None, z_max=None, dz=None, z_array=None,
                                time_min=None, time_max=None, mode=None, dtime=None, selection=None):
        """
        Infer grid_id from given grid parameters.

        Args:
            lat_min (float): Minimum latitude.
            lat_max (float): Maximum latitude.
            dlat (float): Latitude step size.
            lon_min (float): Minimum longitude.
            lon_max(float): Maximum longitude.
            dlon (float): Longitude step size.
            z_min (float): Minimum depth.
            z_max (float): Maximum depth.
            dz (float): Depth step size.
            z_array (numpy.array<float>): Array specifying depth levels (if depth steps are irregular, use this instead
            of z_min, z_max and dz).
            time_min (str): Minimum time.
            time_max (str): Maximum time.
            mode (str): Time mode/general time frame of interest (either "Y" (year), "YM" (year-month), "YD" (year-day),
            "M" (month), "MD" (month-day) or "D" (day)).
            dtime (str): Time step size (in units specified by mode).
            selection (numpy.ndarray<boolean>): Boolean array as mask for the time array to (de)select specific times
            from the time grid."""

        # Determine whether z_array or z_min/z_max/dz was used
        if z_array is None:
            z_statement = f"z_min={z_min} and z_max={z_max} and dz={dz} and z_array='None'"
        else:
            z_statement = f"z_min='None' and z_max='None' and dz='None' and " \
                          f"z_array='{json.dumps(z_array.tolist())}'"

        # Determine which selection was used
        if selection is None:
            sel_statement = "selection='None'"
        else:
            sel_statement = f"selection='{json.dumps(selection.tolist())}'"

        # Assemble query to check parameter specs in the grid information table
        q = f"SELECT grid_id FROM {self.grid_info_table} WHERE " \
            f"lat_min={lat_min} AND lat_max={lat_max} AND dlat={dlat} AND " \
            f"lon_min={lon_min} AND lon_max={lon_max} AND dlon={dlon} AND " \
            f"{z_statement} AND " \
            f"time_min='{time_min}' AND time_max='{time_max}' AND mode='{mode}' AND dtime={dtime} AND " \
            f"{sel_statement};"
        cur = self.connection.execute(q)
        res = cur.fetchall()

        if len(res) == 0:
            logging.info("GridManager: Could not find an existing grid with the given parameters.")
            return

        grid_id = res[0][0]
        return grid_id


class Grid:
    def __init__(self,
                 lat_min=None, lat_max=None, dlat=None,
                 lon_min=None, lon_max=None, dlon=None,
                 z_min=None, z_max=None, dz=None, z_array=None,
                 time_min=None, time_max=None, mode=None, dtime=None, selection=None,
                 bathymetry_grid_path="../../data/bathymetry/gebco_2022_sub_ice_topo/gebco_2022_sub_ice_topo.nc",
                 lat_variable="lat", lon_variable="lon", depth_variable="elevation",
                 grid_id=None):
        """
        Class to create spatio-temporal grids. Latitude/longitude are cell-centered, time/depth are node-centered.

        Args:
            lat_min (float): Minimum latitude.
            lat_max (float): Maximum latitude.
            dlat (float): Latitude step size.
            lon_min (float): Minimum longitude.
            lon_max(float): Maximum longitude.
            dlon (float): Longitude step size.
            z_min (float): Minimum depth.
            z_max (float): Maximum depth.
            dz (float): Depth step size.
            z_array (numpy.array<float>): Array specifying depth levels (if depth steps are irregular, use this instead
            of z_min, z_max and dz).
            time_min (str): Minimum time.
            time_max (str): Maximum time.
            mode (str): Time mode/general time frame of interest (either "Y" (year), "YM" (year-month), "YD" (year-day),
            "M" (month), "MD" (month-day) or "D" (day)).
            dtime (str): Time step size (in units specified by mode).
            selection (numpy.ndarray<boolean>): Boolean array as mask for the time array to (de)select specific times
            from the time grid.
            bathymetry_grid_path (str): Path to bathymetry data.
            lat_variable (str): Name of latitude variable in bathymetry data.
            lon_variable (str): Name of longitude variable in bathymetry data.
            depth_variable (str): Name of depth variable in bathymetry data.
            grid_id (int): ID for the grid to create. If None, a new ID will be created.
        """

        # Use given grid_id or generate one if none was given
        if grid_id:
            self.grid_id = grid_id
            self.grid_name = "grid_" + str(grid_id)
        else:
            self.grid_id = self.generate_grid_id()
            self.grid_name = "grid_" + str(self.grid_id)

        # Init grid parameters
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.dlat = dlat
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.dlon = dlon
        self.z_array = np.sort(np.array(z_array)) if z_array is not None else None
        self.z_min = z_min if z_array is None else self.z_array.min()
        self.z_max = z_max if z_array is None else self.z_array.max()
        self.dz = dz
        self.time_min = time_min
        self.time_max = time_max
        self.mode = mode
        self.dtime = dtime
        self.selection = selection
        self.bathymetry_grid_path = bathymetry_grid_path

        # Check given parameters for consistency
        self.check_input()

        # Apply selection if given
        if self.selection is not None:
            self.selection = np.array(self.selection)

        # Create grid
        self.time_array = self.create_time_array()
        self.grid = self.create_grid(lat_variable, lon_variable, depth_variable)

    def check_input(self):
        # Consistency checks
        assert (self.lat_min is not None)
        assert (self.lat_max is not None)
        assert (self.dlat is not None)
        assert (self.lon_min is not None)
        assert (self.lon_max is not None)
        assert (self.dlon is not None)
        assert (self.time_min is not None)
        assert (self.time_max is not None)
        assert (self.dtime is not None)
        assert (self.z_min is not None)
        assert (self.z_max is not None)
        assert (self.z_min >= 0)
        assert (self.z_max <= 12000)

        if self.z_array is None:
            assert (self.dz is not None)
            # For a regular grid, we need complete grid cells
            assert ((abs(self.lat_min) + abs(self.lat_max)) % self.dlat == 0)
            assert ((abs(self.lon_min) + abs(self.lon_max)) % self.dlon == 0)
            assert ((abs(self.z_min) + abs(self.z_max)) % self.dz == 0)
        else:
            assert (self.dz is None)

        # General latitude/longitude/time ranges
        assert (self.lat_min >= -90)
        assert (self.lat_max <= 90)
        assert (self.lon_min >= -180)
        assert (self.lon_max <= 180)
        assert (self.dtime >= 1)

        if self.selection is not None:
            assert(isinstance(self.selection, np.ndarray) or isinstance(self.selection, list))

        # Check for all possible time modes
        single_modes = ["Y", "M", "D"]  # not yet implemented: , "H", "m", "S"]
        all_modes = [None]
        for i in range(len(single_modes)):
            for j in range(i, len(single_modes)):
                if i == j:
                    all_modes.append(single_modes[i])
                else:
                    all_modes.append(single_modes[i] + single_modes[j])
        assert (self.mode in all_modes)

        # Check if path is valid
        assert (os.path.isfile(self.bathymetry_grid_path))

    def generate_grid_id(self):
        """ Generate a grid_id from current timestamp-  """
        grid_id = round(time.time())  # Use current timestamp in seconds
        return grid_id

    def create_time_array(self):
        """ Create the time array. """
        time_range = DateTimeRange(self.time_min, self.time_max)
        time_array = []

        # Determine all valid dates for the desired time range
        if self.mode == "Y":
            for value in time_range.range(relativedelta(years=+self.dtime)):
                time_array.append(value.strftime("%Y-%m-%d %H:%M:%S"))
        elif self.mode == "YM":
            for value in time_range.range(relativedelta(months=+self.dtime)):
                time_array.append(value.strftime("%Y-%m-%d %H:%M:%S"))
        elif self.mode == "YD":
            for value in time_range.range(relativedelta(days=+int(self.dtime))):
                time_array.append(value.strftime("%Y-%m-%d %H:%M:%S"))
        elif self.mode == "M":
            time_array = np.char.zfill(np.arange(1, 12 + 1, self.dtime).astype(str), 2)
        elif self.mode == "MD":
            for value in DateTimeRange("2000-01-01", "2000-12-31").range(relativedelta(days=+int(self.dtime))):
                time_array.append(value.strftime("%m-%d %H:%M:%S"))
        elif self.mode == "D":
            time_array = np.char.zfill(np.arange(1, 31 + 1, self.dtime).astype(str), 2)

        # Perform selection
        time_array = np.array(time_array)
        if self.selection is not None:
            assert (np.equal(time_array.shape, self.selection.shape))
            time_array = time_array[self.selection.astype(bool)]

        return time_array

    def create_grid(self, lat_variable="lat", lon_variable="lon", depth_variable="elevation"):
        """
        Create grid using given bathymetry data.
        Bases on: https://github.com/willirath/geomar-open-hacky-hour-2021-04/blob/main/2022-06-23/etopo05_to_grid.ipynb
        Bathymetry data must be an nc file that includes latitude, longitude and depth information.

        Args:
            lat_variable (str): Name of latitude variable in bathymetry data.
            lon_variable (str): Name of longitude variable in bathymetry data.
            depth_variable (str): Name of depth variable in bathymetry data.
        """

        grid_dataset = xr.open_dataset(self.bathymetry_grid_path)
        grid_dataset = grid_dataset.rename({
            lat_variable: "LATITUDE",
            lon_variable: "LONGITUDE",
            depth_variable: "LEV_M"
        }
        )

        # Positive z points downwards (convention in COMFORT)
        grid_dataset["LEV_M"] = grid_dataset["LEV_M"] * -1

        # Will be used as grid selectors (cell-centered)
        lat = xr.DataArray(np.arange(self.lat_min + self.dlat / 2, self.lat_max, self.dlat), dims="LATITUDE")
        lon = xr.DataArray(np.arange(self.lon_min + self.dlon / 2, self.lon_max, self.dlon), dims="LONGITUDE")

        # Will be used to generate a boolean array which is True where water is present (cell-centered)
        if self.z_array is None:
            z_arr = np.arange(self.z_min, self.z_max, self.dz)
        else:
            z_arr = self.z_array[:-1]

        z = xr.DataArray(
            z_arr,
            dims="LEV_M",
            coords={"LEV_M": z_arr},
            name="LEV_M",
        )

        z_at_grid = grid_dataset.LEV_M.sel(
            LONGITUDE=lon,
            LATITUDE=lat,
            method="nearest"
        )

        # Make sure we label with the selectors
        z_at_grid.coords["LATITUDE"] = lat
        z_at_grid.coords["LONGITUDE"] = lon

        # Add a dimension which is True if there is water
        water_filled = (z <= z_at_grid).rename("water")
        # water_filled.plot(col="LEV_M", col_wrap=3)

        # Add time dimension
        water_filled = water_filled.expand_dims(DATEANDTIME=self.time_array)
        # water_filled[:, 0, :, :].plot(col="time")

        df = water_filled.to_dataframe().reset_index()
        # write grid to SQL database
        # df = df[df["water"] == 1].drop(["water"], axis=1)  # only keep grid points where there is water

        # Have right dtypes
        df = df.astype({"DATEANDTIME": str, "LEV_M": float, "LATITUDE": float, "LONGITUDE": float, "water": bool})

        return df

    def map_tables(self, conn, param_tables=None, using_database=True, include_z_max=True):
        """
        Mapping given parameter tables to grid.
        Args:
            conn (sqlite3.connection): Connection to database.
            param_tables (list<str>): Parameter table names to map to the grid.
            using_database (boolean): Whether to store mapped tables in the database or return them in-memory.
            include_z_max (boolean): Whether to include depth values at the end of the interval.
        Returns:
            mapped_tables (list<str> or list<pandas.DataFrame): Names of mapped parameter tables in database (if
            using_database) or list of mapped tables (if using_database=False).
        """
        logging.info("Mapping tables to grid...")
        conn.create_aggregate('median', 1, sqlite_util.Median)
        conn.create_aggregate('std', 1, sqlite_util.Std)

        if not param_tables:
            param_tables = get_names_of_all_parameter_tables(conn)

        mapped_tables = []  # Offline computations: contains grids; online computations: contains table names
        for table in param_tables:
            answer = "n"  # Whether to replace table in database if existing
            # If mapped table already exists in database, only continue computations if they should be replaced
            if using_database and does_table_exist(conn=conn, table_name=f"{table}_{self.grid_id}",
                                                   table_type="table"):
                grid_name = f"grid_{self.grid_id}"
                # Let user decide whether to replace existing table
                input_valid = False
                while not input_valid:
                    print(f"Replace table {table}? y/n")
                    answer = input()
                    if answer == "y" or answer == "n":
                        input_valid = True
                    else:
                        print("Invalid input.")
                if answer == "n":
                    mapped_tables.append(f"{table}_{self.grid_id}")
                    continue

            # Load data
            # Where clause --> filter samples for latitude, longitude, depth, time range
            z_eq = "<=" if include_z_max else "<"
            q = f"SELECT LATITUDE, LONGITUDE, LEV_M, DATEANDTIME, VAL " \
                f"FROM {table} WHERE " \
                f"LATITUDE >= {self.lat_min} AND LATITUDE <= {self.lat_max} AND " \
                f"LONGITUDE >= {self.lon_min} AND LONGITUDE <= {self.lon_max} AND " \
                f"LEV_M >= {self.z_min} AND LEV_M {z_eq} {self.z_max} AND " \
                f"DATEANDTIME >= '{self.time_min}' AND DATEANDTIME <= '{self.time_max}' AND " \
                f"VAL IS NOT NULL;"
            logging.info(q)
            cur = conn.execute(q)
            df = pd.DataFrame(cur.fetchall(), columns=[x[0] for x in cur.description])

            # Map latitude
            lat_bins = np.arange(self.lat_min, self.lat_max + self.dlat, self.dlat).astype(float)
            lat_bins[-1] = lat_bins[-1] + 0.001 * (1 if lat_bins[-1] >= 0 else -1)
            # df["LATITUDE_binned"] = pd.cut(df["LATITUDE"], bins=lat_bins, right=False, labels=lat_bins[:-1])
            df["LATITUDE"] = pd.cut(df["LATITUDE"], bins=lat_bins, right=False,
                                    labels=lat_bins[:-1] + self.dlat / 2)

            # Map longitude
            lon_bins = np.arange(self.lon_min, self.lon_max + self.dlon, self.dlon).astype(float)
            lon_bins[-1] = lon_bins[-1] + 0.001 * (1 if lon_bins[-1] >= 0 else -1)
            df["LONGITUDE"] = pd.cut(df["LONGITUDE"], bins=lon_bins, right=False,
                                     labels=lon_bins[:-1] + self.dlon / 2)

            # Map depth
            if self.z_array is not None:
                z_bins = self.z_array.astype(float)
            else:
                z_bins = np.arange(self.z_min, self.z_max + self.dz, self.dz).astype(float)

            if len(z_bins) == 1:
                df["LEV_M"] = z_bins[0]
            else:
                z_bins[-1] = z_bins[-1] + 0.001  # make sure, last values are included while cutting using pandas
                df["LEV_M"] = pd.cut(df["LEV_M"], bins=z_bins, right=False, labels=z_bins[:-1])

            # Map time
            if self.mode in ["Y", "YM", "YD"]:
                time_bins = pd.DatetimeIndex(self.time_array)
                time_bins = time_bins.union([pd.Timestamp(self.time_max) + pd.Timedelta(nanoseconds=1)])
                # Round microseconds
                df["DATEANDTIME"] = pd.to_datetime(df["DATEANDTIME"], format="ISO8601").dt.round("s")
                df["DATEANDTIME"] = pd.cut(pd.to_datetime(df["DATEANDTIME"], format="%Y-%m-%d %H:%M:%S"),
                                           bins=time_bins, labels=time_bins[:-1], right=False)
                # df["DATEANDTIME"] = df["DATEANDTIME"].astype(np.datetime64).dt.strftime("%Y-%m-%d %H:%M:%S")
                df["DATEANDTIME"] = pd.to_datetime(df["DATEANDTIME"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            elif self.mode == "M":
                # Get month from date column
                # df["DATEANDTIME"] = df["DATEANDTIME"].astype(np.datetime64).dt.strftime("%m")
                df["DATEANDTIME"] = pd.to_datetime(df["DATEANDTIME"]).dt.strftime("%m")
            elif self.mode == "MD":
                # Get month-day from date column
                df["DATEANDTIME"] = df["DATEANDTIME"].astype(np.datetime64).dt.strftime("%m-%d %H:%M:%S")
            elif self.mode in ["D"]:
                # Get day from date column
                df["DATEANDTIME"] = df["DATEANDTIME"].astype(np.datetime64).dt.strftime("%d")

            # Averaging cells with same latitude, longitude, depth and time
            df_grouped = df[["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME", "VAL"]] \
                .groupby(by=["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"], observed=True) \
                .agg(["mean", "median", "std", "count"]).reset_index()
            df_grouped.columns = [x[0] if x[0] != "VAL" else x[1] for x in list(df_grouped.columns.to_flat_index())]
            df_grouped = df_grouped.rename(columns={"mean": "VAL"})

            # Check data types
            df_grouped = df_grouped.astype({"DATEANDTIME": str, "LEV_M": float,
                                            "LATITUDE": float, "LONGITUDE": float, "VAL": float})

            if using_database:
                grid_name = f"grid_{self.grid_id}"
                # If grid was not yet added to database, add it
                if not does_table_exist(conn, grid_name, "table"):
                    self.grid.to_sql(grid_name, conn, if_exists="replace", index=True, index_label="idx")
                    print("Warning in gridding.map_tables: Grid was added to the database but not to the grid_info "
                          "table as no GridManager was used.")

                # If mapped table already exists and user wants to replace it, do that
                if answer == "y":
                    # Drop existing table
                    q = f"DROP TABLE {table}_{self.grid_id};"
                    conn.execute(q)

                # Write mapped table to database
                df_grouped.to_sql(f"temp_{table}_{self.grid_id}", conn, if_exists="replace", index=False)

                # Adding the grid index to the table
                q = f"CREATE TABLE {table}_{self.grid_id} AS " \
                    f"SELECT g.idx, " \
                    f"t.LATITUDE, t.LONGITUDE, t.LEV_M, t.DATEANDTIME, t.VAL as {table}, t.median, t.std, t.count " \
                    f"FROM temp_{table}_{self.grid_id} AS t LEFT JOIN grid_{self.grid_id} AS g " \
                    f"USING(LATITUDE, LONGITUDE, LEV_M, DATEANDTIME);"
                conn.execute(q)

                # Drop temporary mapped table
                q = f"DROP TABLE temp_{table}_{self.grid_id};"
                conn.execute(q)

                # Add grid name to return-list
                mapped_tables.append(f"{table}_{self.grid_id}")

            else:
                # Join grid and parameter table and return table
                joined = self.grid.merge(df_grouped, on=["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"],
                                         how="left")
                joined.rename(columns={"VAL": table}, inplace=True)
                mapped_tables.append(joined)

        return mapped_tables


def __drop_land_cells(df_wide):
    """
    Dropping land cells from a given table if that land cell does not contain any measurement value.

    Args:
        df_wide (pandas.DataFrame): Table where to remove land cells.
    Returns:
        df_wide (pandas.DataFrame): Table where land cells are removed.
    """
    # Get parameter columns
    param_tables = [x for x in df_wide.columns if x.startswith("P_")]

    # Drop points that are land and have no parameter value assigned to it
    # Make sure that for each depth level, the grid looks the same for all times
    depths = df_wide["LEV_M"].value_counts().index.tolist()
    times = df_wide["DATEANDTIME"].value_counts().index.tolist()
    for d in depths:
        condition = True
        indexes = pd.DataFrame()
        for param in param_tables:
            for t in times:
                ddtt = df_wide[(df_wide["LEV_M"] == d) & (df_wide["DATEANDTIME"] == t)].reset_index()
                condition = condition & ~ddtt["water"] & ddtt[param].isna()
                indexes = pd.concat([indexes, pd.DataFrame({f"{param}_{t}": ddtt["index"]})], axis=1)
        indexes = indexes[condition]  # Get indexes that can be dropped
        indexes = pd.concat(indexes[col] for col in indexes)  # Flatten to get one list of indexes to drop
        df_wide = df_wide.drop(indexes)
    return df_wide


# def create_wide_table_offline(mapped_tables, dropping_land_cells=True):
#     # get parameter columns
#     param_tables = []
#     for df in mapped_tables:
#         param_tables.append([x for x in df.columns if x.startswith("P_")][0])
#
#     # get dataframes
#     param_idx = []
#     for idx, mt in enumerate(mapped_tables):
#         for col in mt.columns:
#             if col in param_tables:
#                 # print(idx, mt.columns)
#                 param_idx.append([col, idx])
#
#     idxs = [pai[1] for pai in param_idx]
#     params = [pai[0] for pai in param_idx]
#
#     # merge params and grid into one wide table
#     df_wide = mapped_tables[idxs[0]][["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME", "water", param_tables[0]]]
#     for i in range(1, len(idxs)):
#         df_wide = df_wide.merge(mapped_tables[idxs[i]][["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME", params[i]]],
#                                 how="left", on=["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"])
#
#     if dropping_land_cells:
#         df_wide = __drop_land_cells(df_wide)
#
#     return df_wide


def create_wide_table_online(connection, grid_id, param_tables=None):
    """
    Creates step-wise a wide table with the columns:
    latitude, longitude, dateandtime, lev_m, param_0, ..., param_n (where n=[0, #param_tables - 1]).
    Given parameter tables need to be mapped already and stored in the database as paramTableName_gridId.

    Args:
        connection (sqlite.Connection): Connection to the database.
        grid_id (int): ID of the grid based on which to join the parameter tables.
        param_tables (list<str>): Names of the parameter tables to merge. If None, all parameters will be used.
        Default is None.
    Returns:
        new_path (str): Path to the new database.
        table_name (str): Name of the created wide table.
    """
    # If no parameter tables are specified, use all
    if not param_tables:
        param_tables = get_names_of_all_parameter_tables(connection)

    vals = ", ".join([f"{param_table}_{grid_id}.{param_table}" for param_table in param_tables])
    joins = " ".join([f"left join {param_table}_{grid_id} using(idx)" for param_table in param_tables])

    wide_table_name = f"wide_{grid_id}_{round(time.time())}"
    q = f"CREATE TABLE {wide_table_name} AS SELECT " \
        f"grid_{grid_id}.idx, grid_{grid_id}.DATEANDTIME, grid_{grid_id}.LEV_M, grid_{grid_id}.LATITUDE, " \
        f"grid_{grid_id}.LONGITUDE, grid_{grid_id}.water, {vals} " \
        f"FROM grid_{grid_id} " \
        f"{joins};"
    connection.execute(q)

    return wide_table_name


def load_wide_table(conn, wide_table_name, dropping_land_cells=True):
    """
    Loads wide table from database and drops the land cells from that table.

    Args:
        conn (sqlite3.connection): Connection to database.
        wide_table_name (str): Name of the wide table in the database.
        dropping_land_cells (boolean): Whether to remove land cells or not.
    Returns:
        df_wide (pandas.DataFrame): Loaded table.
    """
    df_wide = get_table_as_df(conn=conn, table_name=wide_table_name)
    if dropping_land_cells:
        df_wide = __drop_land_cells(df_wide)
    return df_wide


def get_missing_value_info_offline(df_wide):
    """
    Compute the absolute and relative number of missing values per parameter.
    Args:
        df_wide (pandas.DataFrame): Table to investigate.
    Returns:
        num_nulls (pandas.DataFrame): Table containing missing value information.
    """
    # Get parameter columns
    param_tables = [x for x in df_wide.columns if x.startswith("P_")]

    # Compute absolute number of missing values per parameter
    num_nulls = pd.DataFrame(columns=["parameter", "absolute"])
    for param in param_tables:
        num_nulls_param = df_wide[param].isna().sum()
        num_nulls = pd.concat([num_nulls,
                               pd.DataFrame({"parameter": [param],
                                             "absolute": [num_nulls_param]
                                             })]
                              )

    # Compute relative number of missing values
    num_grid_cells = len(df_wide)
    if num_grid_cells == 0:
        num_nulls["relative"] = 0
    else:
        num_nulls["relative"] = num_nulls["absolute"] / num_grid_cells * 100
        num_nulls = num_nulls.sort_values(by="relative")
    logging.info(f"  Number of NULLs is {num_nulls}")

    return num_nulls


def grid_data(conn, grid_config, bathymetry_path, parameters, output_dir):
    """
    Map given parameters to a specified grid in a database and store resulting table as CSV.
        conn (sqlite3.connection): Connection to database.
        grid_config (dict): Configuration of the grid.
        bathymetry_path (str): Path to bathymetry data.
        parameters (list<str>): List of parameter table names to map to the grid.
        output_dir (str): Directory where the CSV file will be stored.
    Returns:
        wide_table_path (str): Path to the CSV file of the gridded parameters.
    """
    logging.info("Create grid...")

    # Grid configuration
    # parameters = grid_config["parameters"]
    lat_min = grid_config["lat_min"]
    lat_max = grid_config["lat_max"]
    dlat = grid_config["dlat"]
    lon_min = grid_config["lon_min"]
    lon_max = grid_config["lon_max"]
    dlon = grid_config["dlon"]
    z_min = grid_config["z_min"]
    z_max = grid_config["z_max"]
    dz = grid_config["dz"]
    z_array = np.array(grid_config["z_array"])
    time_min = grid_config["time_min"]
    time_max = grid_config["time_max"]
    mode = grid_config["mode"]
    selection = grid_config["selection"]
    dtime = grid_config["dtime"]

    # Variable names for bathymetric data
    lat_variable = "lat"
    lon_variable = "lon"
    depth_variable = "elevation"

    # Apply gridding
    grid_manager = GridManager(connection=conn, grid_info_table="grid_info")
    grid = grid_manager.create_grid(lat_min=lat_min, lat_max=lat_max, dlat=dlat,
                                    lon_min=lon_min, lon_max=lon_max, dlon=dlon,
                                    z_min=z_min, z_max=z_max, dz=dz, z_array=z_array,
                                    time_min=time_min, time_max=time_max, dtime=dtime, mode=mode,
                                    selection=selection,
                                    bathymetry_grid_path=bathymetry_path,
                                    lat_variable=lat_variable, lon_variable=lon_variable,
                                    depth_variable=depth_variable
                                    )

    # Map parameters to grid and create wide table
    logging.info("Map parameters to grid...")
    mapped = grid.map_tables(conn=conn, param_tables=parameters, using_database=True, include_z_max=True)
    wide_table_name = create_wide_table_online(connection=conn, grid_id=grid.grid_id, param_tables=parameters)
    df_wide = load_wide_table(conn=conn, wide_table_name=wide_table_name, dropping_land_cells=True)
    num_nulls = get_missing_value_info_offline(df_wide)

    # Logging
    logging.info(f"Created wide table named {wide_table_name}. \nMapped parameter tables: {mapped}")
    logging.info(f"The amount of missing values per parameter is: \n{num_nulls}")

    # Store as CSV
    logging.info("Writing wide table to CSV...")
    wide_table_path = output_dir + "wide_table.csv"
    df_wide.to_csv(wide_table_path, index=False)
    logging.info(f"Stored gridded, wide table as {wide_table_path}")

    return wide_table_path


def df_to_gridded_da(df, value_col="P_TEMPERATURE"):
    """
    Convert a DataFrame to a 4D xarray.DataArray (time, depth, lat, lon)
    with integer years as time coordinate.
    """
    df = df.copy()

    # Ensure integer year
    df["DATEANDTIME"] = df["DATEANDTIME"].astype(int)

    df = df.sort_values(["DATEANDTIME", "LEV_M", "LATITUDE", "LONGITUDE"])
    times = np.sort(df["DATEANDTIME"].unique())
    depths = np.sort(df["LEV_M"].unique())
    lats = np.sort(df["LATITUDE"].unique())
    lons = np.sort(df["LONGITUDE"].unique())

    if is_numeric_dtype(df[value_col]):
        fill_val = np.nan
        dtype = float
    elif is_string_dtype(df[value_col]):
        fill_val = ""
        dtype = object
    else:
        print("Unkown type")
        return

    data = np.full((len(times), len(depths), len(lats), len(lons)), fill_val, dtype=dtype)

    for ti, t in enumerate(times):
        subset = df[df["DATEANDTIME"] == t]
        for di, d in enumerate(depths):
            sel = subset[subset["LEV_M"] == d]
            if sel.empty:
                continue
            lat_idx = np.searchsorted(lats, sel["LATITUDE"])
            lon_idx = np.searchsorted(lons, sel["LONGITUDE"])
            data[ti, di, lat_idx, lon_idx] = sel[value_col].values

    da = xr.DataArray(
        data,
        dims=("time", "depth", "lat", "lon"),
        coords={"time": times, "depth": depths, "lat": lats, "lon": lons},
        name=value_col,
    )
    return da

