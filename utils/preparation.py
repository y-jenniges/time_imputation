# TAKEN FROM PREVIOUS PAPER
import sqlite3
import logging
import gsw
import pandas as pd
from utils.database import get_table_as_df, get_num_samples, does_table_exist, get_columns
from utils.units import UnitsConverter
from utils.gridding import grid_data

# from utils.imputation import impute_data


def __copy_and_filter_tables(conn, tables, quality_flags, source_db_path):
    """
    Copy tables from a database to another database (if it does not yet exist there) and filter for quality.

    Args:
        conn (sqlite3.connection): Connection to database.
        tables (list<str>): Table names to copy.
        quality_flags (list<list<str>>): Specification for PQF1, PQF2 and SQF used to filter the data for quality.
        source_db_path (str): Cursor of the source database.
    """
    logging.info("  Copying tables to new database...")

    # Create cursor to database
    cursor = conn.cursor()

    # Attach the source database to the destination database
    cursor.execute(f"ATTACH DATABASE '{source_db_path}' AS source_db")

    # Copy tables to destination database (ensure that temperature and salinity are always copied)
    for parameter in list(set(tables + ["P_TEMPERATURE", "P_SALINITY"])):
        # Check if table already exists in new database
        if does_table_exist(conn=conn, table_name=parameter, table_type="table"):
            logging.info(f"    Table {parameter} already exists. Skipping copy...")
        else:
            query = f"CREATE TABLE IF NOT EXISTS {parameter} AS " \
                    f"SELECT * FROM source_db.{parameter} " \
                    f"WHERE {' and '.join([x[0] + x[1] for x in quality_flags])};"
            logging.info("    " + query)
            cursor.execute(query)

    # Copy STATION, CRUISE, DATABASE_TABLES, UNITS for location and time information and info on tables and units
    for table in ["STATION", "CRUISE", "DATABASE_TABLES", "UNITS"]:
        query = f"CREATE TABLE IF NOT EXISTS {table} AS " \
                f"SELECT * FROM source_db.{table};"
        logging.info("    " + query)
        cursor.execute(query)

    # Close cursor
    cursor.close()


def __add_location_time(conn, tables):
    """
    Add latitude, longitude and dateandtime information to tables (if they do not have these columns yet).

    Args:
        conn (sqlite3.connection): Connection to database.
        tables (list<str>): Names of tables to add the location and time information to.
    """
    logging.info("  Adding location and time...")

    # Add location/time info to each table separately
    for table in tables:
        logging.info("    " + table)

        # Create cursor object
        cursor = conn.cursor()

        # Check if table contains lat/lon/time information
        columns = get_columns(conn=conn, table=table)
        if len([x for x in columns if x in ["LATITUDE", "LONGITUDE", "DATEANDTIME"]]) > 0:
            logging.info(f"    Table {table} already contains latitude, longitude and/or time information. Skipping.")
        else:
            # Join query
            query = f"CREATE TABLE IF NOT EXISTS {table}_llt AS " \
                    f"SELECT p.*, STATION.LATITUDE, STATION.LONGITUDE, STATION.DATEANDTIME " \
                    f"FROM {table} AS p " \
                    f"LEFT JOIN STATION ON p.ID=STATION.ID; "
            logging.info("    " + query)
            cursor.execute(query)

            # Drop initial tables
            query = f"DROP TABLE {table};"
            cursor.execute(query)

            # Rename new tables
            query = f"ALTER TABLE {table}_llt RENAME TO {table};"
            cursor.execute(query)

    # Clean database
    logging.info("    Vacuum database...")
    query = "VACUUM;"
    cursor.execute(query)
    cursor.close()


def __average_over_location_time(conn, tables):
    """
    Average VAL column of a table over LATITUDE, LONGITUDE and DATEANDTIME columns.

    Args:
        conn (sqlite3.connection): Connection to database.
        tables (list<str>): Names of tables to process.
    """
    logging.info("  Averaging over location and time...")
    for table in tables:
        logging.info("    " + table)

        # Create cursor object
        cursor = conn.cursor()

        # Get column names
        cols = [x for x in get_columns(conn=conn, table=table) if x != "VAL"]

        # Grouping
        query = f"CREATE TABLE {table}_avg AS " \
                f"SELECT AVG(VAL) AS VAL, {', '.join(cols)} " \
                f"FROM {table} GROUP BY LATITUDE, LONGITUDE, LEV_M, DATEANDTIME;"
        logging.info("    " + query)
        cursor.execute(query)

        # Drop initial tables
        query = f"DROP TABLE {table};"
        cursor.execute(query)

        # Rename new tables
        query = f"ALTER TABLE {table}_avg RENAME TO {table};"
        cursor.execute(query)

    # Clean database
    logging.info("    Vacuum database...")
    query = "VACUUM;"
    cursor.execute(query)
    cursor.close()


def __add_temperature_salinity(conn, tables):
    """
    Joining temperature and salinity information to given tables.

    Args:
        conn (sqlite3.connection): Connection to database.
        tables (list<str>): Tables to add temperature and salinity to.
    """
    logging.info("  Adding temperature/salinity information...")

    for table in tables:
        logging.info("    " + table)

        # Create cursor object
        cursor = conn.cursor()

        # Check if temperature/salinity information is already given
        columns = get_columns(conn=conn, table=table)
        if len([x for x in columns if x in ["temperature", "salinity"]]) > 0:
            logging.info(f"    Table {table} already contains temperature and/or salinity information. Skipping.")
        else:
            # Adding temperature and salinity information as required
            if table == "P_TEMPERATURE":
                ts_select_statement = ", s.VAL as salinity"
                ts_join_statement = f"LEFT JOIN P_SALINITY s ON (p.LEV_M=s.LEV_M and p.ID=s.ID)"
            elif table == "P_SALINITY":
                ts_select_statement = ", t.VAL as temperature"
                ts_join_statement = f"LEFT JOIN P_TEMPERATURE t ON (p.LEV_M=t.LEV_M and p.ID=t.ID)"
            else:
                ts_select_statement = ", t.VAL as temperature, s.VAL as salinity"
                ts_join_statement = f"LEFT JOIN P_TEMPERATURE t ON (p.LEV_M=t.LEV_M and p.ID=t.ID) " \
                                    f"LEFT JOIN P_SALINITY s ON (p.LEV_M=s.LEV_M and p.ID=s.ID)"

            # Add temperature and salinity information
            query = f"CREATE TABLE {table}_ts AS " \
                    f"SELECT p.* {ts_select_statement} " \
                    f"FROM {table} p " \
                    f"{ts_join_statement};"
            logging.info("    " + query)
            cursor.execute(query)

            # Drop initial tables
            query = f"DROP TABLE {table};"
            cursor.execute(query)

            # Rename new tables
            query = f"ALTER TABLE {table}_ts RENAME TO {table};"
            cursor.execute(query)

    # Clean database
    logging.info("    Vacuum database...")
    query = "VACUUM;"
    cursor.execute(query)
    cursor.close()


def __convert_units(conn, tables):
    """
    Converting units of given tables to default units.

    Args:
        conn (sqlite3.connection): Connection to database.
        tables (list<str>): Names of tables to add the location and time information to.
    """
    logging.info("  Converting units...")

    # Define converter and convert
    df_default_units = get_table_as_df(conn, "DATABASE_TABLES")
    df_units = get_table_as_df(conn, "UNITS").sort_values("ID")
    units_converter = UnitsConverter(conn, df_default_units, df_units, value_column="VAL")
    units_converter.convert_units(tables, use_density=True, override_old_tables=True)

    # Tidy up disc usage
    logging.info("    Vacuum database...")
    query = "VACUUM;"
    cursor = conn.cursor()
    cursor.execute(query)
    cursor.close()


def __t_to_pot_t(conn):
    """
    Convert temperature to potential temperature.

    Args:
        conn (sqlite3.connection): Connection to database.
    """
    logging.info("  Convert temperature to potential temperature...")
    t_table = "P_TEMPERATURE"
    cursor = conn.cursor()

    # Get temperature table
    df_t = get_table_as_df(conn, t_table,
                           columns=["ID", "LATITUDE", "LONGITUDE", "LEV_M", "LEV_DBAR", "VAL", "salinity"]).\
        rename(columns={"VAL": "conservative_val"})

    # Apply conversion and drop values that could not be converted and lead to NaN entries
    df_t["salinity_absolute"] = gsw.SA_from_SP(SP=df_t["salinity"], p=df_t["LEV_DBAR"] - 10.1325,
                                               lon=df_t["LONGITUDE"], lat=df_t["LATITUDE"])
    df_t["VAL"] = gsw.conversions.pt0_from_t(SA=df_t["salinity_absolute"], t=df_t["conservative_val"],
                                             p=df_t["LEV_DBAR"])

    # Do not drop values (only salinity abs)
    count_old = len(df_t)
    df_t.drop("salinity_absolute", axis="columns", inplace=True)
    count_new = len(df_t[~df_t["VAL"].isna()])

    # Output how many values were not converted and thus dropped
    logging.info(f"    Conversion not possible for {count_old - count_new} values.")

    # Write potential temperature table to database
    df_t.to_sql("temp", conn, if_exists="fail")

    # Get columns of original temperature table
    q = f"PRAGMA table_info({t_table});"
    ex = cursor.execute(q)
    cols = ["t." + x[1] for x in ex.fetchall() if x[1] != "VAL"]

    # Combine original table and temp table (with potential temperature) so that all columns are in the table
    # q = f"CREATE TABLE temp2 AS SELECT temp.pot_temperature AS VAL, {', '.join(cols)} " \
    q = f"CREATE TABLE temp2 AS SELECT temp.VAL, temp.conservative_val, {', '.join(cols)} " \
        f"FROM temp " \
        f"LEFT JOIN {t_table} AS t USING(ID, LEV_M); "
    cursor.execute(q)

    # Drop original table
    q = f"DROP TABLE {t_table};"
    cursor.execute(q)
    q = f"DROP TABLE temp;"
    cursor.execute(q)

    # Renaming
    q = f"ALTER TABLE temp2 RENAME TO {t_table};"
    cursor.execute(q)

    # Clear database
    logging.info("    Vacuum database...")
    q = "VACUUM;"
    cursor.execute(q)

    # Close cursor
    cursor.close()


def prepare_database(parameters, quality_flags, temperature_to_potential=True,
                     source_db_path="../../data/comfort.sqlite",
                     dest_db_path="output/custom.db"):
    """
    Store data of interest in new database and prepare it for further processing. Operations are executed in the
    database, except for unit conversions.

    Args:
        parameters (list<str>): Parameter tables of interest.
        quality_flags (list<str>): List of quality flags to filter for.
        temperature_to_potential (boolean): Whether to convert temperature to potential temperature.
        source_db_path (str): Path to source database.
        dest_db_path (str): Path to destination database.
    """
    logging.info("Start database preparation...")

    # Define quality flags to filter for
    if not quality_flags:
        quality_flags = [["pqf1", ">0"], ["pqf2", ">2"], ["sqf", ">=-1"]]

    # Connect to COMFORT database
    source_conn = sqlite3.connect(source_db_path)
    logging.info(
        f"Number of samples (original tables): {get_num_samples(conn=source_conn, table_names=parameters)}")

    # Connect to a new database (or create it if it doesn't exist)
    dest_conn = sqlite3.connect(dest_db_path)

    # Copy desired tables to new database and filter for quality
    __copy_and_filter_tables(tables=parameters, quality_flags=quality_flags,
                             conn=dest_conn, source_db_path=source_db_path)
    logging.info(f"Number of samples (filtered tables): {get_num_samples(conn=dest_conn, table_names=parameters)}")

    # Add latitude, longitude, dateandtime information to temperature and salinity tables
    __add_location_time(tables=["P_TEMPERATURE", "P_SALINITY"], conn=dest_conn)
    logging.info(f"Number of samples (TS with space/time): {get_num_samples(conn=dest_conn, table_names=parameters)}")

    # Average temperature and salinity values at the same time and location
    __average_over_location_time(tables=["P_TEMPERATURE",  "P_SALINITY"], conn=dest_conn)
    logging.info(
        f"Number of samples (TS averaged over space/time): "
        f"{get_num_samples(conn=dest_conn, table_names=parameters)}")

    # Add temperature and salinity information to all tables
    __add_temperature_salinity(tables=parameters, conn=dest_conn)
    logging.info(f"Number of samples (params with TS): {get_num_samples(conn=dest_conn, table_names=parameters)}")

    # Add latitude, longitude, dateandtime information to other tables
    __add_location_time(tables=[p for p in parameters if p not in ["P_TEMPERATURE",  "P_SALINITY"]],
                        conn=dest_conn)
    logging.info(f"Number of samples (params with space/time): {get_num_samples(conn=dest_conn, table_names=parameters)}")

    # Unit conversions
    __convert_units(tables=parameters, conn=dest_conn)
    logging.info(f"Number of samples (units converted): {get_num_samples(conn=dest_conn, table_names=parameters)}")

    # Average values at same location and position
    __average_over_location_time(tables=[p for p in parameters if p not in ["P_TEMPERATURE", "P_SALINITY"]],
                                 conn=dest_conn)
    logging.info(f"Number of samples (params averaged over space/time): "
                 f"{get_num_samples(conn=dest_conn, table_names=parameters)}")

    # Convert temperature to potential temperature
    if temperature_to_potential:
        __t_to_pot_t(conn=dest_conn)
        logging.info(f"Number of samples (t-->pot_t): {get_num_samples(conn=dest_conn, table_names=parameters)}")

    # Close connections
    source_conn.close()
    dest_conn.close()


def grid_data_as_df(db_path, grid_config, bathymetry_path, parameters, output_dir="output/"):
    """
    Args:
        db_path (str): Path to database.
        grid_config (dict): Configuration for the spatio-temporal grid.
        bathymetry_path (str): Path to bathymetry information.
        parameters (list<str>): Parameter table names to map to the grid.
        output_dir (str): Path to directory where to store the mapped and later the imputed data.
    Returns:
        df (pandas.DataFrame): Imputed wide table.
    """
    # Connect to database
    conn = sqlite3.connect(db_path)

    # Map parameters to grid
    wide_table_path = grid_data(conn=conn, grid_config=grid_config, bathymetry_path=bathymetry_path,
                                parameters=parameters, output_dir=output_dir)

    # Load imputed table
    df = pd.read_csv(wide_table_path)

    return df, wide_table_path

# def grid_and_impute_data(db_path, grid_config, bathymetry_path, parameters, output_dir="output/"):
#     """
#     Args:
#         db_path (str): Path to database.
#         grid_config (dict): Configuration for the spatio-temporal grid.
#         bathymetry_path (str): Path to bathymetry information.
#         parameters (list<str>): Parameter table names to map to the grid.
#         output_dir (str): Path to directory where to store the mapped and later the imputed data.
#     Returns:
#         df (pandas.DataFrame): Imputed wide table.
#     """
#     # Connect to database
#     conn = sqlite3.connect(db_path)
#
#     # Map parameters to grid
#     wide_table_path = grid_data(conn=conn, grid_config=grid_config, bathymetry_path=bathymetry_path,
#                                 parameters=parameters, output_dir=output_dir)
#
#     # Impute missing values
#     imputed_table_path = impute_data(csv_path=wide_table_path, parameters=parameters,
#                                      drop_columns=["DATEANDTIME", "idx"],
#                                      output_dir=output_dir)
#
#     # Load imputed table
#     df = pd.read_csv(imputed_table_path)
#
#     return df
