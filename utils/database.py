# TAKEN FROM PREVIOUS PAPER
import logging
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# Utility functions to work with the COMFORT database
# -------------------------------------------------------------------------------------------------------------------- #


def get_table_as_df(conn, table_name, columns=None):
    """
    Get a table from the database as pandas dataframe.

    Args:
        table_name (str): Name of the table to fetch.
        conn (sqlite3.Connection): Connection to the database that stores the table.
        columns (list<str>): List of columns to query. If None, all columns will be queried. Default is None.

    Returns:
        df (pandas.DataFrame): Dataframe containing the table information.
    """
    col_selection = "*"
    if columns:
        col_selection = ", ".join(columns)

    # Define query
    query = f"select {col_selection} from {table_name};"
    cur = conn.cursor()
    ex = cur.execute(query)
    cols = [description[0] for description in cur.description]

    # Execute query
    result = ex.fetchall()
    df = pd.DataFrame(result, columns=cols)
    cur.close()

    return df


def get_num_samples(conn, table_names):
    """
    Get the number of samples contained in the given table.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        table_names (list<str>): List of table names to count.
    Returns:
        num_samples (int): Summed number of sample in the given tables.
    """
    cur = conn.cursor()

    query = "SELECT " + " + ".join([f"(SELECT COUNT(*) FROM {x})" for x in table_names]) + ";"
    ex = cur.execute(query)
    num_samples = ex.fetchall()[0][0]
    cur.close()

    return num_samples


def does_table_exist(conn, table_name, table_type="table"):
    """
    Checks if a table/view with the given name exists. Warning: Make sure that the casing is correct. Function is
    case-sensitive!

    Args:
        conn (sqlite3.Connection): Connection to the database.
        table_name (str): Check if this table name is already in the database.
        table_type (str): Type of the database structure to check, e.g. 'view'. Default is 'table'.
    Returns:
        does_table_exist (bool): If the table/view exists in the database.
    """
    cur = conn.cursor()
    query = f"SELECT name FROM sqlite_master WHERE type='{table_type}' AND name='{table_name}';"
    result = cur.execute(query).fetchall()

    return True if result else False


def remove_tables_like(conn, like_pattern="E|_%", escape_char="|", table_type="table", tables_except=None):
    """
    Removes views/tables whose name match the given like_pattern.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        like_pattern (str): Table names should conform to this string pattern. If None, all parameter tables are
        queried. Default is 'E|_%'
        escape_char (str): Escape char used for the like pattern. If None, sqlite3 default is used. Default is '|'.
        table_type (str): Type of the structure (view or table) to remove. Default is 'view'.
        tables_except (list<str>): Remove all tables matching the like pattern except the ones specified in this list.
        Default is None.
    """
    # Get views/tables to remove
    table_names = get_names_of_all_parameter_tables(conn, like_pattern, escape_char, table_type=table_type,
                                                    include_digits=True)
    cur = conn.cursor()

    # Keep tables that were excepted from removal
    if tables_except:
        for te in tables_except:
            if te in table_names:
                table_names.remove(te)

    # Remove tables from database
    for vtn in table_names:
        query = f"DROP {table_type} {vtn};"
        print(f"Structure.remove_tables_like: {query}")
        cur.execute(query)

    cur.close()


def get_names_of_all_parameter_tables(conn, like_pattern="P|_%", escape_char="|", include_digits=False,
                                      table_type="table"):
    """
    Get all tables conforming with the given like_pattern.

    Args:
        conn (sqlite3.Connection): Connection to the sqlite3 database.
        like_pattern (str): Table names should conform to this string pattern. If None, all table names are queried.
        Default is 'P|_%'
        escape_char (str): Escape char used for the like pattern. If None, sqlite3 default are used. Default is '|'.
        include_digits (bool): Weather tables names including numbers should be returned as well.
        table_type (str): The table type to look for (e.g. 'view'). Default is 'table'.
    Returns:
        param_table_names (list<str>): A list of table names that conform to the given pattern.
    """
    digits_filter = ""
    if not include_digits:
        digits_filter = "AND name NOT glob '*_[0-9]*'"

    # Assemble query
    if like_pattern and escape_char:
        query = f"SELECT name FROM sqlite_master " \
                f"WHERE type='{table_type}' AND name LIKE '{like_pattern}' ESCAPE '{escape_char}' " \
                f"{digits_filter};"
    elif like_pattern:
        query = f"SELECT name FROM sqlite_master " \
                f"WHERE type='{table_type}' AND name LIKE '{like_pattern}' " \
                f"{digits_filter};"
    else:
        query = f"SELECT name FROM sqlite_master " \
                f"Where type='{table_type}' " \
                f"{digits_filter};"

    # Execute query
    logging.info(f"Information: {query}")
    cur = conn.cursor()
    ex = cur.execute(query)
    result = ex.fetchall()
    param_table_names = [entry[0] for entry in result]
    cur.close()

    return param_table_names


def get_columns(conn, table):
    """
    Get column names of a table in a database.

    Args:
        conn (sqlite3.Connection): Connection to the sqlite3 database.
        table (str): Name of table.
    """
    # Create cursor and get table info
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")

    # Fetch column names
    columns = [row[1] for row in cur.fetchall()]

    # Close cursor
    cur.close()

    return columns
