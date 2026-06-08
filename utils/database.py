"""
Following code from:
Yvonne Jenniges. (2025). y-jenniges/ocean_clustering_and_validation: Biogeochemical Ocean Regions - Code Base (v1.0.1).
Zenodo. https://doi.org/10.5281/zenodo.15827777
"""
import logging
import pandas as pd


# -------------------------------------------------------------------------------------------------------------------- #
# Utility functions to work with the COMFORT database
# -------------------------------------------------------------------------------------------------------------------- #


def get_table_as_df(conn, table_name, columns=None):
    """
    Fetch a database table as a pandas DataFrame.

    Parameters
    ----------
    conn : sqlite3.Connection
    table_name : str
    columns : list[str] or None, default=None
        Columns to select; None fetches all columns.

    Returns
    -------
    DataFrame
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
    Return the total row count across the given tables.

    Parameters
    ----------
    conn : sqlite3.Connection
    table_names : list[str]

    Returns
    -------
    int
    """
    cur = conn.cursor()

    query = "SELECT " + " + ".join([f"(SELECT COUNT(*) FROM {x})" for x in table_names]) + ";"
    ex = cur.execute(query)
    num_samples = ex.fetchall()[0][0]
    cur.close()

    return num_samples


def does_table_exist(conn, table_name, table_type="table"):
    """
    Check (case-sensitively) whether a table or view exists in the database.

    Parameters
    ----------
    conn : sqlite3.Connection
    table_name : str
    table_type : str, default='table'
        SQLite structure type to check (``'table'`` or ``'view'``).

    Returns
    -------
    bool
    """
    cur = conn.cursor()
    query = f"SELECT name FROM sqlite_master WHERE type='{table_type}' AND name='{table_name}';"
    result = cur.execute(query).fetchall()

    return True if result else False


def remove_tables_like(conn, like_pattern="E|_%", escape_char="|", table_type="table", tables_except=None):
    """
    Drop all tables/views whose names match ``like_pattern``.

    Parameters
    ----------
    conn : sqlite3.Connection
    like_pattern : str, default='E|_%'
        SQLite LIKE pattern for table names.
    escape_char : str, default='|'
        Escape character for the LIKE pattern.
    table_type : str, default='table'
        Structure type to remove (``'table'`` or ``'view'``).
    tables_except : list[str] or None, default=None
        Names to exclude from removal.
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
    Return all table/view names matching ``like_pattern``.

    Parameters
    ----------
    conn : sqlite3.Connection
    like_pattern : str, default='P|_%'
        SQLite LIKE pattern for table names.
    escape_char : str, default='|'
    include_digits : bool, default=False
        When False, table names containing digits are excluded.
    table_type : str, default='table'

    Returns
    -------
    list[str]
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
    Return the column names of ``table``.

    Parameters
    ----------
    conn : sqlite3.Connection
    table : str

    Returns
    -------
    list[str]
    """
    # Create cursor and get table info
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")

    # Fetch column names
    columns = [row[1] for row in cur.fetchall()]

    # Close cursor
    cur.close()

    return columns
