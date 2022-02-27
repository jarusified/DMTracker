
# ------------------------------------------------------------------------------
# pandas dataframe utils
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from logger import get_logger

LOGGER = get_logger(__name__)

def df_info(df):
    return f"{df.shape}; index={list(df.index.names)}; cols={list(df.columns)}"

def df_unique(df, column, proxy={}):
    column = proxy.get(column, column)
    if column not in df.columns:
        return np.array([])
    return df[column].unique()

def df_count(df, column, proxy={}):
    return len(df_unique(df, column, proxy))

def df_minmax(df, column, proxy={}):
    column = proxy.get(column, column)
    return df[column].min(), df[column].max()

def df_columns(df):
    return df.columns

def df_get_column(df, column, index="name", proxy={}):
    column = proxy.get(column, column)
    return df.set_index(index)[column]

def df_fetch_columns(df, columns, proxy={}):
    columns = [proxy.get(_, _) for _ in columns]
    return df[columns]

def df_filter_by_value(df, column, value, index="name", proxy={}):
    assert isinstance(value, (int, float))
    column = proxy.get(column, column)
    df = df.loc[df[column] > value]
    mask = df[index].isin(df[index].unique())
    return df[mask]

def df_filter_by_list(df, column, values, proxy={}):
    assert isinstance(values, list)
    column = proxy.get(column, column)
    mask = df[column].isin(values)
    return df[mask]

def df_filter_by_search_string(df, column, search_strings, proxy={}):
    column = proxy.get(column, column)
    unq, ids = np.unique(df[column], return_inverse=True)
    unq_ids = np.searchsorted(unq, search_strings)
    mask = np.isin(ids, unq_ids)
    return df[mask]

def df_as_dict(df, from_col, to_col):
    assert isinstance(df, pd.DataFrame)
    assert isinstance(from_col, str) and isinstance(to_col, str)
    assert from_col in df.columns and to_col in df.columns
    df = df[[from_col, to_col]]
    df.set_index(from_col, inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    return df.to_dict()[to_col]

def df_lookup_by_column(df, column, value, proxy={}):
    column = proxy.get(column, column)
    return df.loc[df[column] == value]

def df_lookup_and_list(df, col_lookup, val_lookup, col_list, proxy={}):
    col_lookup = proxy.get(col_lookup, col_lookup)
    col_list = proxy.get(col_list, col_list)
    return np.array(
        list(set(df_lookup_by_column(df, col_lookup, val_lookup)[col_list].values))
    )

def df_group_by(df, columns, proxy={}):
    if isinstance(columns, list):
        columns = [proxy.get(_, _) for _ in columns]
        return df.groupby(columns)
    else:
        assert isinstance(columns, str)
        columns = proxy.get(columns, columns)
        return df.groupby([columns])

def df_bi_level_group(df, group_attrs, cols, group_by, apply_func, proxy={}):

    assert len(group_attrs) in [1, 2]
    _cols = [proxy.get(_, _) for _ in cols] + group_by

    # Set the df.index as the group_attrs
    _df = df.set_index(group_attrs)
    _levels = _df.index.unique().tolist()

    # If "rank" is present in the columns, we will group by "rank".
    has_rank = "rank" in _df.columns
    if has_rank:
        has_rank = df["rank"].unique().shape[0] > 1

    # --------------------------------------------------------------------------
    if not has_rank:
        _cols = [c for c in _cols if c != "rank"]
        return {_: _df.xs(_)[_cols] for _ in _levels}

    elif len(group_attrs) == 1:
        if len(group_by) == 0:
            _cols = _cols + ["rank"]
            return {_: _df.xs(_)[_cols] for _ in _levels}
        else:
            return {
                _: (_df.xs(_)[_cols].groupby(group_by).mean()).reset_index()
                for _ in _levels
            }

    elif len(group_attrs) == 2:
        if len(group_by) == 0:
            _cols = _cols + ["rank"]
            return {_: _df.xs(_)[_cols] for (_, __) in _levels}
        else:
            return {
                _: (_df.xs(_)[_cols].groupby(group_by).mean()).reset_index()
                for (_, __) in _levels
            }

    assert False, "Invalid scenario"

def df_column_mean(df, column, proxy={}):
    """
    Apply a function to the df.column

    :param column: column to apply on.
    :param proxy:
    :return:
    """
    assert isinstance(column, str)
    column = proxy.get(column, column)
    return df[column].mean()

def callsites_column_mean(df, column, proxy={}):
    """
    Apply a function to the df.column

    :param column: column to apply on.
    :param proxy:
    :return:
    """
    assert isinstance(column, str)
    column = proxy.get(column, column)
    if column == "time (inc)":
        return df.groupby("name").mean()[column].max()
    elif column == "time":
        return df.groupby("name").mean()[column].sum()

def df_add_column(
        df,
        column_name,
        apply_value=None,
        apply_func=None,
        apply_dict=None,
        dict_default=None,
        apply_on="name",
        update=False,
    ):
    """
    Wrapper to add a column to a dataframe in place.
    :param column_name: (str) Name of the column to add in the dataframe
    :param apply_value: (*) Value to apply on the column
    :param apply_func: (func) Function to apply on the column
    :param apply_dict: (dict) Dict to apply on the column
    :param apply_on: (str) Column to apply the func, value or dict on
    :param dict_default: (dict) default dictionary to apply on
    :param update: (bool) in place update or not
    """
    has_value = apply_value is not None
    has_func = apply_func is not None
    has_dict = apply_dict is not None
    assert 1 == int(has_value) + int(has_func) + int(has_dict)

    already_has_column = column_name in df.columns
    if already_has_column and not update:
        return

    action = "updating" if already_has_column and update else "appending"

    if has_value:
        assert isinstance(apply_value, (int, float, str))
        LOGGER.debug(
            f'{action} column "{column_name}" = "{apply_value}"'
        )
        df[column_name] = apply_value

    if has_func:
        assert callable(apply_func) and isinstance(apply_on, str)
        LOGGER.debug(
            f'{action} column "{column_name}" = {apply_func}'
        )
        df[column_name] = df[apply_on].apply(apply_func)

    if has_dict:
        assert isinstance(apply_dict, dict) and isinstance(apply_on, str)
        LOGGER.debug(
            f'{action} column "{column_name}" = (dict); default=({dict_default})'
        )
        df[column_name] = df[apply_on].apply(
            lambda _: apply_dict.get(_, dict_default)
        )
    
    return df

def df_factorize_column(df, column, update_df=False, proxy={}):
        """
        Wrapper to factorize a column.
        :param column: (name) Column to discretize into categorical indexes.
        :param update_df: (bool) True will update the dataframe with the discretized values.
        :return: c2v, v2c : (dict, dict) Dictionaries mapping the values to
        indexes.
        """
        assert isinstance(df, pd.DataFrame)
        assert isinstance(column, str)

        column = proxy.get(column, column)
        codes, vals = df[column].factorize(sort=True)

        if update_df:
            df[column] = codes

        c2v, v2c = {}, {}
        for i, v in enumerate(vals):
            c2v[i] = v
            v2c[v] = i

        # if there were any invalid values, insert a value for "empty" string
        if -1 in codes:
            c2v[-1] = None
            v2c[None] = -1

        return c2v, v2c