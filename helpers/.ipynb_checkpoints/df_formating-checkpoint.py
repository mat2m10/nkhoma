import os
import pandas as pd
import numpy as np
import re
def convert_to_integer(df, cols):
    for col in cols:
        df[col] = df[col].round().astype("Int64")
    return df

def to_snake_case(col):
    
    col = col.strip().lower()
    col = col.replace("/", "_")
    col = re.sub(r"[^\w\s]", "", col)      # remove punctuation (#, /, -, ())
    col = re.sub(r"\s+", "_", col)         # spaces → underscores
    col = re.sub(r"_+", "_", col)          # collapse multiple underscores
    # fix leading digits (e.g. 1st_assistent → first_assistent)
    col = re.sub(r"^1st_", "first_", col)
    col = re.sub(r"^2nd_", "second_", col)
    col = re.sub(r"^3nd_", "third", col)
    
    return col.strip("_")

def normalize(text):
    if pd.isna(text):
        return pd.NA
    return (
        str(text)
        .lower()
        .strip()
        .replace(",", "")
    )
    
def convert_cols_to_snake_case(df):
    df.columns = [to_snake_case(c) for c in df.columns]
    return df

def drop_row_if_not_complete(df, cols):
    df = df.dropna(subset=cols).copy()
    return df

def drop_if_unnamed(df):
    df = df.loc[:, ~df.columns.str.match(r"^unnamed")]
    return df
    
def excel_time_to_minutes(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, dt.time):
        return x.hour * 60 + x.minute + x.second / 60
    try:
        return pd.to_timedelta(x).total_seconds() / 60
    except Exception:
        return np.nan