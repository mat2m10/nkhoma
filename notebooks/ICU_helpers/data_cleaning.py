import re
import pandas as pd
import numpy as np


# ── 0. Column names ──────────────────────────────────────────────────────────

def to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r'[\s\(\)]+', '_', regex=True)
        .str.replace(r'[^\w]', '', regex=True)
        .str.strip('_')
    )
    return df


# ── 1. Drop empty rows ───────────────────────────────────────────────────────

def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where every clinical column is NaN (the trailing blank rows)."""
    key_cols = ['year_of_birth', 'gender', 'hospital_admission_date']
    existing = [c for c in key_cols if c in df.columns]
    df = df.dropna(subset=existing, how='all').reset_index(drop=True)
    return df


# ── 2. Index / unnamed column ────────────────────────────────────────────────

def drop_unnamed_index(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [c for c in df.columns if c.startswith('unnamed')]
    df = df.drop(columns=unnamed)
    return df


# ── 3. Gender ────────────────────────────────────────────────────────────────

def format_gender(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        'm': 'Male', 'male': 'Male', '1': 'Male',
        'f': 'Female', 'female': 'Female', '2': 'Female',
    }
    col = 'gender'
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
        )
    return df


# ── 4. Age / year of birth ───────────────────────────────────────────────────

def format_age(df: pd.DataFrame) -> pd.DataFrame:
    col = 'age_years'
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].round().astype('Int64')
    elif 'year_of_birth' in df.columns:
        ref_year = pd.Timestamp.now().year
        df['age_years'] = (ref_year - pd.to_numeric(df['year_of_birth'], errors='coerce')).round().astype('Int64')
    return df


# ── 5. Date columns ──────────────────────────────────────────────────────────

DATE_COLS = [
    'hospital_admission_date',
    'hospital_discharge_date',
    'date_of_surgery',
    'icu_admission_date',
    'icu_discharge_date',
]

def format_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='mixed', dayfirst=True, errors='coerce')
    return df


# ── 6. Duration / numeric stay columns ──────────────────────────────────────

NUMERIC_STAY_COLS = [
    'length_of_icu_stay_nights',
    'intubation_nights',
    'hospital_stay_nights',
    'discharge_after_icu_nights',
]

def format_stay_durations(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_STAY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Negative durations are data errors — set to NaN
            df.loc[df[col] < 0, col] = np.nan
            df[col] = df[col].astype('Float64')
    return df


# ── 7. Intubation (Yes/No flag) ──────────────────────────────────────────────

def format_intubation(df: pd.DataFrame) -> pd.DataFrame:
    col = 'intubation'
    if col in df.columns:
        yes_patterns = r'^(yes|y|1|true)'
        no_patterns  = r'^(no|n|0|false)'
        s = df[col].astype(str).str.strip().str.lower()
        df[col] = np.where(
            s.str.match(yes_patterns), 'Yes',
            np.where(s.str.match(no_patterns), 'No', None)
        )
    return df


# ── 8. qSOFA score ───────────────────────────────────────────────────────────

def format_qsofa(df):
    col = 'qsofa'
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').clip(0, 3).astype('Int64')
    return df
    
# ── 9. Categorical text columns ─────────────────────────────────────────────

def format_categoricals(df):
    text_cols = ['type_of_surgery', 'discharge_to', 'disability_at_discharge', 'cause_of_death']
    
    # Acronyms to restore after title-casing
    acronyms = ['Egd', 'Ercp', 'Icu', 'Cpr', 'Hiv', 'Ct', 'Mri', 'Ecg']
    acronym_map = {a: a.upper() for a in acronyms}
    
    for col in text_cols:
        if col in df.columns:
            s = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r'\s+', ' ', regex=True)
                .str.title()
                .replace({'Nan': np.nan, 'None': np.nan, '': np.nan})
            )
            # Fix acronyms: whole-word replace only
            for wrong, right in acronym_map.items():
                s = s.str.replace(rf'\b{wrong}\b', right, regex=True)
            df[col] = s
    return df


# ── 10. Derived / validation columns ────────────────────────────────────────

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-check computed stay lengths against raw date columns.
    Adds boolean flags for rows where values look inconsistent.
    """
    if {'icu_admission_date', 'icu_discharge_date', 'length_of_icu_stay_nights'}.issubset(df.columns):
        computed = (df['icu_discharge_date'] - df['icu_admission_date']).dt.days
        df['icu_stay_check'] = (computed - df['length_of_icu_stay_nights']).abs() > 1

    if {'hospital_admission_date', 'hospital_discharge_date', 'hospital_stay_nights'}.issubset(df.columns):
        computed = (df['hospital_discharge_date'] - df['hospital_admission_date']).dt.days
        df['hosp_stay_check'] = (computed - df['hospital_stay_nights']).abs() > 1

    return df


# ── Master pipeline ──────────────────────────────────────────────────────────

def clean_icu_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = to_snake_case(df)
    df = drop_empty_rows(df)
    df = drop_unnamed_index(df)
    df = format_dates(df)
    df = format_gender(df)
    df = format_age(df)
    df = format_stay_durations(df)
    df = format_intubation(df)
    df = format_qsofa(df)
    df = format_categoricals(df)
    df = add_derived_columns(df)
    return df


if __name__ == '__main__':
    df_raw = pd.read_excel('icu_data.xlsx')   # adjust path / read_csv as needed
    df_clean = clean_icu_df(df_raw)
    print(df_clean.dtypes)
    print(df_clean.head())