import re
import pandas as pd
from difflib import SequenceMatcher


# ═══════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def to_snake_case(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name.lower()


def normalize_name(s) -> str:
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s).strip().upper())


def fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# ═══════════════════════════════════════════════════════════════════════
# ENDOSCOPY LOADER
# ═══════════════════════════════════════════════════════════════════════

_ENDO_SHEETS = {
    "Endoscopy Ledger_ 2022-2023.xlsx": [
        "22 May", "22 June", "22 Aug", "22 Sept", "22 Oct", "22 Nov", "22 Dec",
        "23 JAN", "23 FEB", "23 MAR", "23 APR", "23 MAY",
        "23 SEPT", "23 OCT", "23 NOV", "23 DEC",
    ],
    "Endoscopy Ledger_Consolidated 2024.xlsx": ["2024 FULL LIST"],
    "Endoscopy Ledger_2025 02.04.2026.xlsx": [
        "January", "February ", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ],
}

_ENDO_FINAL_COLS = [
    "id", "patient_number", "patient_name", "full_name", "name",
    "gender", "diagnosis", "surgery",
    "admission_date", "discharge_date", "total_bill_mk", "copay_mk",
    "endoscopy_ledger_balance_mk", "invoice_number", "source_file", "source_sheet",
]

def _load_endo_sheet(filepath: str, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(filepath, sheet_name=sheet_name,
                        header=None, engine="openpyxl")
    header_idx = None
    for i, row in raw.iterrows():
        vals = row.dropna().tolist()
        if vals and str(vals[0]).strip() in ("#", "NAME"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No header row found in sheet '{sheet_name}'")
    df = raw.iloc[header_idx + 1:].copy()
    df.columns = ["id" if str(c).strip() == "#" else to_snake_case(c)
                  for c in raw.iloc[header_idx].tolist()]
    id_col = "id" if "id" in df.columns else "name"
    df = df[df[id_col].notna()]
    df = df[df[id_col].astype(str).str.strip() != ""]
    return df.dropna(how="all").reset_index(drop=True)


def _normalize_endo_df(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if c == "nan" or "moz" in c]
    df = df.drop(columns=drop_cols, errors="ignore")
    balance_aliases = {"balance_mk", "endoscopy_ledger",
                       "bill_to_patientmk", "bill_to_patient"}
    df = df.rename(columns={c: "endoscopy_ledger_balance_mk"
                             for c in df.columns if c in balance_aliases})
    return df.loc[:, ~df.columns.duplicated()]


def load_all_endo(endo_path: str, verbose: bool = True) -> pd.DataFrame:
    """Loads all endoscopy sheets into one clean DataFrame."""
    frames = []
    for filename, sheets in _ENDO_SHEETS.items():
        for sheet in sheets:
            try:
                df = _load_endo_sheet(f"{endo_path}/{filename}", sheet)
                df["source_file"]  = filename
                df["source_sheet"] = sheet
                frames.append(_normalize_endo_df(df))
                if verbose:
                    print(f"✓ endo  {sheet:20s} → {len(df)} rows")
            except Exception as e:
                print(f"✗ endo  {sheet:20s} → {e}")

    combined = pd.concat(frames, ignore_index=True)

    # Unify date columns (2022-2023 used different names)
    combined["admission_date"] = combined.get("admission_date", pd.NaT).fillna(
        combined.get("date_of_treatment", pd.NaT))
    combined["discharge_date"] = combined.get("discharge_date", pd.NaT).fillna(
        combined.get("date_of_discharge", pd.NaT))
    combined = combined.drop(
        columns=["date_of_treatment", "date_of_discharge"], errors="ignore")

    combined["admission_date"] = pd.to_datetime(
        combined["admission_date"], errors="coerce")
    combined["discharge_date"] = pd.to_datetime(
        combined["discharge_date"], errors="coerce")

    combined = combined[combined["name"].notna()]
    combined = combined[combined["admission_date"].notna()]
    combined["patient_name"] = combined["name"]
    combined["full_name"]    = combined["name"].apply(normalize_name)
    combined["patient_number"] = combined.get("patient_number", pd.NA)
    cols = [c for c in _ENDO_FINAL_COLS if c in combined.columns]
    return combined[cols].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
# MERCY FUND LOADER
# ═══════════════════════════════════════════════════════════════════════

_MERCY_SHEETS = {
    "SURGICAL MERCY FUND ALL until Dez 2023.xlsx": ["All"],
    "Mercy Fund Report- Jan-Dec 2024.xlsx":         ["CONSOLIDATED"],
    # 2025 CONSOLIDATED is a copy of 2024 — load monthly sheets instead
    "Mercy Fund Report- Jan-Dec 2025 02.04.2026.xlsx": [
        "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
        "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
    ],
}

_MERCY_FINAL_COLS = [
    "patient_number", "patient_name", "full_name", "gender", "diagnosis", "surgery",
    "admission_date", "discharge_date", "total_bill_mk", "copay_mk",
    "mercy_fund_mk", "discount_mk", "invoice_number",
    "source_file", "source_sheet",
    "days_since_last_mercy", "possible_duplicate_mercy",
]


def _load_mercy_sheet(filepath: str, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(filepath, sheet_name=sheet_name,
                        header=None, engine="openpyxl")
    header_idx = None
    for i, row in raw.iterrows():
        vals = [str(v).strip() for v in row.tolist() if pd.notna(v)]
        if any(v in ("File Number", "Patient Number") for v in vals):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No header row found in sheet '{sheet_name}'")
    df = raw.iloc[header_idx + 1:].copy()
    df.columns = [to_snake_case(c) for c in raw.iloc[header_idx].tolist()]
    df = df[df.iloc[:, 0].notna()]
    df = df[df.iloc[:, 0].astype(str).str.strip() != ""]
    return df.dropna(how="all").reset_index(drop=True)


def _normalize_mercy_df(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns
                 if c == "nan"
                 or "if_patient_did_not" in c
                 or "difference" in c]
    df = df.drop(columns=drop_cols, errors="ignore")
    rename_map = {
        "file_number":        "patient_number",
        "name":               "patient_name",
        "sex":                "gender",
        "surgical_procedure": "surgery",
        "admision_date":      "admission_date",
        "mercy_fund":         "mercy_fund_mk",
        "surgery_done":       "surgery",
        "admission":          "admission_date",
        "discharge":          "discharge_date",
        "patient_copay":      "copay_mk",
        "total_bill":         "total_bill_mk",
        "balance":            "mercy_fund_mk",
        "mercyfund":          "mercy_fund_mk",
        "25_discount":        "discount_mk",
        "25_discount_1":      "discount_mk",
        "invoice_number":     "invoice_number",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.loc[:, ~df.columns.duplicated()]
    return df.replace("-", pd.NA)


def _flag_mercy_duplicates(df: pd.DataFrame,
                            window_days: int = 14) -> pd.DataFrame:
    """
    Flags mercy rows where the same patient_number appears twice within
    window_days — likely double-billing or close re-admission.
    """
    df = df.sort_values(["patient_number", "admission_date"])
    df["_prev_date"] = df.groupby("patient_number")["admission_date"].shift(1)
    df["days_since_last_mercy"] = (df["admission_date"] - df["_prev_date"]).dt.days
    df["possible_duplicate_mercy"] = (
        df["patient_number"].notna() &
        df["days_since_last_mercy"].notna() &
        (df["days_since_last_mercy"] <= window_days)
    )
    return df.drop(columns=["_prev_date"])


def load_all_mercy(mercy_path: str, verbose: bool = True) -> pd.DataFrame:
    """Loads all mercy fund sheets into one clean DataFrame."""
    frames = []
    for filename, sheets in _MERCY_SHEETS.items():
        for sheet in sheets:
            try:
                df = _load_mercy_sheet(f"{mercy_path}/{filename}", sheet)
                df["source_file"]  = filename
                df["source_sheet"] = sheet
                frames.append(_normalize_mercy_df(df))
                if verbose:
                    print(f"✓ mercy {sheet:20s} ({filename[:30]}) → {len(df)} rows")
            except Exception as e:
                print(f"✗ mercy {sheet:20s} ({filename[:30]}) → {e}")

    combined = pd.concat(frames, ignore_index=True)

    combined["admission_date"] = pd.to_datetime(
        combined["admission_date"], errors="coerce")
    combined["discharge_date"] = pd.to_datetime(
        combined["discharge_date"], errors="coerce")

    combined = combined[combined["patient_name"].notna()]
    combined = combined[combined["admission_date"].notna()]

    for col in ["total_bill_mk", "copay_mk", "mercy_fund_mk", "discount_mk"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(
                combined[col], errors="coerce").astype("Float64")

    # Deduplicate — 2024 CONSOLIDATED takes priority over 2025 monthly
    source_priority = {
        "SURGICAL MERCY FUND ALL until Dez 2023.xlsx":     0,
        "Mercy Fund Report- Jan-Dec 2024.xlsx":             1,
        "Mercy Fund Report- Jan-Dec 2025 02.04.2026.xlsx": 2,
    }
    combined["_pri"] = combined["source_file"].map(source_priority)
    combined = (combined
                .sort_values("_pri")
                .drop_duplicates(
                    subset=["patient_number", "admission_date"], keep="first")
                .drop(columns="_pri")
                .reset_index(drop=True))

    # Add matching key
    combined["full_name"] = combined["patient_name"].apply(normalize_name)

    # Flag possible within-fund duplicates
    combined = _flag_mercy_duplicates(combined)
    if verbose:
        print(f"  → possible mercy duplicates flagged: "
              f"{combined['possible_duplicate_mercy'].sum()}")

    cols = [c for c in _MERCY_FINAL_COLS if c in combined.columns]
    return combined[cols].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
# SAFE LOADER
# ═══════════════════════════════════════════════════════════════════════

_SAFE_2024_COLS = {
    1:  "patient_name",
    2:  "patient_number",
    3:  "fund_source",
    4:  "watsi_id",
    5:  "safe_case_number",
    6:  "admission_date",
    7:  "discharge_date",
    8:  "status",
    9:  "age",
    10: "gender",
    11: "diagnosis",
    12: "surgery",
    13: "surgery_type",
    15: "admission_type",
    16: "patient_phone",
    17: "next_of_kin",
    20: "surgery_date",
    22: "payment_status",
    23: "patient_invoice_number",
    24: "invoice_number",
    25: "safe_ask_usd",
    26: "total_bill_mk",
    27: "hospital_price_mk",
    28: "nhif_insurance_mk",
    29: "copay_mk",
    30: "safe_fund_mk",
}

_SAFE_FINAL_COLS = [
    "patient_name", "full_name", "patient_number", "age", "gender",
    "diagnosis", "surgery", "surgery_type", "fund_source",
    "admission_date", "discharge_date", "surgery_date",
    "total_bill_mk", "copay_mk", "safe_ask_usd", "safe_fund_mk",
    "invoice_number", "patient_invoice_number",
    "status", "payment_status", "admission_type",
    "source_file", "source_sheet",
    "days_since_last_safe", "possible_duplicate_safe",
]


def _load_safe_2022(filepath: str) -> pd.DataFrame:
    raw = pd.read_excel(filepath, sheet_name="ALL",
                        header=None, engine="openpyxl")
    df = raw.iloc[1:].copy()
    df.columns = ["patient_name", "age", "gender",
                  "diagnosis", "surgery", "safe_ask_usd"]
    noise = {"Nkhoma 2022", "Budget", "Original Budget",
             "Additional Funding", "Name"}
    df = df[~df["patient_name"].isin(noise)]
    df = df[df["patient_name"].notna()].reset_index(drop=True)
    df["admission_date"] = pd.NaT
    df["discharge_date"] = pd.NaT
    df["surgery_date"]   = pd.NaT
    df["patient_number"] = pd.NA
    df["source_file"]    = "SAFE Program 2022.xlsx"
    df["source_sheet"]   = "ALL"
    return df


def _load_safe_2023(filepath: str) -> pd.DataFrame:
    raw = pd.read_excel(filepath, sheet_name="SAFE Patients 2023",
                        header=None, engine="openpyxl")
    df = raw.iloc[5:].copy()
    df.columns = ["id", "patient_name", "age", "gender",
                  "surgery", "surgery_date", "safe_ask_usd"]
    df = df[df["patient_name"].notna()]
    df = df[df["patient_name"].astype(str).str.strip() != ""]
    df = df.drop(columns=["id"]).reset_index(drop=True)
    df["admission_date"] = pd.NaT
    df["discharge_date"] = pd.NaT
    df["patient_number"] = pd.NA
    df["source_file"]    = "SAFE 2023 REPORT.xlsx"
    df["source_sheet"]   = "SAFE Patients 2023"
    return df


def _load_safe_master(filepath: str, filename: str) -> pd.DataFrame:
    raw = pd.read_excel(filepath, sheet_name="Sheet1",
                        header=None, engine="openpyxl")
    # 2025 file has a header row — skip it
    if str(raw.iloc[0, 1]).strip() == "PATIENT NAME":
        raw = raw.iloc[1:].reset_index(drop=True)
    df = raw.iloc[:, list(_SAFE_2024_COLS.keys())].copy()
    df.columns = list(_SAFE_2024_COLS.values())
    df = df[df["patient_name"].notna()]
    df = df[df["patient_name"].astype(str).str.strip() != ""]
    df["nhif_insurance_mk"] = pd.to_numeric(
        df["nhif_insurance_mk"].replace("No", 0), errors="coerce")
    df["source_file"]  = filename
    df["source_sheet"] = "Sheet1"
    return df.reset_index(drop=True)


def _normalize_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["admission_date", "discharge_date", "surgery_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["age", "safe_ask_usd", "total_bill_mk", "copay_mk",
                "safe_fund_mk", "hospital_price_mk", "nhif_insurance_mk"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "gender" in df.columns:
        df["gender"] = df["gender"].str.strip().str.capitalize()
    return df


def _flag_safe_duplicates(df: pd.DataFrame,
                           window_days: int = 14) -> pd.DataFrame:
    """
    Flags safe rows where the same patient_number appears twice within
    window_days — likely double-billing or close re-admission.
    """
    df = df.sort_values(["patient_number", "surgery_date"])
    df["_prev_date"] = df.groupby("patient_number")["surgery_date"].shift(1)
    df["days_since_last_safe"] = (df["surgery_date"] - df["_prev_date"]).dt.days
    df["possible_duplicate_safe"] = (
        df["patient_number"].notna() &
        df["days_since_last_safe"].notna() &
        (df["days_since_last_safe"] <= window_days)
    )
    return df.drop(columns=["_prev_date"])


def load_all_safe(safe_path: str, verbose: bool = True) -> pd.DataFrame:
    """Loads all SAFE fund sheets into one clean DataFrame."""
    loaders = [
        ("SAFE Program 2022.xlsx",                                        _load_safe_2022),
        ("SAFE 2023 REPORT.xlsx",                                         _load_safe_2023),
        ("Master SAFE Patient Spreadsheet 2024_Nkhoma Hospital.xlsx",     None),
        ("Master SAFE Patient Spreadsheet 2025_Nkhoma Hospital.xlsx", None),
    ]
    frames = []
    for filename, loader_fn in loaders:
        try:
            df = (loader_fn(f"{safe_path}/{filename}") if loader_fn
                  else _load_safe_master(f"{safe_path}/{filename}", filename))
            df = _normalize_safe_df(df)
            frames.append(df)
            if verbose:
                print(f"✓ safe  {filename[:55]:55s} → {len(df)} rows")
        except Exception as e:
            print(f"✗ safe  {filename[:55]:55s} → {e}")

    combined = pd.concat(frames, ignore_index=True)

    # Drop metadata rows
    noise = {"KEY: to fill row", "$1 = K1,751"}
    combined = combined[~combined["patient_name"].isin(noise)].reset_index(drop=True)

    # Add matching key
    combined["full_name"] = combined["patient_name"].apply(normalize_name)

    # Flag possible within-fund duplicates
    combined = _flag_safe_duplicates(combined)
    if verbose:
        print(f"  → possible safe duplicates flagged: "
              f"{combined['possible_duplicate_safe'].sum()}")

    cols = [c for c in _SAFE_FINAL_COLS if c in combined.columns]
    return combined[cols].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
# THEATRE BOOK PREP
# ═══════════════════════════════════════════════════════════════════════

def prepare_tb(tb: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Cleans TB dates, adds full_name, builds match index."""
    tb = tb.copy()
    tb["date_of_surgery"] = pd.to_datetime(tb["date_of_surgery"], errors="coerce")
    tb = tb[tb["date_of_surgery"].dt.year > 2000].reset_index(drop=True)
    tb["full_name"] = (
        tb["first_name"].apply(normalize_name) + " " +
        tb["last_name"].apply(normalize_name)
    ).str.strip()

    tb_index = [
        {
            "theatre_index":    row["theatre_book_index"],
            "full_name":        row["full_name"],
            "tokens":           set(row["full_name"].split()),
            "date":             row["date_of_surgery"],
            "department":       row["department"],
            "procedure":        row["main_procedure_category"],
            "diagnosis":        row["final_diagnosis_category"],
            "urgency":          row["urgency"],
            "surgery_severity": row["surgery_severity"],
        }
        for _, row in tb.iterrows()
    ]
    return tb, tb_index


# ═══════════════════════════════════════════════════════════════════════
# NAME + DATE MATCHING
# ═══════════════════════════════════════════════════════════════════════

# Manually verified false positives — pairs that look similar but are
# different people
_FALSE_POSITIVES = {
    ("IDA WHITE",        "SAID WHITE"),
    ("JAZIEL DANIEL",    "JAZIEL VANILI"),
    ("YOHANE FANUEL",    "YOHANE SAMUEL"),
    ("LACKSON GEORGE",   "LACKSON GWENGWE"),
    ("AISA CHIKUMBUTSO", "DALITSO CHKUMBUTSO"),
    ("JEMUSI JEMITALA",  "JAMES JEMITALA"),
    ("CHIMWEMWE LYNOS",  "CHIMWEMWE IGNOSI"),
    ("DANIEL JASON",     "DANIEL FASHION"),
    ("DANIEL FASHONI",   "DANIEL JASON"),
    ("RYAN CHALAMANDA",  "LAYANI CHALAMANDA"),
    ("SHAMIMU LUCAS",    "SHAMIM LUFASI"),
}


def _is_confident_fuzzy(fund_name: str, theatre_name: str,
                         score: float, date_diff) -> bool:
    """Returns True if a fuzzy name match is reliable enough to accept."""
    if (fund_name, theatre_name) in _FALSE_POSITIVES:
        return False
    parts_f = fund_name.split()
    parts_t = theatre_name.split()
    if len(parts_f) >= 2 and len(parts_t) >= 2:
        if fuzzy_score(parts_f[-1], parts_t[-1]) < 0.65:
            return False
        if fuzzy_score(parts_f[0],  parts_t[0])  < 0.60:
            return False
    if score < 0.82:
        return False
    if date_diff is not None and date_diff > 30:
        return False
    return True


def _find_matches(fund_name: str, fund_date,
                  tb_index: list, date_window: int = 30) -> list:
    """
    Finds theatre book rows that match a fund entry by name + date.
    Strategy:
      1. Token match (≥2 shared tokens) — exact or token type
      2. Fuzzy full-name match (≥0.82) with per-token validation
    """
    fund_tokens = set(normalize_name(fund_name).split())
    fund_dt     = pd.to_datetime(fund_date) if pd.notna(fund_date) else None
    results     = []

    for t in tb_index:
        # Date filter first (cheapest check)
        if fund_dt is not None:
            diff = abs((t["date"] - fund_dt).days)
            if diff > date_window:
                continue
        else:
            diff = None

        shared = fund_tokens & t["tokens"]
        if len(shared) >= 2:
            score      = 1.0
            match_type = "exact" if fund_name == t["full_name"] else "token"
        else:
            score = fuzzy_score(fund_name, t["full_name"])
            if not _is_confident_fuzzy(fund_name, t["full_name"], score, diff):
                continue
            match_type = "fuzzy"

        results.append({**t,
                        "date_diff_days": diff,
                        "match_type":     match_type,
                        "fuzzy_score":    round(score, 3)})
    return results


def _build_links(fund_df: pd.DataFrame, name_col: str, date_col: str,
                 fund_label: str, tb_index: list) -> pd.DataFrame:
    """
    Builds the raw link table: one row per fund-entry × TB-procedure match.
    Unmatched fund entries are kept as single rows with match_status='unmatched'.
    """
    rows = []
    for _, r in fund_df.iterrows():
        hits = _find_matches(r[name_col], r[date_col], tb_index)
        base = {
            "fund":               fund_label,
            "fund_name":          r[name_col],
            "fund_patient_number":r.get("patient_number"),
            "fund_date":          r[date_col],
            "fund_covered_mk":    r.get("mercy_fund_mk") or r.get("safe_fund_mk"),
        }
        if hits:
            for h in hits:
                rows.append({**base,
                             "theatre_index":   h["theatre_index"],
                             "theatre_name":    h["full_name"],
                             "theatre_date":    h["date"],
                             "date_diff_days":  h["date_diff_days"],
                             "department":      h["department"],
                             "tb_procedure":    h["procedure"],
                             "tb_diagnosis":    h["diagnosis"],
                             "urgency":         h["urgency"],
                             "surgery_severity":h["surgery_severity"],
                             "match_type":      h["match_type"],
                             "fuzzy_score":     h["fuzzy_score"],
                             "match_status":    "matched"})
        else:
            rows.append({**base,
                         "theatre_index":   None,
                         "theatre_name":    None,
                         "theatre_date":    None,
                         "date_diff_days":  None,
                         "department":      None,
                         "tb_procedure":    None,
                         "tb_diagnosis":    None,
                         "urgency":         None,
                         "surgery_severity":None,
                         "match_type":      "unmatched",
                         "fuzzy_score":     None,
                         "match_status":    "unmatched"})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# CROSS-FUND DOUBLE-BILLING FLAG
# ═══════════════════════════════════════════════════════════════════════

def _check_cross_fund_overlap(final: pd.DataFrame,
                               window_days: int = 14) -> pd.DataFrame:
    """
    Flags rows where the same patient_number appears in both mercy and safe
    with admissions within window_days — potential double-billing.
    Adds column: cross_fund_flag (bool)
    """
    mercy_dates = (
        final[final["fund"] == "mercy"]
        .drop_duplicates(subset=["patient_number", "fund_admission_date"])
        [["patient_number", "fund_admission_date"]]
        .rename(columns={"fund_admission_date": "mercy_date"})
    )
    safe_dates = (
        final[final["fund"] == "safe"]
        .drop_duplicates(subset=["patient_number", "fund_admission_date"])
        [["patient_number", "fund_admission_date"]]
        .rename(columns={"fund_admission_date": "safe_date"})
    )
    cross = mercy_dates.merge(safe_dates, on="patient_number", how="inner")
    cross["date_gap"] = abs((cross["mercy_date"] - cross["safe_date"]).dt.days)

    double_bill = (cross[cross["date_gap"] <= window_days]
                   [["patient_number"]]
                   .drop_duplicates()
                   .assign(cross_fund_flag=True))

    final = final.merge(double_bill, on="patient_number", how="left")
    final["cross_fund_flag"] = final["cross_fund_flag"].fillna(False)
    return final


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

_FINAL_COLS = [
    # Matching metadata
    "fund", "match_status", "match_type", "fuzzy_score", "unmatched_reason",
    "cross_fund_flag",
    # Patient identity
    "fund_name", "patient_number", "age", "gender",
    # Dates
    "fund_admission_date", "fund_date", "theatre_date", "date_diff_days",
    # Financials
    "total_bill_mk", "copay_mk", "fund_covered_mk",
    "mercy_fund_mk", "discount_mk",
    "safe_ask_usd", "safe_fund_mk",
    "endoscopy_ledger_balance_mk",
    "invoice_number",
    # Fund clinical info
    "fund_diagnosis", "fund_surgery", "fund_source", "admission_type",
    # Theatre book info
    "theatre_index", "theatre_name",
    "department", "tb_diagnosis", "tb_procedure",
    "procedure_free_text", "tb_surgery_type", "indication_for_surgery",
    "urgency", "surgery_severity", "asascore",
    # Duplicate flags
    "days_since_last_mercy", "possible_duplicate_mercy",
    "days_since_last_safe",  "possible_duplicate_safe",
]


def build_final(tb:      pd.DataFrame,
                mercy:   pd.DataFrame,
                safe:    pd.DataFrame,
                endo:    pd.DataFrame,
                verbose: bool = True) -> pd.DataFrame:
    """
    Links mercy and safe fund data to the theatre book via name+date matching.

    Returns one row per fund-entry × TB-procedure match, plus unmatched
    fund rows retained for coverage analysis.

    Parameters
    ----------
    tb      : cleaned theatre book DataFrame (from your TB variable)
    mercy   : output of load_all_mercy()
    safe    : output of load_all_safe()
    verbose : print summary statistics

    Returns
    -------
    final   : linked DataFrame with _FINAL_COLS columns
    """

    # ── 1. Prepare theatre book ───────────────────────────────────────
    tb_clean, tb_index = prepare_tb(tb)

# ── 2. Build raw link tables ──────────────────────────────────────
    mercy_links = _build_links(mercy, "full_name", "admission_date",
                               "mercy", tb_index)
    safe_links  = _build_links(safe,  "full_name", "surgery_date",
                               "safe",  tb_index)
    endo_links  = _build_links(endo,  "full_name", "admission_date",
                               "endo",  tb_index)
    links = pd.concat([mercy_links, safe_links, endo_links], ignore_index=True)
    # ── 3. Drop ObsGyn matches (not covered by these funds) ───────────
    links = links[~(
        (links["match_status"] == "matched") &
        (links["department"] == "ObsGyn")
    )].reset_index(drop=True)

    # ── 4. Tag why rows are unmatched ─────────────────────────────────
    tb_max = tb_clean["date_of_surgery"].max()
    links["unmatched_reason"] = None
    links.loc[
        (links["match_status"] == "unmatched") &
        (pd.to_datetime(links["fund_date"], errors="coerce") > tb_max),
        "unmatched_reason"
    ] = "outside_TB_range"
    links.loc[
        (links["match_status"] == "unmatched") &
        (pd.to_datetime(links["fund_date"], errors="coerce") <= tb_max),
        "unmatched_reason"
    ] = "not_in_TB"

    # ── 5. Join full mercy details back ───────────────────────────────
    mercy_detail = (
        mercy[["full_name", "admission_date", "patient_number", "gender",
               "diagnosis", "surgery", "total_bill_mk", "copay_mk",
               "mercy_fund_mk", "discount_mk", "invoice_number",
               "days_since_last_mercy", "possible_duplicate_mercy"]]
        .drop_duplicates(subset=["full_name", "admission_date"])
    )
    mercy_part = (
        links[links["fund"] == "mercy"]
        .merge(mercy_detail,
               left_on=["fund_name", "fund_date"],
               right_on=["full_name", "admission_date"],
               how="left")
        .drop(columns=["full_name"], errors="ignore")
    )

    # ── 6. Join full safe details back ────────────────────────────────
    safe_detail = (
        safe[["full_name", "surgery_date", "patient_number", "age", "gender",
              "diagnosis", "surgery", "surgery_type", "fund_source",
              "total_bill_mk", "copay_mk", "safe_ask_usd", "safe_fund_mk",
              "invoice_number", "admission_type",
              "days_since_last_safe", "possible_duplicate_safe"]]
        .drop_duplicates(subset=["full_name", "surgery_date"])
    )
    safe_part = (
        links[links["fund"] == "safe"]
        .merge(safe_detail,
               left_on=["fund_name", "fund_date"],
               right_on=["full_name", "surgery_date"],
               how="left")
        .drop(columns=["full_name"], errors="ignore")
    )
    # ── 6b. Join full endo details back ──────────────────────────────────
    endo_detail = (
        endo[["full_name", "admission_date", "patient_number", "gender",
              "diagnosis", "surgery", "total_bill_mk", "copay_mk",
              "endoscopy_ledger_balance_mk", "invoice_number"]]
        .drop_duplicates(subset=["full_name", "admission_date"])
    )
    endo_part = (
        links[links["fund"] == "endo"]
        .merge(endo_detail,
               left_on=["fund_name", "fund_date"],
               right_on=["full_name", "admission_date"],
               how="left")
        .drop(columns=["full_name"], errors="ignore")
    )

    final = pd.concat([mercy_part, safe_part, endo_part], ignore_index=True)

    # ── 7. Join TB procedure details ──────────────────────────────────
    tb_detail = (
        tb_clean[[
            "theatre_book_index", "main_procedure_category",
            "final_diagnosis_category", "procedure_free_text",
            "surgery_type", "indication_for_surgery",
            "urgency", "surgery_severity", "asascore"
        ]]
        .rename(columns={
            "main_procedure_category":  "tb_procedure_detail",
            "final_diagnosis_category": "tb_diagnosis_detail",
            "surgery_type":             "tb_surgery_type",
        })
    )
    final = (
        final
        .merge(tb_detail,
               left_on="theatre_index", right_on="theatre_book_index",
               how="left")
        .drop(columns=["theatre_book_index"], errors="ignore")
    )

    # ── 8. Resolve columns duplicated by merges ───────────────────────
    # tb_procedure / tb_diagnosis: use the freshly joined detail columns
    final["tb_procedure"] = final.pop("tb_procedure_detail").fillna(
        final.get("tb_procedure", pd.NA))
    final["tb_diagnosis"] = final.pop("tb_diagnosis_detail").fillna(
        final.get("tb_diagnosis", pd.NA))

    # urgency / surgery_severity: _y from TB join, _x from links
    for col in ["urgency", "surgery_severity"]:
        final[col] = final.get(f"{col}_y", final.get(col)).fillna(
            final.get(f"{col}_x", pd.NA))
        final = final.drop(
            columns=[f"{col}_x", f"{col}_y"], errors="ignore")

    # ── 9. Unify fund date column ─────────────────────────────────────
    final["fund_admission_date"] = pd.to_datetime(
        final.get("admission_date"), errors="coerce"
    ).fillna(pd.to_datetime(final.get("surgery_date"), errors="coerce"))
    final = final.drop(columns=["admission_date", "surgery_date"],
                       errors="ignore")

    # ── 10. Rename fund clinical columns for clarity ──────────────────
    final = final.rename(columns={
        "diagnosis": "fund_diagnosis",
        "surgery":   "fund_surgery",
    })

    # ── 11. Fill fund_covered_mk from fund-specific columns ───────────
    final["fund_covered_mk"] = (
        final["fund_covered_mk"]
        .fillna(final.get("mercy_fund_mk", pd.NA))
        .fillna(final.get("safe_fund_mk",  pd.NA))
        .fillna(final.get("endoscopy_ledger_balance_mk", pd.NA)) 
    )

    # ── 12. Cross-fund double-billing flag ────────────────────────────
    final = _check_cross_fund_overlap(final)

    # ── 13. Final column selection and ordering ───────────────────────
    cols  = [c for c in _FINAL_COLS if c in final.columns]
    final = final[cols].reset_index(drop=True)

    # ── 14. Summary ───────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*55}")
        print(f"Final shape: {final.shape}")
        print(f"\nMatch status:")
        print(final.groupby(["fund", "match_status"])
              .size().unstack(fill_value=0))
        print(f"\nMatched dept distribution:")
        matched = final[final["match_status"] == "matched"]
        print(matched.groupby(["fund", "department"])
              .size().unstack(fill_value=0))
        n_cross = final["cross_fund_flag"].sum()
        print(f"\nCross-fund double-billing flags: {n_cross}")

    return final