import re
import pandas as pd
import numpy as np
import datetime

def to_snake_case(df):
    def convert(col):
        # Lowercase
        col = col.strip().lower()
        # Replace common patterns
        col = re.sub(r'[\s/()]+', '_', col)   # spaces, slashes, parens → _
        col = re.sub(r'[^a-z0-9_]', '', col)  # remove remaining special chars
        col = re.sub(r'_+', '_', col)          # collapse multiple underscores
        col = col.strip('_')                   # strip leading/trailing _
        return col

    df = df.copy()
    df.columns = [convert(c) for c in df.columns]
    return df

def drop_nan_index(df, col="theatre_book_index"):
    return df.dropna(subset=[col]).reset_index(drop=True)

def drop_unnamed_cols(df):
    return df.loc[:, ~df.columns.str.startswith("unnamed")]

def drop_sparse_cols(df, threshold=0.99):
    min_non_null = len(df) * (1 - threshold)
    return df.dropna(axis=1, thresh=int(min_non_null))

def clean_date_of_surgery(df, col="date_of_surgery"):
    # Drop rows where the value is not a valid date (e.g. summary rows like "Ergebnis")
    df = df[pd.to_datetime(df[col], errors="coerce").notna()].copy()
    # Convert and strip time component
    df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    return df.reset_index(drop=True)

def drop_invalid_dates(df, col="date_of_surgery"):
    today = pd.Timestamp.today().normalize()
    mask = df[col].between(pd.Timestamp("2020-01-01"), today)
    return df[mask].reset_index(drop=True)

def clean_name_cols(df, cols=["first_name", "last_name"]):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()
    return df

def clean_age(df, col="age_years", min_age=0, max_age=120):
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[~df[col].between(min_age, max_age), col] = pd.NA
    df[col] = df[col].round().astype("Int64")
    return df

STAFF_RENAME_MAP = {
    "obs/Gyn": "Obs/Gyn",
    "OBs/Gyn": "Obs/Gyn",
    "other": "Other",
    "lam": "Lam",
    "VITU": "Vitu",
    "vitu": "Vitu",
    "Stuebing": "Stuebing",
    "Steubig": "Stuebing",
    "Steubing": "Stuebing", 
    "alex": "Alex",
    "ALEX": "Alex",
}

def rename_staff(df,
                 staff_cols=["surgeon", "1st_assistent_instructor", "2nd_assistent", "nurse"],
                 surgeon_col="surgeon",
                 surgeon_map=None):
    for col in staff_cols:
        if col in df.columns:
            df[col] = df[col].replace(STAFF_RENAME_MAP)

    if surgeon_map and surgeon_col in df.columns:
        df[surgeon_col] = df[surgeon_col].replace(surgeon_map)

    if "anaestesist" in df.columns:
        df["anaestesist"] = df["anaestesist"].replace("none", pd.NA)

    return df

def clean_text_cols(df):
    """Unify placeholders and normalize whitespace across all text columns."""
    PLACEHOLDERS = {"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "na": pd.NA,
                    "null": pd.NA, "None": pd.NA, "none": pd.NA,
                    "nan": pd.NA, "NaN": pd.NA, "-": pd.NA}
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = (df[col]
                   .astype("string")
                   .replace(PLACEHOLDERS)
                   .str.strip()
                   .str.replace(r"\s+", " ", regex=True))
    return df

def clean_time_cols(df):
    if "sarting_time" in df.columns:
        df = df.rename(columns={"sarting_time": "starting_time"})

    for col in ["starting_time", "finishing_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.time

    if "operation_time_minutes" in df.columns:
        df["operation_time_minutes"] = (
            df["operation_time_minutes"]
            .astype("string")
            .str.extract(r"(\d{2}:\d{2}:\d{2})")[0]
        )
        df.loc[df["operation_time_minutes"] == "00:00:00", "operation_time_minutes"] = pd.NA
        df["operation_time_minutes"] = pd.to_timedelta(df["operation_time_minutes"], errors="coerce")

    return df

def clean_categoricals(df):
    # --- sex ---
    if "sex" in df.columns:
        df["sex"] = df["sex"].str.upper().replace({"FEMALE": "F", "MALE": "M"})
        df.loc[~df["sex"].isin(["F", "M"]), "sex"] = pd.NA

    # --- urgency ---
    if "urgency" in df.columns:
        df["urgency"] = (df["urgency"]
                         .str.title()
                         .replace({"Emerg": "Emergency", "Elective ": "Elective"}))

    # --- histology ---
    if "histology" in df.columns:
        df["histology"] = df["histology"].str.upper()
        df["histology"] = df["histology"].replace({"YES": "Yes", "Y": "Yes", "NO": "No", "N": "No"})
        df.loc[~df["histology"].isin(["Yes", "No"]), "histology"] = pd.NA

    # --- asascore ---
    if "asascore" in df.columns:
        df["asascore"] = (df["asascore"]
                          .str.upper()
                          .str.replace(r"^ASA\s*([1-6])$", r"ASA \1", regex=True))

    # --- department ---
    if "department" in df.columns:
        df["department"] = df["department"].str.strip()
        df["department"] = df["department"].replace({
            "GEn_Surg": "Gen_Surg",
            "ortho":    "Ortho",
        })

    # --- indication_for_surgery ---
    if "indication_for_surgery" in df.columns:
        df["indication_for_surgery"] = df["indication_for_surgery"].str.strip().str.title()
        df["indication_for_surgery"] = df["indication_for_surgery"].replace({
            "Non-Communicable Disease": "Non-Communicable Disease",
            "Infection":                "Infection",
            "Trauma":                   "Trauma",
            "Caesarean Section":        "Caesarean Section",
            "Indication For Surgery":   pd.NA,  # junk value
        })

    # --- surgery_type ---
    if "surgery_type" in df.columns:
        df["surgery_type"] = df["surgery_type"].str.strip().str.title()
        df["surgery_type"] = df["surgery_type"].replace({
            "Any Other Surgery": "Any Other Surgery",
        })

    # --- final_diagnosis_category ---
    if "final_diagnosis_category" in df.columns:
        df["final_diagnosis_category"] = df["final_diagnosis_category"].str.strip().str.title()
        df["final_diagnosis_category"] = df["final_diagnosis_category"].replace({
            "Pregnancy":         "Pregnancy",
            "Pregnancy":         "Pregnancy",
            "Leomyoma":          "Leiomyoma",
            "Hernia":            "Inguinal_Hernia",
            "Inguinal Hernia":   "Inguinal_Hernia",
            "Umbilical Hernia":  "Umbilical_Hernia",
            "Umbilical Hernia":  "Umbilical_Hernia",
            "Other_Gyn":         "Other_Gyn",
            "Other_Gyn":         "Other_Gyn",
            "Other_Surg":        "Other_Surg",
            "Other_Surg":        "Other_Surg",
            "Other_Ortho":       "Other_Ortho",
            "Other_Ortho":       "Other_Ortho",
            "Other ":            "Other_Surg",
            "Other Surgical":    "Other_Surg",
            "Lipoma":            "Lipoma_Subcut_Tumor",
            "Epigastric Hernia": "Ventral_Hernia",
            "Cyst":              "Cyst",
            "Urethral Stricture ": "Urethral_Stricture",
        })

    # --- side ---
    if "side" in df.columns:
        df["side"] = df["side"].str.strip().str.title()
        df["side"] = df["side"].replace({"Na": pd.NA, "Na": pd.NA})
        df.loc[~df["side"].isin(["Left", "Right", "Bilateral", "Other"]), "side"] = pd.NA

    # --- main_procedure_category ---
    # Too varied for full normalization — just fix obvious case/whitespace issues
    if "main_procedure_category" in df.columns:
        df["main_procedure_category"] = df["main_procedure_category"].str.strip()
        df["main_procedure_category"] = df["main_procedure_category"].replace({
            "c/s":   "C/S",
            "other": "Other",
            "OTHER": "Other",
            "OTher": "Other",
            "OTHer": "Other",
            " HERNIA REPAIR":  "Hernia Repair",
            "hernia repair":   "Hernia Repair",
            "henia repair":    "Hernia Repair",
            "HERNIA REPAIR":   "Hernia Repair",
            "LICHTENSTEIN":         "Lichtenstein",
            "LICHENSTEIN PROCEDURE": "Lichtenstein",
            "LICHTENSTEIN PROCEDURE": "Lichtenstein",
            "LICHTNENSTEIN PROCEDURE": "Lichtenstein",
            "LICHTENSTEIN REPAIR":    "Lichtenstein",
            "LICHTENSTEINS":          "Lichtenstein",
            "L Lichtenstein":         "Lichtenstein",
            "R Lichtenstein":         "Lichtenstein",
            "L Lichtenstein Procedure": "Lichtenstein",
            "R Lichtenstein Procedure": "Lichtenstein",
            "Endoscopy: EGD+ Biopsy":    "Endoscopy: EGD±Biopsy",
            "Endoscopy: EGD:intervation": "Endoscopy: EGD+Intervention",
            "LAPARASCOPIC":  "Laparoscopy: Other",
            "laparascopic":  "Laparoscopy: Other",
            "Laparotomy: only explorative ": "Explorative Laparotomy",
            "Ex-Lap ":       "Explorative Laparotomy",
            "Ex-Lap":        "Explorative Laparotomy",
            "EX LAP":        "Explorative Laparotomy",
        })

    # --- surgery_severity ---
    if "surgery_severity" in df.columns:
        df["surgery_severity"] = df["surgery_severity"].str.strip().str.title()
        df["surgery_severity"] = df["surgery_severity"].replace({"Minor": "Minor"})
        df.loc[~df["surgery_severity"].isin(["Major", "Minor", "Intermediate"]), "surgery_severity"] = pd.NA

    return df