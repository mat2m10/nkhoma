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

def clean_surgical_df(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the surgical procedures dataframe.
    
    Args:
        df: Raw surgical dataframe
        mapping: Surgery type mapping dataframe with 'Full Surgery List' and 'Type of Surgery' columns
    
    Returns:
        Cleaned dataframe
    """
    df = to_snake_case(df)
    
    # Drop index and empty columns
    df = df[df['unnamed_0'].notna()].reset_index(drop=True)
    df = df.drop(columns=['unnamed_0', 'unnamed_34', 'unnamed_35'])

    # Patient info
    df['hospital_number'] = pd.to_numeric(df['hospital_number'], errors='coerce').astype('Int64')
    df['name'] = df['name'].str.strip().str.title()
    df['village_of_residence'] = (df['village_of_residence']
        .str.strip()
        .str.title()
        .str.replace(r'Area(\d)', r'Area \1', regex=True)
    )
    df['age_at_surgery'] = df['age_at_surgery'].astype('Int64')
    df['weight_kg'] = pd.to_numeric(df['weight_kg'].replace('\xa0', np.nan), errors='coerce')
    df['ward'] = df['ward'].str.strip().str.upper()
    df['previous_abdominal_surgery_yes_no'] = df['previous_abdominal_surgery_yes_no'].str.strip().str.upper()

    # Surgery info
    df = df.merge(
        mapping[['Full Surgery List', 'Type of Surgery']].rename(columns={
            'Full Surgery List': 'type_of_surgery',
            'Type of Surgery': 'surgery_type'
        }),
        on='type_of_surgery',
        how='left'
    )
    df['urgency'] = df['urgency'].str.strip().str.title()

    indication_map = {
        'Right Inguinal Hrnia': 'Right Inguinal Hernia',
        'Cholecystits': 'Cholecystitis',
        'Biliary Pancreatistis': 'Biliary Pancreatitis',
        'RLQP': 'RLQ Pain',
        'RIH': 'Right Inguinal Hernia',
        'LIH': 'Left Inguinal Hernia',
        'BL inguinal hernia': 'Bilateral Inguinal Hernia',
        'BL inguinal hernia + umbilical hernia': 'Bilateral Inguinal Hernia + Umbilical Hernia',
        'Bilateral inguinal Hernia': 'Bilateral Inguinal Hernia',
        'Chronic Abd Pains': 'Chronic Abdominal Pain',
        'chronic abd pain': 'Chronic Abdominal Pain',
        'peritoneal Carcinomatosis': 'Peritoneal Carcinomatosis',
        'incarcerated LIH': 'Incarcerated Left Inguinal Hernia',
        'recurrent RIH': 'Recurrent Right Inguinal Hernia',
        'epigastric Hernia': 'Epigastric Hernia',
        'acute/chronic cholecystitis': 'Acute/Chronic Cholecystitis',
        'chronic cholecystitis': 'Chronic Cholecystitis',
    }
    df['indication'] = df['indication'].str.strip().replace(indication_map).str.title()
    df['duration_mins'] = (df['duration_mins']
        .replace('\xa0', np.nan)
        .replace('110(50)', 110)
        .pipe(pd.to_numeric, errors='coerce')
        .astype('Int64')
    )

    # Anaesthesia
    muscle_relaxant_map = {
        'Sux': 'Short acting Muscle relaxant',
        'sux': 'Short acting Muscle relaxant',
        
        'Sux +Vecu': 'longacting muscle relaxant',
        'sux +vecu': 'longacting muscle relaxant',
        'Vecu': 'longacting muscle relaxant',
        'vecuronium': 'longacting muscle relaxant',
        'Other': 'longacting muscle relaxant',
        'Longacting Muscle Relaxant': 'longacting muscle relaxant'
        
    }
    df['muscle_relaxant_used'] = df['muscle_relaxant_used'].str.strip().replace(muscle_relaxant_map)

    # Payment
    df['payment_method_scheme_cash'] = (df['payment_method_scheme_cash']
        .replace('\xa0', np.nan)
        .str.strip()
        .str.title()
    )
    df['cost_mwk'] = df['cost_mwk'].replace('\xa0', np.nan).pipe(pd.to_numeric, errors='coerce')
    df['patientpayment_mwk'] = (df['patientpayment_mwk']
        .replace('\xa0', np.nan)
        .replace('nil', 0)
        .pipe(pd.to_numeric, errors='coerce')
    )
    df['otherpayment_mwk'] = df['otherpayment_mwk'].replace('\xa0', np.nan).str.strip().str.title()
    df = df.rename(columns={'otherpayment_mwk': 'other_payment_source'})

    # Admission
    df['length_of_hospital_stay_days'] = df['length_of_hospital_stay_days'].astype('Int64')

    # Surgeons
    surgeon_map = {
        'Dr widmann': 'Dr Widmann',
        'Dr Widmann ': 'Dr Widmann',
        'Dr lam': 'Dr Lam',
        'Dr Lam ': 'Dr Lam',
        'Dr stuebing': 'Dr Stuebing',
        'Dr Stuebing ': 'Dr Stuebing',
        'Dr Stuebibg': 'Dr Stuebing',
        'Drvaylann': 'Dr Vaylann',
        'Dr vaylann': 'Dr Vaylann',
    }
    surgeon_map_2 = {
        **surgeon_map,
        'Dr vaylann/Dr limbe': 'Dr Vaylann',
        'Dr limbe': 'Dr Limbe',
        'Dr wanda': 'Dr Lam',
        'Dr beth': 'Dr Stuebing',
        '\xa0': np.nan,
    }
    resident_map = {
        'Dr caleb': 'Dr Caleb',
        'Dr mumba': 'Dr Wongani',
        'Dr. Vitu': 'Dr Vitu',
        'Dr vitu': 'Dr Vitu',
        'Dr Vitu ': 'Dr Vitu',
        'Dr jonathan': 'Dr Jonathan',
        'Dr wongani': 'Dr Wongani',
        'Brenda': 'Visiting Resident',
        'faith': 'Visiting Resident',
        'Dr James': 'Visiting Resident',
    }
    resident_map_2 = {
        'Dr mada': 'Dr Madalitso',
        'Dr jonathan': 'Dr Jonathan',
        '`': np.nan,
        '\xa0': np.nan,
    }
    df['attending_surgeon_1'] = df['attending_surgeon_1'].str.strip().replace(surgeon_map)
    df['attending_surgeon_2'] = df['attending_surgeon_2'].str.strip().replace(surgeon_map_2)
    df['resident_surgeon_1'] = df['resident_surgeon_1'].str.strip().replace(resident_map)
    df['resident_surgeon_2'] = df['resident_surgeon_2'].str.strip().replace(resident_map_2)

    # Gas and conversion
    df['amount_of_gas_used_l'] = (df['amount_of_gas_used_l']
        .replace({'?': np.nan, '\xa0': np.nan, '        ': np.nan})
        .apply(lambda x: np.nan if isinstance(x, datetime.datetime) else x)
        .pipe(pd.to_numeric, errors='coerce')
    )
    df['conversion'] = df['conversion'].str.strip().str.lower().map({'yes': True, 'no': False})
    df['reason_of_conversion'] = (df['reason_of_conversion']
        .replace('\xa0', np.nan)
        .astype(str)
        .str.strip()
        .replace('nan', np.nan)
        .str.replace('Inadvetent', 'Inadvertent')
    )
    df['time_of_conversion_min_after_incisicon'] = (df['time_of_conversion_min_after_incisicon']
        .replace('\xa0', np.nan)
        .pipe(pd.to_numeric, errors='coerce')
        .astype('Int64')
    )
    df = df.rename(columns={'time_of_conversion_min_after_incisicon': 'time_of_conversion_min_after_incision'})

    # Complications
    df['complication'] = df['complication'].str.strip().str.lower().map({'yes': True, 'no': False})
    df['complication_clavien_dindo'] = (df['complication_clavien_dindo']
        .replace('\xa0', np.nan)
        .astype(str)
        .str.strip()
        .str.lower()
        .replace('nan', np.nan)
    )
    df['complication_description'] = df['complication_description'].replace('\xa0', np.nan).str.strip()
    df['age_group'] = df['age_at_surgery'].apply(lambda x: 'Pediatric' if x < 16 else 'Adult')
    df['teaching_category'] = df['teaching'].apply(
        lambda x: 'No Teaching' if pd.isna(x) or str(x).strip() in ['<10', ''] 
        else ('Partial Teaching' if 10 <= int(x) < 50 
        else 'Teaching')
    )
    mask = df['payment_method_scheme_cash'].isin(['Safe', 'Cash+Mercy', 'Mercy Fund'])
    df.loc[mask, 'payment_method_scheme_cash'] = 'Poor patient fund'
    return df