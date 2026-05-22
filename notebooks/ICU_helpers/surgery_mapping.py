def map_surgeries(df):
    surgery_map = {
        # === Caesarean Section ===
        'C/S': 'Caesarean Section',
        'C/Section': 'Caesarean Section',
        'Caesarian Section': 'Caesarean Section',
        'Ceasarian Section': 'Caesarean Section',
        'Cessarian Section': 'Caesarean Section',

        # === EGD ===
        'Egd': 'EGD',
        'EGD': 'EGD',
        'EGD And Banding': 'EGD',
        'Egd And Banding': 'EGD',
        'EGD And Clipping, Then Exp Lap With Oversew Of Gastric Ulcer': 'EGD',
        'EGD Glueing': 'EGD',
        'Endoscopy, Banding': 'EGD',
        'Banding': 'EGD',

        # === Cholecystectomy ===
        'Laparoscopic Cholecystectomy': 'Laparoscopic Cholecystectomy',
        'Open Cholecystectomy & Specimen': 'Other',

        # === Open Prostatectomy ===
        'Open Prostatectomy': 'Open Prostatectomy',
        'Open Prostectomy': 'Open Prostatectomy',
        'Prostate??': 'Open Prostatectomy',
        'Open Prostatectomy And Take Back': 'Open Prostatectomy',
        'Open Prostatectomy And 2 Take Backs': 'Open Prostatectomy',

        # === Hysterectomy ===
        'Tah': 'Hysterectomy',
        'Tah And Bso': 'Hysterectomy',
        'Tah, J-Stent Ureteral Injury Repair': 'Hysterectomy',
        'Tah,Bso And Underlay Incisional Hernia Repair': 'Hysterectomy',
        'Subtotal Hysterectomy': 'Hysterectomy',
        'Subtotal Hysterectomy + Bladder Repair': 'Hysterectomy',

        # === Debridement ===
        'Debridement': 'Debridement',
        'Chest Debridement': 'Debridement',
        'Cervial/Chest Debridement, Ssg': 'Debridement',
        'Extensive Debridement With I&D': 'Debridement',
        'Scrotal/Perianal Debribement': 'Debridement',

        # === Exploratory Laparotomy (plain) ===
        'Ex Lap': 'Exploratory Laparotomy',
        'Ex Laparatomy': 'Exploratory Laparotomy',
        'Ex-Laparatomy': 'Exploratory Laparotomy',
        'Exp Laparatomy': 'Exploratory Laparotomy',
        'Exploratory Laparatomy': 'Exploratory Laparotomy',
        'Exploratory Laparotomy, Washout, Drain Placement': 'Exploratory Laparotomy',
        'Exploration': 'Exploratory Laparotomy',
        'Ex-Lap, Adhesiolysis': 'Exploratory Laparotomy',
        'Exp-Laparotomy, Re-Lap 10/11/2024': 'Exploratory Laparotomy',
        'Exploratory Laparatomy, Modified Grahams Patch': 'Exploratory Laparotomy',
        'Exploratory Laparatomy, Liver Packing, Then 27/8 Relap': 'Exploratory Laparotomy',
        'Expl Laparotomy, Lavage, Repair And Patch Of Gastric Perf': 'Exploratory Laparotomy',
        'Ex-Lap, Washout': 'Exploratory Laparotomy',
        'Ex-Lap, Bowel Decompression': 'Exploratory Laparotomy',
        'Ex Lap, Psoas Abscess Drainage/Appendectomy': 'Exploratory Laparotomy',
        'Ex-Lap,Rectal Sigmoid Resection': 'Exploratory Laparotomy',
        "Modified Graham'S Patch": 'Exploratory Laparotomy',
        'Abdominal Wshout And Closure': 'Exploratory Laparotomy',
        'Second Look, Waschout': 'Exploratory Laparotomy',

        # === Laparotomy with Bowel Resection ===
        'Damage Control Ileo-Resection And Washout': 'Exploratory Laparotomy with Bowel Resection',
        'Damage Control Ileo-Resection And Washout/ Bowel Anastomosis': 'Exploratory Laparotomy with Bowel Resection',
        'Exploratorory Laparotomy: Adhensions Plus Bowel Resection': 'Exploratory Laparotomy with Bowel Resection',
        'Exploratory Laparotomy, Ileoal Resection,': 'Exploratory Laparotomy with Bowel Resection',
        'Ex Lap, Ileal Transverse Side To End Anastomosis': 'Exploratory Laparotomy with Bowel Resection',
        'Ex Lap, Right Hemicolectomy': 'Exploratory Laparotomy with Bowel Resection',
        'Diagnostic Laparoscopy, Adhesiolysis, Konversion To Ope Right Hemicolectomy, E-E Ileotransversotomy': 'Exploratory Laparotomy with Bowel Resection',
        'Small Bowel Resection, Anastomosis': 'Exploratory Laparotomy with Bowel Resection',
        'Sigmoid Resection': 'Exploratory Laparotomy with Bowel Resection',
        'Right Hemicolectomy, Duodenum Wedge Resection, Gastrojejunostomy': 'Exploratory Laparotomy with Bowel Resection',
        'Eviceration Of Small Bowel': 'Exploratory Laparotomy with Bowel Resection',
        'Exlap, Ileostomy': 'Exploratory Laparotomy with Bowel Resection',
        'Ex Lap, Ileostomy': 'Exploratory Laparotomy with Bowel Resection',

        # === Exploratory Laparotomy with Splenectomy ===
        'Exploratory Laparotomy, Splenectomy': 'Exploratory Laparotomy with Splenectomy',
        'Splenectomy': 'Exploratory Laparotomy with Splenectomy',
        'Spleenectomy': 'Exploratory Laparotomy with Splenectomy',

        # === Thyroidectomy ===
        'Hemithyroidectomy': 'Hemi/Total Thyroidectomy',
        'Total Thyroidectomy': 'Hemi/Total Thyroidectomy',
        'Left Hemithyroidectomy And Isthmussectomy': 'Hemi/Total Thyroidectomy',

        # === No Surgery ===
        'No Surgery': 'No Surgery',
        'No Surgery (Poison)': 'No Surgery',
        'No Surgery( Pleural And Ascitic Tap)': 'No Surgery',
        'Asthma': 'No Surgery',

        # === Other ===
        'Flexible Sigmoidoscopy/ Reopenning': 'Other',
        'Right Chest Tube Placement': 'Chest Tube Placement',
        'Rib Resection, Irrigation, Skin Flap Over Diaphragm, Packing': 'Other',
        'Bladder Mass Excision': 'Other',
        'Buccal Graft Urethroplasty': 'Other',
        'Rib Resection': 'Other',
        'Bilateral Inguinal Hernia Repair': 'Other',
    }

    df['type_of_surgery_grouped'] = df['type_of_surgery'].replace(surgery_map)
    return df