import pandas as pd
import matplotlib.pyplot as plt
import os
def plot_dashboard(df):
    df = df.copy()  # avoid mutating the original

    # --- Complication: normalise to TRUE/FALSE ---
    df["complication"] = df["complication"].map(
        {1: "TRUE", 1.0: "TRUE", 0: "FALSE", 0.0: "FALSE",
         "1": "TRUE", "0": "FALSE", "1.0": "TRUE", "0.0": "FALSE",
         "TRUE": "TRUE", "FALSE": "FALSE"}
    )

    # --- Conversion: map to human-readable labels ---
    conversion_map = {
        1: "Conversion",   1.0: "Conversion",
        0: "No-Conversion", 0.0: "No-Conversion",
        "1": "Conversion", "1.0": "Conversion",
        "0": "No-Conversion", "0.0": "No-Conversion",
        "TRUE": "Conversion", "FALSE": "No-Conversion",
        "YES": "Conversion", "NO": "No-Conversion",
        "CONVERSION": "Conversion", "NO-CONVERSION": "No-Conversion",
    }
    df["conversion"] = (
        df["conversion"].astype(str).str.strip().str.upper()
        .map(lambda v: conversion_map.get(v, v))
    )

    # --- Teaching: remap raw values to ordered labels ---
    teaching_map = {
        # adapt these keys to whatever raw values your data contains
        "NO TEACHING": "No Teaching",
        "NONE": "No Teaching",
        "0": "No Teaching",
        "PARTIAL": "Partial Teaching (<50%)",
        "PARTIAL TEACHING": "Partial Teaching (<50%)",
        "<50": "Partial Teaching (<50%)",
        "TEACHING": "Teaching (50%+)",
        "FULL": "Teaching (50%+)",
        "50+": "Teaching (50%+)",
        "1": "Teaching (50%+)",
    }
    TEACHING_ORDER = ["No Teaching", "Partial Teaching (<50%)", "Teaching (50%+)"]
    df["teaching_category"] = (
        df["teaching_category"].astype(str).str.strip().str.upper()
        .map(lambda v: teaching_map.get(v, v))
    )

    COLORS = ["#4472C4", "#ED7D31", "#A9A9A9", "#FFC000", "#5B9BD5", "#70AD47"]
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    def make_autopct(values):
        """Show both % and n= on pie slices."""
        def autopct(pct):
            total = sum(values)
            n = int(round(pct * total / 100.0))
            return f"{pct:.0f}%\n(n={n})"
        return autopct

    # --- AGE GROUP ---
    counts = df["age_group"].value_counts()
    axes[0].pie(counts, labels=counts.index, autopct=make_autopct(counts),
                colors=COLORS, startangle=90, counterclock=False)
    axes[0].set_title("AGE GROUP", fontweight="bold")
    axes[0].legend(counts.index, loc="center right", bbox_to_anchor=(1.3, 0.5))

    # --- GENDER ---
    counts = df["sex"].value_counts()
    axes[1].pie(counts, labels=counts.index, autopct=make_autopct(counts),
                colors=COLORS, startangle=90, counterclock=False)
    axes[1].set_title("GENDER", fontweight="bold")
    axes[1].legend(counts.index, loc="center right", bbox_to_anchor=(1.3, 0.5))

    # --- PAYMENT METHOD  (NaN → "MISSING") ---
    counts = df["payment_method_scheme_cash"].value_counts(dropna=False)
    counts.index = counts.index.fillna("MISSING")           # ← changed from BLANKS
    axes[2].pie(counts, labels=counts.index, autopct=make_autopct(counts),
                colors=COLORS, startangle=90, counterclock=False)
    axes[2].set_title("PAYMENT METHOD", fontweight="bold")

    # --- TEACHING CATEGORY  (ordered: No → Partial → Full) ---
    counts = df["teaching_category"].value_counts(dropna=False)
    counts.index = counts.index.fillna("MISSING")
    # reindex to enforce the desired slice order; missing categories get 0
    counts = counts.reindex(
        [t for t in TEACHING_ORDER if t in counts.index]
        + [t for t in counts.index if t not in TEACHING_ORDER]
    )
    axes[3].pie(counts, labels=counts.index, autopct=make_autopct(counts),
                colors=COLORS, startangle=90, counterclock=False)
    axes[3].set_title("TEACHING", fontweight="bold")
    axes[3].legend(counts.index, loc="center right", bbox_to_anchor=(1.3, 0.5))

    # --- CONVERSION  (Conversion / No-Conversion) ---
    CONVERSION_ORDER = ["Conversion", "No-Conversion"]
    counts = df["conversion"].value_counts(dropna=False)
    counts.index = counts.index.fillna("MISSING")
    counts = counts.reindex(
        [c for c in CONVERSION_ORDER if c in counts.index]
        + [c for c in counts.index if c not in CONVERSION_ORDER]
    )
    axes[4].pie(counts, labels=counts.index, autopct=make_autopct(counts),
                colors=COLORS, startangle=90, counterclock=False)
    axes[4].set_title("CONVERSION", fontweight="bold")
    axes[4].legend(counts.index, loc="center right", bbox_to_anchor=(1.3, 0.5))

    # --- CASES PER YEAR ---
    cases_per_year = pd.to_datetime(df["date_of_surgery"]).dt.year.value_counts().sort_index()
    axes[5].plot(cases_per_year.index, cases_per_year.values, marker="o", color="#4472C4")
    for x, y in zip(cases_per_year.index, cases_per_year.values):
        total = cases_per_year.sum()
        pct = y / total * 100
        axes[5].annotate(f"n={y}\n({pct:.0f}%)", (x, y),
                         textcoords="offset points", xytext=(0, 8), ha="center")
    axes[5].set_title("CASES PER YEAR", fontweight="bold")
    axes[5].set_xticks(cases_per_year.index)
    axes[5].set_ylim(0, cases_per_year.max() + 20)

    # --- COMPLICATIONS ---
    counts = df["complication"].astype(str).str.strip().str.upper().value_counts(dropna=False)
    counts.index = counts.index.fillna("MISSING")
    axes[6].pie(counts, labels=counts.index, autopct=make_autopct(counts),
                colors=COLORS, startangle=90, counterclock=False)
    axes[6].set_title("COMPLICATIONS", fontweight="bold")
    axes[6].legend(counts.index, loc="center right", bbox_to_anchor=(1.3, 0.5))

    # --- hide unused ---
    axes[7].set_visible(False)

    plt.suptitle("SURGICAL DASHBOARD", fontweight="bold", fontsize=16, y=1.01)
    plt.tight_layout()

    # ── Save full dashboard ───────────────────────────────────────────
    os.makedirs("laparoscopy_plots", exist_ok=True)
    fig.savefig("laparoscopy_plots/00_surgical_dashboard.png", dpi=150, bbox_inches='tight')
    fig.savefig("laparoscopy_plots/00_surgical_dashboard.pdf", bbox_inches='tight')

    # ── Save each subplot individually ───────────────────────────────
    subplot_titles = [
        "01_age_group", "02_gender", "03_payment_method", "04_teaching",
        "05_conversion", "06_cases_per_year", "07_complications",
    ]
    renderer = fig.canvas.get_renderer()
    for ax, title in zip(axes[:7], subplot_titles):
        extent = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        expanded = extent.expanded(1.15, 1.2)
        fig.savefig(f"laparoscopy_plots/{title}.png", bbox_inches=expanded, dpi=150)
        fig.savefig(f"laparoscopy_plots/{title}.pdf", bbox_inches=expanded)

    plt.show()