import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ==============================================================================
# 1. EDIT YOUR FIGURE TITLES & FILE PATH HERE
# ==============================================================================
CSV_PATH = "poisson_model_output.csv"  # Change to "global_tennis_statistics_v3.csv" if needed

TITLES = {
    "fig1_super": "Czechia’s Disproportionate Tennis Capital & Labor Investment vs. Global Benchmarks",
    "fig1_panel_a": "A. Raw Infrastructure & Coaching Stock",
    "fig1_panel_b": "B. Per-Capita Resource Intensity (Per 100k Pop.)",
    "fig2_main": "Cobb-Douglas Production Isoquants — Tennis Capital, Labor & Elite Output",
}

# ==============================================================================
# 2. LOAD DATA & COMPUTE PER-CAPITA METRICS (IF NOT ALREADY IN CSV)
# ==============================================================================
df = pd.read_csv(CSV_PATH)

# Ensure Total_Top100 and per-100k columns exist regardless of which CSV version is loaded
if "Total_Top100" not in df.columns:
    df["Total_Top100"] = (
        df["ATP Top-100 Players (2000–2025)"] + df["WTA Top-100 Players (2000–2025)"]
    )

pop_100k = df["Avg Population (2000–2025, M)"] * 10.0  # Millions -> units of 100,000
if "courts_pc" not in df.columns:
    df["courts_pc"] = df["Tennis Courts (2018)"] / pop_100k
if "coaches_pc" not in df.columns:
    df["coaches_pc"] = df["Tennis Coaches (2018)"] / pop_100k
if "clubs_pc" not in df.columns:
    df["clubs_pc"] = df["Tennis Clubs (2018)"] / pop_100k

cze = df[df["IOC"] == "CZE"].iloc[0]

# ==============================================================================
# 3. FIT COBB-DOUGLAS MODEL VIA POISSON MLE (Y = A * L^alpha * K^beta)
# ==============================================================================
df["log_L"] = np.log(df["Tennis Coaches (2018)"])
df["log_K_courts"] = np.log(df["Tennis Courts (2018)"])

cd_pois = smf.glm(
    "Total_Top100 ~ log_L + log_K_courts",
    data=df,
    family=sm.families.Poisson(),
).fit(cov_type="HC3")

A = np.exp(cd_pois.params["Intercept"])
alpha = cd_pois.params["log_L"]
beta = cd_pois.params["log_K_courts"]

# ==============================================================================
# 4. GENERATE FIGURE 1: RESOURCE INVESTMENT BAR CHART
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

categories = ["Tennis Courts", "Tennis Coaches", "Tennis Clubs"]
cze_raw = [
    cze["Tennis Courts (2018)"],
    cze["Tennis Coaches (2018)"],
    cze["Tennis Clubs (2018)"],
]
mean_raw = [
    df["Tennis Courts (2018)"].mean(),
    df["Tennis Coaches (2018)"].mean(),
    df["Tennis Clubs (2018)"].mean(),
]
med_raw = [
    df["Tennis Courts (2018)"].median(),
    df["Tennis Coaches (2018)"].median(),
    df["Tennis Clubs (2018)"].median(),
]

x = np.arange(len(categories))
w = 0.26

b1 = axes[0].bar(x - w, cze_raw, width=w, label="Czech Republic (Pop: 10.5M)", color="#D62728")
b2 = axes[0].bar(x, mean_raw, width=w, label="39-Country Mean (Avg Pop: 107.6M)", color="#2B5B84")
b3 = axes[0].bar(x + w, med_raw, width=w, label="39-Country Median (Pop: 24.4M)", color="#9AA5B1")

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            h + 220,
            f"{int(round(h)):,}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold" if bars == b1 else "normal",
            color="#D62728" if bars == b1 else "#222222",
        )

axes[0].set_xticks(x)
axes[0].set_xticklabels(categories, fontsize=10.5, fontweight="bold")
axes[0].set_ylabel("Total National Count (2018)", fontsize=10.5)
axes[0].set_title(TITLES["fig1_panel_a"], fontsize=11.5, fontweight="bold")
axes[0].set_ylim(0, max(cze_raw) * 1.18)
axes[0].grid(axis="y", linestyle=":", alpha=0.5)
axes[0].legend(frameon=True, fontsize=8.8)

# Panel B: Per 100,000 Residents
cze_pc = [cze["courts_pc"], cze["coaches_pc"], cze["clubs_pc"]]
mean_pc = [df["courts_pc"].mean(), df["coaches_pc"].mean(), df["clubs_pc"].mean()]
med_pc = [df["courts_pc"].median(), df["coaches_pc"].median(), df["clubs_pc"].median()]

b1_pc = axes[1].bar(x - w, cze_pc, width=w, label="Czech Republic (Rank #1, #2, #4)", color="#D62728")
b2_pc = axes[1].bar(x, mean_pc, width=w, label="39-Country Mean", color="#2B5B84")
b3_pc = axes[1].bar(x + w, med_pc, width=w, label="39-Country Median", color="#9AA5B1")

for bars in [b1_pc, b2_pc, b3_pc]:
    for bar in bars:
        h = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            h + 2.0,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold" if bars == b1_pc else "normal",
            color="#D62728" if bars == b1_pc else "#222222",
        )

axes[1].set_xticks(x)
axes[1].set_xticklabels(["Courts / 100k", "Coaches / 100k", "Clubs / 100k"], fontsize=10.5, fontweight="bold")
axes[1].set_ylabel("Units per 100,000 Residents", fontsize=10.5)
axes[1].set_title(TITLES["fig1_panel_b"], fontsize=11.5, fontweight="bold")
axes[1].set_ylim(0, max(cze_pc) * 1.18)
axes[1].grid(axis="y", linestyle=":", alpha=0.5)
axes[1].legend(frameon=True, fontsize=8.8)

plt.suptitle(TITLES["fig1_super"], fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("fig_resource_investment.png", dpi=300, bbox_inches="tight")
plt.show()

# ==============================================================================
# 5. GENERATE FIGURE 2: COBB-DOUGLAS ISOQUANT MAP
# ==============================================================================
fig, ax = plt.subplots(figsize=(9.5, 7))

ax.scatter(
    df["Tennis Coaches (2018)"],
    df["Tennis Courts (2018)"],
    color="#8C9BA5",
    alpha=0.65,
    s=55,
    edgecolors="white",
    linewidth=0.6,
    label="Other Sample Nations (N=39)",
)

L_grid = np.linspace(300, 32000, 400)
iso_specs = [
    (20, "#6BAED6", "--", "Isoquant Y = 20 Players (e.g., Canada/Colombia tier)"),
    (38, "#2B8CBE", "-.", "Isoquant Y = 38 Players (Expected at CZE inputs)"),
    (58, "#D62728", "-",  "Isoquant Y = 58 Players (Czechia & Argentina Actual)"),
    (94, "#238B45", "--", "Isoquant Y = 94 Players (France Actual)"),
]

for Y_target, col, ls, lbl in iso_specs:
    K_iso = (Y_target / (A * (L_grid ** alpha))) ** (1.0 / beta)
    mask = (K_iso >= 400) & (K_iso <= 300000)
    ax.plot(
        L_grid[mask],
        K_iso[mask],
        color=col,
        linestyle=ls,
        linewidth=2.0 if Y_target == 58 else 1.5,
        label=lbl,
    )

# Cleanly spaced label offsets (dx, dy in points) to avoid text collisions
highlights = {
    "CZE": ("#D62728", "Czechia (Actual: 58)", (-95, -14)),
    "USA": ("#1F3A60", "USA (151)", (-55, -14)),
    "FRA": ("#238B45", "France (94)", (8, 6)),
    "ESP": ("#E6550D", "Spain (76)", (8, 5)),
    "RUS": ("#6A51A3", "Russia (67)", (8, 4)),
    "ARG": ("#8C564B", "Argentina (58)", (8, -10)),
    "GER": ("#525252", "Germany (64)", (8, -6)),
    "GBR": ("#525252", "Great Britain (25)", (8, -10)),
    "ITA": ("#525252", "Italy (52)", (8, -4)),
}

for ioc, (col, text_lbl, offset) in highlights.items():
    row = df[df["IOC"] == ioc].iloc[0]
    ax.scatter(
        row["Tennis Coaches (2018)"],
        row["Tennis Courts (2018)"],
        color=col,
        s=140 if ioc == "CZE" else 85,
        zorder=6,
        edgecolors="black",
        linewidth=0.9,
    )
    ax.annotate(
        text_lbl,
        (row["Tennis Coaches (2018)"], row["Tennis Courts (2018)"]),
        xytext=offset,
        textcoords="offset points",
        fontsize=9.5,
        fontweight="bold" if ioc == "CZE" else "normal",
        color=col,
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(300, 40000)
ax.set_ylim(500, 350000)
ax.set_xlabel("Labor Input (L): Certified Tennis Coaches (Log Scale)", fontsize=11)
ax.set_ylabel("Physical Capital Input (K): Tennis Courts (Log Scale)", fontsize=11)
ax.set_title(TITLES["fig2_main"], fontsize=12.5, fontweight="bold")
ax.grid(True, which="both", linestyle=":", alpha=0.45)
ax.legend(loc="lower right", frameon=True, fontsize=8.8)
plt.tight_layout()
plt.savefig("fig_cobb_douglas_isoquants.png", dpi=300)
plt.show()
