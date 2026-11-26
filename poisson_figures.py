import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Helper: Exact Poisson upper-tail probability P(X >= k | mu) without scipy
def poisson_sf(k_minus_1, mu):
    if k_minus_1 < 0:
        return 1.0
    cdf = sum(
        math.exp(-mu) * (mu ** i) / math.factorial(i)
        for i in range(int(k_minus_1) + 1)
    )
    return max(0.0, 1.0 - cdf)

# 1. Load dataset (or generate poisson_model_output.csv from global_tennis_statistics_v3.csv)
try:
    out_df = pd.read_csv("poisson_model_output.csv")
except FileNotFoundError:
    df = pd.read_csv("global_tennis_statistics_v3.csv")
    df["Total_Top100"] = (
        df["ATP Top-100 Players (2000–2025)"] + df["WTA Top-100 Players (2000–2025)"]
    )
    df["ATP_Top100"] = df["ATP Top-100 Players (2000–2025)"]
    df["WTA_Top100"] = df["WTA Top-100 Players (2000–2025)"]

    df["log_pop"] = np.log(df["Avg Population (2000–2025, M)"])
    df["log_gdp_pc"] = np.log(df["Avg GDP Per Capita (2000–2025, USD)"])
    df["part_rate"] = (
        df["Tennis Players (2018)"] / (df["Avg Population (2000–2025, M)"] * 1e6)
    ) * 100

    pop_100k = df["Avg Population (2000–2025, M)"] * 10
    df["courts_pc"] = df["Tennis Courts (2018)"] / pop_100k
    df["coaches_pc"] = df["Tennis Coaches (2018)"] / pop_100k
    df["clubs_pc"] = df["Tennis Clubs (2018)"] / pop_100k

    z_cols = []
    for col in ["courts_pc", "coaches_pc", "clubs_pc"]:
        z_col = f"z_{col}"
        df[z_col] = (df[col] - df[col].mean()) / df[col].std()
        z_cols.append(z_col)
    df["infra_index"] = df[z_cols].mean(axis=1)

    pois_model = smf.glm(
        "Total_Top100 ~ log_pop + log_gdp_pc + part_rate + infra_index",
        data=df,
        family=sm.families.Poisson(),
    ).fit(cov_type="HC3")
    df["lambda_total"] = pois_model.predict(df)
    df["resid_total"] = df["Total_Top100"] - df["lambda_total"]
    df["p_value_poisson"] = [
        poisson_sf(k - 1, mu)
        for k, mu in zip(df["Total_Top100"], df["lambda_total"])
    ]

    for tour, target in [("ATP", "ATP_Top100"), ("WTA", "WTA_Top100")]:
        m = smf.glm(
            f"{target} ~ log_pop + log_gdp_pc + part_rate + infra_index",
            data=df,
            family=sm.families.Poisson(),
        ).fit(cov_type="HC3")
        df[f"lambda_{tour}"] = m.predict(df)
        df[f"resid_{tour}"] = df[target] - df[f"lambda_{tour}"]

    df.to_csv("poisson_model_output.csv", index=False)
    out_df = df

# ==============================================================================
# 1. Create Summary CSV
# ==============================================================================
summary_df = pd.DataFrame({
    "Country": out_df["Country"],
    "IOC": out_df["IOC"],
    "Actual Total": out_df["Total_Top100"],
    "Predicted Total": out_df["lambda_total"].round(2),
    "Residual": out_df["resid_total"].round(2),
    "Poisson Tail Prob": out_df["p_value_poisson"].apply(lambda p: f"{p:.4g}"),
    "Actual vs. Pred (ATP)": [
        f"{int(a)} vs. {p:.2f} ({a-p:+.2f})"
        for a, p in zip(out_df["ATP_Top100"], out_df["lambda_ATP"])
    ],
    "Actual vs. Pred (WTA)": [
        f"{int(a)} vs. {p:.2f} ({a-p:+.2f})"
        for a, p in zip(out_df["WTA_Top100"], out_df["lambda_WTA"])
    ],
    "Actual vs. Pred (Combined)": [
        f"{int(a)} vs. {p:.2f} ({a-p:+.2f})"
        for a, p in zip(out_df["Total_Top100"], out_df["lambda_total"])
    ],
})
summary_df.to_csv("poisson_summary_table.csv", index=False)

# ==============================================================================
# 2. Create Summary Data Table Figure (Czechia + 5 Benchmark Countries)
# ==============================================================================
bench_iocs = ["CZE", "USA", "ESP", "FRA", "ITA", "RUS"]
table_sub = summary_df[summary_df["IOC"].isin(bench_iocs)].copy()
table_sub["sort_key"] = table_sub["IOC"].apply(lambda x: 0 if x == "CZE" else 1)
table_sub = table_sub.sort_values(
    by=["sort_key", "Actual Total"], ascending=[True, False]
).drop(columns=["sort_key", "IOC"])

fig, ax = plt.subplots(figsize=(13, 3.2))
ax.axis("off")
tbl = ax.table(
    cellText=table_sub.values,
    colLabels=table_sub.columns,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.0, 1.65)
tbl.auto_set_column_width(col=list(range(len(table_sub.columns))))

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    if row == 0:
        cell.set_facecolor("#1F3A60")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
    elif row == 1:
        cell.set_facecolor("#FDECEA")
        cell.get_text().set_weight("bold")
        cell.get_text().set_color("#990000")
    elif row % 2 == 0:
        cell.set_facecolor("#F7F9FC")

plt.title(
    "Table 1: Poisson Regression Results — Czechia vs. Benchmark Tennis Powerhouses (2000–2025)",
    fontsize=11.5,
    fontweight="bold",
    pad=12,
)
plt.tight_layout()
plt.savefig("fig_poisson_summary_table.png", dpi=300, bbox_inches="tight")

# ==============================================================================
# 3. Actual vs. Predicted Scatter Plot with 45-Degree Line
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(
    out_df["lambda_total"],
    out_df["Total_Top100"],
    color="#4A6FE3",
    alpha=0.72,
    s=65,
    edgecolors="white",
    linewidth=0.7,
)

max_val = max(out_df["lambda_total"].max(), out_df["Total_Top100"].max()) * 1.06
ax.plot(
    [0, max_val],
    [0, max_val],
    linestyle="--",
    color="#555555",
    linewidth=1.3,
    label="45° Reference Line (Actual = Predicted)",
)

cze = out_df[out_df["IOC"] == "CZE"].iloc[0]
ax.scatter(
    cze["lambda_total"],
    cze["Total_Top100"],
    color="#D62728",
    s=150,
    zorder=5,
    edgecolors="black",
    linewidth=1.0,
    label="Czech Republic (CZE)",
)

label_iocs = [
    "CZE", "USA", "ESP", "FRA", "ITA", "RUS",
    "ARG", "AUS", "GER", "CHN", "IND", "GBR", "JPN",
]
offsets = {
    "CZE": (7, 5), "USA": (-28, -14), "ESP": (6, 4), "FRA": (6, 4),
    "ITA": (6, -10), "RUS": (6, 4), "ARG": (6, 4), "AUS": (6, -8),
    "GER": (6, -8), "CHN": (6, 4), "IND": (6, 4), "GBR": (6, -10), "JPN": (6, -8),
}
for _, r in out_df[out_df["IOC"].isin(label_iocs)].iterrows():
    is_cze = r["IOC"] == "CZE"
    dx, dy = offsets.get(r["IOC"], (5, 5))
    ax.annotate(
        r["IOC"],
        (r["lambda_total"], r["Total_Top100"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=9.5,
        fontweight="bold" if is_cze else "normal",
        color="#D62728" if is_cze else "#222222",
    )

ax.text(
    max_val * 0.18,
    max_val * 0.78,
    "Overperformers\n(Actual > Predicted)",
    fontsize=10,
    color="#2B5B84",
    style="italic",
)
ax.text(
    max_val * 0.62,
    max_val * 0.22,
    "Underperformers\n(Actual < Predicted)",
    fontsize=10,
    color="#777777",
    style="italic",
)

ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.set_xlabel("Predicted Number of Top-100 Players (Poisson Model)", fontsize=11)
ax.set_ylabel("Actual Unique Top-100 Players (2000–2025)", fontsize=11)
ax.set_title(
    "Actual vs. Predicted Elite Tennis Player Production (2000–2025)",
    fontsize=13,
    fontweight="bold",
)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="upper left", frameon=True)
plt.tight_layout()
plt.savefig("fig_actual_vs_predicted.png", dpi=300)

# ==============================================================================
# 4. Diverging Residual Bar Chart (Centered at 0)
# ==============================================================================
selected_iocs = [
    "USA", "RUS", "ESP", "FRA", "ARG", "CZE", "ITA",
    "AUS", "GER", "GBR", "JPN", "CHN", "IND", "CAN", "NZL", "IRL",
]
res_df = out_df[out_df["IOC"].isin(selected_iocs)].sort_values(
    "resid_total", ascending=True
)

fig, ax = plt.subplots(figsize=(10, 7.5))
colors = [
    "#D62728" if ioc == "CZE" else ("#2E7D32" if val > 0 else "#B0B8C4")
    for ioc, val in zip(res_df["IOC"], res_df["resid_total"])
]

bars = ax.barh(res_df["Country"], res_df["resid_total"], color=colors, height=0.7)
ax.axvline(0, color="black", linewidth=1.2)

max_abs = max(abs(res_df["resid_total"].min()), abs(res_df["resid_total"].max())) * 1.18
ax.set_xlim(-max_abs, max_abs)

for bar, val, ioc in zip(bars, res_df["resid_total"], res_df["IOC"]):
    offset = 1.5 if val >= 0 else -1.5
    ha = "left" if val >= 0 else "right"
    ax.text(
        val + offset,
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.1f}",
        va="center",
        ha=ha,
        fontsize=9,
        fontweight="bold" if ioc == "CZE" else "normal",
        color="#D62728" if ioc == "CZE" else "#222222",
    )

ax.set_xlabel(
    "Model Residual: Actual Minus Predicted Top-100 Players ($Y_c - \\hat{\\lambda}_c$)",
    fontsize=11,
)
ax.set_ylabel("Country", fontsize=11)
ax.set_title(
    "Poisson Model Residuals: Overperformers (+) vs. Underperformers (−)",
    fontsize=13,
    fontweight="bold",
)
ax.grid(axis="x", linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig("fig_poisson_residuals.png", dpi=300)
