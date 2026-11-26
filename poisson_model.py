import numpy as np
import pandas as pd
from scipy.stats import poisson
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 1. Load the merged dataset
df = pd.read_csv("global_tennis_statistics_v3.csv")

# 2. Construct target counts and per-capita predictors
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

# Composite Infrastructure Index (courts, coaches, and clubs per 100k residents)
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

# 3. Fit Poisson Regression: log(lambda_c) = B0 + B1*log(pop) + B2*log(gdp_pc) + B3*part + B4*infra
formula = "Total_Top100 ~ log_pop + log_gdp_pc + part_rate + infra_index"
pois_model = smf.glm(formula, data=df, family=sm.families.Poisson()).fit(
    cov_type="HC3"
)
print(pois_model.summary())

# 4. Predict lambda_c, compute residuals, and evaluate Poisson PMF tail probabilities
df["lambda_total"] = pois_model.predict(df)
df["resid_total"] = df["Total_Top100"] - df["lambda_total"]

# Using P(X = k) = e^(-mu) * mu^k / k! to find P(X >= actual | mu = lambda_c)
df["p_value_poisson"] = poisson.sf(df["Total_Top100"] - 1, mu=df["lambda_total"])

# 5. Fit separate ATP and WTA Poisson regressions
for tour, target in [("ATP", "ATP_Top100"), ("WTA", "WTA_Top100")]:
    m = smf.glm(
        f"{target} ~ log_pop + log_gdp_pc + part_rate + infra_index",
        data=df,
        family=sm.families.Poisson(),
    ).fit(cov_type="HC3")
    df[f"lambda_{tour}"] = m.predict(df)
    df[f"resid_{tour}"] = df[target] - df[f"lambda_{tour}"]

# 6. Display benchmark comparison and save results
bench = ["CZE", "USA", "ESP", "FRA", "ITA", "RUS"]
cols_out = [
    "Country",
    "IOC",
    "Total_Top100",
    "lambda_total",
    "resid_total",
    "p_value_poisson",
    "ATP_Top100",
    "lambda_ATP",
    "WTA_Top100",
    "lambda_WTA",
]
print("\n--- Benchmark Countries: Actual vs. Predicted (2000–2025) ---")
print(df[df["IOC"].isin(bench)][cols_out].round(3).to_string(index=False))

df.to_csv("poisson_model_output.csv", index=False)
