import sys
print("ACTIVE PYTHON:", sys.executable)

import pandas as pd
import wbgapi as wb

import pandas as pd
import wbgapi as wb

# Map your dataset's country names to ISO-3 codes
iso3_codes = [
    "USA", "CHN", "DEU", "FRA", "GBR", "JPN", "IND", "RUS", "ESP", "BRA",
    "CAN", "AUS", "ITA", "ARG", "NLD", "MEX", "CHE", "KOR", "CZE", "AUT",
    "SWE", "NZL", "POL", "BEL", "COL", "HKG", "IDN", "HUN", "ISR", "PRT",
    "FIN", "IRL", "GRC", "NOR", "ROU", "DNK", "HRV", "SVK", "SRB"
]

# Fetch GDP (current US$) for 2000–2025
gdp_df = wb.data.DataFrame("NY.GDP.MKTP.CD", economy=iso3_codes, time=range(2000, 2026))

# Calculate 2000–2025 average GDP (in Billions) to match your "Avg Population" column
gdp_df["Avg_GDP_2000_2025_B"] = gdp_df.mean(axis=1) / 1e9
print(gdp_df[["Avg_GDP_2000_2025_B"]].head())
gdp_df.to_csv("gdp_by_country_2000_2025.csv", index=False)
