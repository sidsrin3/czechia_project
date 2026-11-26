
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
df = pd.read_csv("players_2000_2025_combined.csv")

# 2. Average population (in millions)
populations = {
    "USA": 316.26,
    "FRA": 62.60,
    "ITA": 59.04,
    "ESP": 45.59,
    "CZE": 10.49
}

# Country labels for the charts
country_names = {
    "CZE": "Czechia",
    "USA": "United States",
    "FRA": "France",
    "ITA": "Italy",
    "ESP": "Spain"
}

countries = ["CZE", "USA", "FRA", "ITA", "ESP"]
tours = ["ATP", "WTA"]

# 3. Ensure ranking data is numeric
df["best_rank_2000_2025"] = pd.to_numeric(
    df["best_rank_2000_2025"], errors="coerce"
)

# 4. Keep players from the five countries who reached the top 100
top_100 = df[
    (df["country_ioc"].isin(countries)) &
    (df["best_rank_2000_2025"] <= 100)
].copy()

# 5. Count unique players for each tour and country
counts = {}

for tour in tours:
    tour_data = top_100[top_100["tour"] == tour]

    counts[tour] = (
        tour_data.groupby("country_ioc")["player_id"]
        .nunique()
        .reindex(countries, fill_value=0)
    )

# 6. Calculate population-adjusted rates and percentages
population_series = pd.Series(populations)

rates = {}
percentages = {}

for tour in tours:
    # Top-100 players per million residents
    rates[tour] = counts[tour] / population_series

    # Percentage of the total population
    percentages[tour] = (
        counts[tour] / (population_series * 1_000_000)
    ) * 100

# 7. Function to create grouped bar charts
def make_chart(data, title, ylabel, percentage=False):

    x = np.arange(len(countries))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    # ATP bars
    bars_atp = ax.bar(
        x - width / 2,
        data["ATP"],
        width,
        label="ATP"
    )

    # WTA bars
    bars_wta = ax.bar(
        x + width / 2,
        data["WTA"],
        width,
        label="WTA"
    )

    # Add value labels above each bar
    for bars, tour in [(bars_atp, "ATP"), (bars_wta, "WTA")]:
        for i, bar in enumerate(bars):
            value = data[tour].iloc[i]

            if percentage:
                label = f"{value:.6f}%"
            else:
                label = f"{value:.2f}"

            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2,
                 bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8
            )

    ax.set_title(title)
    ax.set_xlabel("Country")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([country_names[c] for c in countries])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Leave room above bars for labels
    ax.set_ylim(0, max(data[t].max() for t in tours) * 1.18)

    plt.tight_layout()
    plt.show()


# 8. First chart: Top-100 players per million residents
rate_data = pd.DataFrame(rates, index=countries)

make_chart(
    rate_data,
    "ATP and WTA Top-100 Players per Million Residents (2000–2025)",
    "Top-100 players per million residents"
)

# 9. Second chart: Top-100 players as a percentage of population
percentage_data = pd.DataFrame(percentages, index=countries)

make_chart(
    percentage_data,
    "ATP and WTA Top-100 Players as a Percentage of Population (2000–2025)",
    "Percentage of population",
    percentage=True
)

# 10. Print the underlying counts and rates
for tour in tours:
    print(f"\n{tour} RESULTS")
    results = pd.DataFrame({
        "Top-100 Players": counts[tour],
        "Population (millions)": population_series,
        "Players per Million": rates[tour],
        "Percentage of Population": percentages[tour]
    })
    print(results)
