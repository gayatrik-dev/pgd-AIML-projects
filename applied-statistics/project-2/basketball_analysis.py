# basketball_analysis_kaggle.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def load_data(filepath):
    df = pd.read_csv("Basketball.csv")
    return df


def clean_data(df):
    # Initial look
    print("Initial shape:", df.shape)
    print("Null counts:\n", df.isnull().sum())
    print("Data types:\n", df.dtypes)

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Convert numeric columns that might have wrong types
    numeric_cols = [
        "Tournament",
        "Score",
        "PlayedGames",
        "WonGames",
        "DrawnGames",
        "LostGames",
        "BasketScored",
        "BasketGiven",
        "TournamentChampion",
        "Runner-up",
        "TeamLaunch",
        "HighestPositionHeld",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean nulls if any (you can decide whether to drop or impute)
    # Here: Drop rows with crucial nulls
    df = df.dropna(
        subset=[
            "Team",
            "Tournament",
            "PlayedGames",
            "WonGames",
            "BasketScored",
            "BasketGiven",
        ]
    )

    # Reset index
    df = df.reset_index(drop=True)

    print("After cleaning shape:", df.shape)
    return df


def feature_engineering(df):
    # Basket Difference
    df["BasketDifference"] = df["BasketScored"] - df["BasketGiven"]

    # Win Rate
    df["WinRate"] = df["WonGames"] / df["PlayedGames"]

    # Loss Rate
    df["LossRate"] = df["LostGames"] / df["PlayedGames"]

    # Score per Game
    df["ScorePerGame"] = df["Score"] / df["PlayedGames"]

    # Experience / Age of team
    # Assuming current year 2025
    df["ExperienceYears"] = 2025 - df["TeamLaunch"]

    # Top 2 finishes
    df["Top2Finishes"] = df["TournamentChampion"] + df["Runner-up"]
    df["Top2Rate"] = df["Top2Finishes"] / df["Tournament"]

    return df


def univariate_analysis(df):
    # Histograms for numerical features
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    plt.figure(figsize=(15, 10))
    df[num_cols].hist(bins=20)
    plt.suptitle("Univariate Distributions of Numeric Features", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Top teams by Win Rate
    top = df.sort_values("WinRate", ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top, x="WinRate", y="Team", palette="viridis")
    plt.title("Top 10 Teams by Win Rate")
    plt.xlabel("Win Rate")
    plt.ylabel("Team")
    plt.show()


def bivariate_analysis(df):
    # Win Rate vs Basket Difference
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="BasketDifference",
        y="WinRate",
        hue="ExperienceYears",
        size="Top2Finishes",
        alpha=0.7,
    )
    plt.title("Win Rate vs Basket Difference (colored by Team Age)")
    plt.xlabel("Basket Difference")
    plt.ylabel("Win Rate")
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=np.number).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Numeric Features")
    plt.show()


def multivariate_interactive(df):
    fig = px.scatter(
        df,
        x="ExperienceYears",
        y="ScorePerGame",
        size="Top2Finishes",
        color="WinRate",
        hover_name="Team",
        title="Team: Experience vs Score per Game vs Top 2 Finishes vs Win Rate",
        size_max=40,
    )
    fig.show()


def insights_and_recommendations(df):
    print("\n--- Key Insights ---")
    best_win = df.loc[df["WinRate"].idxmax()]
    print(
        f"Best performing team by Win Rate: {best_win['Team']}, WinRate = {best_win['WinRate']:.2f}"
    )

    worst_win = df.loc[df["WinRate"].idxmin()]
    print(
        f"Worst performing team by Win Rate: {worst_win['Team']}, WinRate = {worst_win['WinRate']:.2f}"
    )

    most_champs = df.loc[df["TournamentChampion"].idxmax()]
    print(
        f"Most championships: {most_champs['Team']}, Count = {most_champs['TournamentChampion']}"
    )

    highest_score_per_game = df.loc[df["ScorePerGame"].idxmax()]
    print(
        f"Team with highest Score per Game: {highest_score_per_game['Team']}, ScorePerGame = {highest_score_per_game['ScorePerGame']:.2f}"
    )

    oldest_team = df.loc[df["TeamLaunch"].idxmin()]
    print(
        f"Oldest team (earliest launch year): {oldest_team['Team']}, Launch Year = {int(oldest_team['TeamLaunch'])}"
    )

    high_diff = df.loc[df["BasketDifference"].idxmax()]
    print(
        f"Team with largest positive Basket Difference: {high_diff['Team']}, Difference = {high_diff['BasketDifference']}"
    )


def suggestions_data_quality():
    print("\n--- Suggestions: 5Vs of Data Quality ---")
    print(
        "1. Volume: Collect more frequent data points — e.g., per-game stats (rebounds, steals, turnovers), not only per tournament aggregates."
    )
    print(
        "2. Variety: Include contextual variables (home vs away games, player injuries, roster strength, coaching staff changes)."
    )
    print(
        "3. Velocity: Data should be updated in near real-time after each match or event to allow dashboards and rapid analytics."
    )
    print(
        "4. Veracity: Cross-check match results, baskets, scores from multiple sources; ensure consistency in definitions (what counts as ‘draw’, what counts as ‘played games’)."
    )
    print(
        "5. Value: Expand metrics to include fan engagement, sponsorship revenue, media reach to better understand team value beyond just performance."
    )


def main():
    # 1. Load
    filepath = "Basketball.csv"  # adjust path as needed
    df = load_data(filepath)

    # 2. Clean
    df = clean_data(df)

    # 3. Feature engineering
    df = feature_engineering(df)

    # 4. EDA
    univariate_analysis(df)
    bivariate_analysis(df)
    multivariate_interactive(df)

    # 5. Insights
    insights_and_recommendations(df)

    # 6. Suggestions for data quality
    suggestions_data_quality()


if __name__ == "__main__":
    main()
