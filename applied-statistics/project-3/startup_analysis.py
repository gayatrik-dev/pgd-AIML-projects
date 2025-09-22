import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Step 1: Read the CSV file
df = pd.read_csv("CompanyX_EU.csv")

# Step 2: Data Exploration
# A. Check data types
print("\n--- Data Types ---")
print(df.dtypes)

# B. Check for null values
print("\n--- Null Values ---")
print(df.isnull().sum())

# Step 3: Data Preprocessing & Visualization
# A. Drop rows with null values
df1 = df.dropna()

# B. Convert 'Funding' to numerical value (in millions)
df1.loc[:, "Funds_in_million"] = df1["Funding"].apply(
    lambda x: (
        float(x[1:-1]) / 1000
        if x[-1] == "K"
        else (float(x[1:-1]) * 1000 if x[-1] == "B" else float(x[1:-1]))
    )
)

# C. Box plot for Funds in million
plt.figure(figsize=(10, 6))
sns.boxplot(x=df1["Funds_in_million"])
plt.title("Boxplot of Funds in Million")
plt.xlabel("Funds (Million USD)")
plt.show()

# D. Outliers above upper fence
Q1 = df1["Funds_in_million"].quantile(0.25)
Q3 = df1["Funds_in_million"].quantile(0.75)
IQR = Q3 - Q1
upper_fence = Q3 + 1.5 * IQR
outliers = df1[df1["Funds_in_million"] > upper_fence]
print(f"\n--- Number of Outliers Above Upper Fence ---\n{outliers.shape[0]}")

# E. Frequency of OperatingState
print("\n--- Operating State Frequency ---")
print(df1["OperatingState"].value_counts())

# Step 4: Statistical Analysis
# A. Funds: Operating vs Closed
operating = df1[df1["OperatingState"] == "Operating"]["Funds_in_million"]
closed = df1[df1["OperatingState"] == "Closed"]["Funds_in_million"]

# B. Hypotheses
print("\n--- Hypothesis ---")
print(
    "H0: There is no significant difference in funds raised between Operating and Closed companies."
)
print(
    "H1: There is a significant difference in funds raised between Operating and Closed companies."
)

# C. Test for significance
t_stat, p_val = stats.ttest_ind(operating, closed, equal_var=False)
print(f"\nT-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
if p_val < 0.05:
    print("Conclusion: Reject H0 — Significant difference in funds.")
else:
    print("Conclusion: Fail to reject H0 — No significant difference.")

# D. Make a copy of the dataframe
df_copy = df1.copy()

# E. Frequency distribution of Result
print("\n--- Frequency of 'Result' ---")
print(df_copy["Result"].value_counts())

# F. Percent of Winners and Contestants still operating
winners = df_copy[df_copy["Result"] == "Winner"]
contestants = df_copy[df_copy["Result"] == "Contestant"]

winner_operating_pct = (winners["OperatingState"] == "Operating").mean() * 100
contestant_operating_pct = (contestants["OperatingState"] == "Operating").mean() * 100

print(f"\nPercentage of Winners still operating: {winner_operating_pct:.2f}%")
print(f"Percentage of Contestants still operating: {contestant_operating_pct:.2f}%")

# G. Hypothesis for proportions
print("\n--- Hypothesis for Proportions ---")
print(
    "H0: Proportion of companies still operating is the same for Winners and Contestants."
)
print(
    "H1: Proportion of companies still operating is different for Winners and Contestants."
)

# H. Two-proportion z-test
from statsmodels.stats.proportion import proportions_ztest

success = [
    (winners["OperatingState"] == "Operating").sum(),
    (contestants["OperatingState"] == "Operating").sum(),
]
nobs = [len(winners), len(contestants)]

z_stat, p_val = proportions_ztest(success, nobs)
print(f"\nZ-statistic: {z_stat:.4f}, P-value: {p_val:.4f}")
if p_val < 0.05:
    print("Conclusion: Reject H0 — Significant difference in proportions.")
else:
    print("Conclusion: Fail to reject H0 — No significant difference.")

# I. Filter Events with 'disrupt' from 2013 onwards
df_disrupt = df1[df1["Event"].str.contains("disrupt", case=False)]
df_disrupt["Year"] = df_disrupt["Event"].str.extract(r"(\d{4})").astype(float)
df_disrupt_recent = df_disrupt[df_disrupt["Year"] >= 2013]

print("\n--- Disrupt Events from 2013 Onwards ---")
print(df_disrupt_recent[["Startup", "Event", "Year"]].head())
