# statistics_assignment.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ----------------------------------------
# Question 1: Joint and Conditional Probability
# ----------------------------------------
print("\n------ Question 1 ------")
data = [[400, 100], [200, 1300]]
index = ["Planned_Yes", "Planned_No"]
columns = ["Ordered_Yes", "Ordered_No"]
table = pd.DataFrame(data, index=index, columns=columns)

total = table.values.sum()
joint_prob = table.loc["Planned_Yes", "Ordered_Yes"] / total
cond_prob = table.loc["Planned_Yes", "Ordered_Yes"] / table.loc["Planned_Yes"].sum()

print(f"1.A Joint Probability (Planned & Ordered): {joint_prob:.4f}")
print(f"1.B Conditional Probability (Ordered | Planned): {cond_prob:.4f}")

# ----------------------------------------
# Question 2: Binomial Distribution (Quality Check)
# ----------------------------------------
print("\n------ Question 2 ------")
n = 10
p = 0.05

p_0 = binom.pmf(0, n, p)
p_1 = binom.pmf(1, n, p)
p_leq_2 = binom.cdf(2, n, p)
p_gte_3 = 1 - p_leq_2

print(f"2.A P(None defective): {p_0:.4f}")
print(f"2.B P(Exactly one defective): {p_1:.4f}")
print(f"2.C P(Two or fewer defective): {p_leq_2:.4f}")
print(f"2.D P(Three or more defective): {p_gte_3:.4f}")

# ----------------------------------------
# Question 3: Poisson Distribution (Car Sales)
# ----------------------------------------
print("\n------ Question 3 ------")
λ = 3

p_some = 1 - poisson.pmf(0, λ)
p_2_to_4 = poisson.pmf(2, λ) + poisson.pmf(3, λ) + poisson.pmf(4, λ)

print(f"3.A P(Sells some cars): {p_some:.4f}")
print(f"3.B P(Sells 2 to 4 cars): {p_2_to_4:.4f}")

# 3.C Plot Cumulative Poisson Distribution
x_vals = np.arange(0, 11)
cdf_vals = poisson.cdf(x_vals, λ)

plt.figure(figsize=(8, 4))
plt.plot(x_vals, cdf_vals, marker="o", linestyle="-", color="blue")
plt.title("3.C Cumulative Poisson Distribution (λ=3)")
plt.xlabel("Number of Cars Sold per Week")
plt.ylabel("Cumulative Probability")
plt.grid(True)
plt.tight_layout()
plt.savefig("poisson_cumulative_plot.png")
plt.show()

# ----------------------------------------
# Question 4: Binomial Distribution (Order Accuracy)
# ----------------------------------------
print("\n------ Question 4 ------")
p_correct = 0.868
n_orders = 3

p_all_correct = binom.pmf(3, n_orders, p_correct)
p_none_correct = binom.pmf(0, n_orders, p_correct)
p_2_correct = binom.pmf(2, n_orders, p_correct)
p_at_least_2 = p_2_correct + p_all_correct

print(f"4.A P(All 3 orders correct): {p_all_correct:.4f}")
print(f"4.B P(None correct): {p_none_correct:.4f}")
print(f"4.C P(At least 2 correct): {p_at_least_2:.4f}")

# ----------------------------------------
# Question 5: Retail Inventory Optimization Scenario
# ----------------------------------------
print("\n------ Question 5 ------")
print("Retail Inventory Optimization using Demand Forecasting:")
print("- Problem: Overstocking leads to waste; understocking leads to lost sales.")
print("- Solution:")
print("  • Use historical sales data and statistical models like time series analysis.")
print("  • Predict demand using ARIMA, exponential smoothing, regression.")
print("  • Model uncertainty with Poisson arrivals or Monte Carlo simulations.")
print(
    "  • Outcome: Data-driven decisions on stock levels to improve efficiency and profit."
)

# Simulate historical daily sales data (180 days)
np.random.seed(42)
days = pd.date_range(start="2023-01-01", periods=180)
base_demand = 20
seasonality = 5 * np.sin(np.linspace(0, 3 * np.pi, 180))
noise = np.random.normal(0, 3, 180)
sales = np.maximum(base_demand + seasonality + noise, 0).round()

df = pd.DataFrame({"Date": days, "Sales": sales})
df.set_index("Date", inplace=True)

# Forecast next 30 days using Exponential Smoothing
model = ExponentialSmoothing(
    df["Sales"], trend="add", seasonal="add", seasonal_periods=30
)
fit = model.fit()
forecast = fit.forecast(30)

# Plot forecast
plt.figure(figsize=(12, 5))
df["Sales"].plot(label="Historical Sales")
forecast.plot(label="Forecast (Next 30 Days)", linestyle="--")
plt.title("Forecasting Demand Using Exponential Smoothing")
plt.ylabel("Units Sold")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("demand_forecast_plot.png")
plt.show()

# Monte Carlo simulation to estimate reorder point
n_simulations = 1000
lead_time_days = 5
daily_mean = forecast.mean()
std_dev = df["Sales"].std()

simulated_demand = np.random.normal(
    loc=daily_mean * lead_time_days,
    scale=std_dev * np.sqrt(lead_time_days),
    size=n_simulations,
)

reorder_point = np.percentile(simulated_demand, 95)

print("\nInventory Simulation Results:")
print(f"- Avg Daily Forecasted Demand: {daily_mean:.2f}")
print(f"- 95% Reorder Point over {lead_time_days} days: {reorder_point:.2f}")
print(f"- Suggested Safety Stock: {reorder_point - (daily_mean * lead_time_days):.2f}")

# Poisson distribution for customer arrivals
λ_arrival = 25
x = np.arange(0, 51)
pmf = poisson.pmf(x, λ_arrival)

plt.figure(figsize=(10, 4))
plt.bar(x, pmf, color="skyblue")
plt.title("Poisson Distribution of Customer Arrivals (λ = 25)")
plt.xlabel("Customers per Day")
plt.ylabel("Probability")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("customer_poisson_arrivals.png")
plt.show()
