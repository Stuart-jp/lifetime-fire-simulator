import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random

import io

# Inline CSV data (keep this small — just 100–200 rows is fine)
CSV_DATA = """
Year,Return
1872,10.56%
1873,-1.05%
1874,16.73%
1875,8.65%
1876,-10.11%
1877,10.85%
1878,30.27%
1879,25.86%
1880,26.51%
1881,0.75%
1882,4.52%
1883,4.86%
1884,-3.00%
1885,28.56%
1886,18.67%
1887,-8.09%
1888,2.14%
1889,14.49%
1890,-10.88%
1891,29.00%
1892,4.84%
1893,-8.81%
1894,10.09%
1895,2.04%
1896,3.48%
1897,17.10%
1898,21.69%
1899,-5.80%
1900,23.43%
1901,14.57%
1902,-1.76%
1903,-9.31%
1904,25.61%
1905,19.77%
1906,1.42%
1907,-27.85%
1908,40.36%
1909,7.72%
1910,-0.10%
1911,8.15%
1912,0.77%
1913,-11.94%
1914,-4.23%
1915,33.02%
1916,-3.29%
1917,-36.56%
1918,4.73%
1919,4.87%
1920,-20.52%
1921,29.12%
1922,30.16%
1923,1.03%
1924,26.01%
1925,24.72%
1926,15.21%
1927,38.72%
1928,39.60%
1929,-4.76%
1930,-18.71%
1931,-35.48%
1932,-0.91%
1933,52.94%
1934,-4.21%
1935,42.66%
1936,33.73%
1937,-33.68%
1938,26.10%
1939,1.92%
1940,-10.20%
1941,-18.74%
1942,7.40%
1943,23.23%
1944,17.22%
1945,34.94%
1946,-22.96%
1947,-3.98%
1948,3.75%
1949,18.90%
1950,24.68%
1951,17.38%
1952,17.42%
1953,-1.79%
1954,53.23%
1955,30.83%
1956,3.46%
1957,-13.31%
1958,40.72%
1959,9.99%
1960,-0.91%
1961,25.93%
1962,-9.95%
1963,20.65%
1964,15.17%
1965,10.22%
1966,-13.07%
1967,20.22%
1968,5.96%
1969,-13.78%
1970,-1.55%
1971,10.65%
1972,14.98%
1973,-21.61%
1974,-34.49%
1975,28.26%
1976,17.80%
1977,-13.27%
1978,-2.44%
1979,4.28%
1980,17.51%
1981,-12.82%
1982,16.94%
1983,17.90%
1984,2.05%
1985,26.73%
1986,17.25%
1987,0.62%
1988,11.38%
1989,25.50%
1990,-8.81%
1991,26.45%
1992,4.56%
1993,7.07%
1994,-1.35%
1995,33.96%
1996,18.86%
1997,31.05%
1998,26.45%
1999,17.83%
2000,-12.07%
2001,-13.18%
2002,-23.87%
2003,26.16%
2004,7.28%
2005,1.33%
2006,12.78%
2007,1.25%
2008,-36.99%
2009,23.42%
2010,13.29%
2011,-0.96%
2012,13.75%
2013,30.25%
2014,12.68%
2015,0.61%
2016,9.56%
2017,19.19%
2018,-6.22%
2019,28.72%
2020,5.31%
2021,27.18%
2022,-22.28%
2023,15.29%
"""

@st.cache
def load_data():
    import io
    import pandas as pd

    df = pd.read_csv(io.StringIO(CSV_DATA.strip()))

    # Ensure column names are correct
    df.columns = [col.strip() for col in df.columns]

    # Clean percentage string → float
    df["Return"] = (
        df["Return"]
        .astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .astype(float)
        / 100  # Convert to decimal (e.g. 30.25% → 0.3025)
    )

    df["Year"] = df["Year"].astype(int)
    return df

import pandas as pd
import matplotlib.pyplot as plt

st.title("Lifetime FIRE Simulator")

# uploaded_file = st.file_uploader("Upload CSV with 'Year' and 'Return' columns", type="csv")
df = load_data()

import numpy as np

def describe_percentiles(data, label, money=False):
    # Force convert to a list, even if it's a Series or similar
    try:
        data = list(data)
    except Exception:
        return f"{label}: invalid input"

    if not data:
        return f"{label}: no data"

    if len(data) < 3:
        return f"{label}: insufficient data"

    try:
        p10 = np.percentile(data, 10)
        p50 = np.percentile(data, 50)
        p90 = np.percentile(data, 90)

        if money:
            fmt = lambda x: f"${x:,.0f}"
        else:
            fmt = lambda x: f"{x:.1f}"

        return f"{label}: 10th percentile = `{fmt(p10)}`, median = `{fmt(p50)}`, 90th percentile = `{fmt(p90)}`"
    except Exception as e:
        return f"{label}: error ({e})"

# if uploaded_file:
# df = pd.read_csv(uploaded_file)

# Clean returns
df["Return"] = df["Return"].astype(str).str.strip().str.replace("%", "", regex=False).astype(float)
df = df.sort_values("Year").reset_index(drop=True)
returns = df["Return"].tolist()


# --- Sidebar Inputs ---
st.sidebar.header("Simulation Inputs")

simulation_mode = st.sidebar.radio(
    "Simulation Mode",
    options=["Deterministic (by year)", "Monte Carlo (randomized)"],
    index=0
)

if simulation_mode == "Monte Carlo (randomized)":
    num_monte_carlo_runs = st.sidebar.slider("Number of simulations", 100, 2000, 1000, step=100)

starting_capital = st.sidebar.number_input("Starting Capital ($)", value=0, step=1000)
annual_saving = st.sidebar.number_input("Annual Saving While Working ($)", value=10000, step=1000)
annual_raise_pct = st.sidebar.number_input("Annual Savings Increase (% per year)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
# withdrawal = st.sidebar.number_input("Annual Withdrawal in Retirement ($)", value=40000, step=1000)
target = st.sidebar.number_input("Portfolio Target to Retire ($)", value=1_000_000, step=10000)

early_withdrawal = st.sidebar.number_input("Annual withdrawal (early retirement)", value=40000, step=1000)
late_withdrawal = st.sidebar.number_input("Annual withdrawal (late retirement)", value=30000, step=1000)
cutover_years = st.sidebar.number_input("Years before late retirement starts", value=45, step=1, min_value=1)

max_years = st.sidebar.number_input(
    "Simulation Duration (Years)", value=80, min_value=1, max_value=100, step=1
)
use_log_scale = st.sidebar.checkbox("Use logarithmic scale")

simulations = []
survivors = 0
failed = 0
truncated = 0
working_years_all = []
survivor_final_balances = []
failed_retirement_years = []
retirement_wealths = []

if simulation_mode == "Deterministic (by year)":
    for start_index in range(len(df)):
        portfolio = starting_capital
        year_index = start_index  # reset each loop
        trajectory = []
        working = True
        working_years = 0
        retirement_years = 0
        reason = "truncated"  # default
        trajectory = [portfolio]  # start with initial capital
        # for the cashflow graph
        contributions = []
        withdrawals = []
        if portfolio >= target:
            working = False
        
        for t in range(max_years):
            if year_index >= len(df):
                reason = "truncated"
                break

            r = df.loc[year_index, "Return"]

            if working:
                adjusted_saving = annual_saving * ((1 + annual_raise_pct / 100) ** working_years)
                portfolio = (portfolio + adjusted_saving) * (1 + r)
                contributions.append(adjusted_saving)
                withdrawals.append(0)
                working_years += 1
                if portfolio >= target:
                    working = False
                    retirement_wealths.append(portfolio)
            else:
                # Choose correct withdrawal amount based on retirement phase
                current_withdrawal = early_withdrawal if t < cutover_years else late_withdrawal
                portfolio = (portfolio - current_withdrawal) * (1 + r)
                contributions.append(0)
                withdrawals.append(current_withdrawal)
                retirement_years += 1
                if portfolio <= 0:
                    trajectory.append(0)
                    reason = "failed"
                    break

            trajectory.append(portfolio)
            year_index += 1

        # Classify simulation outcome
        if len(trajectory) == max_years + 1:
            survivors += 1
            reason = "survived"
            survivor_final_balances.append(portfolio)
        elif reason == "failed":
            failed += 1
            failed_retirement_years.append(retirement_years)
        else:
            truncated += 1

        if working_years > 0:
            working_years_all.append(working_years)

        simulations.append((trajectory, reason, working_years, contributions, withdrawals))
else:
    # Monte Carlo logic
    for _ in range(num_monte_carlo_runs):
        portfolio = starting_capital
        trajectory = []
        working = True
        working_years = 0
        retirement_years = 0
        reason = "truncated"
        trajectory = [portfolio]  # start with initial capital
        # for the cashflow chart
        contributions = []
        withdrawals = []

        if portfolio >= target:
            working = False

        retirement_wealth = None

        for t in range(max_years):
            r = random.choice(returns)

            if working:
                adjusted_saving = annual_saving * ((1 + annual_raise_pct / 100) ** working_years)
                portfolio = (portfolio + adjusted_saving) * (1 + r)
                contributions.append(adjusted_saving)
                withdrawals.append(0)
                working_years += 1
                if portfolio >= target:
                    working = False
                    retirement_wealth = portfolio
            else:
                current_withdrawal = early_withdrawal if t < cutover_years else late_withdrawal
                portfolio = (portfolio - current_withdrawal) * (1 + r)
                contributions.append(0)
                withdrawals.append(current_withdrawal)
                retirement_years += 1
                if portfolio <= 0:
                    trajectory.append(0)
                    reason = "failed"
                    break

            trajectory.append(portfolio)

        if len(trajectory) == max_years + 1:
            survivors += 1
            reason = "survived"
            survivor_final_balances.append(portfolio)
        elif reason == "failed":
            failed += 1
            failed_retirement_years.append(retirement_years)
        else:
            truncated += 1

        if working_years > 0:
            working_years_all.append(working_years)
        if retirement_wealth:
            retirement_wealths.append(retirement_wealth)

        simulations.append((trajectory, reason, working_years, contributions, withdrawals))


# --- Headline Stats ---
total = len(simulations)
full_sims = survivors + failed
avg_work_years = sum(working_years_all) / len(working_years_all) if working_years_all else 0


if full_sims > 0:
   fail_pct = failed / full_sims * 100
   survive_pct = survivors / full_sims * 100
   st.subheader(f"Success rate: **{survive_pct:.1f}%**")
   st.markdown(f"Ran **{total}** simulations and **{full_sims}** had enough data to simulate your full career (the rest reached the present!)")
   st.markdown(f"Of the full simulations **{fail_pct:.1f}%** ran out of money and **{survive_pct:.1f}%** survived the entire period")
   if retirement_wealths:
       avg_retirement_wealth = sum(retirement_wealths) / len(retirement_wealths)
       label_text = f"Average wealth at retirement: ${avg_retirement_wealth:,.0f}"
       with st.expander(label_text):
           st.markdown("This is the average portfolio value at the exact moment retirement began. It may exceed your target due to strong market years just before retirement.")
else:
   st.markdown("Not enough data to run any full-length simulations")

truncated_pct = truncated / total * 100 if total > 0 else 0

if working_years_all:
    st.markdown(describe_percentiles(working_years_all, "Time to retire"))

if failed_retirement_years:
    st.markdown(describe_percentiles(failed_retirement_years, "How long the failed scenarios lasted"))

# --- Identify full-length simulations for percentile selection ---
full_simulations = []
full_indices = []

for i, (trajectory, reason, working_years, contributions, withdrawals) in enumerate(simulations):
    if reason in ("survived", "failed"):
        if reason == "survived":
            score = trajectory[-1]  # final value
        else:  # failed
            score = len(trajectory) - working_years  # retirement length
        full_simulations.append((i, score))
        full_indices.append(i)

# Sort only full simulations by score (descending = better)
sorted_full = sorted(full_simulations, key=lambda x: x[1], reverse=True)

# Extract the percentile-ranked original indices
n = len(sorted_full)
highlight_indices = set()
if n >= 1:
    highlight_indices.add(sorted_full[int(n * 0.10)][0])
if n >= 1:
    highlight_indices.add(sorted_full[int(n * 0.50)][0])
if n >= 1:
    highlight_indices.add(sorted_full[int(n * 0.90)][0])

# --- Plot ---
import matplotlib.ticker as mtick

st.subheader("Portfolio Trajectories")

fig, ax = plt.subplots(figsize=(10, 6))
any_lines = False
num_lines = len(simulations)

for i, (sim, reason, working_years, contributions, withdrawals) in enumerate(simulations):
    if not sim or len(sim) < 5:
        continue

    # --- Truncate simulation if using log scale ---
    if use_log_scale and working_years < len(sim):
        # Only apply truncation after retirement begins
        retirement_sim = sim[working_years:]
        cutoff_offset = next((i for i, v in enumerate(retirement_sim) if v < 1000), len(retirement_sim))
        cutoff_index = working_years + cutoff_offset
        sim = sim[:cutoff_index]

        # Ensure we still have enough data to plot
        if len(sim) < 2:
            continue

        # Adjust retirement length if truncated
        working_years = min(working_years, len(sim))

    # --- Prepare data ---
    x = list(range(len(sim)))
    sim_thousands = [v / 1000 for v in sim]

    if working_years > 0:
        x_work = x[:working_years]
        y_work = sim_thousands[:working_years]
        x_ret = x[working_years - 1:]
        y_ret = sim_thousands[working_years - 1:]
    else:
        x_work = []
        y_work = []
        x_ret = x
        y_ret = sim_thousands

    # --- Style ---
    # Scale alpha and width for clarity
    base_alpha = max(0.05, min(0.4, 40 / num_lines))  # dims as more lines are drawn
    base_linewidth = max(0.2, min(1.5, 150 / num_lines))

    if i in highlight_indices:
        alpha = 0.9
        linewidth = 2.0
    else:
        alpha = base_alpha
        linewidth = base_linewidth

    # --- Plot working phase in blue ---
    ax.plot(x_work, y_work, color="blue", linewidth=linewidth, alpha=alpha)

    # --- Plot retirement phase in green/red/gray ---
    color = (
        "green" if reason == "survived"
        else "red" if reason == "failed"
        else "gray"
    )
    ax.plot(x_ret, y_ret, color=color, linewidth=linewidth, alpha=alpha)

    # --- Label endpoint for highlighted lines ---
    if i in highlight_indices and x_ret and y_ret:
        final_value = sim[-1]
        ax.text(
            x_ret[-1], y_ret[-1],
            f"${final_value:,.0f}",
            fontsize=8,
            color=color,
            ha='left', va='center',
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2')
        )
    any_lines = True

if any_lines:
    ax.set_xlabel("Years")
    ax.set_ylabel("Portfolio Value ($000s)")
    ax.set_title("All Scenarios (10th, 50th, 90th Percentile Bold)")
    ax.grid(True)
    if use_log_scale:
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    st.pyplot(fig)
else:
    st.warning("No valid simulations to display. Try adjusting inputs.")

if use_log_scale:
    st.caption(
        "*Note: in log scale, the graph truncates simulations the year before they run out. "
        "This avoids plotting extremely small values that distort the chart.*"
    )

st.subheader("Cash Flow")

median_index = sorted_full[len(sorted_full) // 2][0]

sim_options = {
    "10th Percentile": sorted_full[int(0.1 * len(sorted_full))][0],
    "Median": median_index,
    "90th Percentile": sorted_full[int(0.9 * len(sorted_full))][0],
    "Random": "random"
}

choice = st.radio("Choose a simulation to view cash flows:", list(sim_options.keys()), index=1)

if sim_options[choice] == "random":
    import random
    selected_index = random.choice([i for i, _ in full_simulations])
else:
    selected_index = sim_options[choice]




if simulations:
    _, _, _, contributions, withdrawals = simulations[selected_index]
    # Pad both lists to full length
    # Ensure both are lists
    contributions = list(contributions)
    withdrawals = list(withdrawals)

    # Force exact length by truncating or padding
    contributions = contributions[:max_years] + [0] * (max_years - len(contributions))
    withdrawals = withdrawals[:max_years] + [0] * (max_years - len(withdrawals))
    years = list(range(max_years))
    
    fig_cf, ax_cf = plt.subplots(figsize=(10, 4))
    
    # Plot contributions as positive bars
    ax_cf.bar(years, contributions, label="Contributions", color="steelblue")

    # Plot withdrawals as negative bars
    ax_cf.bar(years, withdrawals, label="Withdrawals", color="indianred")

    ax_cf.set_xlim(0, max_years - 1)

    # Add baseline and style
    ax_cf.axhline(0, color='black', linewidth=0.5)
    ax_cf.set_xlabel("Year")
    ax_cf.set_ylabel("Cash Flow ($)")
    ax_cf.set_title("Annual Contributions and Withdrawals (Selected Simulation)")
    ax_cf.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax_cf.legend()

    # Format y-axis as currency
    ax_cf.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${int(x):,}"))

    ax_cf.set_xlabel("Year")
    ax_cf.set_ylabel("Cash Flow ($)")
    ax_cf.set_title("Annual Contributions and Withdrawals")
    ax_cf.legend()
    ax_cf.grid(True)
    st.pyplot(fig_cf)
    
with st.expander("Methodology"):
    st.markdown("""
    The most well known retirement simulation is the famous Trinity study.
    This study investigated what portion of a portfolio at retirement could be sustainably withdrawn every year, adjusting annually for inflation.
    This approach used historical data to find the famous "4% rule": withdrawing 4% of your retirement savings in your first year of retirement and adjusting that amount for inflation each subsequent year.

    This tool intends to answer a question: are we underweighting retirements starting in boom years and overweighting bust years?
    We know retirees are unlikely to retire immediately after a market crash, put simply: if you had enough money to retire in 2009, you had enough to retire in 2007.
    To achieve this we model an entire "lifetime", starting from any arbitrary point, investing some chosen amount yearly, and retiring at your chosen portfolio size.
    The simulation then forecasts retirement by drawing an income from the portfolio until the end of the lifetime, or failure.

    The tool will then report the number of simulations run, and the success rate of those simulations. 
    
    **Simulation Models:**
    - The default is to use real historical data. This can generate only a limited quantity of simulations.
    - The Monte Carlo mode will randomly sample the historical data to generate arbitrary synthetic sequences.

    It is recommended to use log scale for Monte Carlo simulations, to avoid extreme positive outliers obscuring the details.

    **Simulation Details:**
    - Portfolio starts with user-defined capital and accumulates savings and investement returns each year.
    - Once it reaches the target, withdrawals begin.
    - Withdrawals can change after a specified number of retirement years, this simulates fixed income from pensions.
    - Note the years in "Years before late retirement starts" is based on the full simulation length counting both "working" and "retired" years.
    - The simulation stops after the chosen duration (success), if funds are depleted (failure), or if data runs out.
    
    This tool uses historical real global index returns dating back to 1872. By using real returns, there is no need to adjust for inflation anywhere.

    The median time to retire will be different from the median in the graphed simulations. The medians in the graph are the median by *final balance*, time to retire uses the median by *time to retire*.
    """)
