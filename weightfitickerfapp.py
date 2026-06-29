# Import Streamlit to build the interactive dashboard.
import streamlit as st

# Import NumPy for random simulation, arrays, and numerical calculations.
import numpy as np

# Import pandas for tables, returns, and time-series calculations.
import pandas as pd

# Import yfinance to download historical market prices from Yahoo Finance.
import yfinance as yf

# Import Plotly Graph Objects to create professional interactive charts.
import plotly.graph_objects as go


# Set the Streamlit page title, browser tab name, and wide layout.
st.set_page_config(
    page_title="Hessian AI Quant Portfolio Simulator",
    layout="wide"
)


# Add custom CSS to make the dashboard look more professional.
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 2rem;
        }

        .small-note {
            font-size: 0.90rem;
            color: #9aa4b2;
        }

        .section-box {
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid #30363d;
            background-color: #111827;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Show the main dashboard title.
st.title("Hessian AI: Portfolio Turnover, Friction & Sharpe Simulator")


# Show a professional subtitle under the title.
st.caption(
    "A quant dashboard showing how portfolio turnover, slippage, and brokerage costs reduce real-world Sharpe ratio."
)


# Give a truthful note so interviewers do not attack the word AI.
st.info(
    "Important: this app uses a stochastic Monte Carlo allocation engine. "
    "It is not a trained prediction model yet. In an interview, call this a quant simulation engine, not pure AI."
)


# Create a sidebar header for user controls.
st.sidebar.header("Dashboard Controls")


# Ask the user to enter global ticker symbols.
ticker_input = st.sidebar.text_input(
    "Enter tickers separated by commas",
    "AAPL, RELIANCE.NS, SHOP.TO, NVDA"
)


# Let the user choose how much historical price data to download.
price_period = st.sidebar.selectbox(
    "Historical data period",
    ["6mo", "1y", "2y", "5y"],
    index=1
)


# Let the user set the annual risk-free rate used in Sharpe ratio.
risk_free_rate = st.sidebar.number_input(
    "Annual risk-free rate",
    min_value=0.00,
    max_value=0.20,
    value=0.05,
    step=0.005,
    format="%.3f"
)


# Let the user control how aggressively the allocation engine changes weights.
ai_aggressiveness = st.sidebar.slider(
    "Allocation aggressiveness: daily weight shock (%)",
    min_value=1.0,
    max_value=20.0,
    value=5.0,
    step=0.5
)


# Let the user set slippage cost in basis points.
slippage_bps = st.sidebar.slider(
    "Slippage / market impact (bps)",
    min_value=0,
    max_value=50,
    value=10,
    step=1
)


# Let the user set brokerage and exchange fee cost in basis points.
brokerage_bps = st.sidebar.slider(
    "Brokerage / exchange fees (bps)",
    min_value=0,
    max_value=50,
    value=5,
    step=1
)


# Let the user control the random seed for reproducible simulation results.
seed_value = st.sidebar.number_input(
    "Random seed",
    min_value=1,
    max_value=999999,
    value=42,
    step=1
)


# Split the user's comma-separated ticker input into a clean Python list.
tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]


# Remove duplicate tickers while keeping the original order.
tickers = list(dict.fromkeys(tickers))


# Stop the dashboard if the user enters fewer than two tickers.
if len(tickers) < 2:
    st.warning("Please enter at least two valid tickers.")
    st.stop()


# Cache the market data download so Streamlit does not call Yahoo Finance on every small interaction.
@st.cache_data(ttl=3600, show_spinner=True)
def load_close_prices(ticker_tuple, period):
    # Convert the immutable ticker tuple back into a normal list.
    ticker_list = list(ticker_tuple)

    # Download adjusted historical prices from Yahoo Finance.
    raw_data = yf.download(
        tickers=ticker_list,
        period=period,
        auto_adjust=True,
        group_by="column",
        progress=False
    )

    # If Yahoo Finance returns no data, return an empty DataFrame.
    if raw_data.empty:
        return pd.DataFrame()

    # Check whether yfinance returned a MultiIndex column structure.
    if isinstance(raw_data.columns, pd.MultiIndex):
        # Read the first column level names.
        level_zero = raw_data.columns.get_level_values(0)

        # Read the second column level names.
        level_one = raw_data.columns.get_level_values(1)

        # If "Close" is on the first level, select the Close price table.
        if "Close" in level_zero:
            close_prices = raw_data.xs("Close", axis=1, level=0)

        # If "Close" is on the second level, select the Close price table.
        elif "Close" in level_one:
            close_prices = raw_data.xs("Close", axis=1, level=1)

        # If Close prices cannot be found, return an empty DataFrame.
        else:
            return pd.DataFrame()

    # Handle the simpler non-MultiIndex case.
    else:
        # If the data contains a Close column, select it.
        if "Close" in raw_data.columns:
            close_prices = raw_data[["Close"]]

        # If no Close column exists, return an empty DataFrame.
        else:
            return pd.DataFrame()

    # Convert all column names to strings.
    close_prices.columns = [str(column).upper() for column in close_prices.columns]

    # Remove assets where almost all price values are missing.
    useful_columns = close_prices.columns[close_prices.notna().sum() > 20]

    # Keep only useful assets.
    close_prices = close_prices.loc[:, useful_columns]

    # Forward-fill missing prices caused by holidays or exchange differences.
    close_prices = close_prices.ffill()

    # Drop rows that still contain missing prices.
    close_prices = close_prices.dropna(how="any")

    # Return the cleaned price table.
    return close_prices


# Convert the ticker list into a tuple because cached functions prefer stable inputs.
ticker_tuple = tuple(tickers)


# Load close prices using the cached data function.
prices = load_close_prices(ticker_tuple, price_period)


# Stop the app if fewer than two tickers were successfully downloaded.
if prices.empty or prices.shape[1] < 2:
    st.error("Could not load enough valid assets. Try changing tickers or using a longer period.")
    st.stop()


# Calculate simple daily percentage returns from close prices.
returns = prices.pct_change().dropna(how="any")


# Stop the app if return history is too short.
if len(returns) < 10:
    st.error("Not enough return data after cleaning. Try a longer period or different tickers.")
    st.stop()


# Get the successfully loaded ticker symbols.
valid_tickers = list(prices.columns)


# Show which assets were successfully loaded.
st.success(f"Loaded {len(valid_tickers)} assets: {', '.join(valid_tickers)}")


# Warn the user about global tickers and currency conversion.
st.warning(
    "Global ticker warning: this prototype uses local-market returns. "
    "It does not yet convert all assets into one base currency such as CAD or USD."
)


# Let the user choose how many days to simulate after data is loaded.
days_to_simulate = st.sidebar.slider(
    "Days to simulate",
    min_value=5,
    max_value=len(returns),
    value=min(210, len(returns)),
    step=1
)


# Keep only the most recent return window chosen by the user.
simulation_returns = returns.tail(days_to_simulate)


# Add slippage and brokerage together to get total transaction cost in bps.
total_cost_bps = slippage_bps + brokerage_bps


# Create a helper function that forces weights to be long-only and sum to 1.
def normalize_long_only_weights(raw_weights):
    # Clip negative weights to zero because this prototype assumes no short-selling.
    clipped_weights = np.clip(raw_weights, 0, None)

    # Calculate the sum of clipped weights.
    weight_sum = np.sum(clipped_weights)

    # If all weights are zero, replace them with equal weights.
    if weight_sum == 0:
        return np.repeat(1.0 / len(raw_weights), len(raw_weights))

    # Normalize the weights so they sum to exactly 1.
    return clipped_weights / weight_sum


# Create a helper function to calculate portfolio performance metrics.
def calculate_performance_metrics(daily_returns, annual_risk_free_rate):
    # If the return series is empty, return missing values.
    if daily_returns.empty:
        return {
            "cumulative_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan
        }

    # Calculate the cumulative return over the simulation period.
    cumulative_return = (1 + daily_returns).prod() - 1

    # Calculate the number of trading days in the simulation.
    number_of_days = len(daily_returns)

    # Annualize the cumulative return using 252 trading days.
    annualized_return = (1 + cumulative_return) ** (252 / number_of_days) - 1

    # Calculate daily volatility.
    daily_volatility = daily_returns.std(ddof=1)

    # Annualize daily volatility.
    annualized_volatility = daily_volatility * np.sqrt(252)

    # Convert annual risk-free rate into daily risk-free rate.
    daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1 / 252) - 1

    # Calculate Sharpe ratio only if volatility is positive.
    if daily_volatility > 0:
        sharpe = ((daily_returns - daily_risk_free_rate).mean() / daily_volatility) * np.sqrt(252)

    # If volatility is zero, Sharpe ratio is undefined.
    else:
        sharpe = np.nan

    # Convert daily returns into a cumulative equity curve.
    equity_curve = (1 + daily_returns).cumprod()

    # Calculate running maximum portfolio value.
    running_maximum = equity_curve.cummax()

    # Calculate drawdown from the running maximum.
    drawdown_curve = equity_curve / running_maximum - 1

    # Calculate maximum drawdown.
    max_drawdown = drawdown_curve.min()

    # Return all performance metrics in a dictionary.
    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown
    }


# Create the main portfolio simulation function.
def simulate_portfolio(sim_returns, aggressiveness_percent, cost_bps, seed):
    # Create a random number generator using the chosen seed.
    rng = np.random.default_rng(seed)

    # Count how many assets are in the portfolio.
    number_of_assets = sim_returns.shape[1]

    # Store the asset names.
    asset_names = list(sim_returns.columns)

    # Start the portfolio with equal weights.
    current_weights = np.repeat(1.0 / number_of_assets, number_of_assets)

    # Create an empty list to store starting weights each day.
    start_weight_history = []

    # Create an empty list to store target weights after each rebalance.
    target_weight_history = []

    # Create an empty list to store gross daily portfolio returns.
    gross_return_history = []

    # Create an empty list to store net daily portfolio returns after costs.
    net_return_history = []

    # Create an empty list to store daily turnover.
    daily_turnover_history = []

    # Create an empty list to store daily transaction costs.
    daily_cost_history = []

    # Loop through each date and row of asset returns.
    for current_date, asset_return_row in sim_returns.iterrows():
        # Save the portfolio weights used at the start of the trading day.
        start_weight_history.append(current_weights.copy())

        # Convert the row of asset returns into a NumPy array.
        asset_returns_today = asset_return_row.values.astype(float)

        # Calculate today's gross portfolio return using starting weights.
        gross_return_today = np.dot(current_weights, asset_returns_today)

        # Generate random weight shocks from a normal distribution.
        random_weight_shift = rng.normal(
            loc=0.0,
            scale=aggressiveness_percent / 100.0,
            size=number_of_assets
        )

        # Create raw target weights after applying the random shift.
        raw_target_weights = current_weights + random_weight_shift

        # Convert raw target weights into valid long-only weights that sum to 1.
        target_weights = normalize_long_only_weights(raw_target_weights)

        # Calculate one-way portfolio turnover.
        turnover_today = 0.5 * np.sum(np.abs(target_weights - current_weights))

        # Convert bps cost into decimal cost and multiply by turnover.
        cost_today = turnover_today * (cost_bps / 10000.0)

        # Subtract trading cost from gross return to get net return.
        net_return_today = gross_return_today - cost_today

        # Save the target weights after rebalancing.
        target_weight_history.append(target_weights.copy())

        # Save today's gross return.
        gross_return_history.append(gross_return_today)

        # Save today's net return.
        net_return_history.append(net_return_today)

        # Save today's turnover.
        daily_turnover_history.append(turnover_today)

        # Save today's cost.
        daily_cost_history.append(cost_today)

        # Move the portfolio to the target weights for the next day.
        current_weights = target_weights

    # Store the simulation dates.
    simulation_dates = sim_returns.index

    # Convert gross returns into a pandas Series.
    gross_daily_returns = pd.Series(
        gross_return_history,
        index=simulation_dates,
        name="Gross Daily Return"
    )

    # Convert net returns into a pandas Series.
    net_daily_returns = pd.Series(
        net_return_history,
        index=simulation_dates,
        name="Net Daily Return"
    )

    # Convert daily turnover into a pandas Series.
    daily_turnover = pd.Series(
        daily_turnover_history,
        index=simulation_dates,
        name="Daily Turnover"
    )

    # Convert daily costs into a pandas Series.
    daily_costs = pd.Series(
        daily_cost_history,
        index=simulation_dates,
        name="Daily Cost"
    )

    # Convert starting weights into a DataFrame.
    start_weights_df = pd.DataFrame(
        start_weight_history,
        index=simulation_dates,
        columns=asset_names
    )

    # Convert target weights into a DataFrame.
    target_weights_df = pd.DataFrame(
        target_weight_history,
        index=simulation_dates,
        columns=asset_names
    )

    # Create the gross portfolio equity curve.
    gross_equity = (1 + gross_daily_returns).cumprod()

    # Create the net portfolio equity curve.
    net_equity = (1 + net_daily_returns).cumprod()

    # Create the net portfolio drawdown curve.
    net_drawdown = net_equity / net_equity.cummax() - 1

    # Create a daily summary table.
    daily_summary = pd.DataFrame(
        {
            "Gross Daily Return": gross_daily_returns,
            "Net Daily Return": net_daily_returns,
            "Daily Turnover": daily_turnover,
            "Daily Cost": daily_costs,
            "Gross Equity": gross_equity,
            "Net Equity": net_equity,
            "Net Drawdown": net_drawdown
        }
    )

    # Return all simulation outputs in one dictionary.
    return {
        "gross_daily_returns": gross_daily_returns,
        "net_daily_returns": net_daily_returns,
        "daily_turnover": daily_turnover,
        "daily_costs": daily_costs,
        "start_weights": start_weights_df,
        "target_weights": target_weights_df,
        "gross_equity": gross_equity,
        "net_equity": net_equity,
        "net_drawdown": net_drawdown,
        "daily_summary": daily_summary
    }


# Run the portfolio simulator using the selected dashboard inputs.
results = simulate_portfolio(
    sim_returns=simulation_returns,
    aggressiveness_percent=ai_aggressiveness,
    cost_bps=total_cost_bps,
    seed=seed_value
)


# Calculate gross performance metrics.
gross_metrics = calculate_performance_metrics(
    daily_returns=results["gross_daily_returns"],
    annual_risk_free_rate=risk_free_rate
)


# Calculate net performance metrics.
net_metrics = calculate_performance_metrics(
    daily_returns=results["net_daily_returns"],
    annual_risk_free_rate=risk_free_rate
)


# Calculate cumulative portfolio turnover.
cumulative_turnover = results["daily_turnover"].sum()


# Calculate cumulative transaction cost.
total_friction_cost = results["daily_costs"].sum()


# Calculate Sharpe decay caused by trading friction.
sharpe_decay = gross_metrics["sharpe"] - net_metrics["sharpe"]


# Create a helper function to format percentages.
def format_percent(value):
    # Return N/A if the value is missing.
    if pd.isna(value):
        return "N/A"

    # Convert decimal return into a percentage string.
    return f"{value * 100:.2f}%"


# Create a helper function to format normal numbers.
def format_number(value):
    # Return N/A if the value is missing.
    if pd.isna(value):
        return "N/A"

    # Format the number to two decimals.
    return f"{value:.2f}"


# Add a divider before the executive summary.
st.divider()


# Show the executive summary section.
st.subheader("Executive Summary")


# Create seven columns for headline metrics.
metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5, metric_col_6, metric_col_7 = st.columns(7)


# Show gross annualized return.
metric_col_1.metric(
    "Gross Ann. Return",
    format_percent(gross_metrics["annualized_return"])
)


# Show net annualized return.
metric_col_2.metric(
    "Net Ann. Return",
    format_percent(net_metrics["annualized_return"]),
    delta=f"-{format_percent(total_friction_cost)} friction",
    delta_color="inverse"
)


# Show gross Sharpe ratio.
metric_col_3.metric(
    "Gross Sharpe",
    format_number(gross_metrics["sharpe"])
)


# Show net Sharpe ratio.
metric_col_4.metric(
    "Net Sharpe",
    format_number(net_metrics["sharpe"]),
    delta=f"-{format_number(sharpe_decay)} decay",
    delta_color="inverse"
)


# Show net max drawdown.
metric_col_5.metric(
    "Net Max Drawdown",
    format_percent(net_metrics["max_drawdown"])
)


# Show cumulative turnover.
metric_col_6.metric(
    "Cumulative Turnover",
    format_percent(cumulative_turnover)
)


# Show total cost in basis points.
metric_col_7.metric(
    "Cost / Turnover",
    f"{total_cost_bps} bps"
)


# Add interpretation based on net Sharpe.
if pd.notna(net_metrics["sharpe"]) and net_metrics["sharpe"] > 1:
    st.success("Interpretation: after friction, the strategy still has a decent Sharpe profile.")

# Add warning if Sharpe is positive but weak.
elif pd.notna(net_metrics["sharpe"]) and net_metrics["sharpe"] > 0:
    st.warning("Interpretation: the strategy survives costs, but net Sharpe is weak. Turnover control needs improvement.")

# Add error message if net Sharpe is negative.
else:
    st.error("Interpretation: after friction, the strategy is not attractive. Trading cost is damaging the model.")


# Add a divider before charts.
st.divider()


# Show the chart section heading.
st.subheader("Gross vs Net Portfolio Value")


# Create a Plotly figure for portfolio equity curves.
equity_fig = go.Figure()


# Add the gross equity curve.
equity_fig.add_trace(
    go.Scatter(
        x=results["gross_equity"].index,
        y=results["gross_equity"],
        mode="lines",
        name="Gross Portfolio Value"
    )
)


# Add the net equity curve.
equity_fig.add_trace(
    go.Scatter(
        x=results["net_equity"].index,
        y=results["net_equity"],
        mode="lines",
        name="Net Portfolio Value"
    )
)


# Improve the chart layout.
equity_fig.update_layout(
    template="plotly_dark",
    height=420,
    xaxis_title="Date",
    yaxis_title="Portfolio Value, Starting at 1.00",
    legend_title="Series",
    margin=dict(l=10, r=10, t=30, b=10)
)


# Display the equity curve chart.
st.plotly_chart(equity_fig, use_container_width=True)


# Create two columns for turnover and drawdown charts.
chart_col_1, chart_col_2 = st.columns(2)


# Put cumulative turnover chart in the left column.
with chart_col_1:
    # Show the turnover chart title.
    st.markdown("### Cumulative Turnover")

    # Create a Plotly figure for cumulative turnover.
    turnover_fig = go.Figure()

    # Add cumulative turnover line.
    turnover_fig.add_trace(
        go.Scatter(
            x=results["daily_turnover"].index,
            y=results["daily_turnover"].cumsum() * 100,
            mode="lines",
            name="Cumulative Turnover (%)"
        )
    )

    # Improve turnover chart layout.
    turnover_fig.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Date",
        yaxis_title="Cumulative Turnover (%)",
        margin=dict(l=10, r=10, t=30, b=10)
    )

    # Display the turnover chart.
    st.plotly_chart(turnover_fig, use_container_width=True)


# Put drawdown chart in the right column.
with chart_col_2:
    # Show the drawdown chart title.
    st.markdown("### Net Portfolio Drawdown")

    # Create a Plotly figure for drawdown.
    drawdown_fig = go.Figure()

    # Add net drawdown line.
    drawdown_fig.add_trace(
        go.Scatter(
            x=results["net_drawdown"].index,
            y=results["net_drawdown"] * 100,
            mode="lines",
            name="Net Drawdown (%)"
        )
    )

    # Improve drawdown chart layout.
    drawdown_fig.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        margin=dict(l=10, r=10, t=30, b=10)
    )

    # Display the drawdown chart.
    st.plotly_chart(drawdown_fig, use_container_width=True)


# Add a divider before weights section.
st.divider()


# Show the weights section heading.
st.subheader("Portfolio Weight Tracking")


# Create two columns for final weights and latest weight table.
weights_col_1, weights_col_2 = st.columns([1, 2])


# Put final weight chart in the left column.
with weights_col_1:
    # Show the final weights title.
    st.markdown("### Final Target Weights")

    # Select the latest target weights.
    final_weights = results["target_weights"].iloc[-1]

    # Create a bar chart for final weights.
    weights_fig = go.Figure()

    # Add final weight bars.
    weights_fig.add_trace(
        go.Bar(
            x=final_weights.index,
            y=final_weights.values * 100,
            name="Final Weight (%)"
        )
    )

    # Improve final weight chart layout.
    weights_fig.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Asset",
        yaxis_title="Weight (%)",
        margin=dict(l=10, r=10, t=30, b=10)
    )

    # Display the final weights chart.
    st.plotly_chart(weights_fig, use_container_width=True)


# Put weight history table in the right column.
with weights_col_2:
    # Show the weight table title.
    st.markdown("### Last 10 Rebalanced Weights")

    # Take the last 10 rows of target weights.
    latest_weights = results["target_weights"].tail(10)

    # Convert weights into percentage strings.
    latest_weights_percent = (latest_weights * 100).round(2).astype(str) + "%"

    # Display the latest weights table.
    st.dataframe(latest_weights_percent, use_container_width=True)


# Add a divider before formulas.
st.divider()


# Create an expandable section for formulas.
with st.expander("Model Formulas and Simple Explanation", expanded=True):
    # Explain the purpose of the dashboard.
    st.markdown(
        """
        This dashboard answers one quant question:

        **Does a portfolio strategy still look good after turnover, slippage, and brokerage costs?**

        A model can look profitable before fees, but real money is made after costs.
        """
    )

    # Show the weight shock formula.
    st.markdown("#### 1. Stochastic Weight Shock")

    # Display the weight shock formula.
    st.latex(r"w_t^{raw} = w_{t-1} + \epsilon_t")

    # Display the random noise assumption.
    st.latex(r"\epsilon_t \sim \mathcal{N}(0, \sigma^2)")

    # Explain the weight shock.
    st.markdown(
        """
        The engine randomly changes portfolio weights each day.  
        Higher **aggressiveness** means bigger weight changes.
        """
    )

    # Show the turnover formula.
    st.markdown("#### 2. Daily Turnover")

    # Display turnover formula.
    st.latex(r"\text{Turnover}_t = \frac{1}{2} \sum_i |w_{i,t}^{target} - w_{i,t-1}|")

    # Explain turnover.
    st.markdown(
        """
        Turnover measures how much of the portfolio was traded.  
        High turnover means more trading cost.
        """
    )

    # Show friction formula.
    st.markdown("#### 3. Daily Friction Cost")

    # Display friction cost formula.
    st.latex(r"\text{Cost}_t = \text{Turnover}_t \times \frac{\text{Slippage bps} + \text{Brokerage bps}}{10,000}")

    # Explain friction cost.
    st.markdown(
        """
        Slippage and brokerage are charged on traded volume, not on the full portfolio every day.
        """
    )

    # Show net return formula.
    st.markdown("#### 4. Net Daily Return")

    # Display net return formula.
    st.latex(r"r_t^{net} = r_t^{gross} - \text{Cost}_t")

    # Explain net return.
    st.markdown(
        """
        Net return is the return that matters because it is what remains after trading cost.
        """
    )

    # Show Sharpe formula.
    st.markdown("#### 5. Annualized Sharpe Ratio")

    # Display Sharpe formula.
    st.latex(r"\text{Sharpe} = \frac{\overline{r_d - r_{f,d}}}{\sigma_d} \times \sqrt{252}")

    # Explain Sharpe.
    st.markdown(
        """
        Sharpe measures return per unit of risk.  
        A strategy with high gross Sharpe but weak net Sharpe is usually overtrading.
        """
    )


# Add a divider before daily results.
st.divider()


# Show daily results section.
st.subheader("Daily Simulation Results")


# Create a formatted copy of the daily summary table.
daily_summary_display = results["daily_summary"].copy()


# Display the daily summary table with percentage formatting.
st.dataframe(
    daily_summary_display.style.format(
        {
            "Gross Daily Return": "{:.2%}",
            "Net Daily Return": "{:.2%}",
            "Daily Turnover": "{:.2%}",
            "Daily Cost": "{:.4%}",
            "Gross Equity": "{:.4f}",
            "Net Equity": "{:.4f}",
            "Net Drawdown": "{:.2%}"
        }
    ),
    use_container_width=True
)


# Convert the daily summary table into CSV bytes.
csv_output = results["daily_summary"].to_csv().encode("utf-8")


# Add a download button for the simulation results.
st.download_button(
    label="Download Daily Simulation CSV",
    data=csv_output,
    file_name="portfolio_turnover_friction_simulation.csv",
    mime="text/csv"
)
