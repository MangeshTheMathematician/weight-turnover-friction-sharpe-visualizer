# Import Streamlit because it creates the interactive web dashboard.
import streamlit as st

# Import NumPy because it is used for arrays, random numbers, and mathematical calculations.
import numpy as np

# Import pandas because it is used for tables, returns, and time-series calculations.
import pandas as pd

# Import yfinance because it downloads historical price data from Yahoo Finance.
import yfinance as yf

# Import Plotly Graph Objects because it creates professional interactive charts.
import plotly.graph_objects as go


# Set the browser tab title and make the dashboard use the full page width.
st.set_page_config(
    page_title="Portfolio Turnover, Friction & Net Sharpe Dashboard",
    layout="wide"
)


# Add small CSS improvements to make the dashboard cleaner.
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.75rem;
        }

        .small-note {
            color: #9ca3af;
            font-size: 0.90rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Show the main dashboard title.
st.title("Portfolio Turnover, Friction & Net Sharpe Dashboard")


# Show a short professional subtitle.
st.caption(
    "A quant portfolio simulator showing how turnover, slippage, and brokerage costs reduce real-world strategy performance."
)


# Give a truthful model warning so the app is not oversold as real AI.
st.info(
    "This dashboard uses a stochastic allocation engine. "
    "It simulates daily portfolio weight changes and measures gross versus net performance after trading friction."
)


# Create the sidebar for all user inputs.
st.sidebar.header("Inputs / Controls")


# Ask the user for ticker symbols.
ticker_input = st.sidebar.text_input(
    "Enter tickers separated by commas",
    "AAPL, RELIANCE.NS, SHOP.TO, NVDA"
)


# Let the user choose the historical data window.
price_period = st.sidebar.selectbox(
    "Historical data period",
    ["6mo", "1y", "2y", "5y"],
    index=1
)


# Let the user choose the annual risk-free rate used in Sharpe ratio.
annual_risk_free_rate = st.sidebar.number_input(
    "Annual risk-free rate",
    min_value=0.00,
    max_value=0.30,
    value=0.05,
    step=0.005,
    format="%.3f"
)


# Let the user choose the daily allocation shock size.
allocation_shock_percent = st.sidebar.slider(
    "Daily allocation shock / aggressiveness (%)",
    min_value=0.5,
    max_value=20.0,
    value=5.0,
    step=0.5
)


# Let the user choose market-impact cost in basis points.
slippage_bps = st.sidebar.slider(
    "Slippage / market impact (bps)",
    min_value=0,
    max_value=100,
    value=10,
    step=1
)


# Let the user choose brokerage and exchange cost in basis points.
brokerage_bps = st.sidebar.slider(
    "Brokerage / exchange fees (bps)",
    min_value=0,
    max_value=100,
    value=5,
    step=1
)


# Let the user choose a random seed for reproducible simulation.
seed_value = st.sidebar.number_input(
    "Random seed",
    min_value=1,
    max_value=999999,
    value=42,
    step=1
)


# Convert the comma-separated ticker input into a clean list.
tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]


# Remove duplicate tickers while preserving original order.
tickers = list(dict.fromkeys(tickers))


# Stop the app if fewer than two tickers are entered.
if len(tickers) < 2:
    st.warning("Please enter at least two valid tickers.")
    st.stop()


# Cache the market-data download so the app does not re-download prices after every small input change.
@st.cache_data(ttl=3600, show_spinner=True)
def load_close_prices(ticker_tuple, period):
    # Convert the ticker tuple back into a normal list.
    ticker_list = list(ticker_tuple)

    # Download adjusted market data from Yahoo Finance.
    raw_data = yf.download(
        tickers=ticker_list,
        period=period,
        auto_adjust=True,
        group_by="column",
        progress=False
    )

    # If no data is returned, send back an empty DataFrame.
    if raw_data.empty:
        return pd.DataFrame()

    # Check whether yfinance returned a multi-level column index.
    if isinstance(raw_data.columns, pd.MultiIndex):
        # Read first column level names.
        level_zero = raw_data.columns.get_level_values(0)

        # Read second column level names.
        level_one = raw_data.columns.get_level_values(1)

        # If Close appears on the first level, select the Close price panel.
        if "Close" in level_zero:
            close_prices = raw_data.xs("Close", axis=1, level=0).copy()

        # If Close appears on the second level, select the Close price panel.
        elif "Close" in level_one:
            close_prices = raw_data.xs("Close", axis=1, level=1).copy()

        # If Close prices cannot be found, return empty data.
        else:
            return pd.DataFrame()

    # Handle the simpler single-level column case.
    else:
        # If Close column exists, select it.
        if "Close" in raw_data.columns:
            close_prices = raw_data[["Close"]].copy()

            # If only one ticker exists, rename the column to the ticker.
            if len(ticker_list) == 1:
                close_prices.columns = ticker_list

        # If Close column does not exist, return empty data.
        else:
            return pd.DataFrame()

    # Convert all column names into uppercase strings.
    close_prices.columns = [str(column).upper() for column in close_prices.columns]

    # Keep only assets with enough non-missing price observations.
    useful_columns = close_prices.columns[close_prices.notna().sum() > 20]

    # Filter the close price table to useful assets only.
    close_prices = close_prices.loc[:, useful_columns]

    # Forward-fill missing values caused by exchange holidays or small data gaps.
    close_prices = close_prices.ffill()

    # Drop rows that still contain missing values.
    close_prices = close_prices.dropna(how="any")

    # Return the cleaned close price data.
    return close_prices


# Convert tickers to a tuple because cached functions need stable hashable inputs.
ticker_tuple = tuple(tickers)


# Load cleaned close price data.
prices = load_close_prices(ticker_tuple, price_period)


# Stop the app if fewer than two assets were successfully loaded.
if prices.empty or prices.shape[1] < 2:
    st.error("Could not load enough valid assets. Try different tickers or a longer data period.")
    st.stop()


# Calculate daily percentage returns from closing prices.
returns = prices.pct_change().dropna(how="any")


# Stop the app if the cleaned return history is too short.
if len(returns) < 10:
    st.error("Not enough return data after cleaning. Try a longer period or different tickers.")
    st.stop()


# Get the list of successfully loaded ticker symbols.
valid_tickers = list(prices.columns)


# Show a success message with loaded assets.
st.success(f"Loaded {len(valid_tickers)} assets: {', '.join(valid_tickers)}")


# Warn the user about global tickers and currency conversion.
st.warning(
    "Global ticker warning: this prototype uses local-market returns. "
    "It does not yet convert all assets into one base currency such as CAD or USD."
)


# Let the user choose the number of recent trading days to simulate.
days_to_simulate = st.sidebar.slider(
    "Days to simulate",
    min_value=5,
    max_value=len(returns),
    value=min(210, len(returns)),
    step=1
)


# Keep only the selected recent return window.
simulation_returns = returns.tail(days_to_simulate)


# Add slippage and brokerage to get total trading cost in basis points.
total_cost_bps = slippage_bps + brokerage_bps


# Convert total cost from bps into decimal form.
total_cost_rate = total_cost_bps / 10000.0


# Create a helper function to format money-like portfolio values.
def format_number(value):
    # Return N/A if value is missing.
    if pd.isna(value):
        return "N/A"

    # Return number rounded to two decimals.
    return f"{value:.2f}"


# Create a helper function to format percentages.
def format_percent(value):
    # Return N/A if value is missing.
    if pd.isna(value):
        return "N/A"

    # Convert decimal into percentage string.
    return f"{value * 100:.2f}%"


# Create a helper function to force portfolio weights to be long-only and sum to one.
def normalize_long_only_weights(raw_weights):
    # Replace negative weights with zero.
    clipped_weights = np.clip(raw_weights, 0, None)

    # Calculate total weight after clipping.
    total_weight = np.sum(clipped_weights)

    # If all weights became zero, return equal weights.
    if total_weight == 0:
        return np.repeat(1.0 / len(raw_weights), len(raw_weights))

    # Normalize weights so they add to one.
    return clipped_weights / total_weight


# Create a helper function to calculate drawdown from an equity curve.
def calculate_drawdown(equity_curve):
    # Calculate the running maximum value through time.
    running_maximum = equity_curve.cummax()

    # Calculate percentage fall from the running maximum.
    drawdown = equity_curve / running_maximum - 1

    # Return the drawdown series.
    return drawdown


# Create a helper function to calculate performance metrics from daily returns.
def calculate_performance_metrics(daily_returns, annual_rf_rate):
    # If no daily returns exist, return missing values.
    if daily_returns.empty:
        return {
            "cumulative_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "win_rate": np.nan
        }

    # Count the number of trading days.
    number_of_days = len(daily_returns)

    # Build the equity curve from daily returns.
    equity_curve = (1 + daily_returns).cumprod()

    # Calculate cumulative return.
    cumulative_return = equity_curve.iloc[-1] - 1

    # Calculate annualized return if the equity curve remains positive.
    if equity_curve.iloc[-1] > 0:
        annualized_return = equity_curve.iloc[-1] ** (252 / number_of_days) - 1

    # If equity is non-positive, annualized return is not meaningful.
    else:
        annualized_return = np.nan

    # Calculate daily volatility.
    daily_volatility = daily_returns.std(ddof=1)

    # Annualize daily volatility.
    annualized_volatility = daily_volatility * np.sqrt(252)

    # Convert annual risk-free rate into daily risk-free rate.
    daily_rf_rate = (1 + annual_rf_rate) ** (1 / 252) - 1

    # Calculate Sharpe ratio if volatility is positive.
    if daily_volatility > 0:
        sharpe = ((daily_returns - daily_rf_rate).mean() / daily_volatility) * np.sqrt(252)

    # If volatility is zero, Sharpe is undefined.
    else:
        sharpe = np.nan

    # Calculate drawdown curve.
    drawdown = calculate_drawdown(equity_curve)

    # Calculate maximum drawdown.
    max_drawdown = drawdown.min()

    # Calculate Calmar ratio if max drawdown is negative.
    if max_drawdown < 0:
        calmar = annualized_return / abs(max_drawdown)

    # If no drawdown exists, Calmar is undefined.
    else:
        calmar = np.nan

    # Calculate percentage of positive-return days.
    win_rate = (daily_returns > 0).mean()

    # Return all metrics.
    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate
    }


# Create the main portfolio simulation engine.
def simulate_portfolio(sim_returns, shock_percent, cost_rate, seed):
    # Create a random number generator using the selected seed.
    rng = np.random.default_rng(seed)

    # Count how many assets are in the portfolio.
    number_of_assets = sim_returns.shape[1]

    # Store asset names.
    asset_names = list(sim_returns.columns)

    # Start with equal weights across all assets.
    current_weights = np.repeat(1.0 / number_of_assets, number_of_assets)

    # Create an empty list to store pre-trade weights.
    pre_trade_weight_history = []

    # Create an empty list to store target weights.
    target_weight_history = []

    # Create an empty list to store post-return drifted weights.
    end_weight_history = []

    # Create an empty list to store gross daily returns.
    gross_return_history = []

    # Create an empty list to store net daily returns.
    net_return_history = []

    # Create an empty list to store daily turnover.
    daily_turnover_history = []

    # Create an empty list to store daily cost.
    daily_cost_history = []

    # Loop through each trading date and asset return row.
    for current_date, asset_return_row in sim_returns.iterrows():
        # Store the weights before trading.
        pre_trade_weight_history.append(current_weights.copy())

        # Convert today's asset returns into a NumPy array.
        asset_returns_today = asset_return_row.values.astype(float)

        # Generate a random allocation shock for each asset.
        random_shock = rng.normal(
            loc=0.0,
            scale=shock_percent / 100.0,
            size=number_of_assets
        )

        # Add the random shock to current weights.
        raw_target_weights = current_weights + random_shock

        # Convert raw target weights into valid long-only weights.
        target_weights = normalize_long_only_weights(raw_target_weights)

        # Calculate one-way turnover.
        turnover_today = 0.5 * np.sum(np.abs(target_weights - current_weights))

        # Calculate trading cost as turnover multiplied by cost rate.
        cost_today = turnover_today * cost_rate

        # Calculate gross portfolio return using target weights.
        gross_return_today = np.dot(target_weights, asset_returns_today)

        # Subtract trading cost to get net portfolio return.
        net_return_today = gross_return_today - cost_today

        # Calculate asset values after market movement.
        post_market_asset_values = target_weights * (1 + asset_returns_today)

        # Calculate total post-market portfolio value before cost.
        post_market_total_value = np.sum(post_market_asset_values)

        # Normalize post-market asset values into end-of-day weights.
        if post_market_total_value > 0:
            end_of_day_weights = post_market_asset_values / post_market_total_value

        # If something extreme happens, keep target weights as fallback.
        else:
            end_of_day_weights = target_weights.copy()

        # Store target weights.
        target_weight_history.append(target_weights.copy())

        # Store end-of-day drifted weights.
        end_weight_history.append(end_of_day_weights.copy())

        # Store gross return.
        gross_return_history.append(gross_return_today)

        # Store net return.
        net_return_history.append(net_return_today)

        # Store turnover.
        daily_turnover_history.append(turnover_today)

        # Store trading cost.
        daily_cost_history.append(cost_today)

        # Use end-of-day drifted weights as next day's starting weights.
        current_weights = end_of_day_weights

    # Store simulation dates.
    simulation_dates = sim_returns.index

    # Convert gross return history into a pandas Series.
    gross_daily_returns = pd.Series(
        gross_return_history,
        index=simulation_dates,
        name="Gross Daily Return"
    )

    # Convert net return history into a pandas Series.
    net_daily_returns = pd.Series(
        net_return_history,
        index=simulation_dates,
        name="Net Daily Return"
    )

    # Convert turnover history into a pandas Series.
    daily_turnover = pd.Series(
        daily_turnover_history,
        index=simulation_dates,
        name="Daily Turnover"
    )

    # Convert cost history into a pandas Series.
    daily_cost = pd.Series(
        daily_cost_history,
        index=simulation_dates,
        name="Daily Cost"
    )

    # Convert pre-trade weights into a DataFrame.
    pre_trade_weights = pd.DataFrame(
        pre_trade_weight_history,
        index=simulation_dates,
        columns=asset_names
    )

    # Convert target weights into a DataFrame.
    target_weights = pd.DataFrame(
        target_weight_history,
        index=simulation_dates,
        columns=asset_names
    )

    # Convert end-of-day weights into a DataFrame.
    end_weights = pd.DataFrame(
        end_weight_history,
        index=simulation_dates,
        columns=asset_names
    )

    # Calculate gross equity curve.
    gross_equity = (1 + gross_daily_returns).cumprod()

    # Calculate net equity curve.
    net_equity = (1 + net_daily_returns).cumprod()

    # Calculate net drawdown curve.
    net_drawdown = calculate_drawdown(net_equity)

    # Create a daily summary table.
    daily_summary = pd.DataFrame(
        {
            "Gross Daily Return": gross_daily_returns,
            "Net Daily Return": net_daily_returns,
            "Daily Turnover": daily_turnover,
            "Daily Cost": daily_cost,
            "Gross Equity": gross_equity,
            "Net Equity": net_equity,
            "Net Drawdown": net_drawdown
        }
    )

    # Return every simulation output in one dictionary.
    return {
        "gross_daily_returns": gross_daily_returns,
        "net_daily_returns": net_daily_returns,
        "daily_turnover": daily_turnover,
        "daily_cost": daily_cost,
        "pre_trade_weights": pre_trade_weights,
        "target_weights": target_weights,
        "end_weights": end_weights,
        "gross_equity": gross_equity,
        "net_equity": net_equity,
        "net_drawdown": net_drawdown,
        "daily_summary": daily_summary
    }


# Run the portfolio simulation.
results = simulate_portfolio(
    sim_returns=simulation_returns,
    shock_percent=allocation_shock_percent,
    cost_rate=total_cost_rate,
    seed=seed_value
)


# Calculate gross performance metrics.
gross_metrics = calculate_performance_metrics(
    daily_returns=results["gross_daily_returns"],
    annual_rf_rate=annual_risk_free_rate
)


# Calculate net performance metrics.
net_metrics = calculate_performance_metrics(
    daily_returns=results["net_daily_returns"],
    annual_rf_rate=annual_risk_free_rate
)


# Calculate cumulative turnover across all simulation days.
cumulative_turnover = results["daily_turnover"].sum()


# Calculate average daily turnover.
average_daily_turnover = results["daily_turnover"].mean()


# Calculate total direct friction cost as sum of daily cost rates.
total_friction_cost = results["daily_cost"].sum()


# Calculate Sharpe decay from gross Sharpe to net Sharpe.
sharpe_decay = gross_metrics["sharpe"] - net_metrics["sharpe"]


# Calculate terminal gross portfolio value.
gross_terminal_value = results["gross_equity"].iloc[-1]


# Calculate terminal net portfolio value.
net_terminal_value = results["net_equity"].iloc[-1]


# Calculate terminal drag caused by cost compounding.
terminal_friction_drag = gross_terminal_value - net_terminal_value


# Create the executive summary section.
st.header("1. Executive Summary")


# Create columns for the headline metrics.
m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)


# Show gross annualized return.
m1.metric(
    "Gross Ann. Return",
    format_percent(gross_metrics["annualized_return"])
)


# Show net annualized return.
m2.metric(
    "Net Ann. Return",
    format_percent(net_metrics["annualized_return"])
)


# Show gross Sharpe.
m3.metric(
    "Gross Sharpe",
    format_number(gross_metrics["sharpe"])
)


# Show net Sharpe with Sharpe decay.
m4.metric(
    "Net Sharpe",
    format_number(net_metrics["sharpe"]),
    delta=f"-{format_number(sharpe_decay)}",
    delta_color="inverse"
)


# Show net maximum drawdown.
m5.metric(
    "Max Drawdown",
    format_percent(net_metrics["max_drawdown"])
)


# Show cumulative turnover.
m6.metric(
    "Cumulative Turnover",
    format_percent(cumulative_turnover)
)


# Show total direct friction cost.
m7.metric(
    "Total Friction Cost",
    format_percent(total_friction_cost)
)


# Show total transaction cost per turnover.
m8.metric(
    "Cost / Turnover",
    f"{total_cost_bps} bps"
)


# Add an executive interpretation.
if pd.notna(net_metrics["sharpe"]) and net_metrics["sharpe"] > 1:
    st.success(
        "Interpretation: after trading friction, the strategy still has a solid risk-adjusted return profile."
    )
elif pd.notna(net_metrics["sharpe"]) and net_metrics["sharpe"] > 0:
    st.warning(
        "Interpretation: the strategy survives costs, but net Sharpe is weak. Turnover control is important."
    )
else:
    st.error(
        "Interpretation: after costs, the strategy is not attractive under current assumptions."
    )


# Create the input summary section.
st.header("2. Inputs / Controls")


# Build a table containing the selected dashboard assumptions.
input_summary = pd.DataFrame(
    {
        "Input": [
            "Assets Loaded",
            "Historical Period",
            "Simulation Days",
            "Annual Risk-Free Rate",
            "Daily Allocation Shock",
            "Slippage",
            "Brokerage / Fees",
            "Total Trading Cost",
            "Random Seed"
        ],
        "Value": [
            ", ".join(valid_tickers),
            price_period,
            f"{days_to_simulate} trading days",
            f"{annual_risk_free_rate:.2%}",
            f"{allocation_shock_percent:.2f}%",
            f"{slippage_bps} bps",
            f"{brokerage_bps} bps",
            f"{total_cost_bps} bps",
            str(seed_value)
        ]
    }
)


# Display the input summary table.
st.dataframe(
    input_summary,
    hide_index=True,
    use_container_width=True
)


# Create the formulas section.
st.header("3. Key Formulas")


# Put formulas inside an expander so the dashboard remains clean.
with st.expander("Open formula explanation", expanded=True):
    # Create two columns for formulas.
    formula_col_1, formula_col_2 = st.columns(2)

    # Put return, portfolio, and turnover formulas on the left.
    with formula_col_1:
        # Show asset return formula.
        st.markdown("### Asset Return")
        st.latex(r"r_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1")

        # Show portfolio return formula.
        st.markdown("### Portfolio Return")
        st.latex(r"r_{p,t}^{gross} = \sum_{i=1}^{N} w_{i,t}^{target} r_{i,t}")

        # Show turnover formula.
        st.markdown("### Daily Turnover")
        st.latex(r"Turnover_t = \frac{1}{2}\sum_i |w_{i,t}^{target} - w_{i,t}^{current}|")

        # Show friction formula.
        st.markdown("### Trading Friction")
        st.latex(r"Cost_t = Turnover_t \times \frac{SlippageBps + BrokerageBps}{10000}")

    # Put Sharpe, drawdown, and equity formulas on the right.
    with formula_col_2:
        # Show net return formula.
        st.markdown("### Net Return")
        st.latex(r"r_t^{net} = r_t^{gross} - Cost_t")

        # Show equity curve formula.
        st.markdown("### Equity Curve")
        st.latex(r"E_t = E_{t-1}(1+r_t)")

        # Show Sharpe formula.
        st.markdown("### Annualized Sharpe")
        st.latex(r"Sharpe = \frac{\overline{r_d-r_{f,d}}}{\sigma_d}\sqrt{252}")

        # Show drawdown formula.
        st.markdown("### Drawdown")
        st.latex(r"Drawdown_t = \frac{E_t}{\max(E_0,\dots,E_t)} - 1")


# Create the main visualization section.
st.header("4. Main Visualization")


# Create visualization tabs.
tab_equity, tab_turnover, tab_weights, tab_returns, tab_corr = st.tabs(
    [
        "Gross vs Net Equity",
        "Turnover & Drawdown",
        "Portfolio Weights",
        "Daily Returns & Costs",
        "Correlation"
    ]
)


# Build the gross vs net equity chart.
with tab_equity:
    # Create the equity curve figure.
    fig_equity = go.Figure()

    # Add gross equity curve.
    fig_equity.add_trace(
        go.Scatter(
            x=results["gross_equity"].index,
            y=results["gross_equity"],
            mode="lines",
            name="Gross Portfolio Value"
        )
    )

    # Add net equity curve.
    fig_equity.add_trace(
        go.Scatter(
            x=results["net_equity"].index,
            y=results["net_equity"],
            mode="lines",
            name="Net Portfolio Value"
        )
    )

    # Update equity chart layout.
    fig_equity.update_layout(
        template="plotly_dark",
        height=500,
        title="Gross vs Net Portfolio Value",
        xaxis_title="Date",
        yaxis_title="Portfolio Value, Starting at 1.00",
        hovermode="x unified",
        legend_title="Series"
    )

    # Display equity chart.
    st.plotly_chart(fig_equity, use_container_width=True)

    # Show terminal drag explanation.
    st.info(
        f"Terminal friction drag: gross portfolio ends at {gross_terminal_value:.4f}, "
        f"net portfolio ends at {net_terminal_value:.4f}, "
        f"gap = {terminal_friction_drag:.4f} portfolio units."
    )


# Build turnover and drawdown charts.
with tab_turnover:
    # Create two columns for turnover and drawdown.
    turnover_col, drawdown_col = st.columns(2)

    # Build turnover chart.
    with turnover_col:
        # Create turnover figure.
        fig_turnover = go.Figure()

        # Add cumulative turnover line.
        fig_turnover.add_trace(
            go.Scatter(
                x=results["daily_turnover"].index,
                y=results["daily_turnover"].cumsum() * 100,
                mode="lines",
                name="Cumulative Turnover (%)"
            )
        )

        # Update turnover chart layout.
        fig_turnover.update_layout(
            template="plotly_dark",
            height=420,
            title="Cumulative Turnover",
            xaxis_title="Date",
            yaxis_title="Turnover (%)",
            hovermode="x unified"
        )

        # Display turnover chart.
        st.plotly_chart(fig_turnover, use_container_width=True)

    # Build drawdown chart.
    with drawdown_col:
        # Create drawdown figure.
        fig_drawdown = go.Figure()

        # Add net drawdown line.
        fig_drawdown.add_trace(
            go.Scatter(
                x=results["net_drawdown"].index,
                y=results["net_drawdown"] * 100,
                mode="lines",
                name="Net Drawdown (%)"
            )
        )

        # Update drawdown chart layout.
        fig_drawdown.update_layout(
            template="plotly_dark",
            height=420,
            title="Net Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode="x unified"
        )

        # Display drawdown chart.
        st.plotly_chart(fig_drawdown, use_container_width=True)


# Build weight visualization charts.
with tab_weights:
    # Create two columns for final weights and weight history.
    weight_col_1, weight_col_2 = st.columns([1, 2])

    # Show final target weights as a bar chart.
    with weight_col_1:
        # Select the final target weights.
        final_weights = results["target_weights"].iloc[-1]

        # Create final weight chart.
        fig_weights = go.Figure()

        # Add final weights bar trace.
        fig_weights.add_trace(
            go.Bar(
                x=final_weights.index,
                y=final_weights.values * 100,
                name="Final Weight (%)"
            )
        )

        # Update final weight chart layout.
        fig_weights.update_layout(
            template="plotly_dark",
            height=420,
            title="Final Target Weights",
            xaxis_title="Asset",
            yaxis_title="Weight (%)"
        )

        # Display final weight chart.
        st.plotly_chart(fig_weights, use_container_width=True)

    # Show historical weights as an area chart.
    with weight_col_2:
        # Create a weight history figure.
        fig_weight_history = go.Figure()

        # Loop through every asset.
        for asset in results["target_weights"].columns:
            # Add each asset's weight history.
            fig_weight_history.add_trace(
                go.Scatter(
                    x=results["target_weights"].index,
                    y=results["target_weights"][asset] * 100,
                    mode="lines",
                    stackgroup="one",
                    name=asset
                )
            )

        # Update weight history chart layout.
        fig_weight_history.update_layout(
            template="plotly_dark",
            height=420,
            title="Target Weight History",
            xaxis_title="Date",
            yaxis_title="Weight (%)",
            hovermode="x unified"
        )

        # Display weight history chart.
        st.plotly_chart(fig_weight_history, use_container_width=True)

    # Show the latest target weights table.
    st.markdown("### Last 10 Target Weight Rows")

    # Format latest target weights as percentages.
    latest_weights_display = (results["target_weights"].tail(10) * 100).round(2).astype(str) + "%"

    # Display latest weights table.
    st.dataframe(latest_weights_display, use_container_width=True)


# Build daily returns and costs chart.
with tab_returns:
    # Create daily return and cost figure.
    fig_returns = go.Figure()

    # Add gross daily return.
    fig_returns.add_trace(
        go.Scatter(
            x=results["daily_summary"].index,
            y=results["daily_summary"]["Gross Daily Return"] * 100,
            mode="lines",
            name="Gross Daily Return (%)"
        )
    )

    # Add net daily return.
    fig_returns.add_trace(
        go.Scatter(
            x=results["daily_summary"].index,
            y=results["daily_summary"]["Net Daily Return"] * 100,
            mode="lines",
            name="Net Daily Return (%)"
        )
    )

    # Add daily cost as bars.
    fig_returns.add_trace(
        go.Bar(
            x=results["daily_summary"].index,
            y=results["daily_summary"]["Daily Cost"] * 100,
            name="Daily Cost (%)",
            opacity=0.45
        )
    )

    # Update daily return chart layout.
    fig_returns.update_layout(
        template="plotly_dark",
        height=500,
        title="Daily Gross Return, Net Return, and Trading Cost",
        xaxis_title="Date",
        yaxis_title="Daily Value (%)",
        hovermode="x unified",
        barmode="overlay"
    )

    # Display daily return chart.
    st.plotly_chart(fig_returns, use_container_width=True)


# Build correlation heatmap.
with tab_corr:
    # Calculate correlation matrix of selected asset returns.
    correlation_matrix = simulation_returns.corr()

    # Create correlation heatmap.
    fig_corr = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation")
        )
    )

    # Update heatmap layout.
    fig_corr.update_layout(
        template="plotly_dark",
        height=500,
        title="Asset Return Correlation Matrix",
        xaxis_title="Asset",
        yaxis_title="Asset"
    )

    # Display heatmap.
    st.plotly_chart(fig_corr, use_container_width=True)

    # Show correlation table.
    st.dataframe(
        correlation_matrix.style.format("{:.2f}"),
        use_container_width=True
    )


# Create risk metrics section.
st.header("5. Risk Metrics")


# Create columns for risk metrics.
r1, r2, r3, r4, r5, r6, r7, r8 = st.columns(8)


# Show net cumulative return.
r1.metric(
    "Net Cumulative Return",
    format_percent(net_metrics["cumulative_return"])
)


# Show net annualized volatility.
r2.metric(
    "Net Ann. Volatility",
    format_percent(net_metrics["annualized_volatility"])
)


# Show net Sharpe.
r3.metric(
    "Net Sharpe",
    format_number(net_metrics["sharpe"])
)


# Show net Calmar ratio.
r4.metric(
    "Net Calmar",
    format_number(net_metrics["calmar"])
)


# Show win rate.
r5.metric(
    "Win Rate",
    format_percent(net_metrics["win_rate"])
)


# Show average daily turnover.
r6.metric(
    "Avg Daily Turnover",
    format_percent(average_daily_turnover)
)


# Show total cost.
r7.metric(
    "Total Cost",
    format_percent(total_friction_cost)
)


# Show Sharpe decay.
r8.metric(
    "Sharpe Decay",
    format_number(sharpe_decay)
)


# Create interpretation section.
st.header("6. Interpretation")


# Explain gross versus net.
st.markdown(
    f"""
    The gross strategy ends with an annualized return of **{format_percent(gross_metrics["annualized_return"])}**.

    The net strategy ends with an annualized return of **{format_percent(net_metrics["annualized_return"])}**.

    The difference comes from turnover-driven trading friction.
    """
)


# Explain turnover result.
if cumulative_turnover > 3:
    st.warning(
        "Turnover is very high. A strategy with high turnover must have a strong edge, otherwise costs will damage net Sharpe."
    )
elif cumulative_turnover > 1:
    st.info(
        "Turnover is moderate. Cost control still matters, especially if slippage rises."
    )
else:
    st.success(
        "Turnover is relatively controlled under current assumptions."
    )


# Explain Sharpe result.
if pd.notna(net_metrics["sharpe"]) and net_metrics["sharpe"] > gross_metrics["sharpe"] * 0.75:
    st.success(
        "Net Sharpe remains close to gross Sharpe. Trading costs are not destroying the strategy under current assumptions."
    )
else:
    st.warning(
        "Net Sharpe is materially lower than gross Sharpe. The strategy may be overtrading or too expensive to implement."
    )


# Create limitations section.
st.header("7. Limitations")


# Put limitations inside an expandable area.
with st.expander("Read limitations"):
    # List model limitations.
    st.markdown(
        """
        - This is a research prototype, not investment advice.
        - The allocation engine is stochastic, not a trained forecasting model.
        - The model assumes long-only weights and no leverage.
        - It does not include taxes, borrow costs, liquidity constraints, bid-ask spread, or execution delay.
        - Global tickers are not converted into one common currency.
        - yfinance data may contain missing values or survivorship/data-quality issues.
        - The model does not perform walk-forward validation or out-of-sample testing.
        - Transaction cost is simplified as constant bps per traded value.
        - Real institutional execution uses more advanced market-impact models.
        """
    )


# Create download section.
st.header("8. Download CSV")


# Create CSV bytes for daily summary.
daily_summary_csv = results["daily_summary"].to_csv().encode("utf-8")


# Create CSV bytes for target weights.
target_weights_csv = results["target_weights"].to_csv().encode("utf-8")


# Create CSV bytes for input summary.
input_summary_csv = input_summary.to_csv(index=False).encode("utf-8")


# Create three columns for download buttons.
download_col_1, download_col_2, download_col_3 = st.columns(3)


# Add daily summary download button.
with download_col_1:
    st.download_button(
        label="Download Daily Simulation CSV",
        data=daily_summary_csv,
        file_name="portfolio_turnover_daily_summary.csv",
        mime="text/csv"
    )


# Add target weights download button.
with download_col_2:
    st.download_button(
        label="Download Target Weights CSV",
        data=target_weights_csv,
        file_name="portfolio_target_weights.csv",
        mime="text/csv"
    )


# Add inputs download button.
with download_col_3:
    st.download_button(
        label="Download Input Summary CSV",
        data=input_summary_csv,
        file_name="portfolio_dashboard_inputs.csv",
        mime="text/csv"
    )


# Show final daily simulation table.
st.subheader("Daily Simulation Table")


# Display formatted daily summary table.
st.dataframe(
    results["daily_summary"].style.format(
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
