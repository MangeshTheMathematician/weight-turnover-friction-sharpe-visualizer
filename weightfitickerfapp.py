import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- Page Config & Dark Theme ---
st.set_page_config(layout="wide", page_title="Hessian AI Quant Visualizer")
st.title("Hessian AI: Dynamic Quant Portfolio Simulator")

# --- 1. Asset Selection ---
st.subheader("Step 1: Choose Your Global Assets")
ticker_input = st.text_input("Enter Tickers (comma separated):", "AAPL, RELIANCE.NS, SHOP.TO, NVDA")
tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

@st.cache_data(show_spinner=True)
def get_market_data(ticker_list):
    if len(ticker_list) < 2:
        return pd.DataFrame(), []
    try:
        data = yf.download(ticker_list, period="1y", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data['Close']
        valid_data = data.dropna(axis=1, how='all')
        returns = valid_data.pct_change().dropna()
        return returns, list(valid_data.columns)
    except:
        return pd.DataFrame(), []

returns, valid_tickers = get_market_data(tickers)

if returns.empty or len(valid_tickers) < 2:
    st.warning("Please enter at least TWO valid stock tickers.")
    st.stop()

st.success(f"Successfully loaded {len(valid_tickers)} assets: {', '.join(valid_tickers)}")
st.divider()

# --- 2. Monte Carlo AI Engine & Daily Stepper ---
st.subheader("Step 2: The Portfolio Turnover Engine (Monte Carlo AI)")
st.markdown("""
**How the AI works:** This engine uses a **Monte Carlo Random Walk** to simulate an AI's daily trading decisions. 
You control the "Aggressiveness" (Standard Deviation, $\sigma$). Higher aggressiveness means the AI predicts wilder shifts, changing your stock weights drastically.
""")

col_form1, col_form2 = st.columns(2)
with col_form1:
    st.markdown("**AI Weight Prediction Formula:**")
    st.latex(r"w_{t} = w_{t-1} + \mathcal{N}(0, \sigma^2)")
with col_form2:
    st.markdown("**Daily Turnover Formula:**")
    st.latex(r"\text{Turnover} = \frac{1}{2} \sum |\text{Target} - \text{Initial}|")

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    days_to_sim = st.select_slider("Days to Simulate:", options=[5, 10, 30, 60, 120, len(returns)], value=len(returns))
with col_ctrl2:
    ai_aggressiveness = st.slider("AI Trading Aggressiveness (Daily Shift % / $\sigma$)", 1.0, 20.0, 5.0, 1.0)

# --- THE SIMULATION LOOP ---
num_assets = len(valid_tickers)
current_weights = np.ones(num_assets) / num_assets # Start equal weight

daily_turnover_list = []
portfolio_returns_list = []
weights_history = [current_weights.copy()] # Track weights for the table

np.random.seed(42) # Keep simulation stable for demonstration
for i in range(days_to_sim):
    # AI Gaussian Random Walk
    noise = np.random.normal(0, ai_aggressiveness / 100.0, num_assets)
    target_weights = np.clip(current_weights + noise, 0, 1)
    
    # Normalizer (prevent dividing by zero)
    weight_sum = np.sum(target_weights)
    if weight_sum > 0:
        target_weights /= weight_sum
    else:
        target_weights = np.ones(num_assets) / num_assets
        
    weights_history.append(target_weights.copy())
    
    # Calculate Daily Turnover
    turnover_today = 0.5 * np.sum(np.abs(target_weights - current_weights))
    daily_turnover_list.append(turnover_today)
    
    # Calculate Profit for the day
    day_return = np.sum(target_weights * returns.iloc[i].values)
    portfolio_returns_list.append(day_return)
    
    current_weights = target_weights

cumulative_turnover = np.sum(daily_turnover_list)
cumulative_turnover_history = np.cumsum(daily_turnover_list)

# --- Display Weightage History & Chart ---
st.markdown("### Engine Output: Daily Weightage Tracking & Cumulative Turnover")
col_out1, col_out2 = st.columns([1, 2])

with col_out1:
    st.caption(f"Showing last 5 days of weight shifts (out of {days_to_sim} days)")
    # Create a nice dataframe of the weights
    df_weights = pd.DataFrame(weights_history[-6:], columns=valid_tickers)
    df_weights.index = [f"Day {days_to_sim - 5 + i}" for i in range(6)] if days_to_sim >= 5 else [f"Day {i}" for i in range(len(weights_history))]
    st.dataframe((df_weights * 100).round(2).astype(str) + '%', use_container_width=True)

with col_out2:
    # Plotly Dark Chart for Turnover
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, days_to_sim + 1)), y=[t * 100 for t in cumulative_turnover_history],
                             mode='lines+markers', line=dict(color='#00ffcc', width=3), name='Turnover %'))
    fig.update_layout(title="Cumulative Portfolio Turnover Over Time",
                      xaxis_title="Days Simulated", yaxis_title="Cumulative Turnover (%)",
                      template="plotly_dark", height=300, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

st.metric(label=f"Final Cumulative Turnover ({days_to_sim} Days)", value=f"{cumulative_turnover * 100:.2f}%")
st.divider()

# --- 3. The Toll Booth (Friction & BPS) ---
st.subheader("Step 3: The Toll Booth (Friction Loss)")
st.latex(r"\text{Total Friction Loss} = \text{Cumulative Turnover} \times \left( \frac{\text{Slippage bps} + \text{Brokerage bps}}{10,000} \right)")

col_bps1, col_bps2 = st.columns(2)
with col_bps1:
    slippage_bps = st.slider("Slippage (bps) - Market Impact", 0, 50, 10)
with col_bps2:
    brokerage_bps = st.slider("Brokerage (bps) - Exchange Fees", 0, 50, 5)

total_friction_decimal = cumulative_turnover * ((slippage_bps + brokerage_bps) / 10000.0)
st.warning(f"**Total Friction Loss:** The broker kept **{total_friction_decimal * 100:.2f}%** of your portfolio value due to the AI's {cumulative_turnover*100:.0f}% turnover rate.")
st.divider()

# --- 4. AI Aggressiveness vs. Profit (The Reality Check) ---
st.subheader("Step 4: The Reality Check (Profit & Sharpe Ratio)")
st.markdown("Watch how your **Net Profit** and **Net Sharpe** change below when you scroll up and adjust the **AI Aggressiveness**.")

# Calculate real financial metrics
risk_free_rate = 0.05 * (days_to_sim / 252.0) # Adjusted for timeframe
gross_return = np.prod(1 + np.array(portfolio_returns_list)) - 1
annual_volatility = np.std(portfolio_returns_list) * np.sqrt(252) if len(portfolio_returns_list) > 1 else 0

net_return = gross_return - total_friction_decimal

if annual_volatility > 0:
    gross_sharpe = (gross_return - risk_free_rate) / annual_volatility
    net_sharpe = (net_return - risk_free_rate) / annual_volatility
else:
    gross_sharpe, net_sharpe = 0, 0

colA, colB = st.columns(2)

with colA:
    st.markdown("### The Dream (Gross)")
    st.latex(r"\text{Gross Return} = \prod (1 + \text{Daily Returns}) - 1")
    st.latex(r"\text{Gross Sharpe} = \frac{\text{Gross Return} - \text{Risk Free}}{\text{Volatility}}")
    st.metric(label="Gross Profit (Pre-Fees)", value=f"{gross_return * 100:.2f}%")
    st.metric(label="Gross Sharpe Ratio", value=f"{gross_sharpe:.2f}")

with colB:
    st.markdown("### The Reality (Net)")
    st.latex(r"\text{Net Return} = \text{Gross Return} - \text{Friction Loss}")
    st.latex(r"\text{Net Sharpe} = \frac{\text{Net Return} - \text{Risk Free}}{\text{Volatility}}")
    st.metric(label="Net Profit (Post-Fees)", value=f"{net_return * 100:.2f}%", delta=f"-{total_friction_decimal * 100:.2f}% Friction", delta_color="inverse")
    st.metric(label="Net Sharpe Ratio", value=f"{net_sharpe:.2f}")