Portfolio Turnover, Friction & Net Sharpe Dashboard

Live Dashboard: hessian-ai-turnover-sharpe-visualizer.streamlit.app

1. Project Summary

This project is an interactive Streamlit-based quant portfolio dashboard. It answers one serious quantitative finance question:

Does a trading strategy still look good after real-world trading costs?

A beginner looks only at profit. A quant looks at:
Profit + Risk + Turnover + Cost + Drawdown

A strategy can look fantastic on paper before costs, but become incredibly weak after slippage, brokerage fees, and overtrading. This dashboard downloads historical stock prices, calculates daily returns, simulates daily stochastic portfolio rebalancing, and visually compares Gross Performance (vanity) against Net Performance (reality).

2. Technical Stack

Frontend / App Framework: Streamlit

Data Ingestion: yfinance (Yahoo Finance API)

Numerical Computing: NumPy

Data Manipulation: Pandas

Data Visualization: Plotly Graph Objects

3. How to Run Locally

To run this dashboard on your local machine, follow these steps:

Clone the repository:

git clone [https://github.com/MangeshTheMathematician/weight-turnover-friction-sharpe-visualizer.git](https://github.com/MangeshTheMathematician/weight-turnover-friction-sharpe-visualizer.git)


Navigate to the folder:

cd weight-turnover-friction-sharpe-visualizer


Install the required dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run weightfitickerfapp.py


(Note: You can view the live deployment of this dashboard here: https://hessian-ai-turnover-sharpe-visualizer.streamlit.app/)

4. The "Chain Reaction" Story (How Quants Evaluate Portfolios)

To understand this dashboard, you just have to follow the money through this 7-step chain reaction:

The Target (Weights): You decide what percentage of your money should be in each stock.

The Toll Booth (Turnover & Cost): To hit that target, you have to buy and sell. Every time you move money, the broker charges a toll.

The Paycheck (Net Return): You take the raw profit your stocks made, subtract the Toll Booth fee, and that is your actual take-home pay for the day.

The Snowball (Equity Curve): You add today's paycheck to yesterday's bank account balance. Your money compounds over time.

The Speedometer (Annualized Return): We stretch your Snowball's growth over a 252-day year so we can compare your "speed" with other traders.

The Rollercoaster (Volatility & Drawdown): We measure how violently your portfolio swung up and down, and how deep of a hole you fell into on your worst day.

The Final Grade (Sharpe Ratio): We divide your Speed by your Rollercoaster bumps. Was the extra profit worth the stomach-churning ride?

5. Comprehensive Math, Proofs, & Plain English Examples

5.1 Asset Return

The Concept: How much did the price change compared to yesterday?
The Formula:

$$r_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1$$

Example: Apple goes from $100 to $102. Return = $(102 / 100) - 1 = 0.02 = 2\%$.

5.2 Portfolio Return

The Concept: How much did my entire pie grow, based on how big each slice was?
The Formula:

$$r_{p,t} = \sum_{i=1}^{N} w_{i,t}r_{i,t}$$

The Proof:
If you have total wealth $W$, and put weight $w_i$ into asset $i$, the money in that asset is $W \times w_i$.
After return $r_i$, that asset becomes $Ww_i(1+r_i)$.
Total portfolio value = $Ww_1(1+r_1) + Ww_2(1+r_2) + \dots + Ww_N(1+r_N)$.
Factor out $W$: $W[(w_1+w_2+\dots+w_N) + (w_1r_1+w_2r_2+\dots+w_Nr_N)]$.
Because weights sum to 1 ($100\%$), this simplifies to $W[1 + \sum w_ir_i]$.
Therefore, the portfolio return is $\sum w_ir_i$.

Real-Life Example:
You have $1,000.

AAPL Weight: 40% (Return: +2%) $\rightarrow 0.40 \times 0.02 = +0.008$

NVDA Weight: 35% (Return: -1%) $\rightarrow 0.35 \times -0.01 = -0.0035$

TSLA Weight: 25% (Return: +4%) $\rightarrow 0.25 \times 0.04 = +0.010$
Total Portfolio Return = $0.008 - 0.0035 + 0.010 = 0.0145 = 1.45\%$.

5.3 Target Weights & Turnover (The "Pizza" Rule)

The Concept: Think of your portfolio like a pizza.

Current Weight: What your pizza looks like right now (e.g., 50% Pepperoni, 50% Cheese).

Target Weight: What you want it to look like because you think it will make you more money (e.g., 60% Pepperoni, 40% Cheese).

Turnover: How many toppings did you actually have to scrape off and move to hit your goal?

The Formula:

$$Turnover_t = \frac{1}{2}\sum_i |w_{i,t}^{target} - w_{i,t}^{current}|$$

Why divide by 2?
If you don't divide by 2, the math double-counts your money (once when you sell, once when you buy).

Real-Life Example:
You have a $1,000 portfolio.

Current: AAPL 50%, NVDA 30%, TSLA 20%

Target: AAPL 40%, NVDA 35%, TSLA 25%

Let's subtract (Target - Current) using absolute values (positive numbers only):

AAPL: $|40\% - 50\%| = 10\%$

NVDA: $|35\% - 30\%| = 5\%$

TSLA: $|25\% - 20\%| = 5\%$
Sum = $20\%$.
Turnover = $20\% / 2 = 10\%$.

Does this match reality? Yes! To hit your target, you sold $100 of AAPL, and used that exact same $100 bill to buy $50 of NVDA and $50 of TSLA. You only "turned over" $100 out of your $1,000 portfolio (10%).

5.4 Daily Friction Cost (The "Toll Booth" Rule)

The Concept: Every time you move money, Wall Street takes a tiny bite. You only pay taxes on the pizza slices you actually moved, not the ones sitting on the plate.
The Formula:

$$Cost_t = Turnover_t \times \frac{SlippageBps + BrokerageBps}{10000}$$

(Note: 1 Basis Point (bps) = 0.01% or 0.0001)

Real-Life Example:
Your Turnover was 10% (0.10). Your broker charges 15 bps (0.0015).
Cost = $0.10 \times 0.0015 = 0.00015 = 0.015\%$.

5.5 Gross vs Net Return (The "Paycheck" Rule)

The Concept: * Gross Return: Your salary before taxes.

Net Return: Your take-home pay after the broker takes their fee. Net is all that matters.
The Formula:

$$r_t^{net} = r_t^{gross} - Cost_t$$

5.6 Equity Curve (The "Snowball" Rule)

The Concept: You earn interest on your interest. Tomorrow's starting line is today's finish line.
The Formula:

$$E_t = E_{t-1}(1+r_t^{net})$$

5.7 Annualized Return (The "Speedometer" Rule)

The Concept: The stock market is open ~252 days a year. If you run a strategy for 63 days, we stretch that performance to a full year so we can compare your "speed" with other traders.
The Formula:

$$AnnualizedReturn = (1+CumulativeReturn)^{\frac{252}{N}} - 1$$

5.8 Annualized Volatility (The "Bumpy Road" Rule)

The Concept: Measuring how violently your portfolio swings up and down.
The Formula:

$$\sigma_{annual} = \sigma_{daily} \times \sqrt{252}$$

Why the square root? Randomness (variance) spreads out linearly with time ($Variance_{annual} = 252 \times Variance_{daily}$). Since volatility (standard deviation) is the square root of variance, we must multiply by the square root of time.

5.9 Sharpe Ratio (The "Hot Wing" Rule)

The Concept: Is the risk actually worth the reward?
Imagine a plain chicken wing (Safe Bank Return = 4% with 0 risk). Your strategy is a "Ghost Pepper Wing." It has huge risk (Volatility) but promises huge reward (Return). The Sharpe ratio asks: For every unit of burning pain I suffer, how much extra meat am I getting?
The Formula:

$$Sharpe = \frac{\overline{r_d - r_{f,d}}}{\sigma_d} \times \sqrt{252}$$

(Average Daily Return minus Daily Bank Return, divided by Daily Volatility, annualized).

5.10 Drawdown (The "Heartbreak" Rule)

The Concept: How much money did you lose compared to your best day ever? It measures the maximum pain you felt while trading.
The Formula:

$$Drawdown_t = \frac{E_t}{RunningMax_t} - 1$$

Example: Your portfolio peaks at $200, then crashes to $150.
$(150 / 200) - 1 = 0.75 - 1 = -0.25$ (A 25% Drawdown).

6. Dashboard Features

Dynamic Ticker Input: Fetch historical data for any global equities via yfinance.

Stochastic Allocation Engine: Simulate daily target weight drift using normal distributions to stress-test overtrading.

Gross vs Net Equity Chart: Visually track how $1 diverges when friction is applied.

Cumulative Turnover Tracker: See how rapidly portfolio churning scales over time.

Drawdown Visualizer: Track the exact depth and duration of underwater periods.

Data Export: Download the underlying daily simulation data as a CSV.
