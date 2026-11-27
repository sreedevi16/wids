# Can a Neural Network See Market Regimes Coming?

A 4-week, beginner-friendly project that combines **Hidden Markov Models** and **neural networks** on Indian market data (Nifty 50).

Markets switch between quiet and violent periods. A Hidden Markov Model (HMM) can label each day as **Calm**, **Turbulent** or **Crisis**, but only for the regime we are already in. This project asks:

1. Can a neural network, looking at the last 20 days of data, **predict tomorrow's regime**?
2. Does using those predictions to move money between equity, gold and cash **beat simple portfolios** like 60/40?

## Weekly plan

| Week | Notebook | Topics |
|---|---|---|
| 1 | [`01_data_and_returns.ipynb`](01_data_and_returns.ipynb) | Python + pandas, market data, returns, volatility, drawdowns |
| 2 | [`02_hmm_regimes.ipynb`](02_hmm_regimes.ipynb) | Hidden Markov Models, transition matrices, lookahead bias, model selection (BIC) |
| 3 | [`03_nn_regime_prediction.ipynb`](03_nn_regime_prediction.ipynb) | PyTorch, MLPs, time-series train/val/test splits, class imbalance, baselines |
| 4 | [`04_backtest_portfolio.ipynb`](04_backtest_portfolio.ipynb) | Regime-based allocation, backtesting without lookahead, Sharpe / Sortino / drawdown / Calmar |

Each notebook ends with **"Check your understanding"** questions.

## How to run

Open any notebook in **Google Colab** (File → Open notebook → GitHub → paste this repo's URL) and run the cells from top to bottom. The first cell installs what is needed. Each notebook is self-contained and downloads its own data from Yahoo Finance.

To run locally instead:

```bash
pip install -r requirements.txt
jupyter notebook
```

## Results

_Test period: 1 Jan 2019 onwards. Fill in after running notebooks 03 and 04._

| Strategy | CAGR | Volatility | Sharpe | Max drawdown |
|---|---|---|---|---|
| Buy & hold Nifty | | | | |
| 60/40 | | | | |
| HMM (filtered) | | | | |
| Neural network | | | | |

Regime prediction accuracy (test): neural network __%, "tomorrow = today" baseline __%.

## Key ideas mentees take away

- **Lookahead bias:** models are fitted on pre-2019 data only, and every trading decision uses only information available the day before.
- **Baselines matter:** regimes are sticky, so "tomorrow = today" is a strong baseline. A model is only interesting if it beats it where it matters (regime changes).
- **Honest backtesting:** transaction costs, a single test period and hand-picked allocations all limit what the results prove.

## What I learned

_(Write this yourself after running the notebooks: what surprised you, what did not work, what you would try next.)_

## Limitations and extensions

- No transaction costs; daily rebalancing; cash assumed to earn a fixed 6.5% per year.
- Extensions: more features (e.g. India VIX, moving averages), LSTM or 1-D CNN models, predicting regime *changes* directly, adding transaction costs.
