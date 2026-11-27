A 4-week, beginner-friendly project that combines **Hidden Markov Models** and **neural networks** on Indian market data (Nifty 50).

Markets switch between quiet and violent periods. A Hidden Markov Model (HMM) can label each day as **Calm**, **Turbulent** or **Crisis**, but only for the regime we are already in. This project asks:

1. Can a neural network, looking at the last 20 days of data, **predict tomorrow's regime**?
2. Does using those predictions to move money between equity, gold and cash **beat simple portfolios** like 60/40?

## Weekly plan

| Week | Notebook | Topics |
|---|---|---|
| 1 | [`Week1.py`](Week1.py) | Python + pandas, market data, returns, volatility, drawdowns |
| 2 | [``]() | Hidden Markov Models, transition matrices, lookahead bias, model selection (BIC) |
| 3 | [``]() | PyTorch, MLPs, time-series train/val/test splits, class imbalance, baselines |
| 4 | [``]() | Regime-based allocation, backtesting without lookahead, Sharpe / Sortino / drawdown / Calmar |

.



