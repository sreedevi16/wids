import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (11, 4)

# ---------------------------------------------------------------
# 1. Download data
# ---------------------------------------------------------------
tickers = {
    "stocks": "^NSEI",           # Nifty 50
    "gold": "GOLDBEES.NS",       # Nippon India Gold ETF (gold in rupees)
    "bonds": "LTGILTBEES.NS",    # Nippon India long-term gilt ETF
    "india_vix": "^INDIAVIX",    # India VIX (expected market volatility)
}

data = yf.download(list(tickers.values()), start="2010-01-01", auto_adjust=True, progress=False)["Close"]
data = data.rename(columns={v: k for k, v in tickers.items()})
data = data[list(tickers.keys())]

print(data.head())
print("Missing values per column:")
print(data.isna().sum())

# Fill gaps (holidays / missing days) with the last known price, then drop rows
# from before every asset had started trading.
data = data.ffill().dropna()
print("Data from", data.index[0].date(), "to", data.index[-1].date(), "-", len(data), "trading days")

# ---------------------------------------------------------------
# 2. Prices: growth of 1 rupee in each asset
# ---------------------------------------------------------------
assets = ["stocks", "gold", "bonds"]
growth = data[assets] / data[assets].iloc[0]
growth.plot(title="Growth of 1 rupee: Nifty 50 vs Gold vs Bonds")
plt.ylabel("value")
plt.show()

data["india_vix"].plot(title="India VIX", color="purple")
plt.show()

# ---------------------------------------------------------------
# 3. Returns
# ---------------------------------------------------------------
returns = data[assets].pct_change().dropna()                 # simple returns
log_returns = np.log(data[assets]).diff().dropna()           # log returns

print("\nDaily return statistics:")
print(returns.describe())

nifty_returns = returns["stocks"]
print("\nBest 10 days for Nifty:")
print(nifty_returns.sort_values(ascending=False).head(10))
print("\nWorst 10 days for Nifty:")
print(nifty_returns.sort_values(ascending=True).head(10))

# Are Nifty returns normal (bell-shaped)?
r = log_returns["stocks"]
plt.hist(r, bins=150, density=True, alpha=0.6, label="actual daily returns")
x = np.linspace(r.min(), r.max(), 400)
normal = np.exp(-(x - r.mean()) ** 2 / (2 * r.std() ** 2)) / (r.std() * np.sqrt(2 * np.pi))
plt.plot(x, normal, "r", label="normal distribution")
plt.xlim(-0.06, 0.06)
plt.legend()
plt.title("Nifty daily log returns vs normal distribution (note the fat tails)")
plt.show()

# How do the assets move together?
print("\nCorrelation of daily returns:")
print(returns.corr().round(2))

# ---------------------------------------------------------------
# 4. Volatility
# ---------------------------------------------------------------
vol_1w = log_returns["stocks"].rolling(5).std() * np.sqrt(252)
vol_1m = log_returns["stocks"].rolling(21).std() * np.sqrt(252)
vol_1q = log_returns["stocks"].rolling(63).std() * np.sqrt(252)

fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
(vol_1m * 100).plot(ax=ax[0], color="darkred", title="Nifty 1-month realised volatility (%)")
data["india_vix"].plot(ax=ax[1], color="purple", title="India VIX")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 5. Summary statistics per asset
# ---------------------------------------------------------------
years = len(data) / 252
summary = pd.DataFrame(index=assets)
summary["CAGR"] = (data[assets].iloc[-1] / data[assets].iloc[0]) ** (1 / years) - 1
summary["Annual volatility"] = log_returns.std() * np.sqrt(252)

drawdowns = data[assets] / data[assets].cummax() - 1
summary["Max drawdown"] = drawdowns.min()
summary["Max drawdown date"] = drawdowns.idxmin().dt.date
print("\nSummary:")
print(summary)

drawdowns.plot(title="Drawdown from previous peak")
plt.show()

# ---------------------------------------------------------------
# 6. Features for the later weeks (saved to features.csv)
# ---------------------------------------------------------------
features = pd.DataFrame({
    "ret": log_returns["stocks"],
    "momentum_1w": np.log(data["stocks"]).diff(5),
    "momentum_1m": np.log(data["stocks"]).diff(21),
    "momentum_1q": np.log(data["stocks"]).diff(63),
    "vol_1w": vol_1w,
    "vol_1m": vol_1m,
    "vol_1q": vol_1q,
    "india_vix": data["india_vix"],
}).dropna()

print("\nFeatures:")
print(features.tail())
features.to_csv("features.csv")
print("Saved features.csv with", len(features), "rows")
