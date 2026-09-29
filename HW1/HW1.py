import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

TICKER = "NKE"
BENCH = "SPY"
TICKERS = [TICKER, BENCH]

# Pull a year of daily closes and drop any empty rows
prices = yf.download(TICKERS, period="1y", interval="1d", auto_adjust=True, progress=False)["Close"]
prices = prices[TICKERS].dropna()

print("Tickers:", ", ".join(TICKERS))
print("Trading days:", len(prices))
print("First date:", prices.index[0].date())
print("Last date:", prices.index[-1].date())

# Last close, return over the year, annualised volatility
returns = prices.pct_change().dropna()
for t in TICKERS:
    last_close = prices[t].iloc[-1]
    year_return = prices[t].iloc[-1] / prices[t].iloc[0] - 1
    ann_vol = returns[t].std() * np.sqrt(252)
    print(f"\n{t}")
    print(f"  Last close:        {last_close:.2f}")
    print(f"  Return over year:  {year_return:.2%}")
    print(f"  Annualised vol:    {ann_vol:.2%}")

# Rebase both series to start at 100 and plot
rebased = prices / prices.iloc[0] * 100
assert (rebased.iloc[0] == 100).all()

fig, ax = plt.subplots(figsize=(10, 5))
for t in TICKERS:
    ax.plot(rebased.index, rebased[t], label=t)
ax.set_xlabel("Date")
ax.set_ylabel("Price (rebased, start = 100)")
ax.set_title(f"{TICKER} vs {BENCH}, one year, rebased to 100")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("HW1_chart.png", dpi=150)
plt.show()

# Biggest single-day move for our ticker
biggest_day = returns[TICKER].abs().idxmax()
biggest_move = returns[TICKER].loc[biggest_day]
print(f"\nBiggest single-day move for {TICKER}: {biggest_day.date()} ({biggest_move:+.2%})")
