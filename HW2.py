import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


def present_value(cash_flows, rate):
    """PV of a list of cash flows, the first arriving one year from now."""
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows, start=1))


def bond_price(face, coupon_rate, years, market_rate):
    """Price of an annual-coupon bond; the last year pays coupon plus face."""
    coupon = face * coupon_rate
    flows = [coupon] * years
    flows[-1] += face
    return round(present_value(flows, market_rate), 2)  # cents; avoids float dust


# ---------- Present value and bonds ----------
print(f"present_value([10, 15, 20], 0.10) = {present_value([10, 15, 20], 0.10):.2f}")
print(f"bond_price(1000, 0.04, 10, 0.04) = {bond_price(1000, 0.04, 10, 0.04):.2f}")

print("\n10-year bond, 4% coupon, face 1000")
for r in [0.02, 0.04, 0.0496]:
    print(f"  market rate {r:.2%}: price {bond_price(1000, 0.04, 10, r):.2f}")

rates = np.linspace(0.0, 0.10, 101)
prices_curve = [bond_price(1000, 0.04, 10, r) for r in rates]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(rates * 100, prices_curve)
ax.axhline(1000, color="gray", ls="--", lw=0.8)
ax.set_xlabel("Market rate (%)")
ax.set_ylabel("Bond price ($)")
ax.set_title("10-year 4% coupon bond: price vs market rate")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("HW2_bond_curve.png", dpi=150)
plt.close(fig)

# ---------- Beta from real data ----------
TICKERS = ["AAPL", "XOM", "JNJ"]  # tech, energy, health care
BENCH = "SPY"
ALL = TICKERS + [BENCH]

prices = yf.download(ALL, period="5y", interval="1d", auto_adjust=True, progress=False)["Close"]
prices = prices[ALL].dropna()

print("\nTrading days:")
for t in ALL:
    print(f"  {t}: {prices[t].count()}")

returns = prices.pct_change().dropna()  # returns, never prices
spy_var = returns[BENCH].var()

rows = []
for t in ALL:
    ann_ret = returns[t].mean() * 252
    ann_vol = returns[t].std() * np.sqrt(252)
    beta = returns[t].cov(returns[BENCH]) / spy_var
    rows.append((t, ann_ret, ann_vol, beta))

print(f"\n{'Ticker':<8}{'Return':>10}{'Volatility':>13}{'Beta':>8}")
for t, r, v, b in rows:
    print(f"{t:<8}{r:>10.2%}{v:>13.2%}{b:>8.2f}")

assert abs(returns[BENCH].cov(returns[BENCH]) / spy_var - 1) < 1e-12  # SPY beta = 1

print("\nRank by volatility (high to low):", [r[0] for r in sorted(rows[:-1], key=lambda x: -x[2])])
print("Rank by beta (high to low):      ", [r[0] for r in sorted(rows[:-1], key=lambda x: -x[3])])

fig, ax = plt.subplots(figsize=(8, 5))
for t, r, v, b in rows:
    ax.scatter(b, v * 100, s=80)
    ax.annotate(t, (b, v * 100), textcoords="offset points", xytext=(6, 6))
ax.set_xlabel("Beta vs SPY")
ax.set_ylabel("Annualised volatility (%)")
ax.set_title("Beta vs volatility, 5 years of daily returns")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("HW2_beta_vol.png", dpi=150)
plt.close(fig)
