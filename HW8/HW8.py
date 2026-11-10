"""Homework 8: Price an option from history alone — Stefan Fahmy"""
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TICKER = "SPY"
START, END = "2006-11-09", "2026-11-10"     # 20 years; END is exclusive, so the last bar is the Nov 9, 2026 close
SPOT = 760.88                               # the "current spot" the assignment fixes
OTM = 1.03                                  # the Q3 strike, as a multiple of spot
N = 21                                      # trading days to expiry
YEARS = N / 252
VOL_WINDOW = 60
HERE = Path(__file__).resolve().parent

HIST_COLOR, BS_COLOR, MUTED, INK, GRID = "#2a78d6", "#eb6834", "#c9c8c2", "#52514e", "#e7e6e1"


# ---------- Historical pricer (Q1, Q3) ----------
def n_day_returns(prices, n):
    """Every overlapping n-day return in the series.

    A series of length L gives L - n windows. With n = 1 this is the daily return series.
    """
    p = np.asarray(prices, dtype=float)
    return p[n:] / p[:-n] - 1


def payoff(kind, strike, finals):
    if kind == "call":
        return np.maximum(finals - strike, 0.0)
    if kind == "put":
        return np.maximum(strike - finals, 0.0)
    raise ValueError(f"unknown kind: {kind}")


def price_from_history(prices, spot, strike, days, kind):
    """Apply every historical `days`-day return to `spot`, take the payoff under each, return the mean.

    No distributional assumption, and deliberately no discounting. `prices` comes first, as in the
    starter, so that the function is not tied to one global series.
    """
    finals = spot * (1 + n_day_returns(prices, days))
    return float(payoff(kind, strike, finals).mean())


# ---------- Black-Scholes, as in Homework 6 ----------
def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def black_scholes(kind, spot, strike, years, rate, vol):
    """European option price. Time in years; rate and vol as decimals."""
    d1 = (log(spot / strike) + (rate + vol ** 2 / 2) * years) / (vol * sqrt(years))
    d2 = d1 - vol * sqrt(years)
    if kind == "call":
        return spot * norm_cdf(d1) - strike * exp(-rate * years) * norm_cdf(d2)
    if kind == "put":
        return strike * exp(-rate * years) * norm_cdf(-d2) - spot * norm_cdf(-d1)
    raise ValueError(f"unknown kind: {kind}")


def implied_volatility(price, spot, strike, years, rate, kind, lo=0.0001, hi=5.0, tol=1e-8):
    """The volatility that makes black_scholes match `price`, by bisection. nan if none does."""
    if not black_scholes(kind, spot, strike, years, rate, lo) <= price <= black_scholes(kind, spot, strike, years, rate, hi):
        return float("nan")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if black_scholes(kind, spot, strike, years, rate, mid) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------- Statistics ----------
def skew_and_excess_kurtosis(x):
    z = (x - x.mean()) / x.std(ddof=0)
    return float((z ** 3).mean()), float((z ** 4).mean() - 3)


def realised_vol(prices, window=VOL_WINDOW):
    """Annualised standard deviation of the last `window` daily log returns."""
    log_returns = np.diff(np.log(np.asarray(prices, dtype=float)))[-window:]
    return float(log_returns.std(ddof=1) * sqrt(252))


def tail_episodes(dates, returns, hit, n):
    """Group the windows flagged by `hit` into episodes. A new episode starts once windows stop overlapping."""
    idx = np.flatnonzero(hit)
    if len(idx) == 0:
        return []
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > n) + 1)
    return [(dates[g[0]], dates[g[-1] + n], len(g), float(returns[g].min() if returns[g].mean() < 0 else returns[g].max()))
            for g in groups]


def report_tail(dates, returns, label, hit, n):
    """How much independent evidence is actually behind one tail of the distribution."""
    eps = tail_episodes(dates, returns, hit, n)
    by_offset = [int(hit[k::n].sum()) for k in range(n)]
    print(f"    {label}: {int(hit.sum())} windows ({hit.mean():.2%}), {len(eps)} distinct episodes, "
          f"about {hit.sum() / n:.1f} independent windows "
          f"(non-overlapping counts run {min(by_offset)} to {max(by_offset)} depending on the start day)")
    for first, last, count, extreme in eps:
        print(f"        {first:%Y-%m-%d} to {last:%Y-%m-%d}: {count:3d} windows, extreme {extreme:+.1%}")
    return eps


# ---------- Data ----------
def load_prices():
    d = yf.download(TICKER, start=START, end=END, auto_adjust=False, progress=False)
    return d.index, d["Close"][TICKER].to_numpy(dtype=float), d["Adj Close"][TICKER].to_numpy(dtype=float)


def load_rate():
    """Latest 13-week T-bill yield, the same risk-free input as Homework 6."""
    irx = yf.download("^IRX", start="2026-10-01", end=END, auto_adjust=False, progress=False)["Close"]["^IRX"].dropna()
    return float(irx.iloc[-1]) / 100, irx.index[-1]


# ---------- Plots ----------
def tidy(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_return_distribution(returns, skew, kurt):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hist(returns * 100, bins=90, density=True, color=HIST_COLOR, edgecolor="white", linewidth=0.4,
            label="Historical 21-day returns")
    mean, sd = returns.mean() * 100, returns.std(ddof=1) * 100
    x = np.linspace(returns.min() * 100, returns.max() * 100, 800)
    ax.plot(x, np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * sqrt(2 * np.pi)), color=BS_COLOR, linewidth=2,
            label="Normal, same mean and std")
    ax.set_title(f"SPY 21-day returns against a fitted normal: skewness {skew:.2f}, excess kurtosis {kurt:.2f}",
                 fontsize=11, loc="left", color="#0b0b0b")
    ax.set_xlabel("21-day return (%)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=9)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(HERE / "HW8_return_distribution.png", dpi=170)
    plt.close(fig)


def plot_final_prices(finals, strike, itm_call, itm_put):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    bins = np.linspace(finals.min(), finals.max(), 110)
    bins = np.sort(np.append(bins[np.abs(bins - strike) > 1e-6], strike))     # a bin edge exactly on the strike
    for ax, kind, itm in ((axes[0], "call", itm_call), (axes[1], "put", itm_put)):
        pays = finals > strike if kind == "call" else finals < strike
        ax.hist(finals[~pays], bins=bins, color=MUTED, edgecolor="white", linewidth=0.3, label="Expires worthless")
        ax.hist(finals[pays], bins=bins, color=HIST_COLOR, edgecolor="white", linewidth=0.3, label="Pays off")
        ax.axvline(strike, color=BS_COLOR, linewidth=2, label=f"Strike ${strike:.2f}")
        ax.set_title(f"{kind.capitalize()}: finished in the money in {itm:.1%} of windows", fontsize=11, loc="left",
                     color="#0b0b0b")
        ax.set_xlabel(f"Final price after 21 days, from a {SPOT:.2f} start (dollars)")
        tidy(ax)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].set_ylabel("Historical windows")
    fig.tight_layout()
    fig.savefig(HERE / "HW8_final_prices.png", dpi=170)
    plt.close(fig)


def plot_curves(strikes, hist, bs):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True)
    pct = strikes / SPOT * 100 - 100
    for col, kind in enumerate(("call", "put")):
        ax = axes[0, col]
        ax.plot(pct, hist[kind], color=HIST_COLOR, linewidth=2, label="Historical")
        ax.plot(pct, bs[kind], color=BS_COLOR, linewidth=2, label="Black-Scholes")
        ax.set_title(f"{kind.capitalize()} price", fontsize=11, loc="left", color="#0b0b0b")
        ax.set_ylabel("Price ($)")
        ax.legend(frameon=False, fontsize=9)
        tidy(ax)
        ax = axes[1, col]
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.plot(pct, hist[kind] - bs[kind], color=HIST_COLOR, linewidth=2)
        ax.set_title(f"{kind.capitalize()}: historical minus Black-Scholes", fontsize=11, loc="left", color="#0b0b0b")
        ax.set_xlabel("Strike relative to spot (%)")
        ax.set_ylabel("Difference ($)")
        tidy(ax)
    fig.tight_layout()
    fig.savefig(HERE / "HW8_price_curves.png", dpi=170)
    plt.close(fig)


def plot_otm_puts(strikes, hist, bs):
    keep = strikes <= SPOT
    pct = strikes[keep] / SPOT * 100 - 100
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(pct, hist["put"][keep], color=HIST_COLOR, linewidth=2, marker="o", markersize=5, label="Historical")
    ax.plot(pct, bs["put"][keep], color=BS_COLOR, linewidth=2, marker="o", markersize=5, label="Black-Scholes")
    ax.set_yscale("log")
    ax.set_title("Out-of-the-money put prices on a log scale", fontsize=11, loc="left", color="#0b0b0b")
    ax.set_xlabel("Strike relative to spot (%)")
    ax.set_ylabel("Price ($, log scale)")
    ax.legend(frameon=False, fontsize=9)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(HERE / "HW8_otm_puts.png", dpi=170)
    plt.close(fig)


def main():
    dates, prices, adjusted = load_prices()
    rate, rate_day = load_rate()
    returns = n_day_returns(prices, N)
    daily = n_day_returns(prices, 1)

    # ---------- Q1 ----------
    print(f"Q1  SPY daily closes {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}: {len(prices)} trading days "
          f"(unadjusted closes, the price an option actually settles on)")
    p1, p99 = np.percentile(returns, [1, 99])
    print(f"    N = {N}: {len(returns)} windows")
    print(f"    mean {returns.mean():+.4%}   std {returns.std(ddof=1):.4%}   "
          f"1st percentile {p1:+.2%}   99th percentile {p99:+.2%}")
    print(f"    worst window {returns.min():+.2%} (starting {dates[returns.argmin()]:%Y-%m-%d}), "
          f"best window {returns.max():+.2%} (starting {dates[returns.argmax()]:%Y-%m-%d})")

    # ---------- Q2 ----------
    skew, kurt = skew_and_excess_kurtosis(returns)
    daily_skew, daily_kurt = skew_and_excess_kurtosis(daily)
    print(f"\nQ2  21-day returns: skewness {skew:.3f}, excess kurtosis {kurt:.3f}  (above zero, so fatter tails than normal)")
    print(f"    daily returns:  skewness {daily_skew:.3f}, excess kurtosis {daily_kurt:.3f}")
    z_left = (p1 - returns.mean()) / returns.std(ddof=1)
    z_right = (p99 - returns.mean()) / returns.std(ddof=1)
    print(f"    the 1st percentile sits {z_left:+.2f} std from the mean and the 99th {z_right:+.2f} (a normal gives -2.33 and +2.33)")
    plot_return_distribution(returns, skew, kurt)

    # ---------- Q3 ----------
    strike = SPOT * OTM
    call = price_from_history(prices, SPOT, strike, N, "call")
    put = price_from_history(prices, SPOT, strike, N, "put")
    discount = exp(-rate * YEARS)
    print(f"\nQ3  spot {SPOT:.2f}, strike {strike:.2f} (3% above spot), 21 days, undiscounted")
    print(f"    historical call {call:.4f}")
    print(f"    historical put  {put:.4f}")
    print(f"    call - put = {call - put:+.2f};  spot - strike = {SPOT - strike:+.2f};  "
          f"spot x mean return = {SPOT * returns.mean():+.2f};  those two sum to {SPOT - strike + SPOT * returns.mean():+.2f}")
    print(f"    discounting at {rate:.3%} for {N} trading days multiplies by {discount:.5f}: "
          f"call {call * discount:.4f} ({call * (discount - 1):+.4f}), put {put * discount:.4f} ({put * (discount - 1):+.4f})")
    print(f"    with dividend-adjusted closes instead: call {price_from_history(adjusted, SPOT, strike, N, 'call'):.4f}, "
          f"put {price_from_history(adjusted, SPOT, strike, N, 'put'):.4f}")

    # ---------- Q4 ----------
    finals = SPOT * (1 + returns)
    itm_call, itm_put = float((finals > strike).mean()), float((finals < strike).mean())
    print(f"\nQ4  fraction of the {len(finals)} windows finishing in the money at strike {strike:.2f}:")
    print(f"    call {itm_call:.2%}   put {itm_put:.2%}")
    plot_final_prices(finals, strike, itm_call, itm_put)

    # ---------- Q5 ----------
    vol = realised_vol(prices)
    print(f"\nQ5  Black-Scholes inputs: realised vol over the last {VOL_WINDOW} days {vol:.2%}, "
          f"rate {rate:.3%} (13-week T-bill, {rate_day:%Y-%m-%d}), T = {N}/252 years")
    bs_call = black_scholes("call", SPOT, strike, YEARS, rate, vol)
    bs_put = black_scholes("put", SPOT, strike, YEARS, rate, vol)
    print(f"    {'contract':<10}{'historical':>12}{'Black-Scholes':>15}{'hist - BS':>12}{'% diff':>10}")
    for name, h, b in (("call", call, bs_call), ("put", put, bs_put)):
        print(f"    {name:<10}{h:>12.4f}{b:>15.4f}{h - b:>+12.4f}{(h - b) / b:>+10.1%}")
    print("    % diff is (historical - Black-Scholes) / Black-Scholes")
    print(f"    call - put: historical {call - put:+.2f}, Black-Scholes {bs_call - bs_put:+.2f}, "
          f"no-arbitrage value spot - strike x discount {SPOT - strike * discount:+.2f}")
    print(f"    same contracts at r = 0: Black-Scholes call {black_scholes('call', SPOT, strike, YEARS, 0.0, vol):.4f}, "
          f"put {black_scholes('put', SPOT, strike, YEARS, 0.0, vol):.4f}")

    # ---------- Q6 ----------
    strikes = SPOT * np.arange(85, 115.5, 1.0) / 100
    hist = {k: np.array([price_from_history(prices, SPOT, K, N, k) for K in strikes]) for k in ("call", "put")}
    bs = {k: np.array([black_scholes(k, SPOT, K, YEARS, rate, vol) for K in strikes]) for k in ("call", "put")}
    print("\nQ6  strikes from 15% below spot to 15% above, in 1% steps")
    print(f"    {'strike':>8}{'% spot':>8} |{'call hist':>11}{'call BS':>10}{'diff':>9} |{'put hist':>10}{'put BS':>10}{'diff':>9}{'% diff':>12}")
    for i, K in enumerate(strikes):
        d_c, d_p = hist["call"][i] - bs["call"][i], hist["put"][i] - bs["put"][i]
        pct = f"{d_p / bs['put'][i]:+,.0%}" if bs["put"][i] > 0 else "n/a"
        print(f"    {K:>8.2f}{K / SPOT:>8.0%} |{hist['call'][i]:>11.3f}{bs['call'][i]:>10.3f}{d_c:>+9.3f} |"
              f"{hist['put'][i]:>10.3f}{bs['put'][i]:>10.3f}{d_p:>+9.3f}{pct:>12}")
    for kind in ("call", "put"):
        gap = hist[kind] - bs[kind]
        i = int(np.argmax(np.abs(gap)))
        print(f"    {kind}: largest dollar disagreement at strike {strikes[i]:.2f} ({strikes[i] / SPOT:.0%} of spot): "
              f"historical {hist[kind][i]:.3f}, Black-Scholes {bs[kind][i]:.3f}, difference {gap[i]:+.3f}")
    gaps = np.vstack([hist["call"] - bs["call"], hist["put"] - bs["put"]])
    which, i = np.unravel_index(np.argmax(np.abs(gaps)), gaps.shape)
    print(f"    overall, in dollars: the {('call', 'put')[which]} struck at {strikes[i]:.2f} ({strikes[i] / SPOT:.0%} of spot), "
          f"by {gaps[which, i]:+.3f}")
    otm_ratio = hist["put"] / bs["put"]
    keep = strikes < SPOT
    j = int(np.nanargmax(np.where(keep, otm_ratio, np.nan)))
    print(f"    deep out-of-the-money puts, in proportion: at strike {strikes[j]:.2f} ({strikes[j] / SPOT:.0%} of spot) "
          f"historical {hist['put'][j]:.3f} against Black-Scholes {bs['put'][j]:.6f}, a ratio of {otm_ratio[j]:,.0f} to 1")
    plot_curves(strikes, hist, bs)
    plot_otm_puts(strikes, hist, bs)

    print(f"\n    Black-Scholes vol that reproduces each historical out-of-the-money price (60-day realised is {vol:.2%}):")
    for level in (85, 90, 95, 100, 105, 110, 115):
        K = SPOT * level / 100
        kind = "put" if level <= 100 else "call"
        iv = implied_volatility(price_from_history(prices, SPOT, K, N, kind), SPOT, K, YEARS, rate, kind)
        print(f"      {level:>3}% of spot ({kind:<4}): {iv:.2%}")

    # ---------- Q7 support ----------
    print("\nQ7  how much independent evidence sits in the tails (21-day windows, strikes as a % move from spot)")
    print(f"    {len(returns)} overlapping windows contain about {len(prices) / N:.0f} non-overlapping ones")
    report_tail(dates, returns, "below -15%", returns < -0.15, N)
    report_tail(dates, returns, "below -20%", returns < -0.20, N)
    report_tail(dates, returns, "above +20%", returns > 0.20, N)
    d_tail = (log(0.85) - (rate - vol ** 2 / 2) * YEARS) / (vol * sqrt(YEARS))
    print(f"    chance of finishing more than 15% down: history {(returns < -0.15).mean():.3%}, "
          f"Black-Scholes at {vol:.2%} vol {norm_cdf(d_tail):.4%} ({d_tail:.2f} std)")
    p_tail = float((returns < -0.20).mean())
    naive, effective = sqrt(p_tail * (1 - p_tail) / len(returns)), sqrt(p_tail * (1 - p_tail) / (len(prices) / N))
    print(f"    standard error on the {p_tail:.2%} chance of a 20% fall: {naive:.3%} if the windows were independent, "
          f"{effective:.3%} using the non-overlapping count")
    print("    prices 20% out of the money:")
    for name, kind, K in (("put at 80% of spot", "put", SPOT * 0.80), ("call at 120% of spot", "call", SPOT * 1.20)):
        by_offset = [float(payoff(kind, K, SPOT * (1 + returns[k::N])).mean()) for k in range(N)]
        print(f"      {name}: historical {price_from_history(prices, SPOT, K, N, kind):.4f}, "
              f"Black-Scholes {black_scholes(kind, SPOT, K, YEARS, rate, vol):.2e}; "
              f"non-overlapping windows alone give {min(by_offset):.4f} to {max(by_offset):.4f} depending on the start day")

    # ---------- Check yourself ----------
    print("\nCheck yourself")
    print(f"    trading days {len(prices)} (about 5,030)")
    print(f"    21-day windows {len(returns)} (about 5,010)")
    print(f"    daily excess kurtosis {daily_kurt:.2f} (well above 3)")
    print(f"    call - put {call - put:+.2f} against spot - strike {SPOT - strike:+.2f}; "
          f"the {call - put - (SPOT - strike):+.2f} gap is exactly spot x the mean 21-day return")


if __name__ == "__main__":
    main()
