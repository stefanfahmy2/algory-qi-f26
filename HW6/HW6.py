"""Homework 6: Greeks, combinations, and implied volatility — Stefan Fahmy"""
from datetime import date, datetime
from math import log, sqrt, exp, erf

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

TICKER = "SPY"

# Hypothetical stock for the combinations: spot 100, premiums priced by Black-Scholes
# with three months to expiry, a 4% rate and 25% volatility, rounded to the cent.
S0, T_COMBO, R_COMBO, VOL_COMBO = 100.0, 0.25, 0.04, 0.25


# ---------- Payoffs (from Homework 5) ----------
def intrinsic(kind, strike, spot):
    """What the option is worth if exercised right now. Never negative. kind is "call" or "put"."""
    if kind == "call":
        return np.maximum(spot - strike, 0)
    if kind == "put":
        return np.maximum(strike - spot, 0)
    raise ValueError(f"unknown kind: {kind}")


def payoff(kind, strike, premium, spot):
    """Profit per share at expiry. Works on a single spot or a numpy array of spots."""
    spot = np.asarray(spot, dtype=float)
    if kind == "long_call":
        return intrinsic("call", strike, spot) - premium
    if kind == "long_put":
        return intrinsic("put", strike, spot) - premium
    if kind == "short_call":
        return premium - intrinsic("call", strike, spot)
    if kind == "short_put":
        return premium - intrinsic("put", strike, spot)
    raise ValueError(f"unknown kind: {kind}")


def shares(entry, spot):
    """Profit per share on stock bought at `entry`."""
    return np.asarray(spot, dtype=float) - entry


# ---------- Combinations ----------
def covered_call(entry, strike, premium, spot):
    """Long 100 shares bought at `entry`, short one call struck at `strike`.

    Returns profit per share at expiry. Above the strike this must stop rising.
    """
    return shares(entry, spot) + payoff("short_call", strike, premium, spot)


def protective_put(entry, strike, premium, spot):
    return shares(entry, spot) + payoff("long_put", strike, premium, spot)


def long_straddle(strike, call_premium, put_premium, spot):
    return payoff("long_call", strike, call_premium, spot) + payoff("long_put", strike, put_premium, spot)


def bull_call_spread(low_strike, low_premium, high_strike, high_premium, spot):
    return payoff("long_call", low_strike, low_premium, spot) + payoff("short_call", high_strike, high_premium, spot)


def describe(profit_fn, top=1000.0, step=0.01):
    """Max profit, max loss and break-evens of a position, read off a fine grid of spots.

    Every leg is piecewise linear with kinks at whole-number strikes, so linear
    interpolation between grid points finds the break-evens exactly. A profit that is
    still sloping at the top of the grid is unlimited. Spot cannot go below zero, so
    the bottom of the grid is a real floor.
    """
    spots = np.arange(0, top + step / 2, step)
    p = profit_fn(spots)
    rising = p[-1] - p[-2] > 1e-9
    falling = p[-1] - p[-2] < -1e-9
    max_profit = np.inf if rising else p.max()
    max_loss = -np.inf if falling else p.min()
    i = np.where(np.sign(p[:-1]) * np.sign(p[1:]) < 0)[0]
    breakevens = spots[i] - p[i] * (spots[i + 1] - spots[i]) / (p[i + 1] - p[i])
    breakevens = np.append(breakevens, spots[p == 0])
    return max_profit, max_loss, np.sort(np.round(breakevens, 2))


def fmt(x):
    return "unlimited" if np.isinf(x) else f"{x:.2f}"


# ---------- Black-Scholes and implied volatility ----------
def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def black_scholes(kind, spot, strike, years, rate, vol):
    """European option price.

    Time in YEARS. Rate and vol as decimals, not percentages. That is the usual bug.
    black_scholes("call", 100, 100, 1, 0.04, 0.25) -> 11.84
    black_scholes("put",  100, 100, 1, 0.04, 0.25) ->  7.92
    """
    d1 = (log(spot / strike) + (rate + vol ** 2 / 2) * years) / (vol * sqrt(years))
    d2 = d1 - vol * sqrt(years)
    if kind == "call":
        return spot * norm_cdf(d1) - strike * exp(-rate * years) * norm_cdf(d2)
    if kind == "put":
        return strike * exp(-rate * years) * norm_cdf(-d2) - spot * norm_cdf(-d1)
    raise ValueError(f"unknown kind: {kind}")


def implied_volatility(price, spot, strike, years, rate, kind, lo=0.0001, hi=5.0, tol=1e-8):
    """Solve for the volatility that makes black_scholes match `price`.

    Bisection between about 0.0001 and 5.0 is plenty and is easier to debug than
    Newton's method. Price rises with volatility, which is what makes it work.
    Feeding your own price back in must return the volatility you started with.
    Returns nan when no volatility in [lo, hi] reproduces the price, which happens
    when a quote sits below the option's no-arbitrage floor.
    """
    if not black_scholes(kind, spot, strike, years, rate, lo) <= price <= black_scholes(kind, spot, strike, years, rate, hi):
        return float("nan")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if black_scholes(kind, spot, strike, years, rate, mid) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------- Live data ----------
def days_out(expiry):
    return (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days


def daily_closes(ticker, period="3mo"):
    """Daily closes, with today's close filled in when yfinance leaves the latest daily bar empty."""
    t = yf.Ticker(ticker)
    closes = t.history(period=period)["Close"]
    closes.index = closes.index.tz_localize(None).normalize()
    if pd.isna(closes.iloc[-1]):
        closes.iloc[-1] = t.fast_info["lastPrice"]
    return closes.dropna()


def short_straddles(hist, scale=1.0):
    """Sell a one-week at-the-money straddle every Friday at Black-Scholes with vol = scale * VIX,
    and settle it at the next Friday's close. Returns premium and P&L as fractions of spot."""
    weekly = hist[hist.index.dayofweek == 4]
    rows = []
    for (d0, r0), (d1, r1) in zip(weekly.iloc[:-1].iterrows(), weekly.iloc[1:].iterrows()):
        s0, s1, y = r0[TICKER], r1[TICKER], (d1 - d0).days / 365
        vol, rate = scale * r0["^VIX"] / 100, r0["^IRX"] / 100
        prem = black_scholes("call", s0, s0, y, rate, vol) + black_scholes("put", s0, s0, y, rate, vol)
        rows.append((d1, prem / s0, (prem - abs(s1 - s0)) / s0))
    return pd.DataFrame(rows, columns=["date", "premium", "pnl"]).set_index("date")


def plot_combinations(positions, filename):
    spots = np.linspace(60, 140, 801)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (name, fn, _) in zip(axes.flat, positions):
        ax.plot(spots, fn(spots), lw=2)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(S0, color="gray", ls="--", lw=1, label=f"Spot today ({S0:.0f})")
        _, _, be = describe(fn)
        for b in be:
            ax.plot(b, 0, "o", color="tab:red", ms=5)
        ax.set_title(name, fontsize=9.5)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("Spot at expiry")
    for ax in axes[:, 0]:
        ax.set_ylabel("Profit per share")
    fig.suptitle("Combination payoffs at expiry (red dots are break-evens)")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    # Q1: the four combinations
    bs = lambda kind, k: round(black_scholes(kind, S0, k, T_COMBO, R_COMBO, VOL_COMBO), 2)
    c95, c100, c105, c110 = bs("call", 95), bs("call", 100), bs("call", 105), bs("call", 110)
    p95, p100 = bs("put", 95), bs("put", 100)
    positions = [
        (f"Covered call: 100 shares at 100, short 105 call ({c105:.2f})",
         lambda s: covered_call(S0, 105, c105, s), S0 - c105),
        (f"Protective put: 100 shares at 100, long 95 put ({p95:.2f})",
         lambda s: protective_put(S0, 95, p95, s), S0 + p95),
        (f"Long straddle: 100 call ({c100:.2f}) + 100 put ({p100:.2f})",
         lambda s: long_straddle(100, c100, p100, s), None),
        (f"Bull call spread: long 100 call ({c100:.2f}), short 110 call ({c110:.2f})",
         lambda s: bull_call_spread(100, c100, 110, c110, s), 100 + c100 - c110),
    ]
    print(f"Q1  spot {S0:.0f}; premiums are Black-Scholes, T = {T_COMBO} years, "
          f"r = {R_COMBO:.0%}, vol = {VOL_COMBO:.0%}")
    print(f"    calls: 100 = {c100:.2f}, 105 = {c105:.2f}, 110 = {c110:.2f}    "
          f"puts: 95 = {p95:.2f}, 100 = {p100:.2f}")
    print("    per share (x100 for one contract)")
    print(f"    {'position':<18}{'max profit':>12}{'max loss':>12}  break-evens")
    for name, fn, _ in positions:
        mp, ml, be = describe(fn)
        print(f"    {name.split(':')[0]:<18}{fmt(mp):>12}{fmt(ml):>12}  {', '.join(f'{b:.2f}' for b in be)}")

    # The same numbers by hand, so the grid search is checked against algebra
    cc, pp, st, bc = (describe(fn) for _, fn, _ in positions)
    assert abs(cc[0] - (105 - S0 + c105)) < 1e-6 and abs(cc[1] + (S0 - c105)) < 1e-6
    assert abs(pp[1] + (S0 - 95 + p95)) < 1e-6 and np.isinf(pp[0])
    assert np.allclose(st[2], [100 - c100 - p100, 100 + c100 + p100]) and abs(st[1] + c100 + p100) < 1e-6
    assert abs(bc[0] - (10 - (c100 - c110))) < 1e-6 and abs(bc[1] + (c100 - c110)) < 1e-6
    for (_, _, expected), (_, _, be) in zip([positions[0], positions[1], positions[3]], [cc, pp, bc]):
        assert np.allclose(be, [expected])
    assert covered_call(100, 105, 4, 120) == covered_call(100, 105, 4, 200)
    plot_combinations(positions, "HW6_combinations.png")

    # Q2: calendar spread. The expiry diagram cannot show it, but Black-Scholes can value the
    # long leg on the day the short leg expires. Same hypothetical stock, strike 100.
    long_t, short_t, week = 0.25, 7 / 365, 7 / 365
    long_c = black_scholes("call", S0, 100, long_t, R_COMBO, VOL_COMBO)
    short_c = black_scholes("call", S0, 100, short_t, R_COMBO, VOL_COMBO)
    debit = long_c - short_c
    print(f"\nQ2  calendar spread, strike 100: buy 3-month call {long_c:.2f}, "
          f"sell 1-week call {short_c:.2f}, net debit {debit:.2f}")
    print("    one week later, underlying unchanged at 100:")
    print(f"      short leg expires worthless: +{short_c:.2f}")
    left = black_scholes("call", S0, 100, long_t - week, R_COMBO, VOL_COMBO)
    print(f"      long leg now worth {left:.2f}, lost {long_c - left:.2f}")
    print(f"      spread P&L {left - debit:+.2f} per share")
    spots = np.linspace(85, 115, 301)
    fig, ax = plt.subplots(figsize=(9, 5))
    for vol, ls in [(0.20, ":"), (0.25, "-"), (0.30, "--")]:
        value = np.array([black_scholes("call", s, 100, long_t - week, R_COMBO, vol) for s in spots])
        pnl = value - intrinsic("call", 100, spots) - debit
        ax.plot(spots, pnl, ls=ls, lw=2, label=f"long leg valued at {vol:.0%} vol")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(100, color="gray", ls="--", lw=1, label="Strike (100)")
    ax.set_xlabel("Spot when the short leg expires (one week)")
    ax.set_ylabel("Profit per share")
    ax.set_title("Calendar spread P&L at the short leg's expiry (long leg priced by Black-Scholes)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("HW6_calendar.png", dpi=150)
    plt.close(fig)

    # Q3: Black-Scholes reference case
    S, K, T, R, SIG = 100, 100, 1, 0.04, 0.25
    call, put = black_scholes("call", S, K, T, R, SIG), black_scholes("put", S, K, T, R, SIG)
    print(f"\nQ3  S={S}, K={K}, T={T}, r={R:.0%}, vol={SIG:.0%}")
    print(f"    call {call:.4f}   put {put:.4f}")
    print(f"    C - P = {call - put:.4f},  S - K*exp(-rT) = {S - K * exp(-R * T):.4f}")
    assert round(call, 2) == 11.84 and round(put, 2) == 7.92

    # Q4: implied volatility recovers the input volatility
    print("\nQ4  feed Black-Scholes prices back into implied_volatility")
    for kind, k, vol in [("call", 100, 0.25), ("put", 100, 0.25), ("put", 100, 0.40), ("call", 120, 0.18)]:
        price = black_scholes(kind, S, k, T, R, vol)
        iv = implied_volatility(price, S, k, T, R, kind)
        print(f"    {kind:<5}K={k:<4} vol {vol:.4f} -> price {price:8.4f} -> implied {iv:.4f}")
        assert round(iv, 4) == vol

    # Q5: implied volatility across a live chain
    t = yf.Ticker(TICKER)
    closes = daily_closes(TICKER)
    spot = closes.iloc[-1]
    rate = daily_closes("^IRX").iloc[-1] / 100
    vix = daily_closes("^VIX").iloc[-1]
    expiry = min(t.options, key=lambda e: abs(days_out(e) - 30))
    years = days_out(expiry) / 365
    calls = t.option_chain(expiry).calls
    c = calls[calls["bid"] > 0][["strike", "bid", "ask", "impliedVolatility"]].copy()
    c["mid"] = (c["bid"] + c["ask"]) / 2
    c["iv"] = [implied_volatility(m, spot, k, years, rate, "call") for m, k in zip(c["mid"], c["strike"])]
    puts = t.option_chain(expiry).puts
    p = puts[(puts["bid"] > 0) & (puts["strike"] < spot)][["strike", "bid", "ask"]].copy()
    p["iv"] = [implied_volatility((b + a) / 2, spot, k, years, rate, "put")
               for b, a, k in zip(p["bid"], p["ask"], p["strike"])]
    p = p.dropna(subset=["iv"])
    no_iv = c[c["iv"].isna()]
    c = c.dropna(subset=["iv"]).reset_index(drop=True)
    print(f"\nQ5  {TICKER} calls, expiry {expiry} ({days_out(expiry)} days), spot {spot:.2f}, "
          f"rate {rate:.2%} (13-week T-bill)")
    print(f"    {len(calls)} calls, {(calls['bid'] == 0).sum()} zero bid, "
          f"{len(no_iv)} with a mid below the no-arbitrage floor (no volatility fits), {len(c)} solved")
    for _, r in no_iv.iterrows():
        floor = spot - r.strike * exp(-rate * years)
        print(f"      no fit at strike {r.strike:.1f}: mid {r.mid:.3f}, floor S - K*exp(-rT) = {floor:.3f}")
    atm = c.loc[(c["strike"] - spot).abs().idxmin()]
    pa = puts.loc[puts["strike"] == atm.strike].iloc[0]
    p_atm = implied_volatility((pa.bid + pa.ask) / 2, spot, atm.strike, years, rate, "put")
    print(f"    implied vol at strike {atm.strike:.1f} (nearest spot): {atm.iv:.2%}  "
          f"(put at the same strike: {p_atm:.2%}; yfinance's own call figure: {atm.impliedVolatility:.2%})")
    print(f"    VIX today: {vix:.2f}  (17.1 on 2026-09-14)")
    print(f"    {'strike':>8}{'% of spot':>11}{'mid':>10}{'implied vol':>13}")
    for pct in [0.80, 0.85, 0.90, 0.95, 1.00, 1.025, 1.05, 1.075]:
        r = c.loc[(c["strike"] - pct * spot).abs().idxmin()]
        print(f"    {r.strike:>8.1f}{r.strike / spot:>11.1%}{r.mid:>10.3f}{r.iv:>13.2%}")
    lowest = c.loc[c["iv"].idxmin()]
    print(f"    lowest implied vol {lowest.iv:.2%} at strike {lowest.strike:.1f}; "
          f"highest {c['iv'].max():.2%} at strike {c.loc[c['iv'].idxmax(), 'strike']:.1f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(c["strike"], c["iv"] * 100, marker=".", ms=4, lw=1, label="Calls (my bisection, from mid)")
    ax.plot(p["strike"], p["iv"] * 100, ls="--", lw=1, color="tab:orange", label="Out-of-the-money puts, for comparison")
    ax.axvline(spot, color="gray", ls="--", lw=1, label=f"Spot ({spot:.2f})")
    ax.axhline(vix, color="tab:red", ls=":", lw=1, label=f"VIX ({vix:.2f})")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(f"{TICKER} calls, expiry {expiry}: implied volatility by strike")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("HW6_iv_smile.png", dpi=150)
    plt.close(fig)
    c.to_csv(f"chain_{expiry}.csv", index=False)

    # Q6: realised volatility over the past 30 trading days, annualised
    rets = np.log(closes).diff().dropna().iloc[-30:]
    realised = rets.std() * np.sqrt(252)
    print(f"\nQ6  30 trading days of log returns, {rets.index[0].date()} to {rets.index[-1].date()}")
    print(f"    {'realised vol':>14}{'ATM implied':>14}{'VIX':>8}{'gap':>8}")
    print(f"    {realised:>14.2%}{atm.iv:>14.2%}{vix / 100:>8.2%}{atm.iv - realised:>8.2%}")

    # Q7: does implied usually beat realised? Every Friday since 2005: VIX that day against
    # the realised volatility SPY then delivered over the next 21 trading days (VIX is a
    # 30-calendar-day figure). Then sell a one-week at-the-money straddle every Friday and
    # hold it to the next Friday. VIX sits above at-the-money vol (it prices the whole skew),
    # so the second run scales it down by today's ATM / VIX ratio.
    hist = yf.download([TICKER, "^VIX", "^IRX"], start="2005-01-01", auto_adjust=True, progress=False)["Close"]
    hist.index = hist.index.tz_localize(None)
    hist = hist.dropna()
    fwd = np.log(hist[TICKER]).diff().rolling(21).std().shift(-21) * np.sqrt(252)
    gap = (hist["^VIX"] / 100 - fwd).dropna()
    fridays = gap[gap.index.dayofweek == 4]
    print(f"\nQ7  {fridays.index[0].date()} to {fridays.index[-1].date()}, {len(fridays)} Fridays: "
          f"VIX above the next month's realised vol {(fridays > 0).mean():.0%} of the time, "
          f"by {fridays.median():.2%} at the median")

    ratio = atm.iv / (vix / 100)
    runs = {}
    for scale, label in [(1.0, "at VIX"), (ratio, f"at {ratio:.2f} x VIX (today's ATM / VIX ratio)")]:
        bt = runs[scale] = short_straddles(hist, scale)
        wins = bt["pnl"] > 0
        cum = bt["pnl"].cumsum()
        print(f"    selling a 1-week ATM straddle every Friday, priced {label}, {len(bt)} weeks, P&L as % of spot:")
        print(f"      wins {wins.mean():.0%} of weeks; average win {bt.pnl[wins].mean():.2%}, "
              f"average loss {bt.pnl[~wins].mean():.2%}; mean week {bt.pnl.mean():+.3%}")
        print(f"      premium averages {bt.premium.mean():.2%} of spot, which is the most any week can make")
        print(f"      cumulative {cum.iloc[-1]:+.1%} of spot; deepest drawdown "
              f"{(cum - cum.cummax()).min():.1%}, bottoming in the week to {(cum - cum.cummax()).idxmin().date()}")
        print("      five worst weeks:")
        for d, r in bt.nsmallest(5, "pnl").iterrows():
            print(f"        week to {d.date()}: premium {r.premium:.2%}, P&L {r.pnl:+.2%}, "
                  f"= {-r.pnl / bt.premium.mean():.1f} average weeks of premium")
        worst = bt["pnl"].min()
        print(f"      sold on margin at L x capital, the worst week costs L x {-worst:.2%}: "
              + ", ".join(f"{lev}x -> {lev * worst:.0%}" for lev in (3, 5, 7))
              + f"; the account is gone above {-1 / worst:.1f}x")
    fig, ax = plt.subplots(figsize=(9, 5))
    for scale, bt in runs.items():
        label = "priced at VIX" if scale == 1.0 else f"priced at {scale:.2f} x VIX"
        ax.plot(bt.index, bt["pnl"].cumsum() * 100, lw=1.2, label=label)
    ax.legend()
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Cumulative P&L, % of spot (unlevered, one straddle a week)")
    ax.set_title(f"Selling a 1-week ATM {TICKER} straddle every Friday")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("HW6_short_vol.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
