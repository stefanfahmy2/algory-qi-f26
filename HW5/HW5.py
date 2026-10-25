from datetime import date, datetime

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

TICKER = "SPY"
KINDS = ["long_call", "long_put", "short_call", "short_put"]


# ---------- Value and payoff ----------
def intrinsic(kind, strike, spot):
    """What the option is worth if exercised right now. Never negative. kind is "call" or "put"."""
    if kind == "call":
        return np.maximum(spot - strike, 0)
    if kind == "put":
        return np.maximum(strike - spot, 0)
    raise ValueError(f"unknown kind: {kind}")


def extrinsic(kind, strike, spot, premium):
    """What you are paying for time and uncertainty: premium minus intrinsic."""
    return premium - intrinsic(kind, strike, spot)


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


# ---------- Chain helpers ----------
def days_out(expiry):
    return (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days


def call_values(calls, spot):
    """Mid, intrinsic and extrinsic for every call with a nonzero bid.

    Also drops quotes whose ask is below intrinsic: buying the call and exercising it
    would be free money, so these are stale quotes left on contracts nobody trades.
    """
    c = calls[calls["bid"] > 0][["strike", "bid", "ask"]].copy()
    c["mid"] = (c["bid"] + c["ask"]) / 2
    c["intrinsic"] = intrinsic("call", c["strike"], spot)
    c["extrinsic"] = extrinsic("call", c["strike"], spot, c["mid"])
    stale = c["ask"] < c["intrinsic"]
    return c[~stale].reset_index(drop=True), int(stale.sum())


def print_near_spot(c, spot, n=10):
    near = c.loc[(c["strike"] - spot).abs().nsmallest(n).index].sort_values("strike")
    print(f"    {'strike':>8}{'mid':>10}{'intrinsic':>11}{'extrinsic':>11}")
    for _, r in near.iterrows():
        print(f"    {r.strike:>8.1f}{r.mid:>10.3f}{r.intrinsic:>11.3f}{r.extrinsic:>11.3f}")


def plot_extrinsic(curves, spot, title, filename):
    """curves is a list of (DataFrame, label) pairs, all drawn on one set of axes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for c, label in curves:
        ax.plot(c["strike"], c["extrinsic"], marker=".", ms=4, lw=1, label=label)
    ax.axvline(spot, color="gray", ls="--", lw=1, label=f"Spot ({spot:.2f})")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Extrinsic value per share")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    # Q1
    print("Q1  strike 100, premium 6")
    print(f"    {'kind':<12}{'S=80':>8}{'S=100':>8}{'S=120':>8}")
    for kind in KINDS:
        row = "".join(f"{payoff(kind, 100, 6, s):>8.2f}" for s in (80, 100, 120))
        print(f"    {kind:<12}{row}")
    assert payoff("long_call", 100, 6, 80) == -6
    assert payoff("short_call", 100, 6, 120) == -14

    # Q2
    spots = np.linspace(60, 140, 401)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, kind in zip(axes.flat, KINDS):
        ax.plot(spots, payoff(kind, 100, 6, spots), lw=2)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(100, color="gray", ls="--", lw=1, label="Strike (100)")
        ax.set_title(kind.replace("_", " ").title())
        ax.grid(alpha=0.3)
        ax.legend(loc="upper center")
    for ax in axes[1]:
        ax.set_xlabel("Spot at expiry")
    for ax in axes[:, 0]:
        ax.set_ylabel("Profit per share")
    fig.suptitle("Option payoffs at expiry: strike 100, premium 6")
    fig.tight_layout()
    fig.savefig("HW5_payoffs.png", dpi=150)
    plt.close(fig)

    # Q3: the expiry closest to 30 days out
    t = yf.Ticker(TICKER)
    spot = t.history(period="5d")["Close"].iloc[-1]
    short_exp = min(t.options, key=lambda e: abs(days_out(e) - 30))
    chain = t.option_chain(short_exp)
    calls, puts = chain.calls, chain.puts
    print(f"\nQ3  ticker: {TICKER}")
    print(f"    expiry: {short_exp} ({days_out(short_exp)} days out)")
    print(f"    spot:   {spot:.2f}")
    print(f"    calls:  {len(calls)}, puts: {len(puts)}")

    # Q4
    short, n_stale = call_values(calls, spot)
    print(f"\nQ4  {short_exp}: {len(short)} of {len(calls)} calls kept "
          f"({(calls['bid'] == 0).sum()} zero bid, {n_stale} ask below intrinsic)")
    print_near_spot(short, spot)
    assert np.allclose(short["intrinsic"] + short["extrinsic"], short["mid"])

    # Q5
    peak = short.loc[short["extrinsic"].idxmax()]
    print(f"\nQ5  largest extrinsic: {peak.extrinsic:.3f} at strike {peak.strike:.1f}")
    print(f"    spot for comparison: {spot:.2f}")
    deep = short[short["strike"] <= 0.8 * spot]
    print(f"    mean extrinsic for strikes <= 80% of spot: {deep['extrinsic'].mean():.3f}")

    # What is left at deep strikes is mostly interest: the call holder pays the strike at
    # expiry instead of today, so the call is worth about spot - strike * exp(-rT).
    rate = yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1] / 100
    low = short.iloc[0]
    interest = low.strike * (1 - np.exp(-rate * days_out(short_exp) / 365))
    print(f"    lowest strike {low.strike:.1f}: extrinsic {low.extrinsic:.3f}, "
          f"interest on strike at {rate:.2%} = {interest:.3f}, "
          f"bid-ask spread {low.ask - low.bid:.2f}")

    plot_extrinsic([(short, None)], spot,
                   f"{TICKER} calls, expiry {short_exp}: extrinsic value by strike",
                   "HW5_extrinsic_short.png")

    # Q6: the first expiry at least six months out, same underlying, same spot
    long_exp = next(e for e in t.options if days_out(e) >= 182)
    long_calls = t.option_chain(long_exp).calls
    long, n_stale = call_values(long_calls, spot)
    print(f"\nQ6  long expiry: {long_exp} ({days_out(long_exp)} days out)")
    print(f"    {len(long)} of {len(long_calls)} calls kept "
          f"({(long_calls['bid'] == 0).sum()} zero bid, {n_stale} ask below intrinsic)")
    print_near_spot(long, spot)
    assert np.allclose(long["intrinsic"] + long["extrinsic"], long["mid"])

    long_peak = long.loc[long["extrinsic"].idxmax()]
    print(f"\n    largest extrinsic, {short_exp}: {peak.extrinsic:.3f} at strike {peak.strike:.1f}")
    print(f"    largest extrinsic, {long_exp}: {long_peak.extrinsic:.3f} at strike {long_peak.strike:.1f}")
    print(f"    ratio {long_peak.extrinsic / peak.extrinsic:.2f}, "
          f"sqrt of time ratio {np.sqrt(days_out(long_exp) / days_out(short_exp)):.2f}")
    atm = min(calls["strike"], key=lambda k: abs(k - spot))
    iv_short = calls.loc[calls["strike"] == atm, "impliedVolatility"].iloc[0]
    iv_long = long_calls.loc[long_calls["strike"] == atm, "impliedVolatility"].iloc[0]
    print(f"    implied vol at strike {atm:.1f}: {iv_short:.1%} ({short_exp}), {iv_long:.1%} ({long_exp})")

    both = short.merge(long, on="strike", suffixes=("_short", "_long"))
    above = (both["extrinsic_long"] > both["extrinsic_short"]).sum()
    print(f"    long curve above short at {above} of {len(both)} shared strikes")
    below = both[both["extrinsic_long"] <= both["extrinsic_short"]]
    for _, r in below.iterrows():
        print(f"      not above at strike {r.strike:.1f}: "
              f"short {r.extrinsic_short:.3f}, long {r.extrinsic_long:.3f}")

    plot_extrinsic([(short, f"{short_exp} ({days_out(short_exp)} days)"),
                    (long, f"{long_exp} ({days_out(long_exp)} days)")], spot,
                   f"{TICKER} calls: extrinsic value by strike, short vs long expiry",
                   "HW5_extrinsic_both.png")

    # Keep the exact quotes used, since the live chain changes every day
    short.to_csv(f"chain_{short_exp}.csv", index=False)
    long.to_csv(f"chain_{long_exp}.csv", index=False)


if __name__ == "__main__":
    main()
