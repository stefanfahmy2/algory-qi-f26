import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

SEED = 42
N = 100_000


# ---------- Simulation ----------
def simulate_dice(n, rng):
    """Pick a 4- or 6-sided die at random, roll it. Returns (sides, roll) arrays."""
    sides = rng.choice([4, 6], size=n)
    rolls = rng.integers(1, sides + 1)  # upper bound is exclusive, so 1..sides
    return sides, rolls


def simulate_coins(n, rng):
    """Flip three fair coins n times; payout is heads x tails."""
    heads = rng.integers(0, 2, size=(n, 3)).sum(axis=1)
    return heads * (3 - heads)


rng = np.random.default_rng(SEED)

# Q1: P(4-sided | rolled a 1)
sides, rolls = simulate_dice(N, rng)
rolled_one = rolls == 1
p_four_given_one = (sides[rolled_one] == 4).sum() / rolled_one.sum()
exact_dice = 0.6  # (1/2 * 1/4) / (1/2 * 1/4 + 1/2 * 1/6)
print(f"Q1  seed {SEED}, {N:,} trials, {rolled_one.sum():,} rolls of 1")
print(f"    simulated P(4-sided | rolled 1) = {p_four_given_one:.4f}")
print(f"    exact                           = {exact_dice:.4f}")
print(f"    difference                      = {p_four_given_one - exact_dice:+.4f}")

# Q2: expected three-coin payout
payouts = simulate_coins(N, rng)
exact_coins = 1.5  # 0*(1/8) + 2*(3/8) + 2*(3/8) + 0*(1/8)
print(f"\nQ2  simulated expected payout = {payouts.mean():.4f}")
print(f"    exact                     = {exact_coins:.4f}")
print(f"    difference                = {payouts.mean() - exact_coins:+.4f}")

# Q3: estimate vs number of trials
trial_counts = [100, 1_000, 10_000, 100_000]
estimates = [simulate_coins(n, rng).mean() for n in trial_counts]
print("\nQ3  trials     estimate")
for n, est in zip(trial_counts, estimates):
    print(f"    {n:>7,}    {est:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(trial_counts, estimates, marker="o", label="Simulated expected payout")
ax.axhline(exact_coins, color="gray", ls="--", lw=1, label="Exact answer (1.5)")
ax.set_xscale("log")
ax.set_xlabel("Number of trials (log scale)")
ax.set_ylabel("Expected payout")
ax.set_title("Three-coin payout: simulated estimate vs number of trials")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("HW3_convergence.png", dpi=150)
plt.close(fig)


# ---------- Are daily returns independent? ----------
# Q4
prices = yf.download("SPY", period="10y", interval="1d", auto_adjust=True, progress=False)["Close"]
prices = prices["SPY"].dropna()
returns = prices.pct_change().dropna()
r = returns.to_numpy()

print(f"\nQ4  first date: {returns.index[0].date()}, last date: {returns.index[-1].date()}")
print(f"    trading days (returns): {len(r)}")
print(f"    mean daily return:      {r.mean():.4%}")

# Q5: count by hand. Each pair is (today, tomorrow) = (r[i], r[i+1]); a day is "down" if its return < 0.
pairs = 0
tomorrow_down = 0
today_down = 0
both_down = 0
today_big_drop = 0
big_drop_then_down = 0
tomorrow_big_move = 0
big_drop_then_big_move = 0
for i in range(len(r) - 1):
    today, tomorrow = r[i], r[i + 1]
    pairs += 1
    if tomorrow < 0:
        tomorrow_down += 1
    if today < 0:
        today_down += 1
        if tomorrow < 0:
            both_down += 1
    if abs(tomorrow) > 0.02:
        tomorrow_big_move += 1
    if today < -0.02:
        today_big_drop += 1
        if tomorrow < 0:
            big_drop_then_down += 1
        if abs(tomorrow) > 0.02:
            big_drop_then_big_move += 1

p_down = tomorrow_down / pairs
p_down_given_down = both_down / today_down
print(f"\nQ5  P(tomorrow down)                   = {p_down:.4f}  ({tomorrow_down} down days out of {pairs})")
print(f"    P(tomorrow down | today down)      = {p_down_given_down:.4f}  ({both_down} out of {today_down} down days)")

# Q6
p_down_given_big = big_drop_then_down / today_big_drop
print(f"\nQ6  P(tomorrow down | today down > 2%) = {p_down_given_big:.4f}  ({big_drop_then_down} out of {today_big_drop} days)")

# Q8 support: direction looks close to independent, so also check the size of the move
print(f"\n    P(tomorrow moves > 2% either way)                  = {tomorrow_big_move / pairs:.4f}  ({tomorrow_big_move} out of {pairs})")
print(f"    P(tomorrow moves > 2% either way | today down > 2%) = {big_drop_then_big_move / today_big_drop:.4f}  ({big_drop_then_big_move} out of {today_big_drop})")


# ---------- Expected present value ----------
def present_value(cash_flows, rate):
    """PV of a list of cash flows, the first arriving one year from now."""
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows, start=1))


def expected_present_value(cash_flows, rate, survival_prob):
    """PV where year t's cash flow only arrives if the company survived t - 1 more years."""
    return sum(cf * survival_prob ** (t - 1) / (1 + rate) ** t
               for t, cf in enumerate(cash_flows, start=1))


epv = expected_present_value([10, 10, 10], 0.10, 0.5)
print(f"\nQ7  expected_present_value([10, 10, 10], 0.10, 0.5) = {epv:.2f}")
assert abs(expected_present_value([10, 10, 10], 0.10, 1.0) - present_value([10, 10, 10], 0.10)) < 1e-12
assert round(epv, 2) == 15.10
