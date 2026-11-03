"""Homework 7: Machine learning on a real prediction problem — Stefan Fahmy"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

TICKER = "SPY"
START, END = "2011-11-03", "2026-11-03"     # 15 years; END is exclusive, so the last bar is Nov 2, 2026
VOL_WINDOW, MA_WINDOW = 20, 50
FEATURES = ["ret_1d", "ret_5d", "ret_20d", "ret_60d", "vol_20d", "px_vs_ma50"]
SEED = 0


# ---------- Features (Q1) ----------
def first_usable_day(lookbacks=(1, 5, 20, 60)):
    """Index of the first day with a full history for every feature."""
    return max(max(lookbacks), VOL_WINDOW, MA_WINDOW - 1)


def feature_row(prices, t, lookbacks=(1, 5, 20, 60)):
    """Features for day t, using prices[0..t] only. The latest close read is prices[t]."""
    p = np.asarray(prices[: t + 1], dtype=float)
    row = [p[t] / p[t - k] - 1 for k in lookbacks]
    daily = p[t - VOL_WINDOW + 1: t + 1] / p[t - VOL_WINDOW: t] - 1       # the last 20 daily returns
    row.append(daily.std(ddof=1) * np.sqrt(252))                          # realised vol, annualised
    row.append(p[t] / p[t - MA_WINDOW + 1: t + 1].mean() - 1)             # price vs its 50-day average
    return row


def build_features(prices, lookbacks=(1, 5, 20, 60)):
    """One row per trading day: features first, label last.

    Every feature must use only prices up to and including that day. The label is
    the NEXT day's return, and it is the only forward-looking value allowed.

    Return a list of rows, each [f1, f2, ..., fn, label].
    """
    prices = [float(x) for x in prices]
    return [feature_row(prices, t, lookbacks) + [prices[t + 1] / prices[t] - 1]
            for t in range(first_usable_day(lookbacks), len(prices) - 1)]


# ---------- Split (Q2) ----------
def split_by_time(rows, train=0.6, val=0.2):
    """Split in time order into (training, validation, test). Never shuffle.

    Shuffling lets 2024 train the model that predicts 2016. It produces
    spectacular, meaningless results and it is the most common mistake in the field.
    """
    n = len(rows)
    a, b = int(n * train), int(n * (train + val))
    return rows[:a], rows[a:b], rows[b:]


def up_day_baseline(prices):
    """Share of days with a positive return: the number any classifier must beat.

    It is nearer 54% than 50%, which is why raw accuracy is a weak measure here.
    """
    p = np.asarray(prices, dtype=float)
    return float(np.mean(p[1:] / p[:-1] - 1 > 0))


# ---------- Backtester (from Homework 4) ----------
LOOKBACK = first_usable_day() + 1       # 61 closes: today plus the 60 before it


def always_hold(window):
    return True


def backtest(prices, strategy, lookback=LOOKBACK):
    """Walk forward one day at a time and apply the strategy's decision.

    On day i the strategy is handed prices[i - lookback + 1 : i + 1], the `lookback`
    closes ending with today's. If it says hold you earn prices[i + 1] / prices[i] - 1,
    otherwise zero. Annualised by 252 trading days, compounded.
    """
    p = np.asarray(prices, dtype=float)
    held, rets = [], []
    for i in range(lookback - 1, len(p) - 1):
        hold = bool(strategy(p[i - lookback + 1: i + 1]))
        held.append(hold)
        rets.append(p[i + 1] / p[i] - 1 if hold else 0.0)
    equity = np.cumprod(1 + np.array(rets))
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    return {
        "total_return": equity[-1] - 1,
        "annualised_return": equity[-1] ** (252 / len(rets)) - 1,
        "max_drawdown": float((equity / peak - 1).min()),
        "fraction_invested": float(np.mean(held)),
        "equity": equity,
    }


def make_model_strategy(model):
    """Hold when the model predicts a positive return for tomorrow, cash otherwise."""
    def model_strategy(window):
        x = np.array([feature_row(window, len(window) - 1)])
        return model.predict(x)[0] > 0
    return model_strategy


# ---------- Helpers ----------
def load_prices():
    df = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
    close = df["Close"][TICKER].dropna()
    return close


def xy(rows):
    a = np.asarray(rows, dtype=float)
    return a[:, :-1], a[:, -1]


def r2s(model, train, val):
    return r2_score(train[1], model.predict(train[0])), r2_score(val[1], model.predict(val[0]))


def tuned(make, grid, train, val):
    """Fit one model per hyperparameter value on training, score each on validation."""
    scores = [r2_score(val[1], make(v).fit(*train).predict(val[0])) for v in grid]
    best = grid[int(np.argmax(scores))]
    return best, scores, make(best).fit(*train)


def main():
    close = load_prices()
    prices, dates = close.to_numpy(), close.index
    start = first_usable_day()

    # Q1: the feature table
    rows = build_features(prices)
    row_dates = dates[start: len(prices) - 1]
    table = pd.DataFrame(rows, columns=FEATURES + ["next_day_return"], index=row_dates.date)
    print(f"Q1  {TICKER} daily closes {dates[0].date()} to {dates[-1].date()}: {len(prices)} trading days")
    print(f"    feature table shape {table.shape}  (first {start} days are warm-up, last day has no label)")
    print(f"    columns: {list(table.columns)}")
    with pd.option_context("display.width", 200, "display.float_format", "{:.6f}".format):
        print("    first three rows:")
        print("\n".join("    " + line for line in table.head(3).to_string().splitlines()))

    # Check one row by hand, straight from the price series
    j = 1000
    t = start + j
    d = lambda k: dates[t - k].date()
    print(f"\n    hand check, row {j} ({dates[t].date()}), close {prices[t]:.4f}:")
    hand = [prices[t] / prices[t - k] - 1 for k in (1, 5, 20, 60)]
    for k, h in zip((1, 5, 20, 60), hand):
        print(f"      {f'ret_{k}d':<7} = close {dates[t].date()} / close {d(k)} - 1 = "
              f"{prices[t]:.4f} / {prices[t - k]:.4f} - 1 = {h:+.6f}")
    rets20 = pd.Series(prices[t - 20: t + 1]).pct_change().dropna()
    hand.append(rets20.std() * np.sqrt(252))
    print(f"      vol_20d = stdev of the 20 daily returns {d(19)}..{dates[t].date()} x sqrt(252) = {hand[-1]:.6f}")
    hand.append(prices[t] / prices[t - 49: t + 1].mean() - 1)
    print(f"      px_vs_ma50 = close / mean(closes {d(49)}..{dates[t].date()}) - 1 = {hand[-1]:+.6f}")
    label = prices[t + 1] / prices[t] - 1
    print(f"      label   = close {dates[t + 1].date()} / close {dates[t].date()} - 1 = {label:+.6f}")
    assert np.allclose(hand + [label], rows[j]), "hand check disagrees with build_features"
    print(f"    every feature reads closes up to {dates[t].date()} and none after;")
    print(f"    only the label reads {dates[t + 1].date()}. Matches build_features exactly.")

    # Q2: time-ordered split
    tr_rows, va_rows, te_rows = split_by_time(rows)
    n_tr, n_va = len(tr_rows), len(va_rows)
    sets = {"training": (0, n_tr), "validation": (n_tr, n_tr + n_va), "test": (n_tr + n_va, len(rows))}
    print("\nQ2  split by time, 60/20/20, no shuffling (dates are the day the features are read)")
    for name, (a, b) in sets.items():
        print(f"    {name:<11}{row_dates[a].date()} to {row_dates[b - 1].date()}   {b - a:>5} rows")
    assert row_dates[n_tr - 1] < row_dates[n_tr] and row_dates[n_tr + n_va - 1] < row_dates[n_tr + n_va]
    train, val, test = xy(tr_rows), xy(va_rows), xy(te_rows)

    # Q3: four models, default settings
    models = {
        "linear regression": LinearRegression(),
        "ridge (alpha=1)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random forest": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        "gradient boosting": GradientBoostingRegressor(random_state=SEED),
    }
    print("\nQ3  R-squared predicting next-day return")
    print(f"    {'model':<20}{'training':>10}{'validation':>12}")
    for name, m in models.items():
        r_tr, r_va = r2s(m.fit(*train), train, val)
        print(f"    {name:<20}{r_tr:>10.4f}{r_va:>12.4f}")

    # Q4: tune one hyperparameter each on validation
    alphas = list(np.logspace(-2, 5, 15))
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    n_trees = [1, 2, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 500]
    ridge_a, ridge_s, ridge = tuned(lambda a: make_pipeline(StandardScaler(), Ridge(alpha=a)),
                                    alphas, train, val)
    rf_d, rf_s, rf = tuned(lambda d: RandomForestRegressor(n_estimators=300, max_depth=d,
                                                           random_state=SEED, n_jobs=-1),
                           depths, train, val)
    gb_n, gb_s, gb = tuned(lambda n: GradientBoostingRegressor(n_estimators=n, random_state=SEED),
                           n_trees, train, val)
    print("\nQ4  one hyperparameter each, chosen on validation R-squared")
    print(f"    ridge             alpha = {ridge_a:<10.4g}validation R2 {max(ridge_s):.4f}")
    print(f"    random forest     max_depth = {rf_d:<6}validation R2 {max(rf_s):.4f}")
    print(f"    gradient boosting n_estimators = {gb_n:<3}validation R2 {max(gb_s):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, grid, scores, best, label_, title in [
        (axes[0], alphas, ridge_s, ridge_a, "alpha (standardised features)", "Ridge"),
        (axes[1], depths, rf_s, rf_d, "max_depth", "Random forest (300 trees)"),
        (axes[2], n_trees, gb_s, gb_n, "n_estimators (depth 3, learning rate 0.1)", "Gradient boosting"),
    ]:
        ax.plot(grid, scores, "o-", lw=1.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(best, color="tab:red", ls="--", lw=1, label=f"chosen: {best:.4g}")
        if title != "Random forest (300 trees)":
            ax.set_xscale("log")
        ax.set_xlabel(label_)
        ax.set_ylabel("validation R-squared")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"Tuning on the validation set ({row_dates[n_tr].date()} to {row_dates[n_tr + n_va - 1].date()})")
    fig.tight_layout()
    fig.savefig("HW7_tuning.png", dpi=150)
    plt.close(fig)

    # Q5: best model by validation, once on test
    lin = models["linear regression"]
    finalists = {
        "linear regression": (lin, r2s(lin, train, val)[1]),
        f"ridge, alpha={ridge_a:.4g}": (ridge, max(ridge_s)),
        f"random forest, max_depth={rf_d}": (rf, max(rf_s)),
        f"gradient boosting, n_estimators={gb_n}": (gb, max(gb_s)),
    }
    best_name = max(finalists, key=lambda k: finalists[k][1])
    best = finalists[best_name][0]
    pred = best.predict(test[0])
    up_share = up_day_baseline(prices[start + n_tr + n_va: len(prices)])
    print(f"\nQ5  best on validation: {best_name} (validation R2 {finalists[best_name][1]:.4f})")
    print(f"    test set, run once: R2 {r2_score(test[1], pred):.4f}   "
          f"corr(predicted, actual) {np.corrcoef(pred, test[1])[0, 1]:+.4f}")
    print(f"    predicted up on {np.mean(pred > 0):.1%} of test days; directional accuracy "
          f"{np.mean((pred > 0) == (test[1] > 0)):.1%} vs {up_share:.1%} of test days actually up")

    # Q6: trading rule, backtested on the test period with the Homework 4 backtester.
    # Hand it the 60 closes before the first test day so the window is full on day one.
    test_prices = prices[start + n_tr + n_va - (LOOKBACK - 1):]
    strat = make_model_strategy(best)
    held = [strat(test_prices[i - LOOKBACK + 1: i + 1]) for i in range(LOOKBACK - 1, len(test_prices) - 1)]
    assert np.array_equal(held, pred > 0), "window strategy disagrees with the feature table"
    res = {"model rule": backtest(test_prices, strat), "buy and hold": backtest(test_prices, always_hold)}
    bh_check = test_prices[-1] / test_prices[LOOKBACK - 1] - 1
    assert abs(res["buy and hold"]["total_return"] - bh_check) < 1e-9
    # For comparison only, not selected: the unregularised linear model, which actually moves
    res["linear (unselected)"] = backtest(test_prices, make_model_strategy(lin))
    print(f"\nQ6  hold {TICKER} when the prediction is positive, cash otherwise; test period "
          f"{row_dates[n_tr + n_va].date()} to {dates[-1].date()}")
    print(f"    {'':<20}{'annualised':>11}{'max drawdown':>14}{'invested':>10}{'total':>9}")
    for name, r in res.items():
        print(f"    {name:<20}{r['annualised_return']:>11.2%}{r['max_drawdown']:>14.2%}"
              f"{r['fraction_invested']:>10.1%}{r['total_return']:>9.1%}")
    test_dates = row_dates[n_tr + n_va:]
    for k in np.flatnonzero(~np.array(held)):
        print(f"    model rule in cash on {test_dates[k].date()} (that day {test[0][k][0]:+.2%}), "
              f"missing {dates[start + n_tr + n_va + k + 1].date()} ({test[1][k]:+.2%})")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    eq_dates = dates[start + n_tr + n_va + 1:]
    for name, r in res.items():
        ax.plot(eq_dates, r["equity"], lw=1.6, label=name)
    ax.set_ylabel("growth of $1")
    ax.set_title(f"Test period: {best_name} as a long/cash rule vs buy and hold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("HW7_backtest.png", dpi=150)
    plt.close(fig)

    # Q7: what the models leaned on
    std_lin = make_pipeline(StandardScaler(), LinearRegression()).fit(*train)
    coefs = std_lin[-1].coef_
    print("\nQ7  linear regression on standardised features: coefficient per 1 s.d. move, ranked by size")
    for i in np.argsort(-np.abs(coefs)):
        print(f"    {FEATURES[i]:<12}{coefs[i] * 1e4:>+8.2f} bp")
    if best_name.startswith("ridge"):
        rc = ridge[-1].coef_
        print(f"    chosen ridge (alpha={ridge_a:.4g}), same scale:")
        for i in np.argsort(-np.abs(rc)):
            print(f"    {FEATURES[i]:<12}{rc[i] * 1e4:>+8.2f} bp")
    for name, m in [(f"random forest (max_depth={rf_d})", rf), (f"gradient boosting (n_estimators={gb_n})", gb)]:
        imp = m.feature_importances_
        print(f"    {name} importances, ranked:")
        for i in np.argsort(-imp):
            print(f"    {FEATURES[i]:<12}{imp[i]:>8.3f}")


if __name__ == "__main__":
    main()
