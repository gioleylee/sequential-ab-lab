"""Compare fixed-horizon, repeated, and anytime-valid tests on paired A/B trials."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.special import betaln
from scipy.stats import binom

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("Fixed horizon", "Repeated exact test", "Beta-mixture e-test")
SHAPE = 20


def exact_p(wins, losses):
    discordant = wins + losses
    return np.minimum(1, 2 * binom.cdf(np.minimum(wins, losses), discordant, 0.5))


def log_e(wins, losses):
    return betaln(wins + SHAPE, losses + SHAPE) - betaln(SHAPE, SHAPE) + (wins + losses) * np.log(2)


def simulate(rng, trials, looks, alpha, control, treatment):
    win = treatment * (1 - control)
    loss = control * (1 - treatment)
    wins = np.zeros(trials, dtype=np.int64)
    losses = np.zeros_like(wins)
    first = {method: np.zeros_like(wins) for method in METHODS}
    curves = {method: [] for method in METHODS}
    previous = 0
    for look in looks:
        discordant = rng.binomial(look - previous, win + loss, size=trials)
        new_wins = rng.binomial(discordant, win / (win + loss))
        wins += new_wins
        losses += discordant - new_wins
        previous = look
        exact = exact_p(wins, losses) <= alpha
        evidence = log_e(wins, losses) >= -np.log(alpha)
        crossings = {
            METHODS[0]: exact if look == looks[-1] else np.zeros(trials, dtype=bool),
            METHODS[1]: exact,
            METHODS[2]: evidence,
        }
        for method, crossed in crossings.items():
            first[method][crossed & (first[method] == 0)] = look
            curves[method].append(float(np.mean(first[method] > 0)))
    return {
        method: {
            "rejection_rate": curves[method][-1],
            "monte_carlo_se": float(np.sqrt(curves[method][-1] * (1 - curves[method][-1]) / trials)),
            "median_first_crossing_if_rejected": float(np.median(stops[stops > 0])) if stops.any() else None,
            "cumulative_rejection": curves[method],
        }
        for method, stops in first.items()
    }


def run(seed, trials, horizon, every, output):
    alpha = 0.05
    looks = [*range(every, horizon, every), horizon]
    treatment_rates = (0.20, 0.24, 0.27)
    generators = np.random.SeedSequence(seed).spawn(len(treatment_rates))
    report = {"seed": seed, "trials_per_scenario": trials, "alpha": alpha, "looks": looks,
              "beta_prior": [SHAPE, SHAPE], "control_rate": 0.20, "scenarios": {}}
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True, layout="constrained")
    for axis, treatment, generator in zip(axes, treatment_rates, generators):
        result = simulate(np.random.default_rng(generator), trials, looks, alpha, 0.20, treatment)
        report["scenarios"][str(treatment)] = result
        for method, color in zip(METHODS, ("#64748b", "#b45309", "#0369a1")):
            axis.step([0, *looks], [0, *result[method]["cumulative_rejection"]], where="post",
                      color=color, label=method)
            axis.scatter(horizon, result[method]["rejection_rate"], color=color, s=18)
        axis.axhline(alpha, linestyle=":", color="#111827", alpha=0.5)
        axis.set(title=f"Treatment conversion {treatment:.0%}", xlabel="Observed pairs", ylim=(0, 1.02))
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Fraction of trials that rejected")
    axes[0].legend(loc="upper left", fontsize=8)
    figure.suptitle("What changes when an A/B test is checked repeatedly?")
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "crossings.png", dpi=170)
    plt.close(figure)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for rate, methods in report["scenarios"].items():
        print(rate, {method: round(values["rejection_rate"], 4) for method, values in methods.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--horizon", type=int, default=1_000)
    parser.add_argument("--every", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if min(args.trials, args.horizon, args.every) < 1:
        parser.error("trials, horizon, and every must be positive")
    run(args.seed, args.trials, args.horizon, args.every, args.output)
