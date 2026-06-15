# Aura — importance to Ethereum is a log-space problem

**Deep Funding GG24 · Level I**
by **Anas** · code: https://github.com/i-anasop/L3

![banner](https://raw.githubusercontent.com/i-anasop/L3/main/assets/banner.png)

Hey everyone,

For Level I I wanted to stop guessing at what the jury rewards and read it directly from the scoring code. That one decision shaped the whole model — meet **Aura**.

## Read the scoring before building the model

The Deep Funding scoring mechanism (`deepfunding/scoring`) is short and honest:

```python
def cost_function(logits, samples):
    return sum((logits[b] - logits[a] - c) ** 2 for a, b, c in samples)
```

Each jury sample `(a, b, c)` is two repos and `c` = the log of how much more important `b` is than `a`. Your cost is the squared error between **your predicted log-ratio** `(log wᵦ − log wₐ)` and the jury's. Three things fall out of this, and they define Aura:

1. **The target is log-weight, not weight.** So Aura predicts in log-space and trains to match pairwise log-ratios — the scoring's own geometry.
2. **The score is scale-invariant.** It only sees *differences* of log-weights, so normalization is irrelevant. Only the **shape** matters — the relative spacing of repos in log-space.
3. **Spread is the master knob.** Since only relative spacing counts, the dispersion of the log-weights is the single highest-leverage thing to calibrate.

This is also why anchoring tricks (matching the public CSV to many decimals) are a dead end for the *final* leaderboard: after the event, models are re-scored on held-out jury comparisons. What survives is a model that **generalizes**, not one that memorized public weights.

## How Aura works

A regularized **log-space ridge regression**, trained on the 50 repos with public jury weights, predicting all 98.

Features — chosen by forward selection against the real jury cost:
- `log_stars`, `log_forks`, `log_size` — adoption & surface area
- `age_days` — maturity
- `tier_prior` — ecosystem role (client / spec / library / tooling)
- `pagerank`, `has_graph` — dependency-graph centrality
- `gitcoin_donors` — community funding signal

Pipeline: standardize → ridge (α=10, heavy L2 since there are only 50 labels) → exponentiate → **calibrate the log-spread** to minimize held-out jury cost → normalize.

## Validation — against the actual objective

I scored every model on the **171 public pairwise jury comparisons** using the exact cost function, under **leave-one-out CV** (the same regime as the post-event held-out leaderboard, so the number actually means something).

![validation](https://raw.githubusercontent.com/i-anasop/L3/main/assets/validation.png)

| Model | Jury pair-cost | vs null |
|---|---|---|
| Uniform (null) | 2.868 | — |
| PageRank only | 2.733 | −5% |
| RetroPGF $ only | 2.463 | −14% |
| Gitcoin $ only | 2.431 | −15% |
| Role tier only | 1.915 | −33% |
| Forks only | 1.740 | −39% |
| Stars only | 1.573 | −45% |
| **Aura (full)** | **1.473** | **−49%** |

Aura explains **48.6%** of jury disagreement and beats every individual signal.

![feature selection](https://raw.githubusercontent.com/i-anasop/L3/main/assets/feature_selection.png)

## What I learned (the honest version)

- **Adoption dominates.** `log_stars` alone hits −45%. The jury's sense of importance tracks real usage more than anything else I tried — including funding and graph centrality.
- **Funding helps, but modestly.** RetroPGF and Gitcoin dollars are genuine positive signals (−14/−15%), but coverage across the 98 repos is sparse and raw adoption outranks them.
- **Simpler generalizes better.** I tried a gradient-boosted ensemble; it *lost* to plain ridge under LOO-CV (1.56 vs 1.47). With 50 training points, regularization and feature selection beat model complexity. Aura ships the simpler thing on purpose.
- **Spread calibration is real, not cosmetic.** Because scoring is scale-invariant, tuning the log-weight dispersion directly lowers pair-cost. It's the highest-leverage single parameter in the whole pipeline.

![weights](https://raw.githubusercontent.com/i-anasop/L3/main/assets/weights.png)

## Run it

```bash
git clone https://github.com/i-anasop/L3
cd L3 && pip install -r requirements.txt
cd src && python aura.py     # writes the submission CSV
python validate.py          # reproduces the LOO pair-cost table
```

Thanks to the Deep Funding team — reading the scoring code instead of guessing turned out to be the whole game.

— **Anas**
https://github.com/i-anasop/L3
