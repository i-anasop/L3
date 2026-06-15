<p align="center">
  <img src="assets/banner.png" alt="AURA" width="100%">
</p>

# AURA — Log-Space Importance Model for Ethereum

**Deep Funding GG24 · Level I** · by **Anas**

Aura assigns relative importance weights to the 98 open-source repositories Ethereum depends on. Unlike popularity- or graph-heuristic approaches, Aura is built around one fact: **the jury scores in log-space.**

---

## The key insight

The Deep Funding scoring code (`deepfunding/scoring`) reveals the real objective:

```python
def cost_function(logits, samples):
    return sum((logits[b] - logits[a] - c) ** 2 for a, b, c in samples)
```

Every jury sample is a triple `(a, b, c)` — repo `a`, repo `b`, and `c` = the log of how much more important `b` is than `a`. Your cost is the squared error between **your predicted log-ratio** `(log wᵦ − log wₐ)` and the jury's observed log-ratio. Three consequences shape the entire model:

1. **The target is log-weight, not weight.** Aura predicts in log-space and is trained to match pairwise log-ratios — the scoring's own geometry.
2. **The score is scale-invariant.** It depends only on *differences* of log-weights, so overall normalization is irrelevant. Only the **shape** — the relative spacing of repos in log-space — matters.
3. **Spread is the master parameter.** Since only relative log-spacing counts, the dispersion of the log-weights is the single most important quantity to calibrate.

---

## How Aura works

A regularized **log-space ridge regression** trained on the 50 repos with public jury weights, predicting all 98.

**Features** (selected by forward search against the real jury cost):

| Feature | Signal |
|---|---|
| `log_stars` | adoption — the dominant predictor |
| `log_forks`, `log_size` | adoption & surface area |
| `age_days` | project maturity |
| `tier_prior` | ecosystem role (client / spec / library / …) |
| `pagerank`, `has_graph` | dependency-graph centrality |
| `gitcoin_donors` | community funding signal |

**Pipeline:** standardize → ridge (α=10, heavy L2 for 50 training points) → exponentiate → **calibrate the log-spread** to minimize held-out jury cost → normalize.

---

## Validation — against the real jury objective

Every model is scored on the **171 public pairwise jury comparisons** using the exact cost function, under **leave-one-out CV** (the same generalization regime as the post-event held-out leaderboard).

<p align="center"><img src="assets/validation.png" width="80%"></p>

| Model | Jury pair-cost | vs null |
|---|---|---|
| Uniform (null) | 2.868 | — |
| PageRank only | 2.733 | −5% |
| RetroPGF $ only | 2.463 | −14% |
| Gitcoin $ only | 2.431 | −15% |
| Role tier only | 1.915 | −33% |
| Forks only | 1.740 | −39% |
| Stars only | 1.573 | −45% |
| **AURA (full)** | **1.473** | **−49%** |

Aura explains **48.6%** of jury disagreement and beats every individual signal.

<p align="center"><img src="assets/feature_selection.png" width="78%"></p>

---

## Findings worth stating honestly

- **Adoption dominates.** `log_stars` alone reaches −45%. The jury's notion of importance tracks real-world usage more than any other single feature.
- **Funding signals help but modestly.** RetroPGF and Gitcoin dollars are positive predictors (−14/−15%), but coverage is sparse across the 98 repos and they're outranked by raw adoption.
- **Simpler generalizes better.** A gradient-boosted ensemble *lost* to plain ridge in LOO-CV (1.56 vs 1.47). With 50 training points, heavy regularization and feature selection win. Aura ships the simpler model.
- **Spread calibration is real, not cosmetic.** Because scoring is scale-invariant, tuning the log-weight dispersion directly lowers pair-cost — it is the highest-leverage single knob.

<p align="center"><img src="assets/weights.png" width="80%"></p>

---

## Run it

```bash
pip install -r requirements.txt
cd src && python aura.py        # writes aura_submission.csv
python validate.py             # reproduces the LOO pair-cost table
```

## Repo layout

```
src/aura.py        the model — features → log-weights → calibrated CSV
src/features.py    feature assembly (GitHub + graph + funding + role)
src/tiers.py       ecosystem-role priors
src/validate.py    honest LOO validation against the jury cost
data/              GitHub features, centrality, OSO funding, jury data
assets/            figures
```

— **Anas**
