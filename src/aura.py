"""
AURA — log-space importance model for Deep Funding GG24 Level I.
Author: Anas

Predicts the relative importance of 98 Ethereum repos. Trained in the scoring's
own geometry: log-weights fit to the jury's pairwise log-ratios. Features are
selected by forward search against the EXACT jury cost under leave-one-out CV,
then the log-spread (the dominant parameter under scale-invariant scoring) is
calibrated to minimise held-out pair-cost.
"""
import csv, json, math
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from features import load_all, FEATURE_COLS, slug

DATA = '/mnt/user-data/uploads'
# Features chosen by forward selection on honest LOO pair-cost (see analysis/validate.py)
SELECTED = ['log_stars','age_days_log','has_graph','gitcoin_donors_log',
            'tier_prior','log_size','log_forks','pagerank']
ALPHA = 10.0       # strong L2 — 50 training points demand heavy regularisation
LOG_SPREAD = 0.74  # CV-optimal spread of log-weights (matches jury dispersion)

def _load(base_dir='.'):
    repos, feats = load_all(f'{DATA}/repos_to_predict.csv', f'{base_dir}/data/github_features.json',
                            f'{base_dir}/data/centrality.json', f'{base_dir}/data/oso_graph.json')
    slugs = [slug(r) for r in repos]
    X = np.array([[feats[r][c] for c in SELECTED] for r in repos])
    return repos, slugs, X

def _labels():
    w={}
    with open(f'{DATA}/PublicEvalR2L1_.csv') as f:
        for row in csv.DictReader(f):
            w['/'.join(row['repo'].split('/')[-2:]).lower()] = float(row['weight'])
    return w

def predict(base_dir='.'):
    repos, slugs, X = _load(base_dir)
    lab = _labels()
    yv = np.array([lab.get(s, np.nan) for s in slugs]); tr = ~np.isnan(yv)
    sc = StandardScaler().fit(X[tr])
    r = Ridge(alpha=ALPHA).fit(sc.transform(X[tr]), np.log(yv[tr]))
    base = r.predict(sc.transform(X))
    z = (base - base[tr].mean())/(base[tr].std()+1e-9)
    logw = z*LOG_SPREAD
    w = np.exp(logw); w = w/w.sum()
    return repos, slugs, w, logw

def write_submission(path='aura_submission.csv', base_dir='.'):
    repos, slugs, w, _ = predict(base_dir)
    url={}
    with open(f'{DATA}/repos_to_predict.csv') as f:
        for row in csv.DictReader(f):
            url[slug(row['repo'])] = row['repo'].strip()
    rows=[['repo','parent','weight']]
    for i,s in enumerate(slugs):
        rows.append([url.get(s,'https://github.com/'+s),'ethereum', f'{w[i]:.10f}'])
    csv.writer(open(path,'w',newline='')).writerows(rows)
    return path

if __name__ == '__main__':
    repos, slugs, w, logw = predict()
    print(f"AURA — predicted importance for {len(repos)} repos (sum={w.sum():.6f})")
    top = sorted(range(len(repos)), key=lambda i: w[i], reverse=True)[:15]
    for i in top:
        print(f"  {slugs[i]:<34} {w[i]:.4f}")
    write_submission()
    print("\nWrote aura_submission.csv")
