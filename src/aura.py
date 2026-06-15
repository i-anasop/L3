"""
AURA v2 — importance model for Deep Funding GG24 Level I.
Author: i-anasop

Two signals, optimally combined in log-space:
  (1) a STRUCTURAL model — ridge over adoption / role / funding features, trained
      on the 50 repos with public jury weights (the scoring's own log-geometry);
  (2) a MARKET signal — the deep.seer.pm prediction-market price, the crowd's
      forecast of the *final* jury weights (i.e. the held-out target itself).

The blend weight and log-spread are chosen by cross-validation against the real
leaderboard metric (sum of absolute errors on weights). See src/validate.py.
"""
import csv, math
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from features import load_all, FEATURE_COLS, slug

DATA = '/mnt/user-data/uploads'
SELECTED = ['log_stars','age_days_log','has_graph','gitcoin_donors_log',
            'tier_prior','log_size','log_forks','pagerank']
ALPHA = 10.0          # ridge L2 for the structural model (50 labels -> heavy reg)
BLEND = 0.14          # weight on structural model; (1-BLEND) on the market signal
LOG_SPREAD = 0.91     # CV-optimal dispersion of the blended log-weights

def _labels():
    w={}
    with open(f'{DATA}/PublicEvalR2L1_.csv') as f:
        for row in csv.DictReader(f): w['/'.join(row['repo'].split('/')[-2:]).lower()]=float(row['weight'])
    return w

def _market(base_dir='.'):
    m={}
    with open(f'{base_dir}/data/seer_market.csv') as f:
        for row in csv.DictReader(f): m[row['repo'].lower()]=float(row['market'])
    return m

def _structural_logw(base_dir='.'):
    """Ridge prediction of log-weight from structural features (no market)."""
    repos, feats = load_all(f'{DATA}/repos_to_predict.csv', f'{base_dir}/data/github_features.json',
                            f'{base_dir}/data/centrality.json', f'{base_dir}/data/oso_graph.json')
    slugs=[slug(r) for r in repos]
    X=np.array([[feats[r][c] for c in SELECTED] for r in repos])
    lab=_labels(); yv=np.array([lab.get(s,np.nan) for s in slugs]); tr=~np.isnan(yv)
    sc=StandardScaler().fit(X[tr]); r=Ridge(alpha=ALPHA).fit(sc.transform(X[tr]), np.log(yv[tr]))
    return repos, slugs, r.predict(sc.transform(X))

def predict(base_dir='.'):
    repos, slugs, struct = _structural_logw(base_dir)
    mkt=_market(base_dir)
    logm=np.array([math.log(mkt.get(s,1e-4)) for s in slugs])
    lab=_labels(); pubmask=np.array([s in lab for s in slugs])
    # blend in log-space, then calibrate spread on the public-repo subset
    blend = BLEND*struct + (1-BLEND)*logm
    z=(blend-blend[pubmask].mean())/(blend[pubmask].std()+1e-9)
    logw=z*LOG_SPREAD
    w=np.exp(logw); w=w/w.sum()
    return repos, slugs, w, logw

def write_submission(path='aura_submission.csv', base_dir='.'):
    repos, slugs, w, _ = predict(base_dir)
    url={}
    with open(f'{DATA}/repos_to_predict.csv') as f:
        for row in csv.DictReader(f): url[slug(row['repo'])]=row['repo'].strip()
    rows=[['repo','parent','weight']]
    for i,s in enumerate(slugs): rows.append([url.get(s,'https://github.com/'+s),'ethereum',f'{w[i]:.10f}'])
    csv.writer(open(path,'w',newline='')).writerows(rows)
    return path

if __name__=='__main__':
    repos, slugs, w, logw = predict()
    print(f"AURA v2 — {len(repos)} repos, sum={w.sum():.6f}")
    for i in sorted(range(len(repos)), key=lambda i:w[i], reverse=True)[:15]:
        print(f"  {slugs[i]:<34} {w[i]:.4f}")
    write_submission()
