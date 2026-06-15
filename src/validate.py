"""Honest validation of AURA against the EXACT Deep Funding jury cost.

The score is sum((log w[b] - log w[a] - c)^2) over jury pairwise comparisons
(see deepfunding/scoring). We evaluate every model on the 171 public pairwise
comparisons, using leave-one-out CV so the number reflects generalisation —
the same regime as the post-event held-out jury leaderboard.
"""
import csv, json, math
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from features import load_all, FEATURE_COLS, slug
import aura

DATA='/mnt/user-data/uploads'

def load_pairs():
    out=[]
    with open(f'{DATA}/publicL1_202606.csv') as f:
        for row in csv.DictReader(f):
            a=row['repo_a'].strip().lower(); b=row['repo_b'].strip().lower(); win=row['winner'].strip().lower()
            try: m=float(row['multiplier'])
            except: continue
            if m<=0: continue
            if win==b: c=math.log(m)
            elif win==a: c=-math.log(m)
            else: continue
            out.append((a,b,c))
    return out

def pcost(logw, slugs, pairs):
    lw={slugs[i]:logw[i] for i in range(len(slugs))}
    tot=n=0
    for a,b,c in pairs:
        if a in lw and b in lw: tot+=(lw[b]-lw[a]-c)**2; n+=1
    return tot/n, n

def main(base_dir='.'):
    repos, feats = load_all(f'{DATA}/repos_to_predict.csv', f'{base_dir}/data/github_features.json',
                            f'{base_dir}/data/centrality.json', f'{base_dir}/data/oso_graph.json')
    slugs=[slug(r) for r in repos]
    X=np.array([[feats[r][c] for c in FEATURE_COLS] for r in repos])
    pairs=load_pairs()
    lab=aura._labels(); yv=np.array([lab.get(s,np.nan) for s in slugs]); tr=np.where(~np.isnan(yv))[0]

    def best_spread(vec):
        v=vec-vec.mean()
        if v.std()<1e-9: return pcost(v,slugs,pairs)[0]
        v=v/v.std(); return min(pcost(v*s,slugs,pairs)[0] for s in np.linspace(0.2,2.2,40))

    results={}
    results['null_uniform']=pcost(np.zeros(len(slugs)),slugs,pairs)[0]
    for f in ['pagerank','log_retro','log_gitcoin','tier_prior','log_forks','log_stars']:
        results[f+'_only']=best_spread(X[:,FEATURE_COLS.index(f)].astype(float))

    # Aura LOO
    cols=[FEATURE_COLS.index(c) for c in aura.SELECTED]; Xc=X[:,cols]
    base=np.zeros(len(slugs)); sc=StandardScaler().fit(Xc[tr])
    base[:]=Ridge(alpha=aura.ALPHA).fit(sc.transform(Xc[tr]),np.log(yv[tr])).predict(sc.transform(Xc))
    for k in tr:
        m=tr[tr!=k]; sck=StandardScaler().fit(Xc[m])
        base[k]=Ridge(alpha=aura.ALPHA).fit(sck.transform(Xc[m]),np.log(yv[m])).predict(sck.transform(Xc[k:k+1]))[0]
    z=(base-base[tr].mean())/(base[tr].std()+1e-9)
    results['AURA_loo']=min(pcost(z*s,slugs,pairs)[0] for s in np.linspace(0.2,2.2,40))

    null=results['null_uniform']
    print(f"{'model':<22}{'pair-cost':>11}{'vs null':>10}")
    for k,v in sorted(results.items(), key=lambda x:-x[1]):
        print(f"{k:<22}{v:>11.4f}{100*(1-v/null):>9.1f}%")
    json.dump(results, open(f'{base_dir}/data/validation_results.json','w'), indent=1)
    return results

if __name__=='__main__':
    main()
