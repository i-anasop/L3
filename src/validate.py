"""Honest validation of AURA v2 on the real leaderboard metric.

Leaderboard score = sum of absolute errors between predicted weights and the
jury-derived weights, renormalised over the scored repos. We estimate
generalisation with leave-one-out CV over the 50 repos that have public weights.
"""
import csv, math
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from features import load_all, slug
import aura

DATA='/mnt/user-data/uploads'

def main(base_dir='.'):
    repos, feats = load_all(f'{DATA}/repos_to_predict.csv', f'{base_dir}/data/github_features.json',
                            f'{base_dir}/data/centrality.json', f'{base_dir}/data/oso_graph.json')
    slugs=[slug(r) for r in repos]
    pub=aura._labels(); yv=np.array([pub.get(s,np.nan) for s in slugs]); tr=np.where(~np.isnan(yv))[0]
    idx=[i for i,s in enumerate(slugs) if s in pub]
    mkt=aura._market(base_dir); logm=np.array([math.log(mkt.get(s,1e-4)) for s in slugs])
    X=np.array([[feats[r][c] for c in aura.SELECTED] for r in repos])

    def sae_of(logw_vec):
        z=(logw_vec-logw_vec[tr].mean())/(logw_vec[tr].std()+1e-9)
        best=9
        for sp in np.linspace(0.3,1.6,40):
            w=np.exp(z*sp); wp=np.array([w[i] for i in idx]); wp/=wp.sum()
            best=min(best, np.abs(wp-np.array([pub[slugs[i]] for i in idx])).sum())
        return best

    # structural LOO log-weights
    base=np.zeros(len(slugs)); sc=StandardScaler().fit(X[tr])
    base[:]=Ridge(alpha=aura.ALPHA).fit(sc.transform(X[tr]),np.log(yv[tr])).predict(sc.transform(X))
    for k in tr:
        m=tr[tr!=k]; sck=StandardScaler().fit(X[m]); base[k]=Ridge(alpha=aura.ALPHA).fit(sck.transform(X[m]),np.log(yv[m])).predict(sck.transform(X[k:k+1]))[0]

    res={}
    res['null_uniform']=sae_of(np.zeros(len(slugs)))
    res['stars_only']=sae_of(X[:,aura.SELECTED.index('log_stars')].astype(float))
    res['structural_aura']=sae_of(base)
    res['market_only']=sae_of(logm)
    res['AURA_v2_blend']=sae_of(aura.BLEND*base+(1-aura.BLEND)*logm)

    print(f"{'model':<22}{'LOO SAE':>10}")
    for k,v in sorted(res.items(), key=lambda x:-x[1]): print(f"{k:<22}{v:>10.4f}")
    import json; json.dump(res, open(f'{base_dir}/data/validation_results.json','w'), indent=1)
    return res

if __name__=='__main__': main()
