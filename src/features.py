"""Aura feature assembly — universal features for all 98 L1 repos."""
import csv, json, math
from datetime import datetime, timezone
from tiers import role_features

def slug(r):
    return '/'.join(r.replace('https://github.com/','').rstrip('/').lower().split('/')[:2])

def _age_days(iso):
    if not iso: return 365.0
    try:
        d = datetime.fromisoformat(iso.replace('Z','+00:00'))
        return max(1.0, (datetime.now(timezone.utc) - d).days)
    except: return 365.0

def load_all(repos_path, gh_path, cent_path, oso_path):
    repos=[]
    with open(repos_path) as f:
        for row in csv.DictReader(f): repos.append(row['repo'].strip())
    gh = json.load(open(gh_path))
    cent = json.load(open(cent_path))
    oso = json.load(open(oso_path))
    oson = {n['id'].lower().rstrip('/'): n for n in oso['nodes']}

    feats={}
    for r in repos:
        s = slug(r)
        g = gh.get(r, {})
        c = cent.get(s, {})
        o = oson.get(r.lower().rstrip('/'), {})
        rf = role_features(s)
        stars = g.get('stars',0) or 0
        forks = g.get('forks',0) or 0
        watchers = g.get('watchers',0) or 0
        retro = (o.get('retro_funding_usd') or 0)
        gitcoin = (o.get('gitcoin_grants_usd') or 0)
        feats[r] = {
            # GitHub adoption (universal)
            'log_stars': math.log1p(stars),
            'log_forks': math.log1p(forks),
            'log_watchers': math.log1p(watchers),
            'fork_ratio': forks/(stars+1),
            'log_size': math.log1p(g.get('size',0) or 0),
            'log_issues': math.log1p(g.get('issues',0) or 0),
            'age_days_log': math.log1p(_age_days(g.get('created_at',''))),
            'recency_log': math.log1p(_age_days(g.get('pushed_at',''))),
            'is_archived': 1.0 if g.get('archived') else 0.0,
            'is_fork': 1.0 if g.get('is_fork') else 0.0,
            'n_topics': len(g.get('topics',[]) or []),
            # Graph centrality (sparse — missingness handled by 0 + indicator)
            'pagerank': c.get('pagerank',0.0)*1e3,
            'eigenvector': c.get('eigenvector',0.0),
            'betweenness': c.get('betweenness',0.0)*1e2,
            'in_degree_log': math.log1p(c.get('in_degree',0)),
            'has_graph': 1.0 if c.get('pagerank',0)>0 else 0.0,
            # Funding / economic (sparse — the jury's anchor where present)
            'log_retro': math.log1p(retro),
            'log_gitcoin': math.log1p(gitcoin),
            'gitcoin_donors_log': math.log1p(o.get('gitcoin_unique_donors',0) or 0),
            'retro_rounds': o.get('num_retro_funding_rounds',0) or 0,
            'has_funding': 1.0 if (retro>0 or gitcoin>0) else 0.0,
            # Role prior
            'tier_prior': rf['tier_prior'],
            '_role': rf['role'],
        }
    return repos, feats

FEATURE_COLS = ['log_stars','log_forks','log_watchers','fork_ratio','log_size','log_issues',
    'age_days_log','recency_log','is_archived','is_fork','n_topics',
    'pagerank','eigenvector','betweenness','in_degree_log','has_graph',
    'log_retro','log_gitcoin','gitcoin_donors_log','retro_rounds','has_funding','tier_prior']
