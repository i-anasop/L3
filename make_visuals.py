import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np, json, csv, sys
sys.path.insert(0,'src')

CORAL='#e07a5f'; CORAL_D='#c75d42'; INK='#2b2b2b'; MUTE='#8a8f98'; BG='#ffffff'; PANEL='#faf7f5'
plt.rcParams.update({'font.family':'DejaVu Sans','axes.edgecolor':'#e6e1de','text.color':INK,
    'axes.labelcolor':INK,'xtick.color':MUTE,'ytick.color':MUTE,'figure.facecolor':BG,'axes.facecolor':BG})

# 1) BANNER
fig,ax=plt.subplots(figsize=(22,5.6)); ax.set_xlim(0,22);ax.set_ylim(0,5.6);ax.axis('off')
fig.patch.set_facecolor('#fbf9f8')
F={'A':['010','101','111','101','101'],'U':['101','101','101','101','111'],'R':['110','101','110','101','101']}
glyph='AURA'; BW=0.62;BH=0.66;GP=0.1;LG=0.42
tw=len(glyph)*(3*(BW+GP)-GP+LG)-LG; sx=(22-tw)/2; sy=4.3
for ch in glyph:
    g=F[ch]
    for ri,row in enumerate(g):
        for ci,c in enumerate(row):
            if c=='1':
                x=sx+ci*(BW+GP); y=sy-ri*(BH+GP)
                ax.add_patch(FancyBboxPatch((x,y),BW,BH,boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor=CORAL,edgecolor=CORAL_D,linewidth=1.3))
    sx+=3*(BW+GP)-GP+LG
ax.text(11,0.95,'Log-Space Importance Model for Ethereum',ha='center',fontsize=15,color=INK,style='italic')
ax.text(11,0.4,'Deep Funding GG24  ·  Level I  ·  by Anas',ha='center',fontsize=11,color=MUTE)
plt.savefig('assets/banner.png',dpi=150,bbox_inches='tight',facecolor='#fbf9f8'); plt.close()

# 2) VALIDATION ABLATION (centerpiece)
res=json.load(open('data/validation_results.json'))
order=['null_uniform','pagerank_only','log_retro_only','log_gitcoin_only','tier_prior_only','log_forks_only','log_stars_only','AURA_loo']
labels={'null_uniform':'Uniform (null)','pagerank_only':'PageRank','log_retro_only':'RetroPGF $','log_gitcoin_only':'Gitcoin $',
        'tier_prior_only':'Role tier','log_forks_only':'Forks','log_stars_only':'Stars','AURA_loo':'AURA (full)'}
vals=[res[k] for k in order]; names=[labels[k] for k in order]
fig,ax=plt.subplots(figsize=(11,6))
cols=[MUTE]*7+[CORAL]
bars=ax.barh(range(len(vals)),vals,color=cols,height=0.66,edgecolor='white')
ax.set_yticks(range(len(vals)));ax.set_yticklabels(names,fontsize=11)
ax.invert_yaxis();ax.set_xlabel('Jury pair-cost  (lower = better, leave-one-out CV)',fontsize=11)
null=res['null_uniform']
for i,(b,v) in enumerate(zip(bars,vals)):
    pct=100*(1-v/null)
    ax.text(v+0.03,i,f'{v:.3f}'+(f'   −{pct:.0f}%' if i>0 else ''),va='center',fontsize=10,
            color=CORAL_D if i==len(vals)-1 else MUTE,fontweight='bold' if i==len(vals)-1 else 'normal')
ax.set_title('What predicts the jury?  Each signal vs the real pairwise objective',fontsize=13,fontweight='bold',color=INK,pad=14)
ax.spines[['top','right']].set_visible(False);ax.set_xlim(0,3.3)
plt.tight_layout();plt.savefig('assets/validation.png',dpi=150,bbox_inches='tight');plt.close()

# 3) TOP-20 WEIGHTS
import aura
repos,slugs,w,_=aura.predict('.')
idx=sorted(range(len(slugs)),key=lambda i:w[i],reverse=True)[:20]
fig,ax=plt.subplots(figsize=(11,7))
names=[slugs[i].split('/')[-1] for i in idx]; wv=[w[i] for i in idx]
g=np.linspace(0,1,20)
bars=ax.barh(range(20),wv,color=[(0.88-0.5*t,0.45-0.1*t,0.37) for t in g],height=0.7,edgecolor='white')
ax.set_yticks(range(20));ax.set_yticklabels(names,fontsize=10);ax.invert_yaxis()
ax.set_xlabel('Predicted importance weight',fontsize=11)
for i,v in enumerate(wv): ax.text(v+0.0004,i,f'{v:.3f}',va='center',fontsize=8.5,color=MUTE)
ax.set_title('AURA — top 20 repos by predicted importance to Ethereum',fontsize=13,fontweight='bold',pad=12)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout();plt.savefig('assets/weights.png',dpi=150,bbox_inches='tight');plt.close()

# 4) FEATURE SELECTION PATH
path=[('Stars',1.6525),('+ Age',1.6098),('+ In-graph',1.5953),('+ Gitcoin donors',1.5540),
      ('+ Role tier',1.5314),('+ Repo size',1.4938),('+ Forks',1.4799),('+ PageRank',1.4742)]
fig,ax=plt.subplots(figsize=(11,5.5))
xs=range(len(path));ys=[p[1] for p in path]
ax.plot(xs,ys,'-o',color=CORAL,markersize=9,markerfacecolor='white',markeredgecolor=CORAL,markeredgewidth=2,linewidth=2.2)
ax.fill_between(xs,ys,max(ys)+0.02,color=CORAL,alpha=0.06)
ax.set_xticks(xs);ax.set_xticklabels([p[0] for p in path],rotation=25,ha='right',fontsize=10)
ax.set_ylabel('LOO pair-cost',fontsize=11)
for x,y in zip(xs,ys): ax.text(x,y-0.012,f'{y:.3f}',ha='center',fontsize=9,color=CORAL_D)
ax.set_title('Forward feature selection — each addition lowers honest jury cost',fontsize=13,fontweight='bold',pad=12)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout();plt.savefig('assets/feature_selection.png',dpi=150,bbox_inches='tight');plt.close()

print("Generated 4 visuals: banner, validation, weights, feature_selection")
