"""Deterministic reporting only; no fitting, no model selection, no new evaluation data."""
from pathlib import Path
import argparse,hashlib,json,importlib.metadata
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,confusion_matrix
ROOT=Path(__file__).resolve().parent
COLORS={'DeltaBeta':'#28608b','MI':'#cb8030','L2coef':'#337b55'}
LABELS={'DeltaBeta':'Absolute Δβ','MI':'Mutual information','L2coef':'L2 coefficients'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default=str(ROOT.parent/'results'));args=ap.parse_args();out=Path(args.results);figs=out/'figures';figs.mkdir(exist_ok=True)
    s=pd.read_csv(out/'summary.csv');pred=pd.read_csv(out/'predictions.csv');st=pd.read_csv(out/'stability_summary.csv')
    rank={'overall':{'n_configurations':len(s),'spearman_rho':float(spearmanr(s.balanced_accuracy_mean.round(12),s.external_balanced_accuracy.round(12)).statistic)},'by_k':[],'interpretation':'Descriptive Spearman correlation including tied ranks; BA rounded to12 decimal places only for rank ties to remove floating-point noise, not for reported metrics. Configurations share samples and are dependent; no p-values or independent-method inferential claims.'}
    for k,g in s.groupby('k'):rank['by_k'].append({'k':int(k),'n_configurations':len(g),'spearman_rho':float(spearmanr(g.balanced_accuracy_mean.round(12),g.external_balanced_accuracy.round(12)).statistic)})
    (out/'rank_transfer.json').write_text(json.dumps(rank,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
    def save(fig,name):
        fig.savefig(figs/(name+'.png'),bbox_inches='tight');fig.savefig(figs/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    rows=[(m,a) for m in ['LR','LinearSVM','RF'] for a in ['DeltaBeta','MI','L2coef']];ks=[30,100,500]
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.3),sharey=True,layout='constrained')
    for ax,col,title in zip(axs,['balanced_accuracy_mean','external_balanced_accuracy'],['Internal: mean outer-fold BA','External: frozen models BA']):
        z=np.array([[s[(s.model==m)&(s.selector==a)&(s.k==k)][col].iloc[0] for k in ks] for m,a in rows]);im=ax.imshow(z,vmin=.5,vmax=1,cmap='Blues',aspect='auto');ax.set_xticks(range(3),ks);ax.set_xlabel('Number of CpGs');ax.set_title(title);ax.set_yticks(range(len(rows)),[m+' | '+{'DeltaBeta':'Δβ','MI':'MI','L2coef':'L2coef'}[a] for m,a in rows]);
        for i in range(z.shape[0]):
            for j in range(3):ax.text(j,i,f'{z[i,j]:.3f}',ha='center',va='center',color='white' if z[i,j]>.79 else 'black',fontsize=9)
    fig.colorbar(im,ax=axs,label='Balanced accuracy',shrink=.8);save(fig,'01_internal_external_ba')
    fig,axs=plt.subplots(1,3,figsize=(7.1,2.7),sharey=True,layout='constrained')
    for ax,m in zip(axs,['LR','LinearSVM','RF']):
        for a in COLORS:
            g=s[(s.model==m)&(s.selector==a)].sort_values('k');ax.plot(g.k,g.external_balanced_accuracy,'o-',color=COLORS[a],label=LABELS[a])
        ax.set_xscale('log');ax.set_xticks(ks,ks);ax.set_ylim(.45,1.015);ax.grid(alpha=.2);ax.set_xlabel('Number of CpGs');ax.set_title(m)
    axs[0].set_ylabel('External balanced accuracy');axs[1].legend(fontsize=7,loc='lower left');save(fig,'02_external_ba_by_panel_size')
    fig,ax=plt.subplots(figsize=(3.5,2.8),layout='constrained')
    for a in COLORS:
        g=st[st.selector==a].sort_values('k');ax.errorbar(g.k,g.jaccard_mean,yerr=g.jaccard_fold_sd_descriptive,marker='o',capsize=4,label=LABELS[a],color=COLORS[a])
    ax.set_xscale('log');ax.set_xticks(ks,ks);ax.set_xlabel('Number of CpGs');ax.set_ylabel('Mean panel Jaccard similarity');ax.set_ylim(0,1);ax.grid(alpha=.2);ax.legend(fontsize=7.5);ax.set_title('Training-panel stability',fontsize=9);save(fig,'03_panel_stability')
    fig,axs=plt.subplots(1,4,figsize=(10,2.7),gridspec_kw={'width_ratios':[1.4,1,1,1]},layout='constrained');e=pred[(pred.cohort=='external')&(pred.model=='LR')&(pred.k==100)]
    for j,a in enumerate(COLORS):
        g=e[e.selector==a];fpr,tpr,_=roc_curve(g.y_true,g.score);axs[0].plot(fpr,tpr,color=COLORS[a],label=LABELS[a]);cm=confusion_matrix(g.y_true,g.y_pred,labels=[0,1]);axs[j+1].imshow(cm,cmap='Blues',vmin=0,vmax=82);axs[j+1].set_xticks([0,1],['Normal','CRC']);axs[j+1].set_yticks([0,1],['Normal','CRC']);axs[j+1].set(xlabel='Predicted class',ylabel='Observed class',title=LABELS[a])
        for r in range(2):
            for c in range(2):axs[j+1].text(c,r,str(cm[r,c]),ha='center',va='center',color='white' if cm[r,c]>45 else 'black',fontsize=11)
    axs[0].plot([0,1],[0,1],'--',color='.6');axs[0].set(xlabel='False positive rate',ylabel='True positive rate',title='External ROC: LR, k=100');axs[0].legend(fontsize=7);save(fig,'04_lr100_roc_and_confusions')
    # Convenience outputs carry provenance and never replace full 27-configuration tables.
    chosen=s.sort_values(['model','selector','balanced_accuracy_mean','k'],ascending=[True,True,False,True]).groupby(['model','selector'],as_index=False).head(1)
    chosen.to_csv(out/'descriptive_best_k_by_internal.csv',index=False)
    contrasts=json.loads((out/'primary_comparison.json').read_text());qc=json.loads((out/'qc_summary.json').read_text())
    report={'primary_reference':s[(s.model=='LR')&(s.k==100)].to_dict(orient='records'),'best_internal_configuration_descriptive':s.sort_values(['balanced_accuracy_mean','k'],ascending=[False,True]).iloc[0].to_dict(),'external_range':[float(s.external_balanced_accuracy.min()),float(s.external_balanced_accuracy.max())],'stability':st.to_dict(orient='records'),'rank_transfer':rank,'predeclared_contrasts':contrasts,'counts':qc}
    (out/'key_results.json').write_text(json.dumps(report,indent=2))
    manifest=json.loads((out/'manifest.json').read_text());manifest['units']={'whole_process_peak_rss_mb':'MiB (1024 squared bytes); retained historical field suffix _mb','total_ram_gb':'GiB (1024 cubed bytes); host visibility only','time':'seconds'};manifest['hardware']['cpu_model']=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown') if Path('/proc/cpuinfo').exists() else 'unknown';manifest['hardware']['cgroup_memory_max']=Path('/sys/fs/cgroup/memory.max').read_text().strip() if Path('/sys/fs/cgroup/memory.max').exists() else 'unknown';manifest['hardware']['cgroup_cpu_max']=Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else 'unknown';manifest['versions']['matplotlib']=matplotlib.__version__;manifest['versions']['threadpoolctl']=importlib.metadata.version('threadpoolctl');manifest['timing_note']=manifest['timing_note'].split(' Per-panel scaler')[0]+' Per-panel scaler fit/transform overhead is omitted from attributed component totals, but included in overall wall time.';manifest['postprocessing']={'script':'analyze_results.py','sha256':sha(__file__),'note':'Computes descriptive per-k rank correlation, convenience tables and figures. No fitting or new model selection; best-k table is descriptive and subject to selection optimism.'};manifest['files_sha256']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['manifest.json','run.log']};(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Reporting complete:',figs)
if __name__=='__main__':main()
