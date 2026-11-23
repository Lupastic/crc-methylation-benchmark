"""Reproducible real-data colorectal methylation pilot. See README for limitations."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse,csv,gzip,hashlib,itertools,json,platform,re,shutil,sys,threading,time,urllib.request,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import psutil,scipy,sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold,ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent

def dump_json(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda z:z.item() if isinstance(z,np.generic) else str(z)))
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def download(acc,cfg):
    p=ROOT/'data/raw'/f'{acc}_series_matrix.txt.gz';p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        with urllib.request.urlopen(urllib.request.Request(cfg['dataset_urls'][acc],headers={'User-Agent':'Mozilla/5.0'}),timeout=120) as r,open(p,'wb') as f: shutil.copyfileobj(r,f)
    return p

def load_geo(acc,cfg,out):
    p=download(acc,cfg); cache=ROOT/'data/processed'/acc;cache.mkdir(parents=True,exist_ok=True)
    if (cache/'X.npy').exists():
        X=np.load(cache/'X.npy',mmap_mode='r');probes=np.load(cache/'probes.npy');md=pd.read_csv(cache/'metadata.csv');qc=json.loads((cache/'qc.json').read_text())
    else:
        meta={}; rows=[]
        with gzip.open(p,'rt') as f:
            for line in f:
                if line.startswith('!Sample_title'):meta['title']=next(csv.reader([line],delimiter='\t'))[1:]
                elif line.startswith('!Sample_geo_accession'):meta['sample_id']=next(csv.reader([line],delimiter='\t'))[1:]
                elif line.startswith('!Sample_characteristics_ch1'):rows.append(next(csv.reader([line],delimiter='\t'))[1:])
                elif line.startswith('!series_matrix_table_begin'):break
            header=next(csv.reader([next(f)],delimiter='\t'))[1:]
            md=pd.DataFrame(meta)
            for chars in rows:
                for i,v in enumerate(chars):
                    if ': ' in v:
                        k,value=v.split(': ',1);md.loc[i,k]=value
            assert header==md.sample_id.tolist()
            # Stream all 450K probes; numeric matrix is float32, without retaining text rows.
            matrix=np.empty((len(md),500000),dtype=np.float32);names=[]
            for j,line in enumerate(f):
                if line.startswith('!series_matrix_table_end'):break
                ident,numbers=line.rstrip('\n').split('\t',1)
                vals=np.fromstring('\t'.join(v or 'nan' for v in numbers.replace('null','nan').replace('NA','nan').split('\t')),sep='\t',dtype=np.float32)
                assert len(vals)==len(md),(acc,j,len(vals))
                matrix[:,j]=vals;names.append(ident.strip('"'))
            X=matrix[:,:len(names)].copy();del matrix;probes=np.asarray(names)
        assert md.sample_id.is_unique and len(set(probes))==len(probes)
        assert np.nanmin(X)>=0 and np.nanmax(X)<=1
        profile_hashes=[hashlib.sha256(np.nan_to_num(r,nan=-1).tobytes()).hexdigest() for r in X]
        dup=len(profile_hashes)-len(set(profile_hashes))
        assert dup==0,'Duplicate profiles need explicit grouping review'
        qc={'dataset':acc,'n_samples':len(md),'n_probes':len(probes),'missing_cells':int(np.isnan(X).sum()),'missing_fraction':float(np.isnan(X).mean()),'beta_min':float(np.nanmin(X)),'beta_max':float(np.nanmax(X)),'duplicate_profiles':dup,'raw_bytes':p.stat().st_size,'raw_sha256':sha256(p),'source_url':cfg['dataset_urls'][acc]}
        np.save(cache/'X.npy',X);np.save(cache/'probes.npy',probes);md.to_csv(cache/'metadata.csv',index=False);dump_json(cache/'qc.json',qc)
    md['dataset']=acc
    if acc=='GSE48684':
        md['status']=md['disease status'];md['y']=md.status.map({'cancer':1,'normal-H':0,'normal-C':0,'adenoma':-1})
        md['group_id']=[acc+'_'+str(s)+'_'+re.search(r'(\d+)\s*$',t).group(1) for s,t in zip(md.status,md.title)]
        md['role']=np.where(md.status.isin(['cancer','normal-H']),'development','exploratory')
    else:
        md['status']=md.tissue;md['y']=md.tissue.map({'Tumor':1,'Normal':0,'IVD':-1})
        def infer_group(t):
            t=t.rstrip('_'); m=re.match(r'^(ES|FI)(MSI|C|S)(?:M|P)?[TN](\d+)$',t)
            return acc+'_'+m.group(1)+'_'+m.group(2)+'_'+str(int(m.group(3))) if m else acc+'_unresolved_'+t
        md['group_id']=md.title.map(infer_group);md['role']=np.where(md.status=='IVD','excluded_control','external')
    md.to_csv(out/f'{acc}_metadata.csv',index=False)
    return X,probes,md,qc

def prep_fit(X,cfg):
    keep=np.flatnonzero(np.isnan(X).mean(axis=0)<=cfg['max_train_missing_fraction'])
    med=np.nanmedian(X[:,keep],axis=0);ok=np.isfinite(med);keep,med=keep[ok],med[ok]
    z=np.where(np.isnan(X[:,keep]),med,X[:,keep]);var=z.var(axis=0)
    order=np.lexsort((keep,-var));order=order[var[order]>0][:cfg['variance_prefilter']]
    return {'columns':keep[order],'medians':med[order]}
def prep_transform(X,state):
    z=X[:,state['columns']];return np.where(np.isnan(z),state['medians'],z)
def rank_selectors(z,y,cfg,seed):
    ranks={};scores={};times={}
    for method in cfg['selection_methods']:
        t=time.perf_counter()
        if method=='DeltaBeta':s=np.abs(z[y==1].mean(0)-z[y==0].mean(0))
        elif method=='MI':s=mutual_info_classif(z,y,n_neighbors=cfg['mi_neighbors'],random_state=seed,n_jobs=1)
        else:
            zz=StandardScaler().fit_transform(z);m=LogisticRegression(C=cfg['coefficient_selector_C'],solver='liblinear',class_weight='balanced',max_iter=5000,random_state=seed).fit(zz,y);s=np.abs(m.coef_[0])
        ranks[method]=np.lexsort((np.arange(len(s)),-s));scores[method]=s;times[method]=time.perf_counter()-t
    return ranks,scores,times

def prepare(X,y,train,test,cfg,seed):
    t=time.perf_counter();pp=prep_fit(X[train],cfg);a,b=prep_transform(X[train],pp),prep_transform(X[test],pp);prep_secs=time.perf_counter()-t
    ranks,scores,sel_secs=rank_selectors(a,y[train],cfg,seed);panels={}
    for method in cfg['selection_methods']:
        for k in cfg['panel_sizes']:
            ix=ranks[method][:k];rawa,rawb=a[:,ix],b[:,ix];sc=StandardScaler().fit(rawa)
            panels[method,k]={'raw':(rawa,rawb),'scaled':(sc.transform(rawa),sc.transform(rawb))}
    return {'pp':pp,'ranks':ranks,'scores':scores,'panels':panels,'prep_seconds':prep_secs,'selection_seconds':sel_secs}

def splits(y,groups,n,seed):
    cv=StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed);result=list(cv.split(np.zeros(len(y)),y,groups))
    for a,b in result:
        assert not(set(groups[a])&set(groups[b]));assert len(set(y[a]))==len(set(y[b]))==2
    return result

def grid(name,cfg):return list(ParameterGrid(cfg['RF_grid'])) if name=='RF' else [{'C':c} for c in cfg['C_grid']]
def make_model(name,params,seed):
    if name=='LR':return LogisticRegression(solver='liblinear',class_weight='balanced',max_iter=5000,random_state=seed,**params)
    if name=='LinearSVM':return LinearSVC(class_weight='balanced',dual='auto',max_iter=10000,random_state=seed,**params)
    return RandomForestClassifier(class_weight='balanced',n_jobs=1,random_state=seed,**params)
def matrix_pair(prepared,method,k,name):return prepared['panels'][method,k]['raw' if name=='RF' else 'scaled']
def score_model(m,X,name):return m.decision_function(X) if name=='LinearSVM' else m.predict_proba(X)[:,1]
def measure(y,s,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {'balanced_accuracy':balanced_accuracy_score(y,p),'roc_auc':roc_auc_score(y,s),'average_precision':average_precision_score(y,s),'accuracy':accuracy_score(y,p),'f1':f1_score(y,p,zero_division=0),'precision':precision_score(y,p,zero_division=0),'sensitivity':recall_score(y,p),'specificity':tn/(tn+fp),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
class PeakRSS:
    def __enter__(self):
        self.stop=threading.Event();self.proc=psutil.Process();self.peak=self.proc.memory_info().rss
        def watch():
            while not self.stop.wait(.02):self.peak=max(self.peak,self.proc.memory_info().rss)
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start();return self
    def __exit__(self,*_):self.peak=max(self.peak,self.proc.memory_info().rss);self.stop.set();self.thread.join()

def tune(inner,method,k,name,cfg,fold,searchrows):
    candidates=[]
    for ci,params in enumerate(grid(name,cfg)):
        ba=[]
        for j,(p,yt,yv) in enumerate(inner):
            a,b=matrix_pair(p,method,k,name);m=make_model(name,params,cfg['seed']+fold*100+j).fit(a,yt);ba.append(balanced_accuracy_score(yv,m.predict(b)))
        candidates.append(float(np.mean(ba)));searchrows.append({'fold':fold,'model':name,'selector':method,'k':k,'candidate_index':ci,'parameters':json.dumps(params),'inner_ba_mean':np.mean(ba),**{f'inner_ba_{j+1}':v for j,v in enumerate(ba)}})
    best=int(np.argmax(candidates));return grid(name,cfg)[best],candidates[best]

def grouped_subsample(y,groups,fraction,rng):
    # Group strata defined by all labels in group; preserves any known paired group.
    strata={}
    for g in np.unique(groups):strata.setdefault(tuple(sorted(set(y[groups==g]))),[]).append(g)
    take=[]
    for gs in strata.values():take.extend(rng.choice(gs,max(1,int(len(gs)*fraction)),replace=False).tolist())
    return np.flatnonzero(np.isin(groups,take))

def stability(X,y,groups,train,probes,cfg,fold):
    rng=np.random.default_rng(cfg['seed']+10000+fold);panels={(m,k):[] for m in cfg['selection_methods'] for k in cfg['panel_sizes']};times=[]
    for r in range(cfg['stability_perturbations']):
        ids=train[grouped_subsample(y[train],groups[train],cfg['subsample_fraction'],rng)];t=time.perf_counter();pp=prep_fit(X[ids],cfg);z=prep_transform(X[ids],pp);ranks,_,_=rank_selectors(z,y[ids],cfg,cfg['seed']+fold*1000+r)
        times.append(time.perf_counter()-t)
        for m,k in panels:panels[m,k].append(set(probes[pp['columns'][ranks[m][:k]]]))
    result=[]
    for (m,k),sets in panels.items():
        js=[len(a&b)/len(a|b) for a,b in itertools.combinations(sets,2)]
        result.append({'fold':fold,'selector':m,'k':k,'perturbations':len(sets),'dependent_pair_count':len(js),'jaccard_mean':np.mean(js),'jaccard_pair_sd_descriptive':np.std(js,ddof=1),'jaccard_min':min(js),'jaccard_max':max(js),'shared_stability_seconds':sum(times)})
    return result

def run_nested(X,md,probes,cfg,out):
    y=md.y.to_numpy(int);groups=md.group_id.to_numpy();foldrows=[];predrows=[];searchrows=[];features=[];splitrows=[];stab=[]
    for fold,(train,test) in enumerate(splits(y,groups,cfg['outer_folds'],cfg['seed']),1):
        print(f'OUTER {fold}: train={len(train)} test={len(test)}',flush=True)
        for i in test:splitrows.append({'sample_id':md.sample_id.iloc[i],'group_id':groups[i],'fold':fold,'y':y[i]})
        outer=prepare(X,y,train,test,cfg,cfg['seed']+fold*100);inner=[]
        for j,(a,b) in enumerate(splits(y[train],groups[train],cfg['inner_folds'],cfg['seed']+fold)):
            ai,bi=train[a],train[b];inner.append((prepare(X,y,ai,bi,cfg,cfg['seed']+fold*100+j+1),y[ai],y[bi]))
        for method in cfg['selection_methods']:
            for k in cfg['panel_sizes']:
                for rank,i in enumerate(outer['ranks'][method][:k],1):features.append({'fold':fold,'selector':method,'k':k,'rank':rank,'cpg':probes[outer['pp']['columns'][i]],'train_score':outer['scores'][method][i]})
                for name in cfg['classifiers']:
                    t=time.perf_counter();cpu=time.process_time()
                    with PeakRSS() as rss:
                        params,best=tune(inner,method,k,name,cfg,fold,searchrows);a,b=matrix_pair(outer,method,k,name);m=make_model(name,params,cfg['seed']+fold*100).fit(a,y[train]);s=score_model(m,b,name);p=m.predict(b)
                    wall=time.perf_counter()-t;cpu=time.process_time()-cpu;prep=outer['prep_seconds']+sum(z[0]['prep_seconds'] for z in inner);sel=outer['selection_seconds'][method]+sum(z[0]['selection_seconds'][method] for z in inner)
                    row={'fold':fold,'model':name,'selector':method,'k':k,'n_train':len(train),'n_test':len(test),'n_train_retained_cpg':len(outer['pp']['columns']),'best_parameters':json.dumps(params),'best_inner_ba':best,**measure(y[test],s,p),'model_search_fit_seconds':wall,'model_search_cpu_seconds':cpu,'shared_preprocessing_seconds':prep,'shared_selection_seconds':sel,'attributed_total_seconds':wall+prep+sel,'whole_process_peak_rss_mb':rss.peak/1024**2};foldrows.append(row)
                    for j,i in enumerate(test):predrows.append({'cohort':'internal','sample_id':md.sample_id.iloc[i],'group_id':groups[i],'status':md.status.iloc[i],'fold':fold,'model':name,'selector':method,'k':k,'y_true':y[i],'score':s[j],'y_pred':p[j]})
                    print(f" {name} {method} k{k} BA={row['balanced_accuracy']:.4f}",flush=True)
        print(' Stability perturbations',flush=True);stab.extend(stability(X,y,groups,train,probes,cfg,fold))
        for file,rows in [('fold_metrics.csv',foldrows),('predictions.csv',predrows),('hyperparameter_search.csv',searchrows),('selected_features.csv',features),('splits.csv',splitrows),('stability_by_fold.csv',stab)]:pd.DataFrame(rows).to_csv(out/file,index=False)
    return pd.DataFrame(foldrows),pd.DataFrame(predrows),pd.DataFrame(stab),searchrows

def external_run(X,md,Xext,em,Xextra,xm,probes,cfg,out,searchrows):
    y=md.y.to_numpy(int);groups=md.group_id.to_numpy();alltrain=np.arange(len(y));outer=prepare(X,y,alltrain,alltrain,cfg,cfg['seed']+9000);inner=[]
    for j,(a,b) in enumerate(splits(y,groups,cfg['inner_folds'],cfg['seed']+900)):
        inner.append((prepare(X,y,a,b,cfg,cfg['seed']+900+j),y[a],y[b]))
    frozen=[];models={};selected=[]
    for method in cfg['selection_methods']:
        for k in cfg['panel_sizes']:
            cols=outer['ranks'][method][:k]
            for rank,i in enumerate(cols,1):selected.append({'selector':method,'k':k,'rank':rank,'cpg':probes[outer['pp']['columns'][i]],'train_score':outer['scores'][method][i]})
            for name in cfg['classifiers']:
                t=time.perf_counter()
                with PeakRSS() as rss:
                    params,best=tune(inner,method,k,name,cfg,0,searchrows);a,_=matrix_pair(outer,method,k,name);m=make_model(name,params,cfg['seed']+9000).fit(a,y)
                key=(method,k,name);models[key]=m
                prep=outer['prep_seconds']+sum(z[0]['prep_seconds'] for z in inner);sel=outer['selection_seconds'][method]+sum(z[0]['selection_seconds'][method] for z in inner)
                frozen.append({'model':name,'selector':method,'k':k,'best_parameters':params,'best_inner_ba':best,'model_search_fit_seconds':time.perf_counter()-t,'shared_preprocessing_seconds':prep,'shared_selection_seconds':sel,'whole_process_peak_rss_mb':rss.peak/1024**2})
    # Freeze all settings and panel identities before any external prediction or scoring.
    dump_json(out/'frozen_external_models.json',{'timestamp_utc':pd.Timestamp.now(tz='UTC').isoformat(),'config_sha256':sha256(ROOT/'config.json'),'models':frozen})
    pd.DataFrame(selected).to_csv(out/'external_selected_features.csv',index=False)
    pd.DataFrame(searchrows).to_csv(out/'hyperparameter_search.csv',index=False)
    extz=prep_transform(Xext,outer['pp']);extraz=prep_transform(Xextra,outer['pp']);trainz=prep_transform(X,outer['pp']);rows=[];pred=[];extra=[]
    for rec in frozen:
        method,k,name=rec['selector'],rec['k'],rec['model'];cols=outer['ranks'][method][:k];ze,zx=extz[:,cols],extraz[:,cols]
        if name!='RF':sc=StandardScaler().fit(trainz[:,cols]);ze,zx=sc.transform(ze),sc.transform(zx)
        m=models[method,k,name];t=time.perf_counter();s=score_model(m,ze,name);p=m.predict(ze);prediction_secs=time.perf_counter()-t
        rows.append({**rec,'best_parameters':json.dumps(rec['best_parameters']),'n_train':len(md),'n_test':len(em),**measure(em.y.to_numpy(int),s,p),'prediction_seconds':prediction_secs,'attributed_total_seconds':rec['model_search_fit_seconds']+rec['shared_preprocessing_seconds']+rec['shared_selection_seconds']})
        for j in range(len(em)):pred.append({'cohort':'external','sample_id':em.sample_id.iloc[j],'group_id':em.group_id.iloc[j],'status':em.status.iloc[j],'fold':0,'model':name,'selector':method,'k':k,'y_true':int(em.y.iloc[j]),'score':s[j],'y_pred':int(p[j])})
        sx=score_model(m,zx,name);px=m.predict(zx)
        for j in range(len(xm)):pred.append({'cohort':'exploratory','sample_id':xm.sample_id.iloc[j],'group_id':xm.group_id.iloc[j],'status':xm.status.iloc[j],'fold':0,'model':name,'selector':method,'k':k,'y_true':int(xm.y.iloc[j]),'score':sx[j],'y_pred':int(px[j])})
        for status in xm.status.unique():
            mask=xm.status.to_numpy()==status;extra.append({'status':status,'model':name,'selector':method,'k':k,'n':int(mask.sum()),'positive_predictions':int(px[mask].sum()),'positive_rate':px[mask].mean(),'specificity_if_normal':float(1-px[mask].mean()) if status=='normal-C' else np.nan})
    pd.DataFrame(rows).to_csv(out/'external_metrics.csv',index=False);pd.DataFrame(extra).to_csv(out/'exploratory_metrics.csv',index=False)
    return pd.DataFrame(rows),pd.DataFrame(pred)

def summarize(folds,internal,external,stability,cfg,out):
    keys=['model','selector','k'];metrics=['balanced_accuracy','roc_auc','average_precision','sensitivity','specificity','f1','attributed_total_seconds','model_search_fit_seconds','shared_selection_seconds','whole_process_peak_rss_mb']
    z=folds.groupby(keys)[metrics].agg(['mean','std']);z.columns=[a+'_'+b for a,b in z.columns];z=z.reset_index()
    for key,g in internal.groupby(keys):
        mask=(z.model==key[0])&(z.selector==key[1])&(z.k==key[2]);z.loc[mask,'oof_balanced_accuracy']=balanced_accuracy_score(g.y_true,g.y_pred)
    ex=external[keys+metrics+['tn','fp','fn','tp']].rename(columns={m:'external_'+m for m in metrics+['tn','fp','fn','tp']});z=z.merge(ex,on=keys)
    stab=stability.groupby(['selector','k']).agg(jaccard_mean=('jaccard_mean','mean'),jaccard_fold_sd_descriptive=('jaccard_mean','std')).reset_index();stab.to_csv(out/'stability_summary.csv',index=False);z=z.merge(stab,on=['selector','k']);z.to_csv(out/'summary.csv',index=False)
    rho=spearmanr(z.balanced_accuracy_mean,z.external_balanced_accuracy).statistic;dump_json(out/'rank_transfer.json',{'spearman_rho':rho,'n_configurations':len(z),'interpretation':'Descriptive correlation of dependent model configurations; no independent-observation p-value reported.'})
    return z

def bootstrap(pred,cfg,out):
    e=pred[pred.cohort=='external'];base=e[(e.model=='LR')&(e.k==100)&(e.selector=='DeltaBeta')].sort_values('sample_id');groups=base.group_id.to_numpy();unique=np.unique(groups);bygroup=[np.flatnonzero(groups==g) for g in unique];rng=np.random.default_rng(cfg['bootstrap_seed']);arrays={}
    for m in ['DeltaBeta','MI','L2coef']:
        d=e[(e.model=='LR')&(e.k==100)&(e.selector==m)].sort_values('sample_id');assert d.sample_id.tolist()==base.sample_id.tolist();arrays[m]=(d.y_true.to_numpy(int),d.y_pred.to_numpy(int))
    rows=[]
    for r in range(cfg['bootstrap_replicates']):
        ix=np.concatenate([bygroup[j] for j in rng.integers(0,len(unique),len(unique))]);y=arrays['DeltaBeta'][0][ix]
        if len(np.unique(y))<2:continue
        vals={m:balanced_accuracy_score(y,p[ix]) for m,(_,p) in arrays.items()};rows.append({'replicate':r,**{m+'_ba':v for m,v in vals.items()},'MI_minus_DeltaBeta':vals['MI']-vals['DeltaBeta'],'L2coef_minus_DeltaBeta':vals['L2coef']-vals['DeltaBeta']})
    b=pd.DataFrame(rows);b.to_csv(out/'bootstrap_replicates.csv',index=False);comparison=[]
    for m in ['MI','L2coef']:
        col=m+'_minus_DeltaBeta';point=balanced_accuracy_score(*arrays[m])-balanced_accuracy_score(*arrays['DeltaBeta']);lo,hi=np.quantile(b[col],[.025,.975]);comparison.append({'model':'LR','k':100,'selector':m,'baseline':'DeltaBeta','metric':'balanced_accuracy','difference':point,'ci95_low':lo,'ci95_high':hi})
    obj={'replicates_requested':cfg['bootstrap_replicates'],'replicates_valid':len(b),'n_inferred_clusters':len(unique),'method':'Paired cluster percentile bootstrap of fixed external predictions. Clusters inferred conservatively from title country/histology/numeric ID; unresolved names remain separate. Conditional on fitted models and observed cohort; does not include retraining/model-selection uncertainty. Two predeclared exploratory contrasts, no multiplicity-adjusted hypothesis claims.','comparisons':comparison,'selector_ci95':{m:{'point':balanced_accuracy_score(*arrays[m]),'low':float(b[m+'_ba'].quantile(.025)),'high':float(b[m+'_ba'].quantile(.975))} for m in arrays}}
    dump_json(out/'primary_comparison.json',obj)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default=str(ROOT.parent/'results'));ap.add_argument('--config',default=str(ROOT/'config.json'));args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);cfg=json.loads(Path(args.config).read_text());t=time.perf_counter();cpu=time.process_time()
    dump_json(out/'protocol_frozen.json',{'timestamp_utc':pd.Timestamp.now(tz='UTC').isoformat(),'config':cfg,'run_py_sha256':sha256(__file__),'config_sha256':sha256(args.config)})
    with threadpool_limits(limits=1),PeakRSS() as totalrss:
        print('Loading official processed matrices',flush=True);A,pa,ma,qa=load_geo('GSE48684',cfg,out);B,pb,mb,qb=load_geo('GSE68060',cfg,out)
        common,ia,ib=np.intersect1d(pa,pb,return_indices=True);cpgmask=np.char.startswith(common,'cg');common,ia,ib=common[cpgmask],ia[cpgmask],ib[cpgmask]
        dm=ma[ma.role=='development'].reset_index(drop=True);em=mb[mb.role=='external'].reset_index(drop=True);xm=ma[ma.role=='exploratory'].reset_index(drop=True)
        X=np.asarray(A[np.flatnonzero(ma.role=='development')][:,ia]);Xe=np.asarray(B[np.flatnonzero(mb.role=='external')][:,ib]);Xx=np.asarray(A[np.flatnonzero(ma.role=='exploratory')][:,ia]);del A,B
        qc={'raw_cohorts':[qa,qb],'common_cg_probes':len(common),'development_samples':len(dm),'development_cancer':int(dm.y.sum()),'development_healthy_normal':int((dm.y==0).sum()),'external_samples':len(em),'external_cancer':int(em.y.sum()),'external_normal':int((em.y==0).sum()),'external_inferred_clusters':em.group_id.nunique(),'excluded_IVD':int((mb.role=='excluded_control').sum()),'exploratory_counts':xm.status.value_counts().to_dict(),'external_unresolved_titles':em[em.group_id.str.contains('unresolved')].title.tolist(),'grouping_limitation':'GSE48684 cancer and normal-C equal title numbers are not donor pairs (sex discordance); normal-C excluded from training. Development specimen labels are unique, true donor identity unverified. External groups are conservative title-derived heuristic, not confirmed UUIDs.','development_sex_counts':pd.crosstab(dm.status,dm.gender).to_dict(),'source_preprocessing':'GSE48684 submitted beta values previously SWAN normalized and ComBat adjusted across source study. Our train-only guarantees concern downstream steps only; original preprocessing was not refit within our CV. GSE68060 submitted GenomeStudio beta values.','probe_annotation_limitation':'Only cg-prefixed shared probe IDs retained; no sex-chromosome, SNP-overlap or cross-reactive probe filtering; no gene annotation claim.','platform':'Illumina HumanMethylation450 BeadChip; processed submitted beta matrices; no additional joint normalization or batch correction; no IDAT-level QC.'};dump_json(out/'qc_summary.json',qc)
        print('QC',json.dumps(qc),flush=True)
        folds,pred,stab,search=run_nested(X,dm,common,cfg,out)
        print('Training final models; external labels never enter tuning',flush=True);external,ep=external_run(X,dm,Xe,em,Xx,xm,common,cfg,out,search);allpred=pd.concat([pred,ep],ignore_index=True);allpred.to_csv(out/'predictions.csv',index=False);summary=summarize(folds,pred,external,stab,cfg,out);bootstrap(allpred,cfg,out)
    manifest={'completed_utc':pd.Timestamp.now(tz='UTC').isoformat(),'wall_seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-cpu,'whole_process_peak_rss_mb':totalrss.peak/1024**2,'hardware':{'platform':platform.platform(),'processor':platform.processor(),'logical_cpus':psutil.cpu_count(),'physical_cpus':psutil.cpu_count(logical=False),'total_ram_gb':psutil.virtual_memory().total/1024**3,'threads_used':1},'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'psutil':psutil.__version__},'run_py_sha256':sha256(__file__),'config_sha256':sha256(args.config),'timing_note':'Model-search time covers CV classifier fits and final classifier fit, excluding cached preprocessing and selector rankings. Shared preprocessing and selector times are attributed in full to each configuration; attributed totals must not be summed across rows. Whole-process RSS sampled every20ms includes loaded matrices and caches, not isolated algorithm allocation. Overall wall includes loading, CV, stability, final fit, evaluation and bootstrap; excludes downloads if raw already present and excludes figures.','files_sha256':{p.name:sha256(p) for p in out.glob('*') if p.is_file() and p.name!='manifest.json'}};dump_json(out/'manifest.json',manifest);print('COMPLETE',json.dumps(manifest),flush=True)
if __name__=='__main__': main()
