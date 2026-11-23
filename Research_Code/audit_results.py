"""Recompute reported metrics from saved predictions and verify structural integrity."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from run import measure,sha256,dump_json
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default=str(ROOT.parent/'results'));args=ap.parse_args();out=Path(args.results)
    cfg=json.loads((ROOT/'config.json').read_text());p=pd.read_csv(out/'predictions.csv');f=pd.read_csv(out/'fold_metrics.csv');e=pd.read_csv(out/'external_metrics.csv');sp=pd.read_csv(out/'splits.csv');s=pd.read_csv(out/'summary.csv');st=pd.read_csv(out/'stability_by_fold.csv')
    assert len(f)==135 and len(e)==len(s)==27
    assert len(p[p.cohort=='internal'])==81*27 and len(p[p.cohort=='external'])==118*27 and len(p[p.cohort=='exploratory'])==66*27
    assert sp.sample_id.is_unique and len(sp)==81
    assert sp.groupby('group_id').fold.nunique().max()==1
    checks=0
    for _,r in f.iterrows():
        q=p[(p.cohort=='internal')&(p.model==r.model)&(p.selector==r.selector)&(p.k==r.k)&(p.fold==r.fold)]
        assert q.sample_id.is_unique
        for name,val in measure(q.y_true.to_numpy(),q.score.to_numpy(),q.y_pred.to_numpy()).items():assert np.isclose(val,r[name],atol=1e-12),(r.model,r.selector,r.k,r.fold,name)
        assert set(q.sample_id)==set(sp[sp.fold==r.fold].sample_id);checks+=1
    for _,r in e.iterrows():
        q=p[(p.cohort=='external')&(p.model==r.model)&(p.selector==r.selector)&(p.k==r.k)]
        assert len(q)==118 and q.sample_id.is_unique
        for name,val in measure(q.y_true.to_numpy(),q.score.to_numpy(),q.y_pred.to_numpy()).items():assert np.isclose(val,r[name],atol=1e-12),(r.model,r.selector,r.k,name)
        checks+=1
    for name,keys in [('selected_features.csv',['fold','selector','k']),('external_selected_features.csv',['selector','k'])]:
        d=pd.read_csv(out/name)
        for key,g in d.groupby(keys):assert g.cpg.is_unique and len(g)==int(g.k.iloc[0]) and g.cpg.str.startswith('cg').all()
    assert len(st)==45 and st.jaccard_mean.between(0,1).all()
    protocol=json.loads((out/'protocol_frozen.json').read_text());assert protocol['config_sha256']==sha256(ROOT/'config.json');assert protocol['run_py_sha256']==sha256(ROOT/'run.py')
    model_freeze=json.loads((out/'frozen_external_models.json').read_text());assert len(model_freeze['models'])==27
    for rec in model_freeze['models']:
        q=e[(e.model==rec['model'])&(e.selector==rec['selector'])&(e.k==rec['k'])].iloc[0];assert json.loads(q.best_parameters)==rec['best_parameters']
    report={'status':'PASS','recomputed_metric_rows':checks,'internal_predictions':int((p.cohort=='internal').sum()),'external_predictions':int((p.cohort=='external').sum()),'exploratory_predictions':int((p.cohort=='exploratory').sum()),'checks':['Every internal/external metric recomputed from saved predictions','Identical outer test specimens across all configurations','No recorded development group crosses outer folds','Each CpG panel has exactly k unique cg IDs','Frozen code/config hashes unchanged during experiment','All final parameters agree with pre-external freeze'],'limitation':'These checks validate the saved analysis structure, not the unknown biological donor map or original source-level preprocessing.'}
    dump_json(out/'integrity_audit.json',report)
    manifest=json.loads((out/'manifest.json').read_text());manifest['integrity_audit']=report;manifest['code_files_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in ROOT.rglob('*') if p.is_file() and 'data' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts};manifest['files_sha256']={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name not in ['manifest.json','run.log']};dump_json(out/'manifest.json',manifest);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
