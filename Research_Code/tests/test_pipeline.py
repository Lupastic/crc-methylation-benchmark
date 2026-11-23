"""Small artificial arrays are used ONLY for software unit tests, never study results."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from run import prep_fit,prep_transform,prepare,splits,grouped_subsample,measure
class PipelineTests(unittest.TestCase):
    def config(self):return {'max_train_missing_fraction':.2,'variance_prefilter':25,'selection_methods':['DeltaBeta','MI','L2coef'],'panel_sizes':[3,10],'coefficient_selector_C':1,'mi_neighbors':3}
    def test_heldout_values_and_labels_cannot_change_training_selection(self):
        rng=np.random.default_rng(123);X=rng.uniform(0,1,(24,40));y=np.tile([0,1],12);train=np.arange(18);test=np.arange(18,24)
        a=prepare(X,y,train,test,self.config(),42);X[test]=1000;y[test]=1-y[test];b=prepare(X,y,train,test,self.config(),42)
        np.testing.assert_array_equal(a['pp']['columns'],b['pp']['columns']);np.testing.assert_array_equal(a['pp']['medians'],b['pp']['medians'])
        for m in self.config()['selection_methods']:
            np.testing.assert_array_equal(a['ranks'][m],b['ranks'][m]);np.testing.assert_array_equal(a['panels'][m,3]['scaled'][0],b['panels'][m,3]['scaled'][0])
    def test_imputation_uses_train_median(self):
        cfg=self.config();cfg['max_train_missing_fraction']=.5
        train=np.array([[0.,1.,np.nan],[.2,1.,.4],[.6,1.,.8],[.8,1.,.6]])
        state=prep_fit(train,cfg);self.assertNotIn(1,state['columns']);test=np.array([[np.nan,100,np.nan]])
        np.testing.assert_allclose(prep_transform(test,state),state['medians'][None,:])
    def test_groups_stay_together(self):
        groups=np.repeat(np.arange(30),2);y=np.tile([0,1],30)
        for a,b in splits(y,groups,5,42):self.assertFalse(set(groups[a])&set(groups[b]))
        take=grouped_subsample(y,groups,.8,np.random.default_rng(4))
        self.assertTrue(all((groups[take]==g).sum()==2 for g in set(groups[take])))
    def test_metrics_from_known_confusion(self):
        m=measure(np.array([0,0,1,1]),np.array([.1,.8,.4,.9]),np.array([0,1,0,1]))
        self.assertEqual(m['balanced_accuracy'],.5);self.assertEqual([m[k] for k in ['tn','fp','fn','tp']],[1,1,1,1])
if __name__=='__main__':unittest.main()
