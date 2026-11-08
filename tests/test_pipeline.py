"""Scientific correctness checks, including a real spawned-worker probe isolation test."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[name]='1'
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import copy
import json
import tempfile
import unittest
import numpy as np
import torch
from core import SmallCNN, partition, batch_order, fedavg, initial_state, to_numpy, to_torch
from controllers import gate, ReactiveController
from experiment import read_config, ensure_manifest
from workers import WorkerPool

class Invariants(unittest.TestCase):
    def test_model_contract(self):
        model=SmallCNN()
        self.assertEqual(sum(p.numel() for p in model.parameters()),12810)
        self.assertEqual(tuple(model(torch.zeros(2,1,28,28)).shape),(2,10))

    def test_partition_complete_disjoint_reproducible(self):
        labels=np.tile(np.arange(10),100)
        a=partition(labels,4,.5,7)
        b=partition(labels,4,.5,7)
        self.assertTrue(all(len(p)>0 for p in a))
        np.testing.assert_array_equal(np.sort(np.concatenate(a)),np.arange(len(labels)))
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        with self.assertRaises(ValueError):partition(labels,1001,.5,0)
        with self.assertRaises(ValueError):partition(labels,4,0,0)

    def test_batch_order_does_not_depend_on_method(self):
        ids=np.arange(100)
        a=batch_order(ids,1,2,3,0)
        np.testing.assert_array_equal(a,batch_order(ids,1,2,3,0))
        np.testing.assert_array_equal(np.sort(a),ids)
        self.assertFalse(np.array_equal(a,batch_order(ids,1,2,3,1)))

    def test_weighted_aggregation(self):
        out=fedavg([{'x':np.array([1.,3.],np.float32)}, {'x':np.array([5.,7.],np.float32)}],[1,3])
        np.testing.assert_allclose(out['x'],[4,6])
        with self.assertRaises(ValueError):fedavg([],[])

    def test_gate_rejects_unamortized_speedup(self):
        cfg={'horizon_epochs':4,'client_examples':100,'min_gain':.05,'payback_factor':1.25}
        self.assertFalse(gate(.01,.009,1,4,cfg)['accepted'])
        self.assertTrue(gate(.01,.009,1,4,cfg,gated=False)['accepted'])
        self.assertTrue(gate(.01,.005,.1,4,cfg)['accepted'])
        self.assertFalse(gate(.01,.011,.001,4,cfg)['accepted'])
        self.assertEqual(gate(.01,.005,.1,1,cfg)['horizon_epochs'],1)

    def test_optimizer_roundtrip_preserves_integer_keys(self):
        v={'state':{0:{'momentum_buffer':torch.tensor([1.,2.])}},'param_groups':[{'params':[0]}]}
        out=to_torch(to_numpy(v))
        self.assertTrue(torch.equal(v['state'][0]['momentum_buffer'],out['state'][0]['momentum_buffer']))

    def test_reactive_hysteresis_and_revert(self):
        c=ReactiveController([1,2,4],{'1':100,'2':100,'4':100})
        c.observe(80)
        self.assertEqual(c.threads,4)
        self.assertEqual(c.observe(80)['event'],'try_lower')
        self.assertEqual(c.threads,2)
        self.assertEqual(c.observe(81)['event'],'revert')
        self.assertEqual(c.threads,4)

    def test_incompatible_resume_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ensure_manifest(Path(d),{'experiment_id':'a'})
            with self.assertRaises(ValueError):ensure_manifest(Path(d),{'experiment_id':'b'})

    def test_probe_isolation_and_momentum(self):
        cfg=read_config(Path(__file__).resolve().parents[1]/'configs/smoke.json')
        cfg['threads']=[1,2]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'tiny.npz'
            rng=np.random.default_rng(42)
            x=rng.integers(0,256,(128,1,28,28),dtype=np.uint8)
            y=np.arange(128,dtype=np.int64)%10
            np.savez(path,x=x,y=y,xt=x,yt=y)
            pool=WorkerPool(path,cfg)
            try:
                s=initial_state(8)
                ids=np.arange(128)
                first=pool.call(1,'train',s,ids)
                saved=copy.deepcopy(first['state'])
                reference=pool.call(1,'train',first['state'],ids,first['optimizer'])
                pool.call(2,'probe',first['state'],ids,first['optimizer'])
                pool.call(1,'probe',first['state'],ids,first['optimizer'])
                after=pool.call(1,'train',first['state'],ids,first['optimizer'])
                for key in saved:
                    np.testing.assert_array_equal(saved[key],first['state'][key])
                    np.testing.assert_array_equal(reference['state'][key],after['state'][key])
                # Same thread, exact matched inputs: probe cannot contaminate committed SGD.
                reset=pool.call(1,'train',first['state'],ids,None)
                self.assertTrue(any(not np.array_equal(reset['state'][k],after['state'][k]) for k in saved))
            finally:pool.close()

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
