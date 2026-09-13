import unittest
from reference_ops import interpolate_native, softmax, prefix_block_cost

class ReferenceTests(unittest.TestCase):
    def test_full_observation_identity(self):
        c=[0.5,2.5,4.5,6.5]; v=[[x,x*x] for x in c]
        self.assertEqual(interpolate_native(c,v,c),v)
    def test_nonuniform_physical_interpolation(self):
        self.assertEqual(interpolate_native([0.,2.,10.],[[0.],[20.],[100.]],[1.,6.]),[[10.],[60.]])
    def test_boundaries_clamp(self):
        self.assertEqual(interpolate_native([2.,5.],[[3.],[9.]],[-1.,7.]),[[3.],[9.]])
    def test_single_observation(self):
        self.assertEqual(interpolate_native([2.],[[3.,4.]],[0.,2.,7.]),[[3.,4.]]*3)
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError): interpolate_native([1.,1.],[[1.],[2.]],[1.])
    def test_shape_rejected(self):
        with self.assertRaises(ValueError): interpolate_native([0.,1.],[[1.,2.],[3.]],[0.])
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): interpolate_native([0.],[[float('nan')]],[0.])
    def test_softmax_pair_residual_is_key_bias(self):
        logits=[[0.2,-0.4,1.1],[1.0,0.2,-1.3],[0.1,0.4,0.8]]
        delta=[0.0,2.1,-0.6]; scale=0.3
        for i,row in enumerate(logits):
            pair=softmax([a+scale*(delta[i]-delta[j]) for j,a in enumerate(row)])
            key=softmax([a-scale*delta[j] for j,a in enumerate(row)])
            for u,v in zip(pair,key): self.assertAlmostEqual(u,v,places=14)
    def test_clipwise_independent_operator_commutes(self):
        clips=[[i*16+j for j in range(16)] for i in range(4)]
        def op(c): return [c[2*t]+2*c[2*t+1] for t in range(8)]
        idx=[0,3]
        self.assertEqual([op(clips[i]) for i in idx],[[op(c) for c in clips][i] for i in idx])
    def test_depth_count(self):
        self.assertEqual(prefix_block_cost([24,18,12],[4,4,4]),216)
        self.assertEqual(prefix_block_cost([48,48,48],[4,4,4]),576)
    def test_non_nested_depth_rejected(self):
        with self.assertRaises(ValueError): prefix_block_cost([12,24],[4,4])

if __name__=='__main__': unittest.main()
