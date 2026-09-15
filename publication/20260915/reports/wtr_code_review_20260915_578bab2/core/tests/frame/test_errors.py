import numpy as np
from tools.frame_errors import weighted_ap

def test_cluster_multiplicity_preserves_all_detections():
    # A background-only video supplies the highest-scoring FP; the second
    # supplies one TP and one GT. Replicating the second whole video gives
    # FP, TP, TP, whose interpolated AP is 2/3.
    tp=np.tile([0,1],(5,1));videos=np.array([0,1]);gt=np.array([0,1])
    assert np.allclose(weighted_ap(tp,videos,gt,np.array([1,1])),.5)
    assert np.allclose(weighted_ap(tp,videos,gt,np.array([1,2])),2/3)
    assert np.allclose(weighted_ap(tp,videos,gt,np.array([0,2])),1.)
    assert np.isnan(weighted_ap(tp,videos,gt,np.array([1,0]))).all()
