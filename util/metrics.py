import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import normalized_mutual_info_score

def cluster_acc(Y_pred, Y):
  assert Y_pred.size == Y.size
  D = int(max(Y_pred.max(), Y.max())+1)
  w = np.zeros((D,D), dtype=np.int64)
  for i in xrange(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size

def cluster_nmi(Y_pred, Y):
    assert Y_pred.size == Y.size
    nmi = normalized_mutual_info_score(Y, Y_pred)
    return nmi