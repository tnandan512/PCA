import numpy as np
import numpy.linalg as linalg

'''############################'''
'''Moore Penrose Pseudo Inverse'''
'''############################'''

'''
Compute Moore Penrose Pseudo Inverse
'''
def compute_pinv(X=None,tol=1e-15):
    U, S, Vt = linalg.svd(X, full_matrices = False)
    S = np.diag(S)
    S_inv = np.diag(linalg.inv(S))
    S_inv = np.diag(S_inv[np.where(S_inv > tol)])
    X_inv = np.dot(np.dot(Vt.T, S_inv), U.T)
    return X_inv 
