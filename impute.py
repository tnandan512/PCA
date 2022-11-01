mport scipy as sp
import numpy as np
import scipy.linalg as linalg
from sklearn.metrics import mean_squared_error 

'''
Impute missing values using the mean of each feature
'''
def mean_imputation(X=None):
    D_imputed = X.copy()
    #Impute each missing entry per feature with the mean of each feature
    for i in range(X.shape[1]):
        feature = X[:,i]
        #get indices for all non-nan values
        indices = sp.where(~sp.isnan(feature))[0]
        #compute mean for given feature
        mean = sp.mean(feature[indices])
        #get nan indices
        nan_indices = sp.where(sp.isnan(feature))[0]
        #Update all nan values with the mean of each feature
        D_imputed[nan_indices,i] = mean
    return D_imputed

'''
Impute missing values using SVD
'''
def svd_imputation(X=None,rank=None,tol=.1,max_iter=100):
    #get all nan indices
    nan_indices = sp.where(sp.isnan(X))
    #initialise all nan entries with the a mean imputation
    D_imputed = mean_imputation(X) 
    #repeat approximation step until convergance
    for i in range(max_iter):
        D_old = D_imputed.copy()
        #SVD on mean_imputed data
        [L,d,R] = linalg.svd(D_imputed)
        #compute rank r approximation of D_imputed
        D_r = sp.matrix(L[:,:rank])*sp.diag(d[:rank])*sp.matrix(R[:rank,:])
        #update imputed entries according to the rank-r approximation
        imputed = D_r[nan_indices[0],nan_indices[1]]
        D_imputed[nan_indices[0],nan_indices[1]] = sp.asarray(imputed)[0]
        #use Frobenius Norm to compute similarity between new matrix and the latter approximation
        fnorm = linalg.norm(D_old-D_imputed,ord="fro")
        if fnorm<tol:
            print("\t\t\t[SVD Imputation]: Converged after %d iterations"%(i+1))
            break
        if (i+1)>=max_iter:
            print("\t\t\t[SVD Imputation]: Maximum number of iterations reached (%d)"%(i+1))
    return D_imputed

'''
Find Optimal Rank-r Imputation
'''
def svd_imputation_optimised(X=None,ranks=None,
                             test_size=0.25,random_state=0,
                             return_optimal_rank=True,return_errors=True):
    #init variables
    sp.random.seed(random_state)
    testing_errors = []
    optimal_rank = sp.nan
    minimal_error = sp.inf

    #TODO Update this function to find the optimal rank r for imputation of missing values
    #1. Get all non-nan indices 
    ind = sp.argwhere(sp.isnan(X) == False) 
    
    #2. Use "test_size" % of the non-missing entries as test data 
    test_num = int(np.ceil(len(ind)*test_size))
    rand_ind = list(sp.random.permutation(sp.arange(0,len(ind)))[:test_num])
    test_ind = ind[rand_ind]
    
    #3. Create a new training data matrix
    Xt = np.copy(X)
    for i,j in test_ind:
        Xt[i,j] = np.nan 
    
    #4. Find optimal rank r by minimising the Frobenius-Norm using the train and test data 
    for rank in ranks:
        print("\tTesting rank %d..."%(rank))
        #4.1 Impute Training Data
        X_imp = svd_imputation(Xt, rank = rank)
        
        #4.2 Compute the mean squared error of imputed test data with original test data and store error in array
        true = X[test_ind[:,0], test_ind[:,1]]
        pred = X_imp[test_ind[:,0], test_ind[:,1]]
        error = mean_squared_error(true, pred)
        testing_errors.append(error) 
        print("\t\tMean Squared Error: %.2f"%error)
        
        #4.3 Update rank if necessary 
    
    #5. Use optimal rank for imputing the "original" data matrix
    optimal_rank = ranks[np.argmin(testing_errors)]  
    minimal_error = min(testing_errors) 
    X_imputed = svd_imputation(X, rank = optimal_rank) 

    print("Optimal Rank: %f (Mean-Squared Error: %.2f)"%(optimal_rank,minimal_error))
    
    #return data
    if return_optimal_rank==True and return_errors==True:
        return [X_imputed,optimal_rank,testing_errors]
    elif return_optimal_rank==True and return_errors==False:
        return [X_imputed,optimal_rank]
    elif return_optimal_rank==False and return_errors==True:
        return [X_imputed,testing_errors]
    else:
        return X_imputed
