import scipy as sp
import scipy.linalg as linalg
import pylab as pl

from utils import plot_color
import numpy as np 


'''############################'''
'''Principle Component Analyses'''
'''############################'''
 
'''
Compute Covariance Matrix
Input: Matrix of size #samples x #features
Output: Covariance Matrix of size #features x #features
Note: Do not use scipy or numpy cov. Implement the function yourself.
      You can of course add an assert to check your covariance function
      with those implemented in scipy/numpy.
'''
def computeCov(X=None):
    N, M = X.shape 
    cov = np.zeros((M, M))
    for i in range(M):
        mean_i = np.sum(X[:,i])/N 
        for j in range(M):
            mean_j = np.sum(X[:j])/N 
            cov[i, j] = np.sum((X[:, i] - mean_i) * (X[:, j] - mean_j))/(N-1)
    return cov

'''
Compute PCA
Input: Covariance Matrix
Output: [eigen_values,eigen_vectors] sorted in such a why that eigen_vectors[:,0] is the first principle component
        eigen_vectors[:,1] the second principle component etc...
Note: Do not use an already implemented PCA algorithm. However, you are allowed to use an implemented solver 
      to solve the eigenvalue problem!
'''
def computePCA(matrix=None):
    eigen_values, eigen_vectors = np.linalg.eig(matrix) 
    sorted_index = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sorted_index]
    eigen_vectors = eigen_vectors[:,sorted_index]
    return eigen_values, eigen_vectors

'''
Transform Input Data Onto New Subspace
Input: pcs: matrix containing the first x principle components
       data: input data which should be transformed onto the new subspace
Output: transformed input data. Should now have the dimensions #samples x #components_used_for_transformation
'''
def transformData(pcs=None,data=None):
    eigenvector_subset = pcs[:,0:2]
    transformed_data = data @ eigenvector_subset 
    return transformed_data
'''
Compute Variance Explaiend
Input: eigen_values
Output: return vector with varianced explained values. Hint: values should be between 0 and 1 and should sum up to 1.
'''
def computeVarianceExplained(evals=None):
    total = np.sum(evals)
    var_explained = []
    for i in range(0, len(evals)):
        var_explained.append(evals[i]/total)
    return np.asarray(var_explained) 

'''############################'''
'''Different Plotting Functions'''
'''############################'''

'''
Plot Transformed Data
Input: transformed: data matrix (#samples x 2)
       labels: target labels, class labels for the samples in data matrix
       filename: filename to store the plot
'''
def plotTransformedData(transformed=None,labels=None, filename ='exercise1.pdf'):
    pl.figure() 
    u_labels = pl.unique(labels)
    colours = ['#F7977A','#FDC68A','#A2D39C','#6ECFF6'] #from plot_colors 
    for i, label in enumerate(u_labels):
        ind = pl.where(labels == label)[0]
        pl.scatter(transformed[ind,0], transformed[ind,1], color = colours[i])
    pl.xlabel("Transformed (PC1)")
    pl.ylabel("Transformed (PC2)")
    pl.grid(True)
    pl.legend(u_labels, scatterpoints = 1, numpoints = 1, prop={'size':9},title = 'Class') 
    pl.savefig(filename)

'''
Plot Cumulative Explained Variance 
Input: var: variance explained vector
       filename: filename to store the file
'''
def plotCumSumVariance(var=None,filename="cumsum.pdf"):
    pl.figure() 
    pl.plot(sp.arange(var.shape[0]),sp.cumsum(var)*100)
    pl.xlabel("Principle Component")
    pl.ylabel("Cumulative Variance Explained in %")
    pl.grid(True)
    # Save file 
    pl.savefig(filename) 



'''############################'''
'''Data Preprocessing Functions'''
'''############################'''

'''
Data Normalisation (Zero Mean, Unit Variance)
'''
def dataNormalisation(X=None):
    N, M = X.shape 
    for i in range(M):
        mean_i = np.sum(X[:,i])/N
        X[:,i] = (X[:,i]- mean_i)/np.std(X[:,i])
    return X
