import imageio
import numpy as np
from utils import *

'''
Function for performing the noise reduction through SVD.

Input
----------------
img:	    noisy image
threshold:  singular values threshold

Output
----------------
img_dn:		denoised image
'''

def NoiseReduction(img, threshold):
    	# 1. Perform SVD on the noisy image, i.e., img
    U, S, Vt = np.linalg.svd(img, full_matrices = False)
    	# 2. Nullify the singular values lower than the 
    	#    threshold, i.e., threshold
    S[S < threshold] = 0
    S = np.diag(S)
    	# 3. Reconstruct the image using the modified 
    	#    singular values
    img_denoised = np.dot(np.dot(U,S), Vt)
    return img_denoised 
