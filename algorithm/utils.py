import numpy as np
from scipy.stats import pearsonr
from kerpy.GaussianKernel import GaussianKernel
from independence_testing.HSICSpectralTestObject import HSICSpectralTestObject
import random


def correlation(x, y, alpha=0.001):
    rho, p_value = pearsonr(x, y)
    if p_value < alpha:
        return True, p_value
    else:
        return False, p_value
    

def independence(x, y, alpha=0.001):
    lens = len(x)
    x=x.reshape(lens,1)
    y=y.reshape(lens,1)
    kernelY = GaussianKernel(float(1.0))
    kernelX=GaussianKernel(float(1.0))
    num_samples = lens

    myspectralobject = HSICSpectralTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                          kernelX_use_median=False, kernelY_use_median=False,
                                          rff=True, num_rfx=30, num_rfy=30, num_nullsims=1000)
    p_value = myspectralobject.compute_pvalue(x, y)

    if p_value > alpha:
        return True, p_value
    else:
        return False, p_value


# unbiased estimator of cum(x,x,x,y)
def cum31(x,y):
    n = len(x)
    cumulant = np.mean(x**3 * y) - 3 / (n**2 - n) * (np.sum(x**2) * np.sum(x * y) - np.sum(x**3 * y))
    return cumulant


# unbiased estimator of cum(x,x,y,y)
def cum22(x,y):
    n = len(x)
    cumulant = np.mean(x**2 * y**2) - 1 / (n**2 - n) * (np.sum(x**2) * np.sum(y**2) - np.sum(x**2 * y**2)) - 2 / (n**2 - n) * (np.sum(x*y)**2 - np.sum(x**2 * y**2))
    return cumulant


# pseudo-residual: R(x1,x2|x3)
def pr(x1,x2,x3):
    return x1 - np.cov(x1,x3)[0,1] / np.cov(x2,x3)[0,1] * x2