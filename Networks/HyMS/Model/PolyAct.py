import numpy as np
from scipy import optimize
from concurrent.futures import ProcessPoolExecutor
import time

def loss(x, args):
    lamda, rgb = args
    return np.sum(1/2*pow(rgb - x @ lamda,2))


def minimize(args):
    x,a,b= args
    res = optimize.minimize(loss, x, method = 'BFGS', args = [a,b])
    return res.x

def Modification(X_in,hrrgb,srf,data='chikusei'):
    H,W,band = X_in.shape
    print('\r\033[1;31mActivating...\033[0m', end='')
    start_time = time.time()
    # The original CPU code launched one BFGS process per pixel for a linear
    # least-squares problem.  Solve all 3x3 systems in a batch instead; this
    # is mathematically equivalent and makes the no-CUDA fallback practical.
    wavelengths = np.arange(0.40, 0.40 + 0.01 * band, 0.01, dtype=X_in.dtype)
    basis = np.stack((X_in, X_in * wavelengths, X_in * wavelengths ** 2), axis=-2)
    rgb_basis = basis @ srf  # H, W, polynomial-basis, MSI-band
    gram = rgb_basis @ np.swapaxes(rgb_basis, -1, -2)
    rhs = rgb_basis @ hrrgb[..., None]
    weights = np.linalg.solve(gram + 1e-8 * np.eye(3, dtype=X_in.dtype), rhs)[..., 0]
    X_act = (basis * weights[..., :, None]).sum(axis=-2)
    end_time = time.time()
    print('\r\033[1;32mModification Successfully \033[0m')
    print('Modification Time Cost: {:.3f}s \n'.format(end_time-start_time))
    return X_act
