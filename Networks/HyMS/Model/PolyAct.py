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
    msiband = hrrgb.shape[-1]
    n_poly = 3

    lam_mat = np.empty([n_poly,band])
    for i in range(n_poly):
        for j in range(band):
            lam_mat[i] = pow(0.4+0.01*j,i)
    init = X_in.reshape([-1,1,band]) * lam_mat
    init_rgb = init @ srf
    lam = init_rgb
    rgb1 =hrrgb.reshape([-1,msiband])
    x = [1, 0, 0]
    args = [(x, lam[i], rgb1[i]) for i in range(W*H)]
    print('\r\033[1;31mActivating...\033[0m', end='')
    start_time = time.time()
    with ProcessPoolExecutor() as executor:
        results = executor.map(minimize,args)

    results =np.array(list(results)).reshape([-1,1,n_poly])
    a = np.matmul(results,init.reshape([-1,n_poly,band]))
    X_act = a.reshape([W,H,band])
    end_time = time.time()
    print('\r\033[1;32mModification Successfully \033[0m')
    print('Modification Time Cost: {:.3f}s \n'.format(end_time-start_time))
    return X_act
