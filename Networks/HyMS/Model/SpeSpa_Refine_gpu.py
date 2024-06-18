import cupyx.scipy.sparse.linalg
import numpy as np
import cupy as cp
from scipy import fftpack
import cv2 as cv
from time import time
from scipy.sparse.linalg import svds



def ConvC(X, FK, nl):
    p,n = X.shape
    nc = n//nl
    A = cp.fft.ifft2(cp.fft.fft2(X.reshape([p,nl,nc])) *cp.repeat(FK.reshape([1,nl,nc]),p,axis = 0)).real
    A = A.reshape([p,n])
    return A
def soft_thr(X1,X2,tau,WX,WY):
    V2__ =cp.sign(X1)*np.maximum(cp.abs(X1)-tau*WX,0)
    V3__ = cp.sign(X2)*np.maximum(cp.abs(X2)-tau*WY,0)
    return V2__,V3__

def get_ratio(R):
    r,g,b = R.T
    return r,g,b
def Refinement(Yhim,Ymim,lam_p,lam_r,R,K):
    msbands = Ymim.shape[-1]
    R = cp.array(R)
    print('\r\033[1;31mRefining...\033[0m', end='')
    start_time = time()
    nc,nl = Ymim.shape[:-1]
    bands = Yhim.shape[-1]
    Y = Ymim.reshape([-1,Ymim.shape[-1]]).T
    Y_H = Yhim.reshape([-1,bands]).T
    Y = cp.asarray(Y)
    Y_H = cp.asarray(Y_H)

    #####################################
    #                                   #
    #             P C A                 #
    #                                   #
    #####################################

    Phi,s_,vh_ = cupyx.scipy.sparse.linalg.svds(Y_H,K)
    C = cp.diag(s_) @ vh_
    p = Phi.shape[-1]

    #####################################
    #                                   #
    #           Prepare                 #
    #                                   #
    #####################################

    Phi_pre = cp.linalg.inv(Phi.T @ Phi) @ Phi.T
    IE = cp.linalg.inv( Phi.T @ (R.T @ R) @ Phi + lam_p * cp.eye(p))
    yym = (R @ Phi).T @ Y

    dh = cp.zeros([nl, nc])
    dh[0, 0] = 1
    dh[0, -1] = -1

    dv = cp.zeros([nl, nc])
    dv[0, 0] = 1
    dv[-1, 0] = -1



    FDH = cp.fft.fft2(dh)
    FDHC = cp.conj(FDH)
    FDV =  cp.fft.fft2(dv)
    FDVC = cp.conj(FDV)


    # srf_b, srf_g, srf_r = get_ratio(R.T)
    #
    W_set_x = []
    W_set_y = []
    N_0 = ConvC(Y, FK=FDH, nl=nl)
    N_1 = ConvC(Y, FK=FDV, nl=nl)
    for i in range(msbands):
        W_x = np.exp(-abs(N_0[i]) /( abs(N_0[i]).mean()) )
        W_y = np.exp(-abs(N_1[i]) /( abs(N_1[i]).mean()))
        W_set_x.append(W_x)
        W_set_y.append(W_y)


    WX = cp.zeros([bands, nl * nc])
    WY = cp.zeros([bands, nl * nc])
    # for i in range(bands):
    #     WX[i] = (srf_b[i] * W_set_x[2] + srf_g[i] * W_set_x[1] + srf_r[i] * W_set_x[0])/3
    #     WY[i] =( srf_b[i] * W_set_y[2] + srf_g[i] * W_set_y[1] + srf_r[i] * W_set_y[0])/3
    for i in range(bands):
        sum_numx = 0
        sum_numy = 0
        for j in range(msbands):
            sum_numx += R[j][i] * W_set_x[msbands - 1 - j]
            sum_numy += R[j][i] * W_set_y[msbands - 1 - j]
        WX[i] = sum_numx / msbands
        WY[i] = sum_numy / msbands

    Down_C =  abs(FDH)**2+abs(FDV)**2 + 1
    I_DXDY_I = 1/Down_C
    I_DXDY_DXH = FDHC/Down_C
    I_DXDY_DYH = FDVC/Down_C



    #####################################
    #                                   #
    #                                   #
    #             A D M M               #
    #                                   #
    #                                   #
    #####################################
    C__ = cp.zeros([nc * nl, p]).T
    C_ = cp.zeros([nc*nl,bands]).T
    A1 =  C__.copy()
    A2 = C_.copy()
    A3 = C_.copy()



    for i in range(5):


        '''
         Update V1
         '''
        NU1 = C - A1
        V1 = IE @ (yym + lam_p * NU1)


        NU2 = ConvC(Phi@C, FDH, nl) - A2
        NU3 = ConvC(Phi@C, FDV, nl) - A3
        V2, V3 = soft_thr(NU2, NU3, lam_r / lam_p ,WX,WY)




        A1 = -NU1 + V1
        A2 = -NU2 + V2
        A3 = -NU3 + V3



        '''
        Update E
        arg min  lam_r/2||E - V1- A1||_F^2+
            E    lam_r/2||EDx - V2' - A2'||_F^2+
                 lam_r/2||EDy - V3' - A3'||_F^2+

        '''

        V2_ = Phi_pre @ V2
        V3_ = Phi_pre @ V3
        A2_ = Phi_pre @ A2
        A3_ = Phi_pre @ A3
        C = ConvC(V1+A1,I_DXDY_I,nl) + ConvC(V2_+A2_,I_DXDY_DXH,nl) + ConvC(V3_+A3_,I_DXDY_DYH,nl)
        img = (Phi @ C).T.reshape([nc,nl,bands])

        img[img <= 0] = 1e-5
    end_time = time()
    print('\r\033[1;32mRefinement Successfully \033[0m')
    print('Refinement Time Cost: {:.3f}s \n'.format(end_time - start_time))
    print(''.center(50,'—'))
    return cp.asnumpy(img)

