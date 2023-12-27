
import numpy as np
from scipy import fftpack
import cv2 as cv
from time import time
from scipy.sparse.linalg import svds

def SoftMax(data):
    return 1/(1+np.exp(-data))


def get_psf(size,sigma):
    psf = cv.getGaussianKernel(size,sigma)
    psf = psf @ psf.T
    return psf
def psf2otf(psf, outSize):
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = outSize - psfSize
    psf = np.pad(psf, ((0, padSize[0]), (0, padSize[1])), 'constant')
    for i in range(len(psfSize)):
        psf = np.roll(psf, -int(psfSize[i] / 2), i)
    otf = np.fft.fftn(psf)
    nElem = np.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * np.log2(psfSize[k]) * nffts
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= nOps * np.finfo(np.float32).eps:
        otf = np.real(otf)
    return otf
def ConvC(X, FK, nl):
    p,n = X.shape

    nc = n//nl

    A = fftpack.ifft2(fftpack.fft2(X.reshape([p,nl,nc])) *np.repeat(FK.reshape([1,nl,nc]),p,axis = 0)).real
    A = A.reshape([p,n])
    return A
def get_DhDv(nc,nl):
    dh = np.zeros([nl, nc])
    dh[0, 0] = 1
    dh[0, -1] = -1
    dv = np.zeros([nl, nc])
    dv[0, 0] = 1
    dv[-1, 0] = -1
    return dh,dv

def soft_thr(X1,X2,tau,WX,WY):

    V2__ =np.sign(X1)*np.maximum(np.abs(X1)-tau*WX,0)
    V3__ = np.sign(X2)*np.maximum(np.abs(X2)-tau*WY,0)
    return V2__,V3__


def upsample(img,nl,nc,band,sf):
    aux = np.zeros([nl,nc,band])
    for i in range(band):
        aux[sf//2-1::sf,sf//2-1::sf,i] = img[:,:,i]
    return aux
def get_mask(sf,nc,nl):
    mask = np.zeros([nc,nl])
    mask[sf//2-1::sf,sf//2-1::sf]=1
    return mask
def get_ratio(R):
    r,g,b = R.T

    return r,g,b
def Refinement(Yhim,Ymim,lam_p,lam_r,R):
    print('\r\033[1;31mRefining...\033[0m', end='')
    msbands = Ymim.shape[-1]
    start_time = time()
    nc,nl = Ymim.shape[:-1]
    bands = Yhim.shape[-1]
    Y = Ymim.reshape([-1,Ymim.shape[-1]]).T
    Y_H = Yhim.reshape([-1,bands]).T


    #####################################
    #                                   #
    #             P C A                 #
    #                                   #
    #####################################


    Phi,s_,vh_ = svds(Y_H,5)
    C = np.diag(s_) @ vh_
    p = Phi.shape[-1]

    #####################################
    #                                   #
    #           Prepare                 #
    #                                   #
    #####################################

    Phi_pre = np.linalg.inv(Phi.T @ Phi) @ Phi.T
    IE = np.linalg.inv( Phi.T @ (R.T @ R) @ Phi + lam_p * np.eye(p))
    yym = (R @ Phi).T @ Y

    dh = np.zeros([nl, nc])
    dh[0, 0] = 1
    dh[0, -1] = -1

    dv = np.zeros([nl, nc])
    dv[0, 0] = 1
    dv[-1, 0] = -1



    FDH = fftpack.fft2(dh)
    FDHC = np.conj(FDH)
    FDV =  fftpack.fft2(dv)
    FDVC = np.conj(FDV)

    #
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

    WX = np.zeros([bands, nl * nc])
    WY = np.zeros([bands, nl * nc])
    for i in range(bands):
        sum_numx=0
        sum_numy=0
        for j in range(msbands):
            sum_numx+=R[j][i]*W_set_x[msbands-1-j]
            sum_numy+=R[j][i]*W_set_y[msbands-1-j]
        WX[i] = sum_numx/msbands
        WY[i] =sum_numy/msbands



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
    C__ = np.zeros([nc * nl, p]).T
    C_ = np.zeros([nc*nl,bands]).T
    A1 =  C__.copy()
    A2 = C_.copy()
    A3 = C_.copy()



    for i in range(20):


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
    return img

