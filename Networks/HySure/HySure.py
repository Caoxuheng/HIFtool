# Code author: Xiangmei Hu (Beijing Institute of Graphic Communication)
import numpy as np
from time import time
from scipy.sparse.linalg import svds
from numpy.fft import fft2, ifft2, ifftshift


def ConvC(X, FK, nl):
    p, n = X.shape

    nc = n // nl

    A = ifft2(fft2(X.reshape([p, nl, nc])) * np.repeat(FK.reshape([1, nl, nc]), p, axis=0)).real
    A = A.reshape([p, n])
    return A


def soft_thr(X1, X2, tau):
    NU = np.sqrt(sum(X1 ** 2) + sum(X2 ** 2))
    A = np.maximum(0, NU - tau)
    A = np.repeat((A / (A + tau))[:, np.newaxis], X1.shape[0], axis=1)
    Y3 = A.T * X1
    Y4 = A.T * X2
    return Y3, Y4


def upsample(img, nl, nc, band, sf):
    aux = np.zeros([nl, nc, band])
    '''for i in range(band):
        # 从 Yhim 中选择一个二维矩阵，并对其进行上采样操作
        # 采用了两次上采样，每次的采样因子为 downsamp_factor，并进行了偏移 shift
        # 最终结果存储在 aux[:,:,i] 中
        aux[:, :, i] = zoom(zoom(img[:, :, i], sf, order=0), sf, order=0)'''
    for i in range(band):
        aux[sf // 2 - 1::sf, sf // 2 - 1::sf, i] = img[:, :, i]
    return aux


def get_mask(sf, nc, nl, bands):
    mask = np.zeros([nc, nl])
    mask[sf // 2 - 1::sf, sf // 2 - 1::sf] = 1
    maskim = np.tile(mask[:, :, None], (1, 1, bands))
    mask = maskim.reshape([-1, bands])
    return mask


class HySure():
    def __init__(self, args):
        self.sf = args.sf
        self.lam_p, self.lam_r, self.lam_m = args.lam_p, args.lam_r, args.lam_m

    def equip(self, srf, psf):
        self.R = srf.T
        self.psf = psf

    def __call__(self, LR_HSI, HR_MSI):
        start_time = time()
        '''
        I. Precomputations. 
        '''
        nl, nc, nb = HR_MSI.shape

        bands = LR_HSI.shape[-1]

        Y = HR_MSI.reshape([-1, HR_MSI.shape[-1]]).T
        Y_H = LR_HSI.reshape([-1, bands]).T

        # Dh
        dh = np.zeros([nl, nc])
        dh[0, 0] = 1
        dh[0, -1] = -1
        # Dv
        dv = np.zeros([nl, nc])
        dv[0, 0] = 1
        dv[-1, 0] = -1

        FDH = fft2(dh)
        FDHC = np.conj(FDH)
        FDV = fft2(dv)
        FDVC = np.conj(FDV)

        # Fourier transform of B
        B = np.zeros([nl, nc])
        middlel = round((nl + 1) / 2)
        middlec = round((nc + 1) / 2)
        _fake_sf = self.psf.shape[0]
        B[middlel - _fake_sf // 2 - 1:middlel + _fake_sf // 2,
        middlec - _fake_sf // 2 - 1:middlec + _fake_sf // 2] = self.psf
        B = ifftshift(B).real
        B = B / sum(sum(B))
        FB = fft2(B)
        FBC = np.conj(FB)
        Down_C = abs(FB) ** 2 + abs(FDH) ** 2 + abs(FDV) ** 2 + 1

        I_DXDY_B = FBC / Down_C
        I_DXDY_I = 1 / Down_C
        I_DXDY_DXH = FDHC / Down_C
        I_DXDY_DYH = FDVC / Down_C

        LR_HSI_up = upsample(LR_HSI, nl, nc, bands, self.sf)
        Y_up = LR_HSI_up.reshape([-1, bands]).T

        Phi, s_, vh_ = svds(Y_H, 10)

        # 光谱基数量p
        p = Phi.shape[-1]

        mask = get_mask(self.sf, nc, nl, p)

        '''
        ADMM
        '''
        IE = Phi.T @ Phi + self.lam_p * np.eye(p)

        yyh = Phi.T @ Y_up

        IRE = self.lam_m * Phi.T @ (self.R.T @ self.R) @ Phi + self.lam_p * np.eye(p)
        yym = (self.R @ Phi).T @ Y

        C__ = np.zeros([nl * nc, p]).T

        V1 = C__.copy()
        V2 = C__.copy()
        V3 = C__.copy()
        V4 = C__.copy()
        A1 = C__.copy()
        A2 = C__.copy()
        A3 = C__.copy()
        A4 = C__.copy()

        for i in range(200):
            '''
            Update E
            arg min  self.lam_r/2||EB - V1- A1||_F^2+
                E    self.lam_r/2||E - V2- A2||_F^2
                     self.lam_r/2||EDx - V2' - A2'||_F^2+
                     self.lam_r/2||EDy - V3' - A3'||_F^2
            '''
            C = ConvC(V1 + A1, I_DXDY_B, nl) + ConvC(V2 + A2, I_DXDY_I, nl) + ConvC(V3 + A3, I_DXDY_DXH, nl) + ConvC(
                V4 + A4, I_DXDY_DYH, nl)
            '''
            Update V1
            min (1/2)||Z-Phi V1 S||_F^2+(lamda/2)||EB-V1-A1||_F^2
             V1
             '''
            NU1 = ConvC(C, FB, nl) - A1
            V1 = np.linalg.inv(IE) @ (yyh + self.lam_p * NU1) * mask.T + NU1 * (1 - mask).T

            '''
            Update V2 
            min (self.lam_m/2)||Y-R Phi V2||_F^2+(lamda/2)||E-V2-A2||_F^2
             V2
            '''
            NU2 = C - A2

            V2 = np.linalg.inv(IRE) @ (self.lam_m * yym + self.lam_p * NU2)

            '''
            Update V3,V4 
            '''
            NU3 = ConvC(C, FDH, nl) - A3
            NU4 = ConvC(C, FDV, nl) - A4
            V3, V4 = soft_thr(NU3, NU4, self.lam_r / self.lam_p)
            print(f'\r\033[1;31mHySure Fusion:\titer = {i} out of {200}\033[0m', end='')
            '''Update  A1,A2,A3,A4'''
            A1 = -NU1 + V1
            A2 = -NU2 + V2
            A3 = -NU3 + V3
            A4 = -NU4 + V4

            img = (Phi @ C).T.reshape([nc, nl, bands])

        end_time = time()
        print('\nHySure Time Cost: {:.3f}s \n'.format(end_time - start_time))
        print(''.center(50, '—'))
        return img
