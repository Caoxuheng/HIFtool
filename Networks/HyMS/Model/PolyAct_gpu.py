import numpy as np
from numba import cuda
import time


@cuda.jit
def inv_matrix(corr_mat,corr_mat_inv):
    row,col = cuda.grid(2)
    n = corr_mat[row,col].shape[0]
    for i in range(n):
        for j in range(n):
            if i==j:
                corr_mat_inv[row, col][i, j] = 1


    for i in range(n):
        for j in range(n):
            if i!=j:
                ratio = corr_mat[row,col][j,i]/corr_mat[row,col][i,i]

                for k in range(n):
                    corr_mat[row, col][j,k] -= ratio*corr_mat[row,col][i,k]
                    corr_mat_inv[row, col][j, k] -= ratio * corr_mat_inv[row, col][i, k]
    for i in range(n):
        divisor = corr_mat[row, col][i,i]
        for j in range(n):
            corr_mat[row, col][i,j] /= divisor
            corr_mat_inv[row, col][i, j] /= divisor


@cuda.jit
def get_hessian(HSI1, HSI2, HSI3, MSI, H):
    row, col = cuda.grid(2)
    hsi1 = HSI1[row, col]
    hsi2 = HSI2[row, col]
    hsi3 = HSI3[row, col]
    msi = MSI[row, col]

    # Hessian for w:
    h_w0_w0 = hsi1[0] * hsi1[0] + hsi1[1] * hsi1[1] + hsi1[2] * hsi1[2]
    h_w0_w1 = hsi1[0] * hsi2[0] + hsi1[1] * hsi2[1] + hsi1[2] * hsi2[2]
    h_w0_w2 = hsi1[0] * hsi3[0] + hsi3[1] * hsi3[1] + hsi3[2] * hsi3[2]

    h_w1_w0 = hsi2[0] * hsi1[0] + hsi2[1] * hsi1[1] + hsi2[2] * hsi1[2]
    h_w1_w1 = hsi2[0] * hsi2[0] + hsi2[1] * hsi2[1] + hsi2[2] * hsi2[2]
    h_w1_w2 = hsi2[0] * hsi3[0] + hsi2[1] * hsi3[1] + hsi2[2] * hsi3[2]

    h_w2_w0 = hsi3[0] * hsi1[0] + hsi3[1] * hsi1[1] + hsi3[2] * hsi1[2]
    h_w2_w1 = hsi3[0] * hsi2[0] + hsi3[1] * hsi2[1] + hsi3[2] * hsi2[2]
    h_w2_w2 = hsi3[0] * hsi3[0] + hsi3[1] * hsi3[1] + hsi3[2] * hsi3[2]

    H[row, col, 0, 0] = h_w0_w0
    H[row, col, 0, 1] = h_w0_w1
    H[row, col, 0, 2] = h_w0_w2
    H[row, col, 1, 0] = h_w1_w0
    H[row, col, 1, 1] = h_w1_w1
    H[row, col, 1, 2] = h_w1_w2
    H[row, col, 2, 0] = h_w2_w0
    H[row, col, 2, 1] = h_w2_w1
    H[row, col, 2, 2] = h_w2_w2


@cuda.jit
def actication(HSI1, HSI2, HSI3, MSI, W, H_inv):
    row, col = cuda.grid(2)
    hsi1 = HSI1[row, col]
    hsi2 = HSI2[row, col]
    hsi3 = HSI3[row, col]
    msi = MSI[row, col]
    w1 = W[row, col, 0]
    w2 = W[row, col, 1]
    w3 = W[row, col, 2]

    H_ = H_inv[row, col]
    # Calculate primary error

    rec1 = w1 * hsi1[0] + w2 * hsi2[0] + w3 * hsi3[0]
    rec2 = w1 * hsi1[1] + w2 * hsi2[1] + w3 * hsi3[1]
    rec3 = w1 * hsi1[2] + w2 * hsi2[2] + w3 * hsi3[2]

    c_error = 1 / 2 * (pow(msi[0] - rec1, 2) + pow(msi[1] - rec2, 2) + pow(msi[2] - rec3, 2))
    # Gradient for w:
    d_w0 = -(msi[0] - rec1) * hsi1[0] - (msi[1] - rec2) * hsi1[1] - (msi[2] - rec3) * hsi1[2]
    d_w1 = -(msi[0] - rec1) * hsi2[0] - (msi[1] - rec2) * hsi2[1] - (msi[2] - rec3) * hsi2[2]
    d_w2 = -(msi[0] - rec1) * hsi3[0] - (msi[1] - rec2) * hsi3[1] - (msi[2] - rec3) * hsi3[2]
    W[row, col, 1] = c_error

    epoch = 0
    theta = 1e-7
    iters = 100
    while epoch < iters:
        p1 = -(H_[0, 0] * d_w0 + H_[0, 1] * d_w1 + H_[0, 2] * d_w2)
        p2 = -(H_[1, 0] * d_w0 + H_[1, 1] * d_w1 + H_[1, 2] * d_w2)
        p3 = -(H_[2, 0] * d_w0 + H_[2, 1] * d_w1 + H_[2, 2] * d_w2)
        w1 += p1
        w2 += p2
        w3 += p3
        rec1 = w1 * hsi1[0] + w2 * hsi2[0] + w3 * hsi3[0]
        rec2 = w1 * hsi1[1] + w2 * hsi2[1] + w3 * hsi3[1]
        rec3 = w1 * hsi1[2] + w2 * hsi2[2] + w3 * hsi3[2]

        d_w0 = -(msi[0] - rec1) * hsi1[0] - (msi[1] - rec2) * hsi1[1] - (msi[2] - rec3) * hsi1[2]
        d_w1 = -(msi[0] - rec1) * hsi2[0] - (msi[1] - rec2) * hsi2[1] - (msi[2] - rec3) * hsi2[2]
        d_w2 = -(msi[0] - rec1) * hsi3[0] - (msi[1] - rec2) * hsi3[1] - (msi[2] - rec3) * hsi3[2]

        c_error = 1 / 2 * (pow(msi[0] - rec1, 2) + pow(msi[1] - rec2, 2) + pow(msi[2] - rec3, 2))

        epoch = epoch + 1

    W[row, col, 0] = w1

    W[row, col, 1] = w2

    W[row, col, 2] = w3

def Modification(X_in,MSI,srf,u,data='rgb'):

    a1 = np.arange(0.40, 0.701, 0.01)
    if data.lower()=='pavia':
        a1 = np.arange(0.53,0.8636263,0.0036263)
    elif data.lower()=='wdcm':
        a1 = np.arange(0.383, 0.0069*128+0.383 , 0.0069)
    HSI_1 = X_in
    HSI_2 = X_in * a1
    HSI_3 = X_in * a1 * a1
    h = X_in.shape[0]
    hsi1 = cuda.to_device(HSI_1 @ srf)
    hsi2 = cuda.to_device(HSI_2 @ srf)
    hsi3 = cuda.to_device(HSI_3 @ srf)
    info = [h, h]
    w = cuda.to_device(np.full([h, h, 3], 0.0))
    H = cuda.to_device(np.full([h, h, 3, 3], 0.0))
    H_inv = cuda.to_device(np.full([h, h, 3, 3], 0.0))
    msi = cuda.to_device(MSI)
    thread_block = (u, u)
    block_grid_x = int(np.ceil(info[0] / thread_block[0]))
    block_grid_y = int(np.ceil(info[1] / thread_block[1]))
    block_grid = (block_grid_x, block_grid_y)
    cuda.synchronize()
    print('\r\033[1;31mModifying...\033[0m', end='')
    start_time = time.time()
    get_hessian[thread_block, block_grid](hsi1, hsi2, hsi3, msi, H)
    inv_matrix[thread_block, block_grid](H, H_inv)
    actication[thread_block, block_grid](hsi1, hsi2, hsi3, msi, w, H_inv)
    w__ = w.copy_to_host()
    w_1 = w__[:, :, 0]
    w_2 = w__[:, :, 1]
    w_3 = w__[:, :, 2]
    for i in range(31):
        HSI_1[:, :, i] *= w_1
        HSI_2[:, :, i] *= w_2
        HSI_3[:, :, i] *= w_3
    X_act = HSI_1 + HSI_2 + HSI_3
    end_time = time.time()
    print('\r\033[1;32mModified Successfully \033[0m')
    print('Modification Time Cost: {:.3f}s \n'.format(end_time-start_time))
    return X_act