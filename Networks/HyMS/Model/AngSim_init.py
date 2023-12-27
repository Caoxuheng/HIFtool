from numba import cuda
import numpy as np
import math 
import cv2 as cv
import time

@cuda.jit
def SAM_inter(H_XYZ,L_XYZ,Error_M,scale,s_f):
    row, col = cuda.grid(2)
    index = 0
    for i in range(int(row/s_f),int(row/s_f)+2*scale):
        for j in range(int(col/s_f),int(col/s_f)+2*scale):
            up_temp = 0
            down_temp1 = 0
            down_temp2 = 0
            for dx in range(3):
                up_temp += H_XYZ[row, col, dx] * L_XYZ[i, j, dx]
                down_temp1 += H_XYZ[row, col, dx] * H_XYZ[row, col, dx]
                down_temp2 += L_XYZ[i, j, dx] * L_XYZ[i, j, dx]
            Error_M[row,col,index] = math.acos(up_temp/math.sqrt(down_temp1*down_temp2))
            index+=1

def initialization(LHSI, XYZ,research_scale, srf,u,sf):
    assert srf.min()>=0,f'the minimum value of Spectral Response Function must be >0, but {srf.min()} '
    HSI = cv.copyMakeBorder(LHSI, research_scale, research_scale, research_scale, research_scale, cv.BORDER_REFLECT)
    L_XYZ = HSI @ srf
    H,W = XYZ.shape[:-1]
    H_M = np.max(XYZ, axis=-1)
    L_M = np.max(L_XYZ, axis=-1)
    for i in range(1):
        XYZ[:, :, i] /= H_M
        L_XYZ[:, :, i] /= L_M
    info = XYZ.shape
    XYZ_C = cuda.to_device(XYZ)
    L_XYZ_C = cuda.to_device(L_XYZ)
    SAM_Ermat = cuda.device_array([info[0], info[1], 4 * pow(research_scale, 2)], float)
    thread_block = (u, u)
    block_grid_x = int(np.ceil(info[0] / thread_block[0]))
    block_grid_y = int(np.ceil(info[1] / thread_block[1]))
    block_grid = (block_grid_x, block_grid_y)
    cuda.synchronize()
    start_time=time.time()
    print('\r\033[1;31mInitializing...\033[0m',end='')
    SAM_inter[thread_block, block_grid](XYZ_C, L_XYZ_C, SAM_Ermat, research_scale, sf)
    error = SAM_Ermat.copy_to_host()
    min_error = np.min(error, axis=-1).reshape([info[0], info[1], 1])
    min_error_tensor = np.tile(min_error, (1, 1, 4 * pow(research_scale, 2)))
    idx = np.argwhere(error == min_error_tensor)
    box_idx = np.concatenate(((idx[:, -1] // (2 * research_scale)).reshape([-1, 1]),
                              (idx[:, -1] % (2 * research_scale)).reshape([-1, 1])), axis=-1)
    spec_idx = idx[:, :2] // sf + box_idx
    idx_set = np.concatenate((idx[:, :-1], spec_idx), axis=-1)
    recon = np.ones([H, W, LHSI.shape[-1]])
    for i in range(idx_set.shape[0]):
        recon[idx_set[i, :2][0], idx_set[i, :2][1]] = HSI[idx_set[i, 2:][0], idx_set[i, 2:][1]]
    end_time = time.time()

    print('\r\033[1;32mInitialization Successfully \033[0m')
    print('Initialization Time Cost: {:.3f}s \n'.format(end_time-start_time))
    return recon




