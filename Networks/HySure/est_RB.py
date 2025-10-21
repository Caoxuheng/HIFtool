# -*- coding: utf-8 -*-

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.linalg import toeplitz


def _im2mat(img):
    if img.ndim == 3:
        nl, nc, nb = img.shape
        return img.reshape(nl * nc, nb).T
    elif img.ndim == 2:
        nl, nc = img.shape
        return img.reshape(1, nl * nc)
    else:
        raise ValueError(f"_im2mat expects 2D/3D array, got {img.ndim}D.")

def _mat2im(mat, nl, nc):
    nb, N = mat.shape
    assert N == nl * nc, "_mat2im: size mismatch"
    return mat.T.reshape(nl, nc, nb)

def fspecial_average(h, w):
    k = np.ones((h, w), dtype=np.float64)
    k /= k.sum()
    return k

def _ConvC(Y, FB, nl, nc=None):
    if nc is None:
        nc = FB.shape[1]
    nb, N = Y.shape
    assert N == nl * nc, "_ConvC: size mismatch"

    if FB.shape != (nl, nc) or not np.iscomplexobj(FB):
        FB = fft2(FB)

    out = np.zeros_like(Y, dtype=np.float64)
    for b in range(nb):
        im = Y[b, :].reshape(nl, nc)
        out[b, :] = np.real(ifft2(fft2(im) * FB)).ravel()
    return out

def _upsamp_HS(Ylr, factor, nl, nc, shift):
    nlh, nch, L = Ylr.shape
    out = np.zeros((nl, nc, L), dtype=Ylr.dtype)
    rows = np.arange(shift, nl, factor)
    cols = np.arange(shift, nc, factor)
    assert len(rows) == nlh and len(cols) == nch, "upsamp grid mismatch"
    out[np.ix_(rows, cols, np.arange(L))] = Ylr
    return out

def _downsamp_HS(img, factor, shift):
    nl, nc, nb = img.shape
    rows = np.arange(shift, nl, factor)
    cols = np.arange(shift, nc, factor)
    return img[np.ix_(rows, cols, np.arange(nb))]

def laplacian_fft(nl, nc):
    ky = 2 * np.pi * np.fft.fftfreq(nl)
    kx = 2 * np.pi * np.fft.fftfreq(nc)
    Ky, Kx = np.meshgrid(ky, kx, indexing='ij')
    return 4 - 2 * np.cos(Kx) - 2 * np.cos(Ky)


def sen_resp_est(Yhim, Ymim, downsamp_factor, intersection, contiguous,
                 p, lambda_R, lambda_B, hsize_h, hsize_w, shift, blur_center,is_calSRF,R):
    """
    输入：
      Yhim: (nlh, nch, L) 低分辨率 HSI（删噪声波段后的数据）
      Ymim: (nl,  nc,  nb) 高分辨率 MSI
      intersection: list(len=nb)，每项为该 MS 通道覆盖的 HS 波段索引（0-based，删后索引）
      contiguous:   与 intersection 对齐的“原始连续索引”（不跨断带做差分）
    返回：
      V: (L, p)  SVD 子空间
      R: (nb, L) 相对谱响应
      B: (nl, nc) 2D 空域核（sum(B)=1）
    """
    # ---------- I) 给 MS 强模糊（稳谱估计） ----------
    nl, nc, nb = Ymim.shape
    middlel, middlec = (nl + 1) // 2, (nc + 1) // 2
    lx, ly = hsize_h,  hsize_w  # 模糊窗口半径
    Bm = np.zeros((nl, nc), dtype=np.float64)
    Bm[middlel-ly-1:middlel+ly, middlec-lx-1:middlec+lx] = fspecial_average(2*ly+1, 2*lx+1)
    Bm = np.fft.ifftshift(Bm)
    FBm = fft2(Bm)

    Ym = _im2mat(Ymim)                  # (nb, nl*nc)
    Ymb = _ConvC(Ym, FBm, nl, nc)       # (nb, nl*nc)
    Ymbim = _mat2im(Ymb, nl, nc)        # (nl, nc, nb)

    Ymbim_down = _downsamp_HS(Ymbim, downsamp_factor, shift-1)
    Ymb_down = _im2mat(Ymbim_down)      # (nb, (nl/df)*(nc/df))

    # ---------- II) 给 HSI 对应缩放模糊 ----------
    nlh, nch, L = Yhim.shape
    lx_s = max(1, int(round(lx / downsamp_factor)))
    ly_s = max(1, int(round(ly / downsamp_factor)))
    middlelh, middlech = (nlh + 1) // 2, (nch + 1) // 2

    Bh = np.zeros((nlh, nch), dtype=np.float64)
    Bh[middlelh-ly_s-1:middlelh+ly_s, middlech-lx_s-1:middlech+lx_s] = fspecial_average(2*ly_s+1, 2*lx_s+1)
    Bh = np.fft.ifftshift(Bh)
    FBh = fft2(Bh)

    Yh  = _im2mat(Yhim)                  # (L, nlh*nch)
    Yhb = _ConvC(Yh, FBh, nlh, nch)      # (L, nlh*nch)

    # ---------- III) 估计谱响应 R ----------
    if is_calSRF:
        R = np.zeros((nb, L), dtype=np.float64)

        for ib in range(nb):  # 对应 MATLAB 的 idx_Lm = 1:nb
            S = np.asarray(intersection[ib], dtype=int)  # 覆盖的 HS 波段索引（长度 m）
            m = S.size
            if m == 0:
                continue

            # ---- 构造一阶差分矩阵 D（维度 (m-1) × m，与 MATLAB 完全一致）----
            col = np.zeros((m - 1,), dtype=np.float64)
            row = np.zeros((m,), dtype=np.float64)
            col[0] = 1.0
            row[0] = 1.0
            row[1] = -1.0
            D = toeplitz(col, row)  # 形状 (m-1, m)

            # 不连续波段对应的差分行置零（避免跨“被删”波段做差分）
            cont = np.asarray(contiguous[ib], dtype=int)  # 与 S 对齐的“原始区间”位置
            if cont.size >= 2:
                breaks = np.diff(cont) != 1  # True 的地方是断点
                if np.any(breaks):
                    D[breaks, :] = 0.0

            # ---- 正规方程 (Yhb_S Yhb_S^T + λ D^T D) r = Yhb_S Ymb_down(ib,:)^T ----
            Yhb_S = Yhb[S, :]  # (m, N)


            DDt = D.T @ D  # (m, m)
            A = Yhb_S @ Yhb_S.T + lambda_R * DDt  # (m, m)
            b = np.matmul( Yhb_S , Ymb_down[ib, :].T)  # (m, )
            r = np.linalg.inv(A) @ b
            # r = np.linalg.solve(A, b)  # (m, )

            R[ib, S] = r

    # ---------- IV) 子空间降噪（SVD） ----------
    # 生成 mask（与 LR->HR 对齐）
    mask = np.zeros((nl, nc), dtype=bool)
    mask[shift:nl:downsamp_factor, shift:nc:downsamp_factor] = True

    # 上采样到 HR 尺寸，便于后续一致性项
    Yhim_up = _upsamp_HS(Yhim, downsamp_factor, nl, nc, shift)  # (nl, nc, L)
    Yh_up = _im2mat(Yhim_up)                                    # (L, nl*nc)

    U_svd, s, _ = np.linalg.svd(Yh, full_matrices=False)
    V = U_svd[:, :p]                 # (L, p)
    Yh_up = (V @ V.T) @ Yh_up        # 子空间降噪

    # ---------- V) 估计空间核 B（带一阶差分正则） ----------
    hH, hW = hsize_h, hsize_w
    col = np.zeros(hW-1); row = np.zeros(hH)
    col[0] = 1.0; row[0] = 1.0; row[1] = -1.0
    Dv = toeplitz(col, row)  # (hW-1, hH)
    Dh = Dv.T                # (hH-1, hW)
    Ah = np.kron(Dh.T, np.eye(hW)); ATAh = Ah.T @ Ah
    Av = np.kron(np.eye(hH), Dv);   ATAv = Av.T @ Av

    # 将降噪后的 HSI 投到 MSI 通道空间：ryh ≈ R * X
    ryh = R @ Yh_up                      # (nb, nl*nc)
    ryhim = _mat2im(ryh, nl, nc)          # (nl, nc, nb)

    ymymt = np.zeros((hH*hW, hH*hW))
    rtyhymt = np.zeros((hH*hW,))
    r_half, c_half = hH//2, hW//2

    for ib in range(nb):
        for r in range(r_half, nl - r_half):
            for c in range(c_half, nc - c_half):
                if not mask[r, c]:
                    continue
                patch = Ymim[r-r_half:r+r_half+(hH%2), c-c_half:c+c_half+(hW%2), ib]
                v = patch.reshape(-1)
                ymymt += np.outer(v, v)
                rtyhymt += ryhim[r, c, ib] * v

    A_reg = ymymt + lambda_B * (ATAh + ATAv)
    b_vec = np.linalg.solve(A_reg, rtyhymt)
    b_vecim = b_vec.reshape(hH, hW)

    # 嵌入到整幅大小（居中）
    B = np.zeros((nl, nc), dtype=np.float64)
    ml, mc = middlel, middlec
    if hH % 2 == 0: r0, r1 = ml - (hH-1)//2 - 1 - blur_center, ml + (hH-1)//2 - blur_center
    else:           r0, r1 = ml - (hH-1)//2     - blur_center, ml + (hH-1)//2 - blur_center
    if hW % 2 == 0: c0, c1 = mc - (hW-1)//2 - 1 - blur_center, mc + (hW-1)//2 - blur_center
    else:           c0, c1 = mc - (hW-1)//2     - blur_center, mc + (hW-1)//2 - blur_center
    B[r0:r1+1, c0:c1+1] = b_vecim
    B = np.fft.ifftshift(B)

    # ---------- VI) DC 归一化 ----------
    B_vol = B.sum()
    if B_vol == 0:
        B_vol = 1.0
    B /= B_vol
    R /= B_vol

    return V, R, B

