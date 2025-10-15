import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def psnr(pred_y, true_y):
    diff_square = (pred_y - true_y) ** 2
    mse = diff_square.mean()

    if mse == 0:
        return 100
    PIXEL_MAX = 1.0

    return 10 * torch.log10(PIXEL_MAX / mse)

def normalization(a):
    ran = a.max() - a.min()
    a = (a - a.min()) / ran
    return a, ran


