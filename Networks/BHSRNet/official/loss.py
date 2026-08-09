# -----------------------------------------------------------------------------
# If you use this code in your research, please cite our paper:
# Blur-Resistant Hyperspectral Image Super-Resolution via Dual-Degradation Fusion Model
# Thanks
# -----------------------------------------------------------------------------
import torch.nn as nn
import torch

def to_gray(tensor):
    gray_tensor = torch.mean(tensor, dim=1, keepdim=True)
    return gray_tensor

def tv_loss(image, reduction='mean'):

    batch_size = image.shape[0]
    h_diff = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])  # (B, C, H-1, W)
    w_diff = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])  # (B, C, H, W-1)
    if reduction == 'mean':
        loss = (h_diff.mean() + w_diff.mean()) / batch_size
    elif reduction == 'sum':
        loss = (h_diff.sum() + w_diff.sum()) / batch_size
    return loss

class Losses(nn.Module):

    def __init__(self, scale, model_name='zsl', blur=0): 
        super(Losses, self).__init__()
        self.mse_loss = nn.MSELoss() 
        self.l1_loss = nn.L1Loss()
        self.model_name = model_name 
        self.scale = scale  
        if scale==8:
            self.weight = 0.05 if blur>0  else 0
        else:
            self.weight = 0.35 if blur>0 else 0.05

    def forward(self, Z_pred, Z, y_, y_blur,epoch):


        l1_loss = None
        weight=0.005*10/(len(Z_pred))
        for i in range(len(Z_pred)):
            if l1_loss is None:
                l1_loss = self.l1_loss(Z_pred[i], Z)*weight
            else:
                l1_loss += self.l1_loss(Z_pred[i], Z)*weight
                weight = weight * 2  

        total_loss = l1_loss

        GL = GradientLoss(device=Z.device, in_channels=1)  
        if epoch<40:
            total_loss = l1_loss + 0.01 * GL(to_gray(y_), to_gray(Z))
        else:
            total_loss = l1_loss + self.weight * GL(to_gray(y_), to_gray(Z))
 
        return total_loss

class GradientLoss(nn.Module):
    def __init__(self, device, in_channels=3):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.sobel_x = self.sobel_x.repeat(1, in_channels, 1, 1).to(device)  
        self.sobel_y = self.sobel_y.repeat(1, in_channels, 1, 1).to(device) 
        
        self.conv_x = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False).to(device)  
        self.conv_y = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False).to(device)  
        
        self.conv_x.weight = nn.Parameter(self.sobel_x)
        self.conv_y.weight = nn.Parameter(self.sobel_y)
        
    def forward(self, pred, target):

        assert pred.shape[1] == self.conv_x.in_channels, \
            f"input channel {pred.shape[1]} and loss config {self.conv_x.in_channels} not match"

        pred_grad_x = self.conv_x(pred)
        pred_grad_y = self.conv_y(pred)
        target_grad_x = self.conv_x(target)
        target_grad_y = self.conv_y(target)

        loss = torch.mean(torch.abs(pred_grad_x - target_grad_x) + torch.abs(pred_grad_y - target_grad_y))
        return loss

