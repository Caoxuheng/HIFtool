import torch.nn.functional as F
import math
import numpy as np
import torch as th
import torch.nn as nn
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
class Edge(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        sobel_kernel_x = th.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=th.float32)
        sobel_kernel_y = th.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=th.float32)
        self.sobel_x.weight.data = sobel_kernel_x.view(1, 1, 3, 3)
        self.sobel_y.weight.data = sobel_kernel_y.view(1, 1, 3, 3)
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
    def compute_edge_map(self, rgb):
        gray = th.mean(rgb, dim=1, keepdim=True)
        grad_x = self.sobel_x(gray)
        grad_y = self.sobel_y(gray)
        edge_magnitude = th.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        b, _, h, w = edge_magnitude.shape
        min_val = edge_magnitude.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        max_val = edge_magnitude.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        edge_map = (edge_magnitude - min_val) / (max_val - min_val).clamp_min(1e-8)
        return edge_map + 1.0
    def forward(self,rgb):
        edge_map = self.compute_edge_map(rgb)
        return  edge_map

class EdgeAwareSpectralResidual(nn.Module):
    def __init__(self, band_dim: int):
        super().__init__()
        self.band_dim = band_dim
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        sobel_kernel_x = th.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=th.float32)
        sobel_kernel_y = th.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=th.float32)
        self.sobel_x.weight.data = sobel_kernel_x.view(1, 1, 3, 3)
        self.sobel_y.weight.data = sobel_kernel_y.view(1, 1, 3, 3)
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

    def compute_edge_map(self, rgb):
        gray = th.mean(rgb, dim=1, keepdim=True)
        grad_x = self.sobel_x(gray)
        grad_y = self.sobel_y(gray)
        edge_magnitude = th.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        b, _, h, w = edge_magnitude.shape
        min_val = edge_magnitude.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        max_val = edge_magnitude.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        edge_map = (edge_magnitude - min_val) / (max_val - min_val).clamp_min(1e-8)
        return edge_map

    def forward(self, x_hr, y_lr, rgb_hr):
        edge_map = self.compute_edge_map(rgb_hr)
        muti_res = y_lr - x_hr
        return muti_res, edge_map

class EMRDIFF:
    def __init__(self, configs):
        opt = configs['params']
        self.schedule_name = opt['schedule_name']
        power = opt['schedule_kwargs']['power']
        min_noise_level = opt['min_noise_level']
        kappa = opt['kappa']
        num_diffusion_timesteps = opt['steps']
        etas_end = opt['etas_end']
        band_dim = opt.get('band_dim', 128)

        if self.schedule_name == 'exponential':
            etas_start = min(min_noise_level / kappa, min_noise_level)
            increaser = math.exp(1 / (num_diffusion_timesteps - 1) * math.log(etas_end / etas_start))
            base = np.ones([num_diffusion_timesteps, ]) * increaser
            power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True) ** power
            power_timestep *= (num_diffusion_timesteps - 1)
            self.sqrt_etas = np.power(base, power_timestep) * etas_start
            self.etas = self.sqrt_etas ** 2
        elif self.schedule_name == 'olss':
            etas_start = min(min_noise_level / kappa, min_noise_level)
            self.sqrt_etas = np.linspace(etas_start, etas_end, num_diffusion_timesteps, dtype=np.float64)
            self.etas = self.sqrt_etas ** 2
        elif self.schedule_name == 'uniform':
            etas_start = min(min_noise_level / kappa, min_noise_level)
            self.sqrt_etas = np.linspace(etas_start, etas_end, num_diffusion_timesteps, dtype=np.float64)
            self.etas = self.sqrt_etas ** 2
        else:
            raise KeyError(f'{self.schedule_name} is not a valid schedule')

        self.fixed_noise_scale = opt.get('fixed_noise_scale', 1.0)
        self.kappa = opt['kappa']
        self.normalize_input = opt['normalize_input']
        self.latent_flag = opt['latent_flag']
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.band_dim = band_dim

        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas
        self.posterior_variance = self.kappa ** 2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
            self.posterior_variance[1], self.posterior_variance[1:]
        )
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.residual_calculator = EdgeAwareSpectralResidual(band_dim=band_dim)
        self.map = []
    def forward_addnoise(self, x_start, y, t, noise, rgb_hr=None):
        resy, edge_map = self.residual_calculator(x_start, y, rgb_hr)
        weighted_noise = noise
        modulated_noise = weighted_noise * edge_map
        return (
                _extract_into_tensor(self.etas, t, x_start.shape) * resy + x_start
                + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape) * modulated_noise
        )

    def inverse_denoise(self, x_start, x_t, t, noise, edge_map=None):
        if edge_map is None:
            edge_map = th.ones_like(noise[:, :1])

        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        modulated_noise = noise * edge_map
        output = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_start
                + nonzero_mask * th.exp(0.5 * _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape)) * modulated_noise
        )
        return output

    def prior_sample(self, y, noise, edge_map=None):
        if edge_map is None:
            edge_map = th.ones_like(noise[:, :1])
        t = th.tensor([self.num_diffusion_timesteps - 1, ] * y.shape[0], device=y.device).long()
        weighted_noise = noise  * edge_map
        return y + _extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * weighted_noise





