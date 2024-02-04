# -*- coding: utf-8 -*-
"""
training configuration
"""

import argparse
import torch

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

### common parameters
parser.add_argument('--blind',type=str, default=False,help='is blind fusion?')

parser.add_argument('--sf',type=int,default=32, help='scale factor')
parser.add_argument('--hsi_channel',type=int,default=128, help='HSI spectral band number')
parser.add_argument('--msi_channel',type=int,default=4, help='MSI spectral band number')

parser.add_argument('--isCal_SRF',type=bool, default=False,help='Yes means the SRF is not known and our method can adaptively learn it; No means the SRF is known as a prior information.')
parser.add_argument('--isCal_PSF',type=bool, default=False,help='Yes means the PSF is not known and our method can adaptively learn it; No means the PSF is known as a prior information.')

### parameters about the three-stage training procedure

#the first stage
parser.add_argument("--lr_stage1", type=float, default=5e-5,help='learning rate')
parser.add_argument("--epoch_stage1", type=int, default=20000, help='total epoch')
parser.add_argument("--decay_begin_epoch_stage1", type=int, default=10000, help='epoch which begins to decay,so the lr is 1e-3 in the first 10000 epochs and then it decays from 10000th epoch to 20000th epoch. When 20000 epochs are finished, the lr decays to 0')


#the second stage
parser.add_argument("--lr_stage2", type=float, default=1e-5,help='learning rate')
parser.add_argument("--epoch_stage2", type=int, default=50000, help='total epoch')
parser.add_argument("--decay_begin_epoch_stage2", type=int, default=5000, help='epoch which begins to decay,so the lr is 1e-3 in the first 20000 epochs and then it decays from 20000th epoch to 30000th epoch. When 30000 epochs are finished, the lr decays to 0')


#the last stage
parser.add_argument("--lr_stage3", type=float, default=1e-5,help='learning rate')
parser.add_argument("--epoch_stage3", type=int, default=50000, help='total epoch')
parser.add_argument("--decay_begin_epoch_stage3", type=int, default=5000, help='epoch which begins to decay,so the lr is 6e-5 in the first 5000 epochs and then it decays from 5000th epoch to 15000th epoch. When 15000 epochs are finished, the lr decays to 0')
###


args=parser.parse_args()

device = torch.device(  'cuda:{}'.format(0)  ) if  torch.cuda.is_available() else torch.device('cpu')
args.device=device
# Because the full width at half maxima of Gaussian function used to generate the PSF is set to scale factor in our experiment, 
# there exists the following relationship between  the standard deviation and scale_factor :
args.sigma = args.sf / 2.35482

