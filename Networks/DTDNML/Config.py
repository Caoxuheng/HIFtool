# -*- coding: utf-8 -*-
"""
training configuration
"""

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

### common parameters
parser.add_argument('--blind',type=str, default=False,help='is blind fusion?')

parser.add_argument('--sf',type=int,default=32, help='scale factor')
parser.add_argument('--hsi_channel',type=int,default=31, help='HSI spectral band number')
parser.add_argument('--msi_channel',type=int,default=3, help='MSI spectral band number')
parser.add_argument('--patch_size',type=int,default=512, help='Patch size')

parser.add_argument('--isCal_SRF',type=bool, default=False,help='Yes means the SRF is not known and our method can adaptively learn it; No means the SRF is known as a prior information.')
parser.add_argument('--isCal_PSF',type=bool, default=True,help='Yes means the PSF is not known and our method can adaptively learn it; No means the PSF is known as a prior information.')

parser.add_argument('--isTrain',type=bool, default=True,help='')
parser.add_argument("--avg_crite", type=str, default="No")
parser.add_argument('--concat',type=str, default="Yes",help='')
parser.add_argument('--gpu_ids',type=list, default=[0],help='GPU ID')
parser.add_argument('--lr',type=float, default=1e-4,help='learning_rate')

parser.add_argument('--num_theta',type=float, default=30,help='num_theta')
parser.add_argument('--lambda_A',type=float, default=0.1,help='num_theta')
parser.add_argument('--lambda_B',type=float, default=0,help='spectral manifold')
parser.add_argument('--lambda_C',type=float, default=0,help='spatial manifold')
parser.add_argument(
                "--lambda_D",
                type=float,
                default=1.0,
                help="weight for LR-MSI constraints",
            )
parser.add_argument('--lambda_F',type=float, default=100,help='num_theta')


### parameters about the three-stage training procedure



args=parser.parse_args()
