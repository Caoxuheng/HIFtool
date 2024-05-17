import argparse
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--sf', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--hsi_channel', type=int, default=4, help='output channel number')
parser.add_argument('--msi_channel', type=int, default=1, help='output channel number')
parser.add_argument('--K', type=float, default=3, help='alpha')
parser.add_argument('--isCal_SRF',type=bool, default=True,help='Yes means the SRF is not known and our method can adaptively learn it; No means the SRF is known as a prior information.')
parser.add_argument('--isCal_PSF',type=bool, default=True,help='Yes means the PSF is not known and our method can adaptively learn it; No means the PSF is known as a prior information.')
parser.add_argument('--pre_epoch', type=int, default=300, help='')


opt = parser.parse_args()
