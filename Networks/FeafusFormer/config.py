import argparse
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--sf', type=int, default=32, help="super resolution upscale factor")
parser.add_argument('--hsi_channel', type=int, default=31, help='output channel number')
parser.add_argument('--msi_channel', type=int, default=3, help='msi channel number')
parser.add_argument('--K', type=float, default=3, help='alpha')
parser.add_argument('--isCal_SRF',type=bool, default=True,help='Yes means the SRF is not known and our method can adaptively learn it; No means the SRF is known as a prior information.')
parser.add_argument('--isCal_PSF',type=bool, default=True,help='Yes means the PSF is not known and our method can adaptively learn it; No means the PSF is known as a prior information.')
parser.add_argument('--pre_epoch', type=int, default=300, help='')
parser.add_argument('--init_epoch', type=int, default=500, help='degradation initialisation iterations')
parser.add_argument('--residual_bound', type=float, default=0.05,
                    help='maximum magnitude of the learned HR-HSI residual')
parser.add_argument('--baseline_weight', type=float, default=0.10,
                    help='L1 weight that anchors fusion output to bicubic LR-HSI')


opt = parser.parse_known_args()[0]
