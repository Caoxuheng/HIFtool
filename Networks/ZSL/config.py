import argparse
parser = argparse.ArgumentParser(description='ZSL hyperparameter')
parser.add_argument('--sf', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--hsi_channel', type=int, default=4, help='output channel number')
parser.add_argument('--msi_channel', type=int, default=1, help='msi channel number')
parser.add_argument('--p', type=float, default=10, help='alpha')
parser.add_argument('--init_lr1', type=float, default=1e-4, help='-')
parser.add_argument('--init_lr2', type=float, default=5e-4, help='=')
parser.add_argument('--decay_power', type=float, default=1.5, help='=')
parser.add_argument('--BATCH_SIZE', type=int, default=64, help='=')

opt = parser.parse_args()
