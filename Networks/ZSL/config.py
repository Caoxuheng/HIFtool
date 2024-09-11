import argparse
parser = argparse.ArgumentParser(description='ZSL hyperparameter')
parser.add_argument('--sf', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--hsi_channel', type=int, default=4, help='output channel number')
parser.add_argument('--msi_channel', type=int, default=1, help='msi channel number')
parser.add_argument('--p', type=float, default=10, help='alpha')

opt = parser.parse_args()
