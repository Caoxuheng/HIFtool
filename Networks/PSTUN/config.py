import argparse
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--sf', type=int, default=32, help="super resolution upscale factor")
parser.add_argument('--patch_size',type=int,default=128, help='patch size for training')
parser.add_argument('--hsi_channel', type=int, default=128, help='output channel number')
parser.add_argument('--msi_channel', type=int, default=4, help='output channel number')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument('--nEpochs', type=int, default=0, help='number of epochs to train for')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--mode', default=1, type=int, help='Train or Test.')
parser.add_argument('--local_rank', default=1, type=int, help='None')
parser.add_argument('--use_distribute', type=int, default=1, help='None')

opt = parser.parse_args()
