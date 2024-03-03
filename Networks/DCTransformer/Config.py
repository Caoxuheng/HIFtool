import argparse
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--sf', type=int, default=32, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--patch_size',type=int,default=96, help='patch size for training')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--hsi_channel', type=int, default=31, help='output channel number')
parser.add_argument('--msi_channel', type=int, default=3, help='output channel number')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument('--nEpochs', type=int, default=0, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--save_folder', default='TrainedNet/', help='Directory to keep training outputs.')
parser.add_argument('--outputpath', type=str, default='result/', help='Path to output img')
parser.add_argument('--mode', default=1, type=int, help='Train or Test.')
parser.add_argument('--local_rank', default=1, type=int, help='None')
parser.add_argument('--use_distribute', type=int, default=1, help='None')
parser.add_argument('--srfpath',type=str,default='NikonD700.npy', help='where you save your HSI reconstruction results')

opt = parser.parse_args()
