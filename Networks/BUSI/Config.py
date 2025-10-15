from argparse import ArgumentParser

parser = ArgumentParser(description='FullCNN-Net')

parser.add_argument('--flag',
                    # required=True,
                    default='fake_and_real_peppers_ms',
                    help="flag for log, or dataset img name for reminder")
parser.add_argument('--sf', type=int, default=32, help='scale factor')
parser.add_argument('--msi_channel', type=int, default=3, help='spectral channel of msi')
parser.add_argument('--hsi_channel', type=int, default=31, help='spectral channel of hsi')



parser.add_argument('--mis', type=str, default="unixy", help="reminder")
parser.add_argument('--layer_num', type=int, default=6, help='phase number of ResCNN-Net')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--rgb_wei', type=float, default=1, help='ryb loss weight')
parser.add_argument('--isCal_SRF',type=bool, default=False,help='Yes means the SRF is not known and our method can adaptively learn it; No means the SRF is known as a prior information.')
parser.add_argument('--isCal_PSF',type=bool, default=True,help='Yes means the PSF is not known and our method can adaptively learn it; No means the PSF is known as a prior information.')

parser.add_argument('--L', type=int, default=64, help='position encoding')
parser.add_argument('--eta', type=float, default=1.0, help='weight')
parser.add_argument('--ker_sz', type=int, default=32, help='kernel border size')
parser.add_argument('--imsz', type=int, default=512, help='rgb border size')
parser.add_argument('--hsi_slice_xy', type=str, default='0,0', help='check dataset_pre.py line 76-78 for explaination')
args = parser.parse_args()