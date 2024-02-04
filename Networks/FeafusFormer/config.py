import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--msi_channel',type=int,default=1, help='spectral band number of MSI')
parser.add_argument('--hsi_channel',type=int,default=4, help='spectral band number of HSI')
parser.add_argument('--size',type=list,default=[1,4,1600,1600], help='')
parser.add_argument('--K',type=int,default=3, help='Hyperparameter of subspace-based loss function')
parser.add_argument('--isCal_PSF',type=bool,default=True, help='if blind spatial degradation')
parser.add_argument('--isCal_SRF',type=bool,default=True, help='if blind spectral degradation')
parser.add_argument('--sf',type=int,default=4, help='scale_factor')
parser.add_argument('--pre_epoch',type=int,default=5000, help='')
parser.add_argument('--max_epoch',type=int,default=500, help='')

opt=parser.parse_args()
