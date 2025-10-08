import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--sf',type=int,default=32, help='scale factor')
parser.add_argument('--msi_channel',type=int,default=3, help='MSI spectral band number')
parser.add_argument('--hsi_channel',type=int,default=31, help='HSI spectral band number')
parser.add_argument('--patch_size',type=int,default=80, help='patch size for training')


parser.add_argument('--n_depth',type=int,default=3, help='depth number of the autoencoder embedded in prior learning module')
parser.add_argument('--n_feat',type=int,default=64, help='feature number of the autoencoder')
args=parser.parse_args()
