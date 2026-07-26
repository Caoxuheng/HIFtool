import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--sf',type=int,default=32, help='scale factor')
parser.add_argument('--msi_channel',type=int,default=3, help='MSI spectral band number')
parser.add_argument('--hsi_channel',type=int,default=31, help='HSI spectral band number')
parser.add_argument('--patch_size',type=int,default=80, help='patch size for training')
parser.add_argument('--n_depth',type=int,default=3, help='depth number of the autoencoder embedded in prior learning module')
parser.add_argument('--n_feat',type=int,default=64, help='feature number of the autoencoder')
parser.add_argument('--caformer_mode', choices=('supervised', 'blind'), default='supervised',
                    help='supervised checkpoint inference or uHNTC-guided blind adaptation')
parser.add_argument('--caformer_stages', type=int, default=3,
                    help='total CaFormer stages; HIFTool CaFormer_3 uses 3')
parser.add_argument('--blind_steps', type=int, default=1000,
                    help='per-observation blind adaptation updates')
parser.add_argument('--blind_lr', type=float, default=3e-5,
                    help='per-observation blind adaptation learning rate')
parser.add_argument('--uhntc_init_steps', type=int, default=500,
                    help='steps for each uHNTC degradation initialization phase')
parser.add_argument('--blind_tile_size', type=int, default=0,
                    help='optional HR tile size; 0 evaluates the complete image')
args=parser.parse_known_args()[0]
