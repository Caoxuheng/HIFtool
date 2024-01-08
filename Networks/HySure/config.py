import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--sf',type=int,default=32, help='scale_factor, set to 8, 16, and 32 in our experiment')
parser.add_argument("--lam_p", type=float, default=0.05,help='')
parser.add_argument("--lam_r", type=float, default=5e-5,help='')
parser.add_argument("--lam_m", type=float, default=1,help='')
args=parser.parse_args()
