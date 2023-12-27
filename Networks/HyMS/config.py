import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--sf',type=int,default=32, help='scale_factor, set to 8, 16, and 32 in our experiment')
parser.add_argument('--d',type=int, default=2,help='search scale')
parser.add_argument("--beta", type=float, default=1e-2,help='regularization parameter beta 𝛽')
parser.add_argument("--gamma", type=float, default=1e-5,help='plenty parameter beta γ ')
args=parser.parse_args()
