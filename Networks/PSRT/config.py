import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sf', type=int, default=32, help='scale factor')
    parser.add_argument('--patch_size', type=int, default=32*5)
    parser.add_argument('--msi_channel', type=int, default=4)
    parser.add_argument('--hsi_channel', type=int, default=128)
    parser.add_argument('--n_bands', type=int, default=80)
    parser.add_argument('--clip_max_norm', type=int, default=10)



    # learning settingl

    # rsicd: 3e-4, ucm: 1e-4,
    parser.add_argument('--lr', type=float, default=1e-4)


    args = parser.parse_args()
    return args
