import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sf', type=int, default=32, help='scale factor')
    parser.add_argument('--msi_channel', type=int, default=4)
    parser.add_argument('--hsi_channel', type=int, default=120)
    parser.add_argument('--h_size', type=list, default=[320, 320], help='spatial size of HR-MSI')


    parser.add_argument('--Depth', type=int, default=3)
    parser.add_argument('--KS_1', type=int, default=3)
    parser.add_argument('--KS_2', type=int, default=3)
    parser.add_argument('--KS_3', type=int, default=3)

    args = parser.parse_args()
    return args
