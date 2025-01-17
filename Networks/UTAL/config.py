import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sf', type=int, default=32, help='scale factor')
    parser.add_argument('--msi_channel', type=int, default=3)
    parser.add_argument('--hsi_channel', type=int, default=31)
    parser.add_argument('--h_size', type=list, default=[256, 256], help='spatial size of HR-MSI')


    parser.add_argument('--Depth', type=int, default=3)
    parser.add_argument('--KS_1', type=int, default=3)
    parser.add_argument('--KS_2', type=int, default=3)
    parser.add_argument('--KS_3', type=int, default=3)


    # Meta-Train
    parser.add_argument('--pre_srf', type=str, default='Networks/UTAL/knowledge/P_N_V2.mat',help='A predefine spectral response function[for mat format]')
    parser.add_argument('--pre_srf_key', type=str, default='P',
                        help='the key of pre_srf.mat')
    parser.add_argument('--fusion_model_path', type=str, default='UTAL/cave/700.pth',
                        help='the path of well-trained supervised network where u store')
    parser.add_argument('--save_path', type=str, default='UTAL/cave/meta',
                        help='the path of well-trained unsupervised network where ud store')


    args = parser.parse_args()
    return args
