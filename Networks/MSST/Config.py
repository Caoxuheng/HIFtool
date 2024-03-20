import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def argsParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sf', type=int, default=32, help='scale factor')
    parser.add_argument('--patch_size', type=int, default=200)
    parser.add_argument('--msi_channel', type=int, default=4)
    parser.add_argument('--hsi_channel', type=int, default=93)

    #
    ### hsi msi device setting
    parser.add_argument('--cuda', type=str2bool, default=False,
                        help='Use CPU to run code')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='The number of GPU used in training')

    ### model setting
    # pavia
    parser.add_argument('--hsi_heads', type=int, default=1)
    parser.add_argument('--msi_heads', type=int, default=4)
    parser.add_argument('--patch_size_n', type=int, default=8)
    parser.add_argument('--msi_embed_dim', type=int, default=256)
    parser.add_argument('--hsi_embed_dim', type=int, default=32)
    # Chikusei
    # parser.add_argument('--hsi_heads', type=int, default=4)
    # parser.add_argument('--msi_heads', type=int, default=4)
    # parser.add_argument('--patch_size_n', type=int, default=8)
    parser.add_argument('--n_feats', type=int, default=64)
    parser.add_argument('--hsi_res_blocks', type=int, default=5)
    parser.add_argument('--msi_res_blocks', type=int, default=5)
    parser.add_argument('--hsi_num_layers', type=int, default=4)
    parser.add_argument('--msi_num_layers', type=int, default=4)

    ### training setting
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=5000,
                        help='The number of training epochs')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save period')
    parser.add_argument('--val_every', type=int, default=5,
                        help='Validation period')
    parser.add_argument('--test', type=str2bool, default=False,
                        help='Test mode')


    args = parser.parse_args()

    return args
