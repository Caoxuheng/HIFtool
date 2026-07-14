import argparse
from pathlib import Path

import imgvision as iv
import numpy as np
import torch
from torch.utils.data import DataLoader

from Dataloader_tool import Large_dataset
from Networks import model_generator


DIRECT_EVAL_METHODS = ('PSRT', 'MoGDCN', 'MSST', 'Fusformer', 'CaFormer',
                       'DCTransformer', 'PSTUN', 'UTAL')
ADAPTIVE_METHODS = ('BUSI', 'DBSR', 'DTDNML', 'FeafusFormer', 'UDALN', 'HyMS', 'HySure')


def method_category(method):
    if any(name in method for name in DIRECT_EVAL_METHODS):
        return 'direct'
    if any(name in method for name in ADAPTIVE_METHODS):
        return 'adaptive'
    if 'ZSL' in method:
        return 'unavailable'
    return 'unknown'


def find_checkpoint(method, dataset):
    candidates = []
    for directory in (Path(method) / dataset, Path('PretrainModel') / method / dataset):
        if directory.is_dir():
            candidates.extend(directory.glob('*.pth'))
    if not candidates:
        raise FileNotFoundError(
            f'No checkpoint found for {method}/{dataset}. Use --checkpoint to provide one explicitly.'
        )

    def checkpoint_key(path):
        try:
            return int(path.stem)
        except ValueError:
            return -1

    return max(candidates, key=checkpoint_key)


def prepare_spatial_model(model, height, width, device):
    """Refresh fixed-size masks used by UTAL before a full-image or tiled call."""
    if hasattr(model, 'Re_generate_Mask'):
        current = tuple(model.MaskX.shape[-2:])
        if current != (height, width):
            model.opt.h_size = [height, width]
            model.Re_generate_Mask(str(device))


def forward_tiled(model, lrhsi, hrmsi, scale, tile_size=None):
    if tile_size is None or max(hrmsi.shape[-2:]) <= tile_size:
        prepare_spatial_model(model, hrmsi.shape[-2], hrmsi.shape[-1], lrhsi.device)
        prediction = model(lrhsi, hrmsi)
        return prediction[0] if isinstance(prediction, (tuple, list)) else prediction
    if tile_size % scale:
        raise ValueError('tile_size must be divisible by the scale factor.')

    batch, channels = lrhsi.shape[:2]
    height, width = hrmsi.shape[-2:]
    if height % tile_size or width % tile_size:
        raise ValueError('tile_size must divide both HRMSI spatial dimensions.')
    output = torch.empty((batch, channels, height, width), device=lrhsi.device, dtype=lrhsi.dtype)
    lr_tile = tile_size // scale
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            prepare_spatial_model(model, tile_size, tile_size, lrhsi.device)
            prediction = model(
                lrhsi[:, :, top // scale:top // scale + lr_tile, left // scale:left // scale + lr_tile],
                hrmsi[:, :, top:top + tile_size, left:left + tile_size],
            )
            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]
            output[:, :, top:top + tile_size, left:left + tile_size] = prediction
    return output


def evaluate(model, data_loader, checkpoint_path, device, output_dir=None, tile_size=None, scale=32,
             max_samples=None):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['net'])
    print(f'Network is successfully loaded from {checkpoint_path}')

    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(data_loader, 1):
            if max_samples is not None and iteration > max_samples:
                break
            ground_truth, lrhsi, hrmsi = batch[0], batch[1].to(device), batch[2].to(device)
            prediction = forward_tiled(model, lrhsi, hrmsi, scale, tile_size)
            output = prediction[0].detach().cpu().permute(1, 2, 0).numpy()
            ground_truth = ground_truth[0].permute(1, 2, 0).numpy()
            sample_name = str(data_loader.dataset.test_name[iteration - 1])
            iv.spectra_metric(output, ground_truth).Evaluation(sample_name)
            if output_dir:
                np.save(output_dir / f'{sample_name}.npy', output)


def main():
    parser = argparse.ArgumentParser(description='Evaluate an HIFtool model on CAVE or HARVARD.')
    parser.add_argument('--method', default='PSRT', help='Model name accepted by Networks.model_generator.')
    parser.add_argument('--dataset', default='CAVE', choices=['CAVE', 'HARVARD'])
    parser.add_argument('--scale', type=int, default=32, help='Expected spatial scale factor.')
    # ``--sf`` is also consumed by every legacy network config while ``--scale``
    # remains the clearer public spelling for this evaluator.
    parser.add_argument('--sf', dest='scale', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--checkpoint', type=Path, help='Checkpoint path; defaults to the latest matching checkpoint.')
    parser.add_argument('--dataset-root', default='D:/CaoXuheng/dataset')
    parser.add_argument('--cave-split', default='official', choices=['official', 'sequential_60_40'])
    parser.add_argument('--patch-size', type=int, help='Full-image evaluation patch size.')
    parser.add_argument('--tile-size', type=int,
                        help='Optional HRMSI tile size for memory-constrained inference.')
    parser.add_argument('--max-samples', type=int, help='Optional limit for a smoke test.')
    parser.add_argument('--device', default='cuda', help='Torch device, e.g. cuda or cpu.')
    parser.add_argument('--save-output', action='store_true', help='Save reconstructed cubes next to the checkpoint.')
    parser.add_argument('--list-models', action='store_true',
                        help='List models supported by this evaluator and exit.')
    args = parser.parse_args()

    if args.list_models:
        print('Direct checkpoint evaluation:', ', '.join(DIRECT_EVAL_METHODS))
        print('Per-image adaptive/model-based methods:', ', '.join(ADAPTIVE_METHODS))
        print('Unavailable upstream source: ZSL')
        return

    category = method_category(args.method)
    if category != 'direct':
        if category == 'adaptive':
            raise ValueError(
                f'{args.method} is a per-image adaptive/model-based method, not a checkpoint model. '
                'Use its dedicated optimisation workflow rather than Network_eval_VL.py.'
            )
        if category == 'unavailable':
            raise NotImplementedError('ZSL is incomplete in the upstream repository and cannot be evaluated.')
        raise ValueError(f'Unknown model method: {args.method}')

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available in this Python environment.')
    device = torch.device(args.device)
    model, opt = model_generator(args.method, str(device))
    if opt.sf != args.scale:
        raise ValueError(f'{args.method} is configured for {opt.sf}x, not the requested {args.scale}x.')
    opt.dataset_root = args.dataset_root
    opt.cave_split = args.cave_split

    patch_size = args.patch_size or (512 if args.dataset == 'CAVE' else 1024)
    checkpoint_path = args.checkpoint or find_checkpoint(args.method, args.dataset)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint_path}')

    output_dir = None
    if args.save_output:
        output_dir = checkpoint_path.parent / f'eval_output_{args.scale}x'
        output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Large_dataset(opt, patch_size, name=args.dataset, type='test', lazy=True)
    data_loader = DataLoader(dataset=dataset, num_workers=0, batch_size=1, shuffle=False,
                             pin_memory=device.type == 'cuda', drop_last=False)
    evaluate(model, data_loader, checkpoint_path, device, output_dir, args.tile_size, args.scale,
             args.max_samples)


if __name__ == '__main__':
    main()
