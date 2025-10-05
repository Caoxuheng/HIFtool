A concise guide to PSRT training and evaluation—covering key arguments, defaults, data splits, and reproducible commands for HARVARD datasets

## PSRT.Config

| Argument          | Type / Default        | Description                                                                                                                                         |
| ----------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--sf`            | `int`, default `32`   | **Scale factor** (HR→LR spatial ratio). Controls the resolution gap between modalities.                                                             |
| `--patch_size`    | `int`, default `32*4` | **Base patch size** tied to the dataset interface. Note: there is another `--patch_size` below (training). The **last provided value on CLI wins**. |
| `--msi_channel`   | `int`, default `3`    | Channels of the HR MSI (e.g., RGB=3).                                                                                                               |
| `--hsi_channel`   | `int`, default `31`   | Spectral channels of the LR HSI (e.g., 31/28/103).                                                                                                  |
| `--n_bands`       | `int`, default `10`   | Spectral subgroup size (used by band grouping/attention, if applicable).                                                                            |
| `--clip_max_norm` | `int`, default `10`   | L2 **gradient clipping** threshold to improve stability.                                                                                            |

## Network_training_VL
| Argument        | Type / Default                           | Description                                                                         |
| --------------- | ---------------------------------------- | ----------------------------------------------------------------------------------- |
| `--method`      | `str`, default `PSRT`                    | Method name (used in logs/saving).                                                  |
| `--dataset`     | `str`, default `CAVE` (`CAVE`/`HARVARD`) | Dataset selector.                                                                   |
| `--batch_size`  | `int`, default `16`                      | Mini-batch size.                                                                    |
| `--epochs`      | `int`, default `1000`                    | Total training epochs.                                                              |
| `--ckpt_step`   | `int`, default `50`                      | Save a checkpoint every N epochs.                                                   |
| `--lr`          | `float`, default `1e-4`                  | Initial learning rate.                                                              |
| `--patch_size`  | `int`, default `128`                     | **Training crop size** (recommended to use this one as the single source of truth). |
| `--resume`      | `flag`, default `False`                  | Enable resume-from-checkpoint.                                                      |
| `--start_epoch` | `int`, default `0`                       | Starting epoch when resuming (used to locate ckpt like `1750.pth`).                 |
| `--general`     | `flag`, default `True`                   | General training over the whole training set.                                       |
| `--specific`    | `flag`, default `False`                  | Per-image fine-tuning / small-set evaluation.                                       |
| `--meta`        | `flag`, default `False`                  | Enable meta-training routine instead of normal training.                            |

## Network_eval_VL
Method       = 'PSRT'  
dataset_name = 'HARVARD'  
patch_size   = 1024    # Larger crop / full-image inference for cleaner visualization  
bestepoch    = 600   # Select the best epoch   

## loader_L
self.test_name = sorted(glob('Multispectral Image Dataset/HARVARD/**/*.mat', recursive=True))  
test_list = [i-1 for i in [4,8,13,19,20,25,27,31,35,42,43,44,48,52,58,59,60,67,70]]  
val_list = [i-1 for i in [1, 3, 10,21,45,63,36,72]]    
train_set = list(set(range(77)) - set(test_list) - (set(val_list)))  
## Dataset
[Harvard](https://vision.seas.harvard.edu/hyperspec/d2x5g3/)
