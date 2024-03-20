import random


def ShuffleIndex(index: list, sample_ratio: float):
    mask_list = []
    if len(index) < 4:
        raise ValueError("ipnuts must be more than 4")
    else:
        remain_length = int(round((1 - sample_ratio) * len(index)))
        temp_index = index.copy()
        while len(temp_index) > remain_length:
            mask = random.choice(temp_index)
            mask_list.append(mask)
            temp_index.remove(mask)

        sample_list = [x for x in index if x not in mask_list]  # get the remain index not in cls token and not in mask_index
        assert len(mask_list) == int(round(len(index) * sample_ratio)), "mask length must be same as the ratio!!!"
    return mask_list, sample_list

def MaskEmbed(token_emb, mask_ratio):

    # token_emb.shape = [b, d, h, w]
    len = token_emb.shape[1]
    token_index = [i for i in range(0, len)]
    mask_index, sample_index = ShuffleIndex(token_index, mask_ratio)
    x = token_emb
    x[:, mask_index, :, :] = 0.0
    return x, sample_index, mask_index