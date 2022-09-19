import math

import torch
from torch import nn
import numpy as np
import copy
import torch.nn.functional as F
from utils.accessor import load_json

def masked_operation(feature, mask, dim, operation):
    # feature: (*, S, *, D), mask: (*, S, *), valid bit: 1
    if operation is "mean":
        return (feature * mask.unsqueeze(-1)).sum(dim=dim) / (mask.sum(dim, keepdim=True) + 1e-9)
    elif operation is "sum":
        return (feature * mask.unsqueeze(-1)).sum(dim=dim)
    else:
        raise NotImplementedError




def fold(inputs, split_dim, stack_dim, size):
    return torch.stack(inputs.split(split_size=size, dim=split_dim), dim=stack_dim)

def right_shift(inputs, dim, length):
    ext_flag = False
    max_len = inputs.size(dim)
    zero_list = [0] * 2 * (len(inputs.size()) - dim - 1)
    pad_list = [*zero_list, length, 0]
    if inputs.dim() == 1:
        ext_flag = True
        inputs = inputs.unsqueeze(0)
        dim = dim + 1
    result = F.pad(inputs, pad_list).split(split_size=(max_len, length), dim=dim)[0]
    return result.squeeze(0) if ext_flag else result


def left_shift(inputs, dim, length):
    ext_flag = False
    max_len = inputs.size(dim)
    zero_list = [0] * 2 * (len(inputs.size()) - dim - 1)
    pad_list = [*zero_list, 0, length]
    if inputs.dim() == 1:
        ext_flag = True
        inputs = inputs.unsqueeze(0)
        dim = dim + 1
    result = inputs.split(split_size=(length, max_len - length), dim=dim)[1]
    result = F.pad(result, pad_list)
    return result.squeeze(0) if ext_flag else result



def clones(module, number):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(number)])


def random_mask(lengths, max_len, mask_nums):
    batch_size = lengths.size(0)
    prob = torch.rand(batch_size, max_len).to(lengths.device)
    valid_mask = sequence_mask(lengths, max_len)
    prob.masked_fill_(~valid_mask, 0.0)
    _, idx = torch.sort(prob, descending=True)
    mask_idx = [idx[batch][:mask_nums[batch]] for batch in range(batch_size)]
    mask_idx = torch.stack([torch.cat([mask,
                                       sample(mask, max_len - mask_nums[batch])])
                            for batch, mask in enumerate(mask_idx)], dim=0)
    return mask_idx


def re_sample(feature, feature_length, target_length):
    """Re-sample features into a fixed-length"""
    length_idx = np.round(np.linspace(0, feature_length - 1, target_length)).T.astype(np.int64)
    new_features = feature[length_idx, :]
    return new_features


def sequence_mask(lengths, max_length):
    """Generate sequence masks from variant lengths"""
    if max_length is None:
        max_length = lengths.max()
    if isinstance(lengths, torch.Tensor):
        inter = torch.ones((len(lengths), max_length)).to(device=lengths.device).cumsum(dim=1).t() > lengths.type(
            torch.float32)
        mask = (~inter).t().type(torch.bool)
    elif isinstance(lengths, np.ndarray):
        inter = np.ones((len(lengths), max_length)).cumsum(axis=1).T > lengths.astype(np.float32)
        mask = (~inter).T.astype(np.bool)
    else:
        raise NotImplementedError
    return mask


def no_peak_mask(length):
    """Generate mask to avoid Attention Modules attend to unpredicted positions"""
    np_mask = np.triu(np.ones((1, length, length)), k=1).astype('uint8')
    mask = torch.from_numpy(np_mask) == 0
    return mask


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_masks = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_masks) == 0


def sample(data, num_sample, accept_probability=1, default=None, weight=None, replace=True):
    """Sample from given data and weights"""
    assert (accept_probability == 1) or (default is not None), "Incompatible parameters!"
    if weight is not None:
        if isinstance(weight, torch.Tensor):
            weight = weight.cpu().numpy()
        weight = weight / np.sum(weight)
    if np.random.rand() <= accept_probability:
        idx = np.random.choice(np.arange(len(data)), int(num_sample), replace=replace, p=weight)
        if type(data) is list:
            return list(np.array(data)[idx])
        elif type(data) in [np.ndarray, torch.Tensor]:
            return data[idx]
        raise NotImplementedError
        # return torch.tensor(np.random.choice(data, num_sample, replace=replace, p=biaffine))
    else:
        return default


def calc_ends(center, length, min_pos=None, max_pos=None):
    start = center - length / 2.0
    end = center + length / 2.0
    if (min_pos is not None) or (max_pos is not None):
        assert (min_pos is not None) and (max_pos is not None), "Invalid parameters"
        start = math.floor(max(min(start, max_pos - 1), min_pos))
        end = math.ceil(max(min(end, max_pos), min_pos + 1))
    return start, end


def make_unidirectional_mask(pad_mask):
    """Generate Unidirectional Mask considering both pad and future positions"""
    # param: pad_mask in shape (batch_size, 1, time_step) / (batch_size, time_step)
    # no_peak_mask in shape (1, time_step, time_step)
    if pad_mask is None:
        return None
    if pad_mask.dim() == 2:
        pad_mask = pad_mask.unsqueeze(1)
    return pad_mask.bool() & no_peak_mask(pad_mask.size(-1)).to(pad_mask.device)


def apply_to_sample(f, sample):
    if hasattr(sample, '__len__') and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return (_apply(x) for x in x)
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def move_to_cpu(sample):
    def _move_to_cpu(tensor):
        # PyTorch has poor support for half tensors (float16) on CPU.
        # Move any such tensors to float32.
        if tensor.dtype == torch.float16:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.cpu()

    return apply_to_sample(_move_to_cpu, sample)


def worker_init_fn(worker_id):
    def set_seed(seed):
        import random
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed + 1)
        torch.manual_seed(seed + 3)
        torch.cuda.manual_seed(seed + 4)
        torch.cuda.manual_seed_all(seed + 4)

    set_seed(8 + worker_id)


if __name__ == "__main__":
    a_1 = torch.arange(9) + 1  # done
    a_2 = torch.stack((torch.arange(9), torch.arange(9) + 1), dim=0)
    a_3 = torch.ones(4, 4, 4)
    r_1 = left_shift(a_3, dim=2, length=1)
    r_1_ = left_shift(a_3, dim=2, length=3)
    print(r_1)
    print(a_3)
    print(r_1_)