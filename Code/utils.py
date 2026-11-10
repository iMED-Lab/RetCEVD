import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def update_class_memories(
    positive_features,
    negative_features,
    positive_memory,
    negative_memory,
    maximum_size: int,
):
    positive_memory.extend(feature.detach().clone() for feature in positive_features)
    negative_memory.extend(feature.detach().clone() for feature in negative_features)
    if len(positive_memory) > maximum_size:
        positive_memory[:] = positive_memory[-maximum_size:]
    if len(negative_memory) > maximum_size:
        negative_memory[:] = negative_memory[-maximum_size:]


def set_requires_grad(module: torch.nn.Module, enabled: bool):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)
