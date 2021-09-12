"""Dataset setting and data loader for SVHN."""


import torch
from torchvision import datasets, transforms

import params


def get_stl_10(split):
    """Get SVHN dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])
    pre_process =  transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((224,224)),
                                        transforms.Normalize((0.5,), (0.5,))])
    # dataset and data loader
    stl_10_dataset = datasets.STL10(root=params.data_root,
                                   split=split,
                                   transform=pre_process,
                                   download=True)

    stl_10_data_loader = torch.utils.data.DataLoader(
        dataset=stl_10_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return stl_10_data_loader
