import torch.nn as nn
import torch.optim as optim
import torch
import params
from utils import make_variable, save_model

import numpy as np
import os

def apply_encoder(encoder, data_loader, dest, train_or_eval):
    """return an encoded dataset."""

    print('start calculating the encoded values on the src/tgt dataset ...')
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.eval()

    ####################
    # 2. train network #
    ####################

    encoded_xs = []
    encoded_ys = []

    for step, (images, labels) in enumerate(data_loader):
        images = make_variable(images.squeeze_())
        labels = make_variable(labels.squeeze_())
        encoded = torch.squeeze(encoder(images))

        for label, single_encoded in zip(labels, encoded):
            encoded_xs.append(single_encoded)
            encoded_ys.append(label)

    encoded_xs = np.asarray(encoded_xs)
    encoded_ys = np.asarray(encoded_ys)

    print('encoded xs has shape ' + str(encoded_xs.shape))
    print('encoded ys has shape ' + str(encoded_ys.shape))

    np.save('snapshots//' + dest + '_' + train_or_eval + '_encoded_xs.npy', encoded_xs)
    np.save('snapshots//' + dest + '_' + train_or_eval + '_encoded_ys.npy', encoded_ys)


    return encoded_xs, encoded_ys
