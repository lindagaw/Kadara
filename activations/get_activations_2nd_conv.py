"""Pre-train successor for source dataset."""
import torch.nn as nn
import torch.optim as optim

import params
from utils import make_variable, save_model

import os
import numpy as np
from sklearn.metrics import accuracy_score

def apply_successor(successor, data_loader, src_or_tgt, dev_or_eval):
    """Evaluate successor for source domain."""
    # set eval state for Dropout and BN layers
    successor.eval()
    successor.cuda()

    activations = []
    ys = []

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        preds = successor(images)

        for pred, label in zip(preds, labels):
            activations.append(pred.detach().cpu().numpy())
            ys.append(np.expand_dims(label.detach().cpu().numpy(), axis=0))

    activations = np.asarray(activations)
    ys = np.asarray(ys)

    print('the activations after the 1st conv have shape {}'.format(activations.shape))
    np.save('snapshots//' + src_or_tgt + '_' + dev_or_eval + '_2nd_conv_activations.npy', activations)

    print('the activations after the 1st conv have labels with shape {}'.format(ys.shape))
    np.save('snapshots//' + src_or_tgt + '_' + dev_or_eval + '_2nd_conv_activations_labels.npy', ys)

    return activations, ys
