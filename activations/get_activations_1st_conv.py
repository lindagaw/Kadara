"""Pre-train descendant for source dataset."""
import torch.nn as nn
import torch.optim as optim
import torch

import params
from utils import make_variable, save_model

import os
import numpy as np
from sklearn.metrics import accuracy_score

def apply_descendant(descendant, data_loader, src_or_tgt, dev_or_eval):
    """Evaluate descendant for source domain."""
    # set eval state for Dropout and BN layers
    descendant.eval()
    descendant.cuda()

    activations = []
    ys = []

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:

        images = make_variable(images, volatile=True).squeeze_()
        labels = make_variable(labels).squeeze_()

        preds = descendant(images)

        for pred, label in zip(preds, labels):
            activations.append(pred.detach().cpu().numpy())
            ys.append(np.expand_dims(label.detach().cpu().numpy(), axis=0))

    activations = torch.from_numpy(np.asarray(activations))
    ys = torch.from_numpy(np.asarray(ys))

    print('the activations after the 1st conv have shape {}'.format(activations.shape))
    torch.save(activations, 'snapshots//' + src_or_tgt + '_' + dev_or_eval + '_1st_conv_activations.pt')

    print('the activations after the 1st conv have labels with shape {}'.format(ys.shape))
    torch.save(ys, 'snapshots//' + src_or_tgt + '_' + dev_or_eval + '_1st_conv_activations_labels.pt')

    return activations, ys
