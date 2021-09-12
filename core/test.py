"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable
from sklearn.metrics import accuracy_score

import numpy as np
from scipy.spatial import distance
import os

def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    ys_true = []
    ys_pred = []

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        preds = classifier(torch.squeeze(encoder(images)))
        loss += criterion(preds, labels).data

        for pred, label in zip(preds, labels):
            ys_pred.append(torch.argmax(pred).detach().cpu().numpy())
            ys_true.append(label.detach().cpu().numpy())

    acc = accuracy_score(ys_true, ys_pred)

    loss /= len(data_loader)
    #acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
