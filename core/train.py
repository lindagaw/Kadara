"""Pre-train encoded for source dataset."""
import torch.nn as nn
import torch.optim as optim
import torch
import params
from utils import make_variable, save_model

import os
import numpy as np
from sklearn.metrics import accuracy_score

def train_encoded(encoded, data_loader, data_loader_eval):
    """Train encoded for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoded.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoded.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        for step, (images, labels) in enumerate(data_loader):
            # make images and labels variable
            images = make_variable(images.squeeze_())
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()
            # compute loss for critic
            try:
                preds = encoded(images)
            except:
                preds = encoded(images[:,0,:,:].unsqueeze(1))
            loss = criterion(preds, labels)

            # optimize source encoded
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.data))

        # eval model
        if ((epoch + 1) % params.eval_step_pre == 0):
            print(">>> validation accuray <<<")
            eval_encoded(encoded, data_loader)
        if ((epoch + 1) % params.eval_step_pre == 0):
            print(">>> testing accuray <<<")
            eval_encoded(encoded, data_loader_eval)

    return encoded


def eval_encoded(encoded, data_loader):
    """Evaluate encoded for source domain."""
    # set eval state for Dropout and BN layers
    encoded.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        preds = encoded(images)
        #print(preds.shape)
        #print(labels.shape)
        #loss += criterion(preds, labels).data
        pred_cls = preds.data.max(1)[1]

        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
