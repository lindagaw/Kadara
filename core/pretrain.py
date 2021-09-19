"""Pre-train encoder and classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim
import torch
import params
from utils import make_variable, save_model


def train_src(encoder, classifier, data_loader, data_loader_eval):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        for step, (images, labels) in enumerate(data_loader):

            print('hello')
            # make images and labels variable
            images = make_variable(images.squeeze_())
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            encoded = torch.squeeze(encoder(images))

            #print(encoded.shape)

            preds = classifier(encoded)
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()
            print('goodbye')

            # print step info
            #if ((step + 1) % params.log_step_pre == 0):
            print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                  .format(epoch + 1,
                          params.num_epochs_pre,
                          step + 1,
                          len(data_loader),
                          loss.data))

        # eval model on test set

        if ((epoch + 1) % params.eval_step_pre == 0):
            print('Outputting the validation accuracy ...')
            eval_src(encoder, classifier, data_loader)
        if ((epoch + 1) % params.eval_step_pre == 0):
            print('Outputting the testing accuracy ...')
            eval_src(encoder, classifier, data_loader_eval)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, "symbiosis-GAN-source-encoder-{}.pt".format(epoch + 1))
            save_model(
                classifier, "symbiosis-GAN-source-classifier-{}.pt".format(epoch + 1))

    # # save final model
    save_model(encoder, "symbiosis-GAN-source-encoder-final.pt")
    save_model(classifier, "symbiosis-GAN-source-classifier-final.pt")

    return encoder, classifier


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images.squeeze_())
        labels = make_variable(labels.squeeze_())

        encoded = torch.squeeze(encoder(images))

        preds = classifier(encoded)
        loss += criterion(preds, labels).data

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
