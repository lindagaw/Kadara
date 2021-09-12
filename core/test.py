"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable
from sklearn.metrics import accuracy_score

import numpy as np
from scipy.spatial import distance
import os

def get_distribution(src_encoder, tgt_encoder, src_classifier, tgt_classifier, critic, data_loader, which_data_loader):

    if os.path.isfile('snapshots//' + which_data_loader + '_mahalanobis_std.npy') and \
        os.path.isfile('snapshots//' + which_data_loader + '_mahalanobis_mean.npy') and \
        os.path.isfile('snapshots//' + which_data_loader + '_iv.npy') and \
        os.path.isfile('snapshots//' + which_data_loader + '_mean.npy'):

        print("Loading previously computed mahalanobis distances' mean and standard deviation ... ")
        mahalanobis_std = np.load('snapshots//' + which_data_loader + '_mahalanobis_std.npy')
        mahalanobis_mean = np.load('snapshots//' + which_data_loader + '_mahalanobis_mean.npy')
        iv = np.load('snapshots//' + which_data_loader + '_iv.npy')
        mean = np.load('snapshots//' + which_data_loader + '_mean.npy')

    else:

        print("Start calculating the mahalanobis distances' mean and standard deviation ... ")
        vectors = []
        for (images, labels) in data_loader:
            images = make_variable(images, volatile=True).squeeze_()
            labels = make_variable(labels).squeeze_()
            torch.no_grad()
            src_preds = src_classifier(torch.squeeze(src_encoder(images))).detach().cpu().numpy()
            tgt_preds = tgt_classifier(torch.squeeze(tgt_encoder(images))).detach().cpu().numpy()
            critic_at_src = critic(torch.squeeze(src_encoder(images))).detach().cpu().numpy()
            critic_at_tgt = critic(torch.squeeze(tgt_encoder(images))).detach().cpu().numpy()
            for image, label, src_pred, tgt_pred, src_critic, tgt_critic \
                            in zip(images, labels, src_preds, tgt_preds, critic_at_src, critic_at_tgt):
                vectors.append(np.linalg.norm(src_critic.tolist() + tgt_critic.tolist()))
                #print('processing vector ' + str(src_critic.tolist() + tgt_critic.tolist()))

        mean = np.asarray(vectors).mean(axis=0)
        cov = np.cov(vectors)
        try:
            iv = np.linalg.inv(cov)
        except:
            iv = cov
        mahalanobis = np.asarray([distance.mahalanobis(v, mean, iv) for v in vectors])
        mahalanobis_mean = np.mean(mahalanobis)
        mahalanobis_std = np.std(mahalanobis)
        np.save('snapshots//' + which_data_loader + '_mahalanobis_mean.npy', mahalanobis_mean)
        np.save('snapshots//' + which_data_loader + '_mahalanobis_std.npy', mahalanobis_std)
        np.save('snapshots//' + which_data_loader + '_iv.npy', iv)
        np.save('snapshots//' + which_data_loader + '_mean.npy', mean)

    print("Finished obtaining the mahalanobis distances' mean and standard deviation on " + which_data_loader)
    return mahalanobis_mean, mahalanobis_std, iv, mean

def is_in_distribution(vector, mahalanobis_mean, mahalanobis_std, mean, iv):
    upper_coefficient = 0.1
    lower_coefficient = 0.1

    upper = mahalanobis_mean + upper_coefficient * mahalanobis_std
    lower = mahalanobis_mean - lower_coefficient * mahalanobis_std

    mahalanobis = distance.mahalanobis(vector, mean, iv)

    if lower < mahalanobis and mahalanobis < upper:
        return True
    else:
        return False



def eval_ADDA(src_encoder, tgt_encoder, src_classifier, tgt_classifier, critic, data_loader):

    src_mahalanobis_std = np.load('snapshots//' + 'src' + '_mahalanobis_std.npy')
    src_mahalanobis_mean = np.load('snapshots//' + 'src' + '_mahalanobis_mean.npy')
    src_iv = np.load('snapshots//' + 'src' + '_iv.npy')
    src_mean = np.load('snapshots//' + 'src' + '_mean.npy')

    tgt_mahalanobis_std = np.load('snapshots//' + 'tgt' + '_mahalanobis_std.npy')
    tgt_mahalanobis_mean = np.load('snapshots//' + 'tgt' + '_mahalanobis_mean.npy')
    tgt_iv = np.load('snapshots//' + 'tgt' + '_iv.npy')
    tgt_mean = np.load('snapshots//' + 'tgt' + '_mean.npy')

    """Evaluation for target encoder by source classifier on target dataset."""
    tgt_encoder.eval()
    src_encoder.eval()
    src_classifier.eval()
    tgt_classifier.eval()
    # init loss and accuracy
    # set loss function
    criterion = nn.CrossEntropyLoss()
    # evaluate network

    y_trues = []
    y_preds = []

    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()
        torch.no_grad()

        src_preds = src_classifier(torch.squeeze(src_encoder(images))).detach().cpu().numpy()
        tgt_preds = tgt_classifier(torch.squeeze(tgt_encoder(images))).detach().cpu().numpy()
        critic_at_src = critic(torch.squeeze(src_encoder(images))).detach().cpu().numpy()
        critic_at_tgt = critic(torch.squeeze(tgt_encoder(images))).detach().cpu().numpy()

        for image, label, src_pred, tgt_pred, src_critic, tgt_critic \
                        in zip(images, labels, src_preds, tgt_preds, critic_at_src, critic_at_tgt):

            vector = np.linalg.norm(src_critic.tolist() + tgt_critic.tolist())

            # ouf of distribution:
            if not is_in_distribution(vector, tgt_mahalanobis_mean, tgt_mahalanobis_std, tgt_mean, tgt_iv) \
                and not is_in_distribution(vector, src_mahalanobis_mean, src_mahalanobis_std, src_mean, src_iv):
                continue
            # if in distribution which the target:
            elif is_in_distribution(vector, tgt_mahalanobis_mean, tgt_mahalanobis_std, tgt_mean, tgt_iv):
                y_pred = np.argmax(tgt_pred)
            else:
                y_pred = np.argmax(src_pred)

            #y_pred = np.argmax(tgt_pred)
            y_preds.append(y_pred)
            y_trues.append(label.detach().cpu().numpy())


    print("Avg Accuracy = {:2%}".format(accuracy_score(y_true=y_trues, y_pred=y_preds)))


def eval_tgt_with_probe(encoder, critic, src_classifier, tgt_classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    src_classifier.eval()
    tgt_classifier.eval()
    # init loss and accuracy
    loss = 0.0
    acc = 0.0
    f1 = 0.0

    ys_pred = []
    ys_true = []
    # set loss function
    criterion = nn.CrossEntropyLoss()
    flag = False
    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        probeds = critic(encoder(images))

        for image, label, probed in zip(images, labels, probeds):
            if torch.argmax(probed) == 1:
                pred = torch.argmax(src_classifier(torch.squeeze(encoder(torch.unsqueeze(image, 0))))).detach().cpu().numpy()
            else:
                pred = torch.argmax(tgt_classifier(torch.squeeze(encoder(torch.unsqueeze(image, 0))))).detach().cpu().numpy()

        ys_pred.append(np.squeeze(pred))
        ys_true.append(np.squeeze(label.detach().cpu().numpy()))

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)
    #f1 /= len(data_loader.dataset)
    print("Avg Accuracy = {:2%}".format(accuracy_score(y_true=ys_true, y_pred=ys_pred)))


def eval_tgt_with_probe(encoder, critic, src_classifier, tgt_classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    src_classifier.eval()
    tgt_classifier.eval()
    # init loss and accuracy
    loss = 0
    acc = 0
    f1 = 0

    ys_pred = []
    ys_true = []
    # set loss function
    criterion = nn.CrossEntropyLoss()
    flag = False
    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        probeds = critic(torch.squeeze(encoder(images)))

        for image, label, probed in zip(images, labels, probeds):
            if torch.argmax(probed) == 1:
                pred = torch.argmax(src_classifier(torch.squeeze(encoder(torch.unsqueeze(image, 0))))).detach().cpu().numpy()
            else:
                pred = torch.argmax(tgt_classifier(torch.squeeze(encoder(torch.unsqueeze(image, 0))))).detach().cpu().numpy()

        ys_pred.append(np.squeeze(pred))
        ys_true.append(np.squeeze(label.detach().cpu().numpy()))

    acc = accuracy_score(ys_true, ys_pred)


    print("Avg Loss = {}, Accuracy = {:2%}".format(loss, acc))

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
