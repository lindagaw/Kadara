"""Main script for ADDA."""
import pretty_errors
import params

from torchvision import datasets, transforms, models

from core import eval_src, eval_tgt, train_src, train_tgt, train_tgt_classifier
from core import train_progenitor, eval_progenitor
from core import apply_encoder, train_encoded, eval_encoded

from models import Discriminator, LeNetClassifier, LeNetEncoder
from models import Progenitor, Descendant, Successor
from models import LeNet_Conv_1_Encoder, LeNet_Conv_1_Classifier, LeNet_Conv_2_Encoder, LeNet_Conv_2_Classifier

from utils import get_data_loader, init_model, init_random_seed, load_chopped_state_dict

from datasets import get_conv_1_activations, get_conv_2_activations
from datasets import get_office_home, get_office_31
from datasets import get_cifar_10, get_stl_10
from datasets import get_src_encoded, get_tgt_encoded

import torch
import torch.nn as nn

import os
import gc
gc.collect()
torch.cuda.empty_cache()

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset

    #src_data_loader = get_office_31(dataset = 'office-31-amazon', train=True)
    #src_data_loader_eval = get_office_31(dataset = 'office-31-amazon', train=False)
    #tgt_data_loader = get_office_31(dataset = 'office-31-webcam', train=True)
    #tgt_data_loader_eval = get_office_31(dataset = 'office-31-webcam', train=False)

    tgt_data_loader = get_cifar_10(train=True)
    tgt_data_loader_eval = get_cifar_10(train=False)
    src_data_loader = get_stl_10(split='train')
    src_data_loader_eval = get_stl_10(split='test')

    progenitor = models.resnet50(pretrained=True)
    progenitor.fc = torch.nn.Linear(2048, 10)

    #progenitor = nn.DataParallel(progenitor)
    progenitor = progenitor.to(torch.device('cuda:0'))

    src_encoder = torch.nn.Sequential(*(list(progenitor.children())[:-1]))
    src_classifier = torch.nn.Linear(2048, 10).to(torch.device('cuda:0'))

    tgt_encoder = torch.nn.Sequential(*(list(progenitor.children())[:-1]))
    tgt_classifier = torch.nn.Linear(2048, 10).to(torch.device('cuda:0'))

    critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims),
                        restore=params.d_model_restore)

    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)

    src_encoder, src_classifier = train_src(
        src_encoder, src_classifier, src_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")
    print(critic)

    tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader)

    tgt_encoder, tgt_classifier = train_tgt_classifier(
        tgt_encoder, tgt_classifier, tgt_data_loader)


    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> only source encoder <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> only target encoder <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> enforced transfer without ood <<<")
    eval_tgt(tgt_encoder, tgt_classifier, tgt_data_loader_eval)

    print("=== Starting to apply the source encoder on the source dataset ===")
    apply_encoder(src_encoder, src_data_loader, 'src', 'train')
    apply_encoder(src_encoder, src_data_loader_eval, 'src', 'eval')
    print("=== Starting to apply the target encoder on the source dataset ===")
    apply_encoder(tgt_encoder, tgt_data_loader, 'tgt', 'train')
    apply_encoder(tgt_encoder, tgt_data_loader_eval, 'tgt', 'eval')

    encoded_tgt_data_loader = get_tgt_encoded(train=True)
    encoded_tgt_data_loader_eval = get_tgt_encoded(train=False)
    encoded_src_data_loader = get_src_encoded(train=True)
    encoded_src_data_loader_eval = get_src_encoded(train=False)

    classifier = torch.nn.Linear(2048, 10).to(torch.device('cuda:0'))

    train_encoded(classifier, encoded_src_data_loader, encoded_src_data_loader_eval)
    train_encoded(classifier, encoded_tgt_data_loader, encoded_tgt_data_loader_eval)

    print("=== Evaluation result of Symbiosis GAN ===")
    eval_encoded(classifier, encoded_tgt_data_loader_eval)


    #TODO:
    '''
    experiment #1:

    1. calculate E_s(X_s) and E_t(X_t) to form a new dataloader.
    2. train a classifier on the new data_loader
    3. Use the new classifier and E_t() to classify X_t_eval

    experiment #2:
    What if you don't use the Encoders; just train directly on lumping X_s and X_t together?


    '''
