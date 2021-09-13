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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset

    #src_data_loader = get_office_31(dataset = 'office-31-amazon', train=True)
    #src_data_loader_eval = get_office_31(dataset = 'office-31-amazon', train=False)
    #tgt_data_loader = get_office_31(dataset = 'office-31-webcam', train=True)
    #tgt_data_loader_eval = get_office_31(dataset = 'office-31-webcam', train=False)

    src_data_loader = get_cifar_10(train=True)
    src_data_loader_eval = get_cifar_10(train=False)
    tgt_data_loader = get_stl_10(split='train')
    tgt_data_loader_eval = get_stl_10(split='test')


    src_encoder = torch.nn.Sequential(*(list(models.resnet50(pretrained=True).children())[:-1]))
    src_classifier = torch.nn.Linear(2048, 10)
    tgt_encoder = torch.nn.Sequential(*(list(models.resnet50(pretrained=True).children())[:-1]))
    tgt_classifier = torch.nn.Linear(2048, 10)
    critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims),
                        restore=params.d_model_restore)

    '''
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        src_encoder = nn.DataParallel(src_encoder)
        src_classifier = nn.DataParallel(src_classifier)
        tgt_encoder = nn.DataParallel(tgt_encoder)
        tgt_classifier = nn.DataParallel(tgt_classifier)
        critic = nn.DataParallel(critic)
    '''
    src_encoder.to(device)
    src_classifier.to(device)
    tgt_encoder.to(device)
    tgt_classifier.to(device)
    critic.to(device)

    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)

    if os.path.isfile("snapshots//symbiosis-GAN-source-encoder-final.pt") and \
        os.path.isfile("snapshots//symbiosis-GAN-source-classifier-final.pt"):
        src_encoder = init_model(src_encoder,
                            restore="snapshots//symbiosis-GAN-source-encoder-final.pt")
        src_classifier = init_model(src_classifier,
                            restore="snapshots//symbiosis-GAN-source-classifier-final.pt")
    else:
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

    if os.path.isfile("snapshots//symbiosis-GAN-critic-final.pt") and \
        os.path.isfile("snapshots//symbiosis-GAN-target-encoder-final.pt"):
        critic = init_model(critic, "snapshots//symbiosis-GAN-critic-final.pt")
        tgt_encoder = init_model(tgt_encoder, "snapshots//symbiosis-GAN-target-encoder-final.pt")
    else:
        tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader)

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

    #classifier = nn.DataParallel(torch.nn.Linear(2048, 10)).to(device)
    classifier = torch.nn.Linear(2048, 10).to(device)

    train_encoded(classifier, encoded_src_data_loader, encoded_src_data_loader_eval)
    train_encoded(classifier, encoded_tgt_data_loader, encoded_tgt_data_loader_eval)

    print("=== Evaluation result of Symbiosis GAN ===")
    eval_encoded(classifier, encoded_tgt_data_loader_eval)


    #TODO:
    '''
    experiment #2:
    What if you don't use the Encoders; just train directly on lumping X_s and X_t together?
    '''
