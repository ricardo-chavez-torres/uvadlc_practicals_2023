################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
# import json
from pyparsing import Opt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from cifar100_utils import get_train_validation_set, get_test_set, set_dataset


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False 
    model.fc = nn.Linear(512, num_classes)
    for param in model.fc.parameters():
        param.requires_grad = True 

    # Randomly initialize and modify the model's last layer for CIFAR100.
    model.fc.weight.data.normal_(0, 0.01)
    model.fc.bias.data.zero_()

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None, models_dir=None, results_dir = None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train_dataset, val_dataset = get_train_validation_set(data_dir, validation_size=5000, augmentation_name=augmentation_name)
    trainloader = DataLoader(train_dataset,batch_size,shuffle=True, num_workers=2)
    valloader = DataLoader(val_dataset,batch_size,shuffle=True, num_workers=2)

    # Initialize the optimizer (Adam) to train the last layer of the model.
    
    optimizer = torch.optim.Adam(model.fc.parameters(),lr = lr)
    
    #loss
    criterion = nn.CrossEntropyLoss()

    # Training loop with validation after each epoch. Save the best model.
    best_model_dict = model.state_dict()
    best_val_accuracy = 0
    train_metrics = {
        "loss": [],
        "accuracy": [],
        }
    val_metrix = {
        "loss": [],
        "accuracy": [],
    }
    
    for epoch in range(epochs):
        model.train()
        for x,y in trainloader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        val_accuracy = evaluate_model(model, valloader, device)
        if val_accuracy > best_val_accuracy:

            best_model_dict = model.state_dict()
            if models_dir:
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)
                torch.save(best_model_dict, Path(models_dir, checkpoint_name + ".pth"))
            best_val_accuracy = val_accuracy

    # Load the best model on val accuracy and return it.
    model.load_state_dict(best_model_dict)
    print(f"Best validation accuracy: {best_val_accuracy:.3f}")
    
    if results_dir:
        with open(Path(results_dir,checkpoint_name + ".txt"), 'a') as file:
            file.write(f"\nBest validation accuracy: {best_val_accuracy}\n")
    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    with torch.no_grad():
        model.eval()
        correct = 0
        for x,y in data_loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            # loss = criterion(logits,y)
            predicted_labels = torch.argmax(logits, dim=1)
            correct += (predicted_labels==y).sum().item()
        accuracy = correct / len(data_loader.dataset)


    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name, test_noise, models_dir,checkpoint_name, results_dir):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    #set wandb to track experiments if wandb in environment
    # try:
    #     import wandb
    #     with open('config.json', 'r') as config_file:
    #         config_data = json.load(config_file)
    #         wandb_api_key = config_data.get('wandb_api_key', None)
    #         wandb.login(key=wandb_api_key)
    #     use_wandb = True
    #     wandb.init(
    #         # set the wandb project where this run will be logged
    #         project="my-awesome-project",
            
    #         # track hyperparameters and run metadata
    #         config={
    #         "lr": lr,
    #         "architecture": "CNN",
    #         "dataset": "CIFAR-100",
    #         "epochs": 10,
    #         }
    #     )
    # except ImportError:
    #     # The optional library is not installed, but it's not critical for the script.
    #     use_wandb = False
    #     pass

    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training in device {device}")
    # Load the model
    model = get_model(num_classes=100)
    model = model.to(device)
    # Get the augmentation to use
    # pass

    # Train the model
    model = train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name, models_dir, results_dir)

    # Evaluate the model on the test set
    test_dataset = get_test_set(data_dir = data_dir, test_noise = test_noise)
    testloader = DataLoader(test_dataset,batch_size,shuffle=False)
    test_accuracy = evaluate_model(model, testloader, device)
    print()
    print(f"Test accuracy: {test_accuracy:.3f}")
    if results_dir:
        with open(Path(results_dir, checkpoint_name + ".txt"), 'a') as file:
            file.write(f"Test accuracy: {test_accuracy}\n")


    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--models_dir', default='models/', type=str,
                        help='name of models checkpoint directory')
    parser.add_argument('--checkpoint_name', default='', type=str,
                        help='name of checkpoint file and the results file')
    parser.add_argument('--results_dir', default='results/', type=str,
                        help='name of results directory')
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar100', 'cifar10'],
                        help='Dataset to use.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')
    parser.add_argument('--test_noise', default=False, action="store_true",
                        help='Whether to test the model on noisy images or not.')

    args = parser.parse_args()
    kwargs = vars(args)
    set_dataset(kwargs.pop('dataset'))

    if kwargs["results_dir"]:
        if not os.path.exists(kwargs["results_dir"]):
            os.makedirs(kwargs["results_dir"])

        with open(Path(kwargs["results_dir"], kwargs["checkpoint_name"] + ".txt"), 'a') as file:
            file.write(str(kwargs))
    print("Arguments and hyperparameters:", kwargs)
    main(**kwargs)
