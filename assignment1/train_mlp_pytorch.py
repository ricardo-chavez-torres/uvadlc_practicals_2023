  ################################################################################
# MIT License
#
# Copyright (c) 2023 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes] sith rows as true classes and columns as predicted classes
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    
    predicted_labels = torch.argmax(predictions, dim=1)

    n_classes = predictions.shape[1]
    conf_mat = torch.zeros(n_classes, n_classes)

    for i in range(targets.size(0)):
        conf_mat[int(targets[i]), int(predicted_labels[i])] += 1

    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    metrics = {}
    N = confusion_matrix.sum()
    tp = torch.diag(confusion_matrix)
    fp = confusion_matrix.sum(dim=0) - tp
    fn = confusion_matrix.sum(dim=1) - tp
    metrics["accuracy"] = tp.sum() / N
    metrics["precision"] = tp / (tp + fp)
    metrics["recall"] = tp / (tp + fn)
    metrics["f1_beta"] = ((1 + beta ** 2) * metrics["precision"] * metrics["recall"]) / ((beta ** 2) * metrics["precision"] + metrics["recall"])
    
    metrics["precision"][torch.isnan(metrics["precision"])] = 0.0
    metrics["recall"][torch.isnan(metrics["recall"])] = 0.0
    metrics["f1_beta"][torch.isnan(metrics["f1_beta"])] = 0.0
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    N = len(data_loader.dataset)
    predictions = torch.zeros((N, num_classes))
    targets = torch.zeros(N)
    idx = 0
    

    # Set the model to evaluation mode
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.view(x.shape[0], -1)  # Flatten the input
            y_preds = model(x)
            loss = nn.functional.cross_entropy(y_preds, y)
            running_loss += loss.item() * len(y)

            shift = len(y)
            predictions[idx:idx + shift, :] = y_preds
            targets[idx:idx + shift] = y
            idx += shift
    
    # Convert predictions and targets to numpy arrays
    # predictions_np = predictions.cpu().numpy()
    # targets_np = targets.cpu().numpy()

    # Calculate confusion matrix and metrics
    conf_mat = confusion_matrix(predictions, targets)
    metrics = confusion_matrix_to_metrics(conf_mat, beta=1.)
    metrics["loss"] = running_loss / N
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    n_classes = len(cifar10["train"].dataset.classes) #10
    n_inputs = cifar10["train"].dataset[0][0].numel()

    # TODO: Initialize model and loss module
    model = MLP(n_inputs = n_inputs, 
                n_hidden = hidden_dims,
                n_classes = n_classes)
    model.to(device)
    loss_module = nn.CrossEntropyLoss()
    # TODO: Training loop including validation
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    N_train = len(cifar10_loader["train"].dataset)

    best_accuracy = 0
    train_metrics = []
    val_metrics = []
    for epoch in tqdm(range(epochs)):
        idx = 0
        train_logits = torch.zeros((N_train, n_classes))
        train_targets = torch.zeros(N_train)
        epoch_loss = 0
        model.train()
        for x,y in cifar10_loader["train"]:
            x, y = x.to(device), y.to(device)
            x = x.view(x.shape[0], -1)
            logits = model(x)
            loss = loss_module(logits, y)
            loss.backward()
            epoch_loss += loss.item() * len(y)
            # TODO: Do optimization with the simple SGD optimizer
            optimizer.step()
            optimizer.zero_grad()
            shift = len(y)
            train_logits[idx:idx + shift,:] = logits
            train_targets[idx:idx + shift] = y
            idx += shift

        epoch_loss /= N_train
        train_conf_mat = confusion_matrix(train_logits, train_targets)
        train_metrics.append(confusion_matrix_to_metrics(train_conf_mat, beta=1.))
        train_metrics[-1]["loss"] = epoch_loss

        model.eval()

        val_metrics.append(evaluate_model(model, cifar10_loader["validation"], num_classes=n_classes))
        
        if best_accuracy < val_metrics[-1]["accuracy"]:
            best_accuracy = val_metrics[-1]["accuracy"]
            best_model_state_dict = deepcopy(model.state_dict())

    val_accuracies =  [i["accuracy"] for i in val_metrics]
    # TODO: Test best model
    model.load_state_dict(best_model_state_dict)
    test_metrics = evaluate_model(model, cifar10_loader["test"], num_classes=n_classes)
    test_accuracy = test_metrics["accuracy"]
    # TODO: Add any information you might want to save for plotting
    logging_info = {
        "train": {k: [dic[k] for dic in train_metrics] for k in train_metrics[0]},
        "val": {k: [dic[k] for dic in val_metrics] for k in val_metrics[0]},
        "test": test_metrics,
    }
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_info = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    print(f"test_accuracy: {test_accuracy:.2f}")
    plt.style.use('seaborn-v0_8')
    plt.figure()
    plt.plot(np.arange(kwargs["epochs"]) + 1 ,logging_info["train"]["loss"], label = "train loss")
    plt.plot(np.arange(kwargs["epochs"]) + 1 ,logging_info["val"]["loss"], label = "val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("train and val loss curves")
    plt.legend()
    plt.show()
    