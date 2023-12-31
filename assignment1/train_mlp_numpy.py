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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import copy
import torch


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    n_classes = predictions.shape[1]
    y = np.argmax(predictions, axis=1)
    conf_mat =np.zeros((n_classes, n_classes),dtype=np.int32)

    # for i in range(len(y)):
    #     conf_mat[targets[i], y[i]] += 1

    for true_class in range(n_classes):
        true_class_samples = (targets == true_class)
        for pred_class in range(n_classes):
            pred_class_samples = (y == pred_class)
            conf_mat[true_class, pred_class] = np.sum(true_class_samples & pred_class_samples)

    #vectorized way to code it
    # pairs = np.stack((targets, y))
    # idxs, counts = np.unique(pairs, axis=0,return_counts=True)
    # conf_mat =np.zeros((n_classes, n_classes),dtype=np.int32)
    # np.add.at(conf_mat, (idxs[:,0].astype(int), idxs[:,1].astype(int)), counts) 

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
    tp = np.diag(confusion_matrix)
    fp = np.sum(confusion_matrix, axis=0) - tp
    fn = np.sum(confusion_matrix, axis=1) - tp
    
    metrics["accuracy"] = tp.sum() / N
    metrics["precision"] = tp / (tp + fp)
    metrics["recall"] = tp / (tp + fn)
    metrics["f1_beta"] = ((1 + beta ** 2) * metrics["precision"] * metrics["recall"]) / ((beta ** 2) * metrics["precision"] + metrics["recall"])
    
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10, plot_cm = False, return_conf = False):
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
    predictions = np.zeros((N, num_classes))
    targets = np.zeros(N)
    idx = 0
    running_loss = 0
    for x,y in data_loader:
        y_preds = model.forward(x.reshape(x.shape[0],-1))
        running_loss += CrossEntropyModule().forward(y_preds, y) * len(y)
        shift = len(y)
        predictions[idx:idx + shift,:] = y_preds
        targets[idx:idx + shift] = y
        idx += shift
    
    conf_mat = confusion_matrix(predictions, targets)
    metrics = confusion_matrix_to_metrics(conf_mat, beta=1.)
    if plot_cm:
        fig = plt.figure()
        ax = plt.gca()
        im = ax.matshow(conf_mat, cmap='cividis')
        fig.colorbar(im)
        ax.set_ylabel("True Classes")
        ax.set_xlabel("Predicted Classes")
        ax.set_xticks(np.arange(10))
        ax.set_xticklabels(data_loader.dataset.classes, rotation = 90)
        ax.set_yticks(np.arange(10))
        ax.set_yticklabels(data_loader.dataset.classes)
        ax.set_title("Confusion matrix")
        #ax.tight_layout()
        ax.xaxis.set_ticks_position('bottom')
        plt.grid(False)
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                plt.text(j, i, str(int(conf_mat[i, j])), va='center', ha='center', color='white')


        plt.savefig("1-6_confusion_matrix.png")

    metrics["loss"] = running_loss / N
    #######################
    # END OF YOUR CODE    #
    #######################
    if return_conf:
        return metrics, conf_mat
    else:
        return metrics



def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
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

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # hidden_dims = [128]
    # lr = 0.1
    # epochs = 10
    # seed = 42
    # data_dir = 'data/'
    # batch_size = 128

    n_classes = len(cifar10["train"].dataset.classes) #10
    n_inputs = cifar10["train"].dataset[0][0].numel()

    # TODO: Initialize model and loss module
    model = MLP(n_inputs = n_inputs, 
                n_hidden = hidden_dims,
                n_classes = n_classes)
    loss_module = CrossEntropyModule()
    train_metrics = []
    val_metrics = []
    best_val_metrics = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1_beta": 0,
    }
    best_models = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_beta": None,

    }

    # TODO: Training loop including validation
    for epoch in tqdm(range(epochs)):
        
        N_train = len(cifar10_loader["train"].dataset)
        train_predictions = np.zeros((N_train, n_classes))
        train_targets = np.zeros(N_train)
        epoch_loss = 0
        idx = 0
        for x, y in cifar10_loader["train"]:
            y_preds = model.forward(x.reshape(x.shape[0],-1))
            loss = loss_module.forward(y_preds, y)
            epoch_loss += loss * len(y)
            dL = loss_module.backward(y_preds, y)
            model.backward(dL)
            for module in model.modules:
                if hasattr(module, "grads") and hasattr(module, "params"):
                    for param in module.params:
                        module.params[param] -= lr * module.grads[param]
            model.clear_cache()

            shift = len(y)
            train_predictions[idx:idx + shift,:] = y_preds
            train_targets[idx:idx + shift] = y
            idx += shift

        train_conf_mat = confusion_matrix(train_predictions, train_targets)
        train_metrics.append(confusion_matrix_to_metrics(train_conf_mat, beta=1.))
        train_metrics[-1]["loss"] = epoch_loss / N_train

        val_metrics.append(evaluate_model(model, cifar10_loader["validation"], num_classes=n_classes))
        # for key in best_val_metrics:
        key = "accuracy"
        if best_val_metrics[key] < val_metrics[-1][key]:
            best_val_metrics[key] = val_metrics[-1][key]
            best_models[key] = copy.deepcopy(model)
    
    




    val_accuracies = [i["accuracy"] for i in val_metrics]
    # TODO: Test best model
    
    test_metrics,test_conf_matrix = evaluate_model(best_models["accuracy"], cifar10_loader["test"], num_classes=n_classes, plot_cm = True, return_conf = True)
    
    
    f_01 = confusion_matrix_to_metrics(test_conf_matrix, beta=0.1)["f1_beta"]
    f_1 = confusion_matrix_to_metrics(test_conf_matrix, beta=1.)["f1_beta"]
    f_10 = confusion_matrix_to_metrics(test_conf_matrix, beta=10.)["f1_beta"]

    # Setting up the figure and axis
    fig, ax = plt.subplots()
    categories = 10
    bar_width = 0.2
    x_positions = np.arange(categories)

    # Plotting bars for each F1beta
    ax.bar(x_positions - bar_width, f_01, width=bar_width, label='beta = 0.1')
    ax.bar(x_positions, f_1, width=bar_width, label='beta = 1')
    ax.bar(x_positions + bar_width, f_10, width=bar_width, label='beta = 10')

    # Adding labels and title
    ax.set_xlabel('Categories')
    ax.set_ylabel('F1-beta')
    ax.set_title('F1-beta per category')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(cifar10_loader["test"].dataset.classes, rotation = 90)
    
    # Adding legend
    ax.legend()
    plt.savefig("1-6_F1beta.png")

    test_accuracy = test_metrics["accuracy"]
    # TODO: Add any information you might want to save for plotting
    logging_info = {
        "train": {k: [dic[k] for dic in train_metrics] for k in train_metrics[0]},
        "val": {k: [dic[k] for dic in val_metrics] for k in val_metrics[0]},
        "test": test_metrics,
    }
    torch.save(logging_info, "logging_info")
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    sns.set_theme()
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
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
    plt.savefig("1-3_loss_curves.png")
