#!/usr/bin/env python

import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchinfo

RANDOM_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nl",
        "--number-of-hidden-layers",
        help="number of hidden layers of the model, the layers between the input and output layers",
        metavar="L",
        type=int,
        required=True,
        dest="number_of_hidden_layers",
    )
    parser.add_argument(
        "-nn",
        "--number-of-nodes-in-each-hidden-layer",
        help="number of hidden layers of the model, the layers between the input and output layers",
        metavar="N",
        type=int,
        required=True,
        dest="number_of_nodes_in_each_hidden_layer",
    )
    parser.add_argument(
        "-f",
        "--figure",
        help="path of file where to save the figure",
        metavar="F",
        type=str,
        required=False,
        dest="figure",
        default="curve_fit.jpg",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        help="path of file where to save the figure",
        metavar="LR",
        type=float,
        required=False,
        dest="learning_rate",
        default=1.0e-3,
    )
    parser.add_argument(
        "-i",
        "--number-of-iterations",
        help="number of epochs/iterations of the learning",
        metavar="NI",
        type=int,
        required=False,
        dest="number_of_iterations",
        default=8000,
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="enable debug messages",
        type=bool,
        required=False,
        action=argparse.BooleanOptionalAction,
        dest="debug",
    )
    parser.add_argument(
        "--csv",
        help="enable csv output",
        type=bool,
        required=False,
        action=argparse.BooleanOptionalAction,
        dest="csv",
    )
    parser.set_defaults(debug=False)
    args = parser.parse_args()
    return args


def generate_data(fn):
    # random initialization
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    # data
    X = torch.linspace(-5, 5, 100).view(-1, 1)
    Y = fn(X)
    YN = Y + 1.0 * torch.randn(X.size())  # with noise
    return (X, Y, YN)


def make_model(number_of_hidden_layers, number_of_nodes_in_each_hidden_layer):
    model = nn.Sequential()
    current_input_size = 1
    for _ in range(number_of_hidden_layers):
        model.append(nn.Linear(current_input_size, number_of_nodes_in_each_hidden_layer))
        model.append(nn.ReLU())
        current_input_size = number_of_nodes_in_each_hidden_layer
    model.append(nn.Linear(current_input_size, 1))
    return model


def make_summary(model, debug):
    verbosity = 1 if debug else 0
    model_info = torchinfo.summary(model, (1,), verbose=verbosity)
    return model_info


def optimize(model, learning_rate, number_of_iterations, debug):
    ## initialize
    loss_function = nn.MSELoss()  # loss function - mean square error
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_values_history = []

    ## optimization procedure
    loop = range(number_of_iterations)
    if debug:
        loop = tqdm(loop, desc="optimize", ascii=True)
    for _ in loop:
        # reinitialize gradients
        optimizer.zero_grad()
        # making predictions with forward pass
        Y_pred = model.forward(X)
        # calculating the loss between original and predicted data points
        loss = loss_function(Y_pred, YN)
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updateing the parameters after each iteration
        optimizer.step()
        # storing the calculated loss in a list
        loss_values_history.append(loss.item())

    return loss_values_history


def plot(X, Y, YN, model, file_path):
    fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(3 * 6, 6))

    with torch.no_grad():
        YP = model.forward(X)  # model prediction
        diff = torch.abs(YP - Y) / Y
        diffN = torch.abs(YP - YN) / Y

    # data with prediction
    ax = axes[0]
    ax.plot(X.numpy(), Y.numpy(), "k", label="Y")
    ax.plot(X.numpy(), YN.numpy(), "b.", label="YN")
    ax.plot(X.numpy(), YP.numpy(), "r--", label="NN")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid("True")

    # loss history
    ax = axes[1]
    ax.semilogy(loss_values_history, "r")
    ax.grid("True")
    ax.grid("True", which="minor", linestyle="--")
    ax.set_xlabel("Epochs/Iterations")
    ax.set_ylabel("Loss")

    # difference between the data and the prediction
    ax = axes[2]
    ax.semilogy(X.numpy(), diff.numpy(), "r-", label="|YPT - Y| / Y")
    ax.semilogy(X.numpy(), diffN.numpy(), "b-", label="|YPT - YN| / Y")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid("True")

    fig.tight_layout()
    fig.savefig(file_path)
    return


def log(model_info, error, args):
    # model_info: torchinfo.ModelStatistics
    # https://github.com/TylerYep/torchinfo/blob/main/torchinfo/model_statistics.py
    if args.csv:
        print(f"{args.number_of_hidden_layers:d},{args.number_of_nodes_in_each_hidden_layer:d},{model_info.trainable_params:d},{args.number_of_iterations:d},{args.learning_rate:.2e},{error:.4f}")
    else:
        print(f"{'number_of_hidden_layers':40s} = {args.number_of_hidden_layers:d}")
        print(f"{'number_of_nodes_in_each_hidden_layer':40s} = {args.number_of_nodes_in_each_hidden_layer:d}")
        print(f"{'trainable_params':40s} = {model_info.trainable_params:d}")
        print(f"{'number_of_iterations':40s} = {args.number_of_iterations:d}")
        print(f"{'learning_rate':40s} = {args.learning_rate:e}")
        print(f"{'error':40s} = {error:.4f}")
    return


if __name__ == "__main__":
    args = parse_args()
    X, Y, YN = generate_data(torch.exp)
    model = make_model(
        args.number_of_hidden_layers, args.number_of_nodes_in_each_hidden_layer
    )
    model_info = make_summary(model, args.debug)
    loss_values_history = optimize(model, args.learning_rate, args.number_of_iterations, args.debug)
    plot(X, Y, YN, model, args.figure)
    log(model_info, loss_values_history[-1], args)
