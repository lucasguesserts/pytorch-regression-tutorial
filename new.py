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
torch.set_num_threads(1)


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
    X = torch.linspace(0, 5, 1000).view(-1, 1)
    Y = fn(X)
    # remove zeros
    valid = (torch.abs(Y) > 1.0e-5)
    X = X[valid].view(-1, 1)
    Y = Y[valid].view(-1, 1)
    return (X, Y)


def make_model(number_of_hidden_layers, number_of_nodes_in_each_hidden_layer):
    model = nn.Sequential()
    current_input_size = 1
    for _ in range(number_of_hidden_layers):
        model.append(
            nn.Linear(current_input_size, number_of_nodes_in_each_hidden_layer)
        )
        model.append(nn.ReLU())
        current_input_size = number_of_nodes_in_each_hidden_layer
    model.append(nn.Linear(current_input_size, 1))
    return model


def make_summary(model, debug):
    verbosity = 1 if debug else 0
    model_info = torchinfo.summary(model, (1,), verbose=verbosity)
    return model_info


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        last_column = data.shape[1]-1
        self.length = len(data)
        self.features = data[:,:last_column] # all but the last column
        self.targets = data[:,last_column] # the last column

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.features[idx], self.targets[idx])


def make_loader(X, Y, batch_size):
    data = torch.hstack([X, Y])
    dataset = Dataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader



def optimize(X, Y, model, learning_rate, number_of_iterations, debug):
    ## initialize
    number_of_epochs = number_of_iterations
    # loss_function = lambda yp, y: torch.mean(torch.abs(yp - y) / torch.abs(y))
    loss_function = nn.MSELoss()  # loss function - mean square error
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_values_history = []
    loader = make_loader(X, Y, 100000)

    Y = Y.view(-1, 1)
    ## optimization procedure
    loop = range(number_of_epochs)
    if debug:
        loop = tqdm(loop)
    for epoch in loop:
        for Xt, Yt in loader:
            # reinitialize gradients
            optimizer.zero_grad()
            # making predictions with forward pass
            YP = model.forward(Xt)
            # calculating the loss between original and predicted data points
            loss = loss_function(YP, Yt)
            # backward pass for computing the gradients of the loss w.r.t to learnable parameters
            loss.backward()
            # updateing the parameters after each iteration
            optimizer.step()
            # storing the calculated loss in a list
            loss_values_history.append(loss.item())
            if debug: loop.set_postfix(loss=loss.item())
        loop.set_description(f"Epoch {epoch}")
    return loss_values_history


def plot(X, Y, model, n_trainable_params, file_path):
    number_of_axes = 4
    fig, axes = plt.subplots(
        nrows=1,
        ncols=number_of_axes,
        squeeze=True,
        figsize=(number_of_axes * 6, 6),
    )
    fig.suptitle(str(n_trainable_params) + " - " + "_".join(file_path.split("/")[-1].split(".")[:-1]))

    with torch.no_grad():
        YP = model.forward(X)  # model prediction
        diff = torch.abs(YP - Y) / torch.abs(Y)

    # data with prediction
    ax = axes[0]
    ax.set_title("Real and Predicted Values - Linear Scale")
    ax.plot(X.numpy(), Y.numpy(), "k.", label="real")
    ax.plot(X.numpy(), YP.numpy(), "r--", marker=".", label="predicted")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid("True")

    # data with prediction
    ax = axes[1]
    ax.set_title("Absolute of the Real and Predicted Values - Log Scale")
    ax.semilogy(X.numpy(), torch.abs(Y).numpy(), "k.", label="real")
    ax.semilogy(X.numpy(), torch.abs(YP).numpy(), "r--", marker=".", label="predicted")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid("True")
    ax.grid("True", which="minor", linestyle="--")

    # loss history
    ax = axes[2]
    ax.set_title("Model Loss History")
    ax.semilogy(loss_values_history, "r")
    ax.grid("True")
    ax.grid("True", which="minor", linestyle="--")
    ax.set_xlabel("Epochs / Iterations")
    ax.set_ylabel("Loss")

    # difference between the data and the prediction
    ax = axes[3]
    ax.set_title("Relative Difference")
    ax.semilogy(X.numpy(), diff.numpy(), "r-", label="|YPT - Y| / |Y|")
    ax.set_xlabel("x")
    ax.set_ylabel("relative difference")
    ax.legend()
    ax.grid("True")

    fig.tight_layout()
    fig.savefig(file_path)
    return


def log(model_info, error, args):
    # model_info: torchinfo.ModelStatistics
    # https://github.com/TylerYep/torchinfo/blob/main/torchinfo/model_statistics.py
    if args.csv:
        print(
            f"{args.number_of_hidden_layers:d},{args.number_of_nodes_in_each_hidden_layer:d},{model_info.trainable_params:d},{args.number_of_iterations:d},{args.learning_rate:.2e},{error:e}"
        )
    else:
        print(f"{'number_of_hidden_layers':40s} = {args.number_of_hidden_layers:d}")
        print(
            f"{'number_of_nodes_in_each_hidden_layer':40s} = {args.number_of_nodes_in_each_hidden_layer:d}"
        )
        print(f"{'trainable_params':40s} = {model_info.trainable_params:d}")
        print(f"{'number_of_iterations':40s} = {args.number_of_iterations:d}")
        print(f"{'learning_rate':40s} = {args.learning_rate:e}")
        print(f"{'error':40s} = {error:e}")
    return


if __name__ == "__main__":
    args = parse_args()
    # fn = lambda x: x * torch.sin(x) + 2 * torch.tanh(x) * torch.sin(2*x) + 0.1 * x**2 * torch.cos(4*x)
    fn = lambda x: x
    X, Y = generate_data(fn)
    model = make_model(
        args.number_of_hidden_layers, args.number_of_nodes_in_each_hidden_layer
    )
    model_info = make_summary(model, args.debug)
    loss_values_history = optimize(
        X,
        Y,
        model,
        args.learning_rate,
        args.number_of_iterations,
        args.debug,
    )
    plot(X, Y, model, model_info.trainable_params, args.figure)
    log(model_info, loss_values_history[-1], args)
