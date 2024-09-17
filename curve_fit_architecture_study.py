import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchinfo

RANDOM_SEED = 42


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
    for _ in range(number_of_hidden_layers):
        model.append(nn.Linear(1, number_of_nodes_in_each_hidden_layer))
        model.append(nn.ReLU())
    model.append(nn.Linear(number_of_nodes_in_each_hidden_layer, 1))
    return model


def make_summary(model):
    torchinfo.summary(model, (1,))
    return


def optimize(model):
    ## initialize
    loss_function = nn.MSELoss()  # loss function - mean square error
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    number_of_iterations = 8000
    loss_values_history = []

    ## optimization procedure
    for _ in tqdm(range(number_of_iterations), desc="optimize", ascii=True):
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


def plot(X, Y, YN, model):
    fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(3 * 6, 6))

    with torch.no_grad():
        YP = model.forward(X) # model prediction
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
    fig.savefig('curve_fit.jpg')
    return


if __name__ == "__main__":
    X, Y, YN = generate_data(torch.exp)
    model = make_model(1, 16)
    make_summary(model)
    loss_values_history = optimize(model)
    print(f"mean square error = {loss_values_history[-1]:.4f}")
    plot(X, Y, YN, model)
