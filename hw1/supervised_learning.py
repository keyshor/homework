import torch

import numpy as np
import torch.nn as nn


class NeuralNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, action_bound):

        # initialize module
        super(NeuralNetwork, self).__init__()

        # define layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # action bound
        self.action_bound = action_bound

    def forward(self, inp):

        # convert input to a tensor
        inp = torch.tensor(inp, dtype=torch.float)

        # compute action
        hidden = torch.relu(self.input_layer(inp))
        hidden = torch.relu(self.hidden_layer(hidden))
        action = torch.tanh(self.output_layer(hidden))

        return self.action_bound * action


# mini batch gradient descent
# instances : n * d sized 2-D array or list, n >= 2
# labels : n * a sized 2D array or list
def learn_nn(instances, labels, network, iter=500, lr=0.0003, batch_size=100):

    # adam optimizer
    optimizer = torch.optim.Adam(network.parameters())
    loss_fn = torch.nn.MSELoss()

    # data in one table
    learning_data = np.concatenate([np.array(instances), np.array(labels)], 1)
    n = len(instances)
    d = instances[0].size

    # number of mini batches
    num_batches = ((n - 1) // batch_size) + 1

    # learning loop
    for _ in range(iter):

        # randomly permute the data
        data = np.random.permutation(learning_data)

        # Average loss
        average_loss = 0.0

        # loop over mini batches
        for i in range(num_batches):

            batch = data[batch_size * i: min(batch_size * (i+1), n)]
            x = batch[:, :d]
            y = torch.tensor(batch[:, d:], dtype=torch.float).requires_grad_(False)
            y_hat = network(x)
            loss = loss_fn(y, y_hat)
            average_loss += loss

            # gradient step
            network.zero_grad()
            loss.backward()
            optimizer.step()

        # Print
        print(average_loss / n)
