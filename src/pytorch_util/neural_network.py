import math
from collections import OrderedDict
from typing import List

import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_shape,  # Tuple-like
        num_output_classes: int,
        hidden_layer_sizes: List[int],
    ):
        """
        A utility to create a basic fully-connected NN. Multi-dimensional input is permitted
        but will be flattened.  Very limited knobs are provided: just the number of hidden layers
        (inferred from hidden_layer_sizes) and the number of nodes in each.

        Hidden layers will have a relu activation, except the final output will be softmax.

        TODO: provide regularization options in a simple way.
        """
        super().__init__()
        d = OrderedDict()
        flatten_layer = nn.Flatten()
        d["flatten"] = flatten_layer
        layer_in_size = math.prod(input_shape)
        layer_num = 1
        for index, hidden_layer_size in enumerate(hidden_layer_sizes):
            linear_label = f"linear{layer_num}"
            d[linear_label] = nn.Linear(
                in_features=layer_in_size, out_features=hidden_layer_size
            )
            if index != len(hidden_layer_sizes) - 1:
                relu_label = f"relu{layer_num}"
                d[relu_label] = nn.ReLU()
            layer_in_size = hidden_layer_size
            layer_num += 1
        self._flatten = nn.Flatten()
        linear_label = f"linear{layer_num}"
        d[linear_label] = nn.Linear(
            in_features=layer_in_size, out_features=num_output_classes
        )
        d["softmax"] = nn.Softmax(dim=1)
        self._neural_network = nn.Sequential(d)
        self._num_output_classes = num_output_classes
        print(self._neural_network)

    # For compatibility with common training loops.  DO NOT CALL DIRECTLY
    def forward(self, x):
        """Passes x through the created network."""
        x = self._flatten(x)
        probs = self._neural_network(x)
        return probs
