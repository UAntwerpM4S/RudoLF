import torch
import torch.nn as nn

class SurrogateMLP(nn.Module):
    def __init__(self, input_dim=8, output_dim=6,
                 hidden_sizes=(128, 128, 128),
                 activation=nn.ReLU):
        super().__init__()

        layers = []
        prev = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h

        layers.append(nn.Linear(prev, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
