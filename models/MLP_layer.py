import torch.nn as nn

class MLPLayer(nn.Module):
    def __init__(self, linear_size):
        super(MLPLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(linear_size, linear_size),
            nn.ReLU(inplace=True),
            nn.Linear(linear_size, linear_size),
            nn.LayerNorm(linear_size)
        )

    def forward(self, x):
        return self.mlp(x)
