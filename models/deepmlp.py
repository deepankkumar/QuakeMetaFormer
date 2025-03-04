import torch.nn as nn

class DeepMLP(nn.Module):
    def __init__(self, linear_size, num_layers=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(linear_size, linear_size))
            layers.append(nn.ReLU(inplace=True))
            # Optionally, add dropout between layers:
            layers.append(nn.Dropout(p=0.1))
        layers.append(nn.LayerNorm(linear_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # print("DeepMLP")
        return self.mlp(x)
