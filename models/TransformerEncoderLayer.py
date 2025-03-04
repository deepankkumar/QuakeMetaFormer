import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, linear_size, num_heads=4, num_layers=2):
        super(TransformerEncoderLayer, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=linear_size, nhead=num_heads),
            num_layers=num_layers
        )

    def forward(self, x):
        # print when using the model that transformer is being used
        # print("Using transformer")
        return self.transformer(x.unsqueeze(1)).squeeze(1)  # Add/remove sequence length dimension
