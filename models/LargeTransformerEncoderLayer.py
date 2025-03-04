import torch.nn as nn

class LargerTransformerEncoder(nn.Module):
    def __init__(self, linear_size, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=linear_size, 
            nhead=num_heads,
            dropout=dropout,
            activation='gelu'  # you can experiment with this
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (batch, features) -> we add a sequence dim.
        return self.transformer(x.unsqueeze(1)).squeeze(1)
