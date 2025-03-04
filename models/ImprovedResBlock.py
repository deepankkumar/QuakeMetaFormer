import torch
import torch.nn as nn

class ImprovedResBlock(nn.Module):
    def __init__(self, model_dim, expansion_factor=4, dropout=0.1):
        super(ImprovedResBlock, self).__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.fc1 = nn.Linear(model_dim, model_dim * expansion_factor)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(model_dim * expansion_factor, model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        # Pre-Norm and expansion
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        # Residual connection and final normalization
        out = residual + x
        out = self.norm2(out)
        return out
