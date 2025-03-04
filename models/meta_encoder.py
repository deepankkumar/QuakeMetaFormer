import torch.nn as nn
import torch

# class ResNormLayer(nn.Module):
#     def __init__(self, linear_size,):
#         super(ResNormLayer, self).__init__()
#         self.l_size = linear_size
#         self.nonlin1 = nn.ReLU(inplace=True)
#         self.nonlin2 = nn.ReLU(inplace=True)
#         self.norm_fn1 = nn.LayerNorm(self.l_size)
#         self.norm_fn2 = nn.LayerNorm(self.l_size)
#         self.w1 = nn.Linear(self.l_size, self.l_size)
#         self.w2 = nn.Linear(self.l_size, self.l_size)

#     def forward(self, x):
#         y = self.w1(x)
#         y = self.nonlin1(y)
#         y = self.norm_fn1(y)
#         y = self.w2(y)
#         y = self.nonlin2(y)
#         y = self.norm_fn2(y)
#         out = x + y
#         return out

class ResNormLayer(nn.Module):
    def __init__(self, model_dim, expansion_factor=4, dropout=0.1):
        super(ResNormLayer, self).__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.fc1 = nn.Linear(model_dim, model_dim * expansion_factor)
        self.fc2 = nn.Linear(model_dim * expansion_factor, model_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Pre-Norm: Normalize before the feed-forward sub-layer
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        out = residual + x  # Residual connection
        # Optionally, you can also apply a second norm here (post-norm)
        
        out = self.norm2(out)
        return out