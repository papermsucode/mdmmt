import torch.nn as nn

class IdLayer(nn.Module):
    def forward(self, x):
        return x
