import torch
import torch.nn as nn

class fnafmDecoder(nn.Module):
    def __init__(self, input_dim=640, output_dim=4):
        super(fnafmDecoder, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    

def tensor_to_sequence(tensor):
    nucleotides = ['A', 'U', 'G', 'C']
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    sequence = ''.join([nucleotides[vector.argmax().item()] for vector in tensor])
    return sequence