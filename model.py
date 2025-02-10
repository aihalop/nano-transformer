import torch


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        d = 128
        n = 50000
        self.embedding = torch.nn.Embedding(n, d)

    def forward(self, x):
        print(x)
        print("embedding: ", self.embedding(x))
        return 0.0
