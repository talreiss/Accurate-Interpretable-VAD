import torch
import clip
import numpy as np

class CLIP(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model, _ = clip.load("ViT-B/16", device=device)
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}")
        print("Context length:", self.model.context_length)
        print("Vocab size:", self.model.vocab_size)

    def forward(self, x):
        return self.model.encode_image(x)