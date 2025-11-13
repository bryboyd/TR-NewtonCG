import torch

def mse_loss(a: torch.Tensor, b: torch.Tensor | float | int) -> torch.Tensor:
    return torch.mean(torch.square(a - b))
