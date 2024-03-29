import torch


def rmse(target, pred):
    MSE = torch.nn.functional.mse_loss(target, pred, reduction="sum")
    return torch.sqrt(MSE)