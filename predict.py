import numpy as np
import os
import pathlib
import skimage.io as io
import skimage.transform as tf
import skimage.color as color
import torch

# import my Library (Pytorch Framework)
from haroun import Data, Model, ConvPool
from haroun.augmentation import augmentation
from haroun.losses import rmse
from main import Network

net = Network()
checkpoint = torch.load("module_with_valid.pth")
net.load_state_dict(checkpoint)

