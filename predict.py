import numpy as np
import os
import pathlib
import skimage.io as io
import skimage.transform as tf
import skimage.color as color
import torch
from main import Network
from torchvision import transforms

# import Torch_model package
from Torch_model.data import Data
from Torch_model.model import Model
from Torch_model.neural import ConvPool
from Torch_model.augmentation import augmentation
from Torch_model.losses import rmse


def load_test_data(path="real_and_fake_face/"):
    images = []
    labels = []

    for directory in os.listdir(path):
        data_path = path + directory

        for im in os.listdir(data_path)[:]:
            image = io.imread(f"{data_path}/{im}")
            image = tf.resize(image, (64, 64))
            images.append(image)
            if directory == "training_fake":
                labels.append("fake")
            elif directory == "training_real":
                labels.append("real")

    images = np.array(images)
    labels = np.array(labels)

    # images, labels = augmentation(images, labels, flip_y=True, flip_x=True, brightness=True)

    return images, labels

def load_model():
    net = Network()
    net.load_state_dict(torch.load("module_with_valid.pth"))
    net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = Model(net, "adam", rmse, device)
    return clf








def load_single_image(path):
    image = io.imread(path)
    image = tf.resize(image, (1, 3, 64, 64))
    X = torch.from_numpy(image).to(torch.float32)
    # convert_tensor = transforms.ToTensor()
    # X = convert_tensor(X)
    return X


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # valid_data = Data(loader=load_test_data("valid/"), classes={'real': 0, 'fake': 1})
    # valid_data.dataset(split_size=0.05, shuffle=True, random_state=42,
    #                    images_format=torch.float32, labels_format=torch.float32,
    #                    permute=True, one_hot=True, device=device)
    fake_path = 'valid/training_fake/easy_234_1000.jpg'
    real_path = 'valid/training_real/real_01037.jpg'
    # predict_single_image()
    # dataset = valid_data.train_inputs
    # print(dataset[0].size())

    net = load_model().net
    # X = torch.rand((1, 3, 64, 64))
    # Y = net(X)
    # print(Y)

    # X = load_single_image(real_path)
    X = load_single_image(fake_path)
    Y = net(X)
    print(Y)
