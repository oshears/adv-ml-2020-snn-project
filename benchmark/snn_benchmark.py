# Python Imports
import os
import argparse

# PyTorch Imports
from torchvision import transforms

# BindsNET Imports
from bindsnet.encoding import PoissonEncoder, BernoulliEncoder, RankOrderEncoder
from bindsnet.datasets import MNIST
from bindsnet.learning import PostPre, WeightDependentPostPre, Hebbian, MSTDP 

# Project Imports
from ..models.snn_models import IF_Network, LIF_Network, SRM0_Network, DiehlAndCook_Network



# Benchmark Arguments
time = 250
dt = 1.0
intensity = 128



# Encoding Combinations

# Load MNIST data.

# Poisson Encoder
train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Bernoulli Encoder
train_dataset = MNIST(
    BernoulliEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Rank Order Encoder
train_dataset = MNIST(
    RankOrderEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Network Models w/ Different Learning Rule Combinations

# IF Networks
network = IF_Network(update_rule=PostPre)
network = IF_Network(update_rule=WeightDependentPostPre)
network = IF_Network(update_rule=Hebbian)

# LIF Networks
network = LIF_Network(update_rule=PostPre)
network = LIF_Network(update_rule=WeightDependentPostPre)
network = LIF_Network(update_rule=Hebbian)

# SRM0 Networks
network = SRM0_Network(update_rule=PostPre)
network = SRM0_Network(update_rule=WeightDependentPostPre)
network = SRM0_Network(update_rule=Hebbian)

# DiehlAndCook Networks
network = DiehlAndCook_Network(update_rule=PostPre)
network = DiehlAndCook_Network(update_rule=WeightDependentPostPre)
network = DiehlAndCook_Network(update_rule=Hebbian)