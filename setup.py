#!/usr/bin/env python
import os

from setuptools import setup

setup(
    name="neural-attentive-circuits",
    version="1.0",
    description="Neural Attentive Circuits code repository",
    author="Nasim Rahaman, Martin Weiss",
    author_email="martin.clyde.weiss@gmail.com",
    install_requires=[
        "numpy",
        "torch",
        "networkx",
        "wandb",
        "speedrun",
        "timm==0.4.12",
        "einops",
        "speedrun @ git+ssh://git@github.com/inferno-pytorch/speedrun@dev#egg=speedrun",
        "wormulon @ git+ssh://git@github.com/mweiss17/wormulon@main#egg=wormulon",
    ],
    extras_require={},
)
