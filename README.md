# Neural Attentive Circuits (NACs)
Neural Attentive Circuits are a general-purpose modular neural architecture. 
This codebase provides the training and inference code to replicate the imagenet experiments in the [paper](arxivlinkgoeshere).

### Installation
To install the NACs codebase follow these directions.

1. Create a conda environment:``conda create -n nac-public python=3.8``
2. Activate the environment: ``source activate nac-public``
3. Clone the repository: ``git clone git@github.com:mweiss17/NeuralAttentiveCircuits.git``
4. Install the dependencies: ``pip install -e .``
5. Install the ImageNet-1k 2012 dataset: https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
6. Install the Tiny-ImageNet dataset: https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4
7. Install the Tiny-ImageNet-R dataset: https://zenodo.org/record/6653675

### Train a NAC!
In order to reduce the boilerplate for our experiments, we used two libraries packages [SpeedRun](https://github.com/inferno-pytorch/speedrun) and [Wormulon](https://github.com/mweiss17/wormulon/tree/main/wormulon).
To run an experiment, these must be installed (and should be installed using a pip installation of setup.py).
The magic words to train a NAC are as follows: ``python3 train_supervised.py experiments/<experiment-name> --inherit templates/INET-NC-X``


### Play with a NAC!
If you just want to use a pre-trained NAC, then you can download them from torch-hub.

| Model Name            | Dataset                     | IID Acc@1 (%) | OOD Acc@1 (%) | Link       |
|-----------------------|-----------------------------|-----------|---------|------------|
| NAC Scale-Free        | (Tiny) Imnet / Imnet-R      | 60.76     | 19.52   | [todo]()   | 
| NAC Planted-Partition | (Tiny) Imnet / Imnet-R               | 60.71     | 19.42   | [todo]()   | 
| NAC Ring-of-Cliques   | (Tiny) Imnet / Imnet-R               | 60.54     | 20.03   | [todo]()   | 
| NAC Erdos-Renyi       | (Tiny) Imnet / Imnet-R               | 60.33     | 19.83   | [todo]()   | 

