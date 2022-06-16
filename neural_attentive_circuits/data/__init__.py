import os
import getpass


def resolve(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find the paths: {paths}")


DATASET_PATH_REGISTRY = {
    "CIFAR": None,
    "IMNET": "/datasets01/imagenet_full_size/061417",
    "INAT": None,
    "INAT19": None,
    "IMNET-R": "/checkpoint/nasimrahaman/data/imagenet-r",
    "TINY-IMNET": resolve(
        "/datasets01/tinyimagenet/081318",
        "/work/nrahaman/nc/data/tiny-imagenet-200",
    ),
    "TINY-IMNET-R": "/checkpoint/nasimrahaman/data/tiny-imagenet-r",
}


CACHE_PATH_REGISTRY = {"IMNET": f"/checkpoint/{getpass.getuser()}/misc"}
