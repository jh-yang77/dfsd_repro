import numpy as np
import torch
import dfsd_repro.models.densenet as dn
import torchvision
import dfsd_repro.utils.svhn_loader as svhn
import torchvision.transforms as transforms
from dfsd_repro.models.route_test import *
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from dfsd_repro.models.resnet18_32x32 import *
from dfsd_repro.config import package_path

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
def get_dataloader(in_dataset,batch_size,args):

    if in_dataset == "CIFAR-10":
        testset = torchvision.datasets.CIFAR10(root=package_path("id_datasets", "CIFAR-10"), train=False, download=False, transform=transform_test)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
        num_classes = 10

    elif in_dataset == "CIFAR-100":
        testset = torchvision.datasets.CIFAR100(root=package_path("id_datasets", "CIFAR-100"), train=False, download=False, transform=transform_test)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
        num_classes = 100

    elif in_dataset == "imagenet":
        testloaderIn = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(package_path("id_datasets", "imagenet", "val"), transform_test_largescale),
            batch_size=batch_size, shuffle=False, num_workers=2)
        num_classes = 1000
    return testloaderIn,num_classes


def get_model(args,num_classes):

    if args.model_arch == 'densenet':
        key = args.in_dataset.lower().replace("-", "")
        info = None if args.p is None else np.load(package_path("features", f"{key}_train_{args.model_arch}_feature_mean.npy"))
        model = dn.DenseNet3(args.layers, num_classes, 12, reduction=0.5, bottleneck=True, dropRate=0.0, normalizer=None, p=args.p, info=info,ash=False)
        checkpoint = torch.load(
            package_path("checkpoints", args.in_dataset, "densenet", f"checkpoint_{args.epochs}.pth.tar"))
        model.load_state_dict(checkpoint['state_dict'])

    elif args.model_arch == 'resnet18':
        key = args.in_dataset.lower().replace("-", "")
        info = None if args.p is None else np.load(package_path("features", f"{key}_train_{args.model_arch}_feature_mean.npy"))
        model = ResNet18(num_classes= num_classes, p=args.p, info=info)
        checkpoint = torch.load(package_path("checkpoints", "best.ckpt"))
        model.load_state_dict(checkpoint)

    elif args.model_arch == 'resnet50':
        info = None if args.p is None else np.load(package_path("imagenet_feature", "imagenet_train_resnet50_feature_mean.npy"))
        num_classes = 1000
        from dfsd_repro.models.resnet import resnet50
        model = resnet50(num_classes=num_classes, pretrained=True, p=args.p, info=info, clip_threshold=args.clip_threshold)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)
    return model

def get_outdataset(out_dataset, in_dataset, batch_size):
    if out_dataset == "SVHN":
        testsetout = svhn.SVHN(package_path("ood_datasets", "svhn"), split="test", transform=transform_test, download=False)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=2)
    elif out_dataset == "dtd":
        transform = transform_test_largescale if in_dataset in {"imagenet"} else transform_test
        testsetout = torchvision.datasets.ImageFolder(root=package_path("ood_datasets", "dtd", "images"), transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    elif out_dataset == "places365":
        testsetout = torchvision.datasets.ImageFolder(root=package_path("ood_datasets", "places365"), transform=transform_test)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    elif out_dataset == "CIFAR-100":
        testsetout = torchvision.datasets.CIFAR100(root=package_path("id_datasets", "CIFAR-100"), train=False, download=False, transform=transform_test)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    elif out_dataset == "inat":
        testloaderOut = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(package_path("ood_datasets", "iNaturalist"), transform=transform_test_largescale),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )
    elif out_dataset == "places":
        testloaderOut = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(package_path("ood_datasets", "Places"), transform=transform_test_largescale),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )
    elif out_dataset == "sun":
        testloaderOut = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(package_path("ood_datasets", "SUN"), transform=transform_test_largescale),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )
    elif out_dataset == "GNImages-imagenet":
        dataset = GaussianNoiseDataset(num_images=1000, image_size=(224, 224), mean=0.0, std=1.0, transform=transform_test)
        testloaderOut = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    elif out_dataset == "GNImages-cifar":
        dataset = GaussianNoiseDataset(num_images=1000, image_size=(32, 32), mean=0.0, std=1.0, transform=transform_test)
        testloaderOut = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    elif out_dataset == "tiny-imagenet":
        testloaderOut = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(package_path("ood_datasets", "tiny-imagenet"), transform=transform_test),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )
    elif out_dataset == "MNIST":
        testset = torchvision.datasets.MNIST(package_path("ood_datasets", "MNIST"), train=False, download=False, transform=transform_test)
        testloaderOut = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    elif out_dataset == "OpenImage-O":
        testloaderOut = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(package_path("ood_datasets", "openimage_o"), transform=transform_test_largescale),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )
    else:
        testsetout = torchvision.datasets.ImageFolder(package_path("ood_datasets", out_dataset), transform=transform_test)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=2)
    return testloaderOut

class GaussianNoiseDataset(Dataset):
    def __init__(self, num_images=1000, image_size=(64, 64), mean=0.0, std=1.0, transform=None):
        self.num_images = num_images
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.transform = transform

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        noise = np.random.normal(loc=self.mean, scale=self.std, size=(3, *self.image_size)).astype(np.float32)
        if self.transform:
            noise = self.transform(noise)
        return noise
