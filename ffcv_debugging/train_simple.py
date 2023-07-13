# imports that are needed for this - 
import wandb
import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
import sys
sys.path.append("../")
from trainer_gen import train_model
from models import ConvNet
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout, NormalizeImage, Squeeze
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder
import numpy as np
import os
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set some global vars
train_shuffle =  False

# make standard CIFAR10 Dataloaders
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

ds_train = datasets.CIFAR10("../data/cifar10/", train=True, download=True, transform = transform)
ds_test = datasets.CIFAR10("../data/cifar10/", train=False, download=True, transform = transform)

dl_train = DataLoader(ds_train, shuffle = train_shuffle, batch_size = 100, num_workers = 4)
dl_test = DataLoader(ds_test, shuffle = False, batch_size = 100, num_workers = 4)


# make FFCV dataloaders
train_path = "../data/cifar10/beton_files/train.beton"
test_path = "../data/cifar10/beton_files/test.beton"
if os.path.exists(train_path) and os.path.exists(test_path):
    print(f"Beton files exist {train_path} and {test_path}")
else:
    ds_train_ffcv = datasets.CIFAR10("../data/cifar10/", train=True, download=True)
    ds_test_ffcv = datasets.CIFAR10("../data/cifar10/", train=False, download=True)
    writer = DatasetWriter(train_path, {'image': RGBImageField(max_resolution=256), 'label': IntField()})
    writer.from_indexed_dataset(ds_train_ffcv)
    writer = DatasetWriter(test_path, {'image': RGBImageField(max_resolution=256), 'label': IntField()})
    writer.from_indexed_dataset(ds_test_ffcv)

# Data decoding and augmentation
image_pipeline = [SimpleRGBImageDecoder(), ToTensor(), ToTorchImage(), ToDevice(torch.device('cuda')), NormalizeImage(np.array(mean) * 255, np.array(std) * 255, np.float32)]


#image_pipeline = [RandomResizedCropRGBImageDecoder((32,32)), ToTensor(), ToTorchImage(), ToDevice(torch.device('cuda')), NormalizeImage(np.array(mean) * 255, np.array(std) * 255, np.float32)]

label_pipeline = [IntDecoder(), ToTensor(), ToDevice(torch.device('cuda')), Squeeze()]

# Pipeline for each data field
pipelines = {'image': image_pipeline, 'label': label_pipeline}

# Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
order_train_ffcv = OrderOption.SEQUENTIAL if 
dl_train_ffcv = Loader(train_path, batch_size=100, num_workers=4, order=OrderOption.RANDOM, pipelines=pipelines)
dl_test_ffcv = Loader(test_path, batch_size=100, num_workers=4, order=OrderOption.SEQUENTIAL, pipelines=pipelines)

# instantiate the DL model 
model = ConvNet(channel=3, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size=(32, 32), bias = True)

# evalute the training for both standard and ffcv dataloader and compare the results
train_model(dl_train, dl_test, model)

