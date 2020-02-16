import os
import torch
import numpy as np
from io import BytesIO
import scipy.misc
#import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from matplotlib import pyplot as plt
from PIL import Image
from dataset import TripletDataset



def cycle(iterable):
    while True:
        for x in iterable:
            yield x




class dataloaderTriplet:
    def __init__(self, path, transform, num_triplets, batchsize, resolution):
        self.path = path
        self.batchsize = batchsize
        self.num_workers = 4
        self.transform = transform
        self.resolution= resolution
        self.num_triplets= num_triplets
        self.dataset = TripletDataset(self.path, self.transform, num_triplets = self.num_triplets, resolution = self.resolution)
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batchsize,
            shuffle=False,
            num_workers=self.num_workers)

        
        

    def __iter__(self):
        return iter(self.dataloader)
    
    def __next__(self):
        return next(self.dataloader)

    def __len__(self):
        return len(self.dataloader.dataset)

 
    

