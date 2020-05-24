import numpy as np
import cv2
import io
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision import transforms
import zipfile
import torchvision.datasets.folder
from PIL import Image
import torch

from torchvision.transforms import ToTensor
from torchvision import transforms
def getNoTransform():
    return transforms.Compose([
        transforms.Resize((224,224)),
        ToTensor()
    ])
# This is the common class for both test and train data sets. Based on indexes, this class serves the sample.
class MDEASDataloader(Dataset):
  def __init__(self, dataInfo, ratio, bg_transform=ToTensor(), \
               fg_bg_transform=ToTensor(), mask_transform=ToTensor(), depth_transform=ToTensor(), TrainFlag=True):
    'Initialization'

    dataLen = int(ratio * len(dataInfo))
    if(TrainFlag):
      self.dataset = dataInfo[:dataLen]
    else:
      self.dataset = dataInfo[-dataLen:]
      
    self.bg_transform     = bg_transform
    self.fg_bg_transform  = fg_bg_transform
    self.mask_transform   = mask_transform
    self.depth_transform  = depth_transform

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.dataset)

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    data = self.dataset[index]
    # Load data and get label
    bg    = Image.open(data[0])
    fg_bg = Image.open(data[1])
    mask  = Image.open(data[2])
    depth = Image.open(data[3])

    try:
      if self.bg_transform:
        bg = self.bg_transform(bg)
      if self.fg_bg_transform:
        fg_bg = self.fg_bg_transform(fg_bg)
      if self.mask_transform:
        mask = self.mask_transform(mask)
      if self.depth_transform:
        depth = self.depth_transform(depth)
    except:
      print('Error while transform:')
    
    sample = {'bg': bg, 'fg_bg': fg_bg, 'mask': mask, 'depth':depth}
    return sample

import csv
from sklearn.utils import shuffle



def GetTrainTestData(labelInfo, ratio, DEBUG=False,  trainBS=50, testBS=50):
  with open( labelInfo, 'r') as labelData:
    labels = csv.reader(labelData, delimiter=';')
    all_data = list(labels)
    all_shuffled = shuffle( all_data, random_state=2)

    if DEBUG: all_shuffled = all_shuffled[:10000]

    train_data = MDEASDataloader( all_shuffled, ratio, getNoTransform(), getNoTransform(), getNoTransform(), getNoTransform())
    test_data  = MDEASDataloader( all_shuffled, 1.0-ratio, getNoTransform(), getNoTransform(), getNoTransform(), getNoTransform(),TrainFlag= False)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=trainBS,
                                          shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=testBS,
                                              shuffle=False, num_workers=4)

    return train_loader, test_loader



