import numpy as np 
import pandas as pd 
import os 
from os.path import splitext 
from typing import Union, List 
from PI import Image 
from torch.utils import data 
from troch.utils.data import DataLoader 
import albumentations as A 
from albumentations.pytorch import ToTensorV2

class DatasetTemplate(data.Dataset):
  #Supports reading and transforming images and segmentation labels. Labels are pre-stored as numpy array with data type np.int8. 
  def __init__(self, img_dir, label_dir, transform):
    self.img_dir, self.label_dir = img_dir, label_dir 
    self.img_names = []
    self.transform = transform 

  def __getitem__(self, index):
    img_name = self.img_names[index]
    img = self._get_image(img_name)
    label = self._get_label(img_name)
    img, label = self._transform(img, label)
    return img, label, img_name 

  def __len__(self):
    return len(self.img_names)

  def _get_image(self, img_name): 
    base = img_name.rsplit('.', 1)[0]
    label_dir = f'{self.label_dir}/{base}.npy'
    img = transformed['image']
    label = transformed['mask']
    return img, label


class CSVSplitDataset(DatasetTemplate):
  
