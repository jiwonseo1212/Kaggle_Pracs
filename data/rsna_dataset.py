import os
import re
import gc
import cv2
import random
import math
from glob import glob
import numpy as np

import torch
from .base_dataset import BaseDataset
from torch.utils.data.dataloader import default_collate
from monai.transforms import Randomizable, apply_transform

import pydicom as dicom


class RsnaBaseDataset(BaseDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.transform = None

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.dataset.shape[0]

    def collater(self, samples):
        if hasattr(self.dataset, "collater"):
            return self.dataset.collater(samples)
        else:
            return default_collate(samples)
    
    @property
    def sizes(self):
        return self.dataset.sizes

    def size(self, index):
        return self.dataset.size(index)
    
    def ordered_indices(self):
        return self.dataset.ordered_indices()

    def randomize(self) -> None:
            '''-> None is a type annotation for the function that states 
            that this function returns None.'''
            
            MAX_SEED = np.iinfo(np.uint32).max + 1
            self.seed = self.R.randint(MAX_SEED, dtype="uint32")


class RSNADataset(RsnaBaseDataset, Randomizable): 
    def __init__(self, dataset, mode, transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.target_cols =  ['C1', 'C2', 'C3','C4', 'C5', 'C6', 'C7', 'patient_overall']

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [ self.atoi(c) for c in re.split(r'(\d+)', text) ]   

    def __getitem__(self, index):
        # Set Random Seed
        self.randomize()
        
        dt = self.dataset.iloc[index, :]
        study_paths = glob(f"../input/rsna-fracture-detection/zip_png_images/{dt.StudyInstanceUID}/*")
        study_paths.sort(key=self.natural_keys)
        
        # Load images
        study_images = [cv2.imread(path)[:,:,::-1] for path in study_paths]
        # Stack all scans into 1
        stacked_image = np.stack([img.astype(np.float32) for img in study_images], 
                                 axis=2).transpose(3,0,1,2)
        
        if self.transform:
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=self.seed)
                
            stacked_image = apply_transform(self.transform, stacked_image)
            
        if self.mode=="test":
            return {"image": stacked_image}
        else:
            targets = torch.tensor(dt[self.target_cols]).float()
            return {"image": stacked_image,
                    "targets": targets}

class EffnetDataSet(torch.utils.data.Dataset):
    def __init__(self, dataset, path, transforms=None):
        super().__init__()
        self.dataset = dataset
        self.path = path
        # self.transforms = transforms

    def load_dicom(path):
        """
        This supports loading both regular and compressed JPEG images. 
        See the first sell with `pip install` commands for the necessary dependencies
        """
        img=dicom.dcmread(path)
        img.PhotometricInterpretation = 'YBR_FULL'
        data = img.pixel_array    
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data=(data * 255).astype(np.uint8)
        return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB), img  
        
    def __getitem__(self, i):
        path = os.path.join(self.path, self.dataset.iloc[i].StudyInstanceUID, f'{self.dataset.iloc[i].Slice}.dcm')

        try:
            img = self.load_dicom(path)[0]
            # Pytorch uses (batch, channel, height, width) order. Converting (height, width, channel) -> (channel, height, width)
            img = np.transpose(img, (2, 0, 1))
            if self.transforms is not None:
                img = self.transforms(torch.as_tensor(img))
                
        except Exception as ex:
            print(ex)
            return None

        if 'C1_fracture' in self.dataset:
            frac_targets = torch.as_tensor(self.dataset.iloc[i][['C1_fracture', 'C2_fracture', 'C3_fracture', 'C4_fracture',
                                                            'C5_fracture', 'C6_fracture', 'C7_fracture']].astype(
                'float32').values)
            vert_targets = torch.as_tensor(
                self.dataset.iloc[i][['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']].astype('float32').values)
            frac_targets = frac_targets * vert_targets  # we only enable targets that are visible on the current slice
            return img, frac_targets, vert_targets
        return img

