import os
import glob

from pathlib import Path
from typing import Union, Optional, Callable, Sequence

import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from ..utils import relabel

class SplitDataset(Dataset):
    """Auxiliary class to split a dataset into train and test sets.
    """
    def __init__(self, dataset, transform = None):
        self.ds = dataset
        self.transform = transform

    def __getitem__(self, idx):
        d, l = self.ds[idx]
        if self.transform:
            d = self.transform(d)
        return d, l

    def __len__(self):
        return len(self.ds)


class ACDCDataset(Dataset):
    """Helper class to load de ACDC data as a torch dataset.
    """
    def __init__(self, root_dir:Union[str,Path], target_size:tuple=(256, 256), transform=None):
        self.root_dir = root_dir
        self.target_size = target_size
        self.transform = transform
        
        self.file_slices = []  # List of (img_path, label_path, slice_idx)
        
        for subset in os.listdir(root_dir):  # Analyze all subdirectories
            subset_path = os.path.join(root_dir, subset)
            if not os.path.isdir(subset_path):
                continue
            
            for patient in os.listdir(subset_path):
                patient_path = os.path.join(subset_path, patient)
                if not os.path.isdir(patient_path):
                    continue
                
                # Find only the labels (_gt.nii.gz)
                label_files = glob.glob(os.path.join(patient_path, "*_gt.nii.gz"))
                for label_path in label_files:
                    img_path = label_path.replace("_gt.nii.gz", ".nii.gz")
                    if not os.path.exists(img_path):
                        continue  # Make sure that the image file exists
                    
                    num_slices = nib.load(label_path).shape[2]  # type: ignore
                    for slice_idx in range(num_slices):
                        self.file_slices.append((img_path, label_path, slice_idx))
    
    def __len__(self):
        return len(self.file_slices)

    def __getitem__(self, idx):
        img_path, label_path, slice_idx = self.file_slices[idx]
        
        image = nib.load(img_path).get_fdata()    # type: ignore
        label = nib.load(label_path).get_fdata()  # type: ignore    
        
        image = torch.tensor(image[:, :, slice_idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label[:, :, slice_idx], dtype=torch.int).unsqueeze(0)
        
        image = self.resize_and_pad(image)
        label = self.resize_and_pad(label)


        if self.transform:
            image = self.transform(image)
            # No transform to the label!
        label = relabel(label)
            
        return image, label
    

    def resize_and_pad(self, tensor):
        _, h, w = tensor.shape
        target_h, target_w = self.target_size
        
        # Trim if its bigger
        if h > target_h:
            start_h = (h - target_h) // 2
            tensor = tensor[:, start_h:start_h + target_h, :]
        if w > target_w:
            start_w = (w - target_w) // 2
            tensor = tensor[:, :, start_w:start_w + target_w]
        
        # Use padding if its smaller
        pad_h = max(0, target_h - tensor.shape[1])
        pad_w = max(0, target_w - tensor.shape[2])
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        return tensor


def get_ACDC_dataloader(root_dir:Union[str,Path]='data', target_size:tuple=(256, 256),
                        batch_size:int=4, shuffle=True, transform:Optional[Callable]=None, num_workers=0):
    dataset = ACDCDataset(root_dir, target_size=target_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader


def get_ACDC_train_test_dataloaders(root_dir:Union[str,Path]='data', target_size:tuple=(256, 256),
                                    train_batch_size:int=4, test_batch_size:int=4,
                                    train_transform:Optional[Callable]=None, test_transform:Optional[Callable]=None,
                                    test_size:float=0.2,
                                    shuffle=True, num_workers=0):
    
    dataset = ACDCDataset(root_dir, target_size=target_size)
    
    # Split dataset into train and test
    train_size = int((1 - test_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataset = SplitDataset(train_dataset, transform=train_transform)  # type: ignore
    test_dataset = SplitDataset(test_dataset, transform=test_transform)     # type: ignore
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size,
                                  shuffle=shuffle, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size,
                                 shuffle=shuffle, num_workers=num_workers)
    
    return train_dataloader, test_dataloader

def get_ACDC_cal_tun_tst_dataloaders(root_dir:Union[str,Path]='data', target_size:tuple=(256, 256),
                                     cal_batch_size:int=10, tune_batch_size:int=10, test_batch_size:int=10,
                                     cal_transform:Optional[Callable]=None, tune_transform:Optional[Callable]=None, test_transform:Optional[Callable]=None,
                                     splits:Sequence=(0.1,0.4,0.5),
                                     shuffle=True, num_workers=0) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    dataset = ACDCDataset(root_dir, target_size=target_size)
    
    # Split dataset into train and test
    cal_dataset, tune_dataset, test_dataset = torch.utils.data.random_split(dataset, splits)
    cal_dataset = SplitDataset(cal_dataset, transform=cal_transform)    # type: ignore
    tune_dataset = SplitDataset(tune_dataset, transform=tune_transform)    # type: ignore
    test_dataset = SplitDataset(test_dataset, transform=test_transform)    # type: ignore

    cal_dataloader = DataLoader(cal_dataset, batch_size=cal_batch_size,
                                  shuffle=shuffle, num_workers=num_workers)
    tune_dataloader = DataLoader(tune_dataset, batch_size=tune_batch_size,
                                 shuffle=shuffle, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size,
                                 shuffle=shuffle, num_workers=num_workers)
    
    return cal_dataloader, tune_dataloader, test_dataloader

class NiiDataset(Dataset):
    def __init__(self, image_dir, label_dir, target_size=(256, 256), transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.target_size = target_size
        self.transform = transform
        self.file_names = [f for f in os.listdir(image_dir) if f.endswith(".nii.gz")]
        
        self.slices_per_file = {}  # Number of slices per file
        self.total_slices = 0
        self.file_slices = []  # (archivo, slice_idx) for indexing
        
        for file_name in self.file_names:
            img_path = os.path.join(self.image_dir, file_name)
            num_slices = nib.load(img_path).shape[2]  # type: ignore # Slice number in the z-axis
            self.slices_per_file[file_name] = num_slices
            for slice_idx in range(num_slices):
                self.file_slices.append((file_name, slice_idx))
            self.total_slices += num_slices

    def __len__(self):
        return self.total_slices

    def __getitem__(self, idx):
        img_name, slice_idx = self.file_slices[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_name = img_name.replace(str(self.image_dir), str(self.label_dir))
        label_path = os.path.join(self.label_dir, label_name)
        
        image = nib.load(img_path).get_fdata()    # type: ignore
        label = nib.load(label_path).get_fdata()  # type: ignore
        
        image = torch.tensor(image[:, :, slice_idx], dtype=torch.float32).unsqueeze(0)  # Extract specific slice
        label = torch.tensor(label[:, :, slice_idx], dtype=torch.float32).unsqueeze(0)

        # Aplicar padding para alcanzar target_size
        pad_h = max(0, self.target_size[0] - image.shape[1])
        pad_w = max(0, self.target_size[1] - image.shape[2])
        
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        label = F.pad(label, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        if self.transform:
            image = self.transform(image)
            #label = self.transform(label)
        
        return image, label


def get_dataloader(image_root='test_data', label_root='reference_data', batch_size=4, shuffle=True, num_workers=0, transform=None):
    
    dataset = NiiDataset(image_root, label_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader

def get_train_test_dataloader(image_root='test_data', label_root='reference_data', batch_size=4, shuffle=True, num_workers=0, train_transform=None, test_transform=None, test_size=0.2):
    dataset = NiiDataset(image_root, label_root, transform=None)

    # Split dataset into train and test
    train_size = int((1 - test_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataset = SplitDataset(train_dataset, transform=train_transform)  # type: ignore
    test_dataset = SplitDataset(test_dataset, transform=test_transform)     # type: ignore

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_dataloader, test_dataloader
