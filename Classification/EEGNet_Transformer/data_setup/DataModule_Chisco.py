import numpy as np
import os
from pathlib import Path
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import List
from pathlib import Path

import random

from torch.utils.data import Subset
import numpy as np
import os
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from data_setup.Dataset_Chisco import Dataset_Small
import random
import torch
from torch.utils.data import DataLoader, Subset
import random

"TRIAL based Datamodule:"

class DataModule(pl.LightningDataModule):
    def __init__(
            self, 
            data_dir: str, 
            test_dir: str = None,
            val_run: str = None,
            test_run: str = None,
            batch_size: int = 16, 
            num_workers: int = 0, 
            seed: int = 42, 
            special = None,
            **kwargs):
        
        super().__init__()
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.val_run = val_run
        self.test_run = test_run
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.special = special #allows to use grayscale_images or fourier_spectrograms instead of data.npy
       
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset = Dataset_Small(Path(self.data_dir), label="group", train=True)
            
            # Perform stratified random split
            train_idx, val_idx = self._stratified_random_split(self.dataset, split=[0.8, 0.2], seed=self.seed) 
            
            # Use Subset to create train and validation datasets
            self.train_dataset = Subset(self.dataset, train_idx)
            self.val_dataset = Subset(self.dataset, val_idx)

            print(f"Training samples: {len(self.train_dataset)}, Validation samples: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            if self.test_dir:
                self.test_dataset = Dataset_Small(Path(self.test_dir), label="group", train=False)
            else:
                pass


    def _stratified_random_split(self, dataset, split: List = [0.8, 0.2], seed: int = None):
        #Splits a dataset into train and validation set while preserving the class distribution.
        np.random.seed(seed) if seed else None
        train_idx = []
        val_idx = []
        labels = dataset.labels.cpu().numpy() ## CHANGED With cpu()
        print("jgvvgv", labels.shape)
        for label in np.unique(labels):
            label_loc = np.argwhere(labels == label).flatten()
            np.random.shuffle(label_loc)
            n_train = int(split[0]*len(label_loc))
            train_idx.append(label_loc[:n_train])
            val_idx.append(label_loc[n_train:])
        train_idx = np.concatenate(train_idx)
        val_idx = np.concatenate(val_idx)
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        return train_idx, val_idx
    
    def train_dataloader(self):
        print(f"Batch size: {self.batch_size}")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle = True)# sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle = False) #sampler=self.val_sampler)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        print("Not implemented yet")
        pass

