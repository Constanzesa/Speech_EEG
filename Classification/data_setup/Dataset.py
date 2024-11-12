import numpy as np
from pathlib import Path
from typing import List, Literal
import torch
import os


class Dataset_Large():
    """
    Pytorch Dataset Large

    This expects multiple recordings and selects one recording as validation and the rest as training data.

    Args:
        data_dir: Path to directory containing the data.
            Expects a directory with multiple run-directories. Concatenates all npy files into one dataset.
        label: Whether to use group labels or labels.
        train: Whether to use the training or test set.
            if True, n-1 runs are returned as dataset
            if False (test), only the hold-out set is loaded for validation.
    """
    def __init__(
            self, 
            data_dir: Path, 
            # label_dir:Path,
            label: Literal["label"], 
            train: bool = True, 
            val_run: str = None,
            special: str = None
            ):
        # if label not in ["group", "label"]:
        #     raise ValueError("option must be either 'group' or 'label'")
                    
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_names = "labels" #if label == "group" else "labels"
     
        # Load data and labela
        session_name = "S01"    
        data_dir = "/Users/arnavkapur/Desktop/EEG_Speech/DATA/PREPROCESSED"
        base_path = os.path.join(data_dir, session_name,'sub')
        
        if train: 
            data_dir = os.path.join(base_path, train, "data.npy")
            label_dir = os.path.join(base_path, train, "labels.npy")
        else:
            data_dir = os.path.join(base_path, 'val', "data.npy")   
            label_dir = os.path.join(base_path,'val', "labels.npy")
        
        self.data = np.load(data_dir, allow_pickle=True)  # Shape: (trials, channels, samples)
        self.labels = np.load(label_dir, allow_pickle=True)  # Shape: (trials,)
        
        # Convert to PyTorch tensors
        # self.data = torch.from_numpy(self.data).type(torch.float32).to(self.device)
        # self.labels = torch.from_numpy(self.labels).type(torch.LongTensor).to(self.device)  # Convert labels to long tensor
        # self.labels = self.labels - 1  # Subtract 1 to make labels 0-indexed

        if special is None:
            self.data = torch.from_numpy(self.data).to(self.device)
        self.labels = torch.from_numpy(self.labels).long().to(self.device) 

        print(f"Data shape: {self.data.shape}, Labels shape: {self.labels.shape}")
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    