import numpy as np
from pathlib import Path
from typing import List, Literal
import torch
import os


"DATA SET Read in for our data-session:"

class Dataset_Small:
    def __init__(self, data_dir: Path, label: Literal["labels"], train: bool = True):
        self.label_names = "labels" 
        self.data = []
        self.labels = []

        trials = sorted(data_dir.glob("S*"))

        print(f"Found sessions: {len(trials)}")

        for file_path in trials:
            data_path = file_path / "data.npy"
            _labels_path = file_path / "labels.npy" 
            _labels = np.load(_labels_path, allow_pickle=True)  
                
            if train:
                selection = np.concatenate([np.argwhere(_labels == label_id).flatten()[:-5] for label_id in np.unique(_labels)])
            else:
                selection = np.concatenate([np.argwhere(_labels == label_id).flatten()[-5:] for label_id in np.unique(_labels)])
            selected_data = np.load(data_path, allow_pickle=True)[selection]
            selected_labels = _labels[selection]

            max_length = 7800
            # Adjust the length of each trial
            adjusted_data = []
            for trial in selected_data:
                if trial.shape[1] > max_length:
                    trial = trial[:, :max_length]  # Truncate to max_length
                elif trial.shape[1] < max_length:
                    padding = max_length - trial.shape[1]
                    trial = np.pad(trial, ((0, 0), (0, padding)), mode="constant")  # Pad to max_length
                adjusted_data.append(trial)
            adjusted_data = np.array(adjusted_data)

            # # Append each session's adjusted data and labels to the list
            self.data.append(adjusted_data)
            self.labels.append(selected_labels)
            # # selected_data = np.load(data_path, allow_pickle=True)
            # # selected_labels = _labels
            
            # # Append each session's data and labels to the list
            # self.data.append(selected_data)
            # self.labels.append(selected_labels)
            # self.max_trial_length = max([self.data[d].shape[2] for d in range(len(self.data))])
            # print("MAX TRIAL LENGTH", self.max_trial_length)
            # for d in range(len(self.data)):
            #     # Pad each trial to the maximum length
            #     padding = self.max_trial_length - self.data[d].shape[2]
            #     self.data[d] = np.pad(self.data[d], ((0, 0), (0, 0), (0, padding)), mode="constant")

        self.data = np.concatenate(self.data, axis=0)  
        self.labels = np.concatenate(self.labels, axis=0) 
        print("DATA SHAPE", self.data.shape)
        print("Labels SHAPE",self.labels.shape)
        self.data = torch.from_numpy(self.data).float() #swap axes to get (n_trials, channels, samples) 
        self.labels = torch.from_numpy(self.labels).long() #turn into one-hot encoding torch.nn.functional.one_hot( , num_classes = -1)

    def __len__(self):
        return len(self.labels)
    
    # def __getitem__(self, idx):
    #     return self.data[idx], self.labels[idx]    

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        length = torch.tensor(data.shape[1])  # Length based on the number of time samples
        return data, label, length


# class Dataset_Small:
#     def __init__(self, data_dir: Path, label: Literal["labels"], train: bool = True, fixed_length: int = None):
#         self.data = []
#         self.labels = []

#         # List all trials (folders with S* pattern)
#         trials = sorted(data_dir.glob("S*"))
#         print(f"Found sessions: {len(trials)}")

#         for file_path in trials:
#             data_path = file_path / "data.npy"
#             labels_path = file_path / "labels.npy"

#             # Load data and labels
#             data = np.load(data_path, allow_pickle=True)
#             labels = np.load(labels_path, allow_pickle=True)
#             assert len(data) == len(labels), "Mismatch between data and labels!"

#             # Select samples based on train/validation split
#             if train:
#                 selection = np.concatenate([np.argwhere(labels == label_id).flatten()[:-5] for label_id in np.unique(labels)])
#             else:
#                 selection = np.concatenate([np.argwhere(labels == label_id).flatten()[-5:] for label_id in np.unique(labels)])

#             selected_data = data[selection]
#             selected_labels = labels[selection]

#             # Append data and labels
#             self.data.append(selected_data)
#             self.labels.append(selected_labels)

#         # Pad or truncate trials
#         self.data = np.concatenate(self.data, axis=0)  # Combine all sessions
#         self.labels = np.concatenate(self.labels, axis=0)

#         if fixed_length:
#             self.data = np.array([
#                 np.pad(trial, ((0, 0), (0, max(0, fixed_length - trial.shape[1]))), mode="constant")[:, :fixed_length]
#                 for trial in self.data
#             ])

#         print(f"Data shape after processing: {self.data.shape}")

#         # Convert to PyTorch tensors
#         self.data = torch.from_numpy(self.data).float()  # Shape: (n_trials, channels, samples)
#         self.labels = torch.from_numpy(self.labels).long()

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         data = self.data[idx]
#         label = self.labels[idx]
#         length = torch.tensor(data.shape[1])  # Length based on the number of time samples
#         return data, label, length
