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
            
            # # Append each session's data and labels to the list
            # self.data.append(selected_data)
            # self.labels.append(selected_labels)
            # self.max_trial_length = max([self.data[d].shape[2] for d in range(len(self.data))])
            # print("MAX TRIAL LENGTH", self.max_trial_length)
            # for d in range(len(self.data)):
            #     # Pad each trial to the maximum length
            #     padding = self.max_trial_length - self.data[d].shape[2]
            #     self.data[d] = np.pad(self.data[d], ((0, 0), (0, 0), (0, padding)), mode="constant")
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

        self.data = np.concatenate(self.data, axis=0)  
        print("COMBINED DATA", self.data.shape)
        self.labels = np.concatenate(self.labels, axis=0) 
        print("COMBINED LABELS", self.lables.shape)

        self.data = torch.from_numpy(self.data).float() #swap axes to get (n_trials, channels, samples) 
        self.labels = torch.from_numpy(self.labels).long() #turn into one-hot encoding torch.nn.functional.one_hot( , num_classes = -1)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

   

# class Dataset_Large:
#     def __init__(
#                 self, 
#                 data_dir: Path, 
#                 label: Literal["labels"], 
#                 train: bool = True, 
#                 val_run: str = None,
#                 special: str = None
#                 ):
        
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             self.label_names = "labels" 
#             self.data = []
#             self.labels = []

#             # load all sessions: 
#             trials = sorted(data_dir.glob("S*"))
#             print(f"Found sessions: {len(trials)}")
#             print(f"Sessions found; {trials}")

#             for file_path in trials:
#                 data_path = file_path / "data.npy"
#                 label_path = file_path / "labels.npy"

#                 # Load session data and labels
#                 data_sessions = np.load(data_path, allow_pickle=True)
#                 label_sessions = np.load(label_path, allow_pickle=True)

#                 self.data.append(data_sessions)
#                 self.labels.append(label_sessions)

#                 self.labels = np.concatenate(self.labels)
#                 self.data = np.concatenate(self.data)

#                 self.data = torch.from_numpy(self.data.swapaxes(1,2)).to(self.device) #swap axes to get (n_trials, channels, samples) 
#                 self.labels = torch.from_numpy(self.labels).long().to(self.device) #turn into one-hot encoding torch.nn.functional.one_hot( , num_classes = -1)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.data[idx], self.labels[idx]


"DATASET Readin for THINGS_EEG DATA: (right now for 5 classes)"

# class Dataset_Large:
#     def __init__(
#                 self, 
#                 data_dir: Path, 
#                 label: Literal["labels"], 
#                 train: bool = True, 
#                 val_run: str = None,
#                 special: str = None
#                 ):
        
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             self.label_names = "labels" 
#             self.data = []
#             self.labels = []

#             # load all sessions: 
#             # trials = sorted(data_dir.glob("S*"))
#             # print(f"Found sessions: {len(trials)}")
#             # print(f"Sessions found; {trials}")

#             # for file_path in trials:

#             data_path = data_dir / "data.npy"
#             label_path = data_dir / "labels.npy"

#             # Load session data and labels
#             data_sessions = np.load(data_path, allow_pickle=True)
#             label_sessions = np.load(label_path, allow_pickle=True)

#             self.data.append(data_sessions)
#             self.labels.append(label_sessions)
#              # Concatenate the list into a single NumPy array
#             self.data = np.concatenate(self.data, axis=0)  # Concatenate along the first axis
#             self.labels = np.concatenate(self.labels, axis=0)
#             # self.labels = np.concatenate(self.labels)
#             # self.data = np.concatenate(self.data)

#             self.data = torch.from_numpy(self.data).to(self.device) #swap axes to get (n_trials, channels, samples) 
#             self.labels = torch.from_numpy(self.labels).long().to(self.device) #turn into one-hot encoding torch.nn.functional.one_hot( , num_classes = -1)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.data[idx], self.labels[idx]\
        

"DATASET Readin for 2 Class Classification:"

# class Dataset_Large:
#     def __init__(
#         self,
#         data_dir: Path,
#         label: Literal["labels"],
#         train: bool = True,
#         binary_classes: List[int] = None,  # Specify two classes for binary classification
#         val_run: str = None,
#         special: str = None
#     ):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.label_names = "labels"
#         self.data = []
#         self.labels = []
#         self.binary_classes = binary_classes  # New attribute for binary classification
        
#         trials = sorted(data_dir.glob("S*"))[:5]

#         print(f"Found sessions: {len(trials)}")

#         for file_path in trials:
#             data_path = file_path / "data.npy"
#             _labels_path = file_path / "labels.npy"
#             _labels = np.load(_labels_path, allow_pickle=True)  
            
#             # Apply binary classification mapping
#             if self.binary_classes:
#                 assert len(self.binary_classes) == 2, "binary_classes must contain exactly two class labels."
#                 _labels = self._map_to_binary_classes(_labels)
            
#             # # Select training or validation data
#             # if train:
#             #     selection = np.concatenate([np.argwhere(_labels == label_id).flatten()[:-5] for label_id in np.unique(_labels)])
#             # else:
#             #     selection = np.concatenate([np.argwhere(_labels == label_id).flatten()[-5:] for label_id in np.unique(_labels)])
            
#             # selected_data = np.load(data_path, allow_pickle=True)[selection]
#             # selected_labels = _labels[selection]
#             if train: 
#                 selected_data = np.load(data_path, allow_pickle=True)
#                 selected_labels = _labels

        
            
#             # Append each session's data and labels to the list
#             self.data.append(selected_data)
#             self.labels.append(selected_labels)

#         self.max_trial_length = max([self.data[d].shape[2] for d in range(len(self.data))])
#         print("MAX TRIAL LENGTH", self.max_trial_length)
#         for d in range(len(self.data)):
#             # Pad each trial to the maximum length
#             padding = self.max_trial_length - self.data[d].shape[2]
#             self.data[d] = np.pad(self.data[d], ((0, 0), (0, 0), (0, padding)), mode="constant")

#         self.data = np.concatenate(self.data, axis=0)
#         self.labels = np.concatenate(self.labels, axis=0)
#         print("DATA SHAPE", self.data.shape)
#         print("Labels SHAPE",self.labels.shape)

#         self.data = torch.from_numpy(self.data).float().to(self.device)  # Swap axes to get (n_trials, channels, samples)
#         self.labels = torch.from_numpy(self.labels).long().to(self.device)
        
#     def _map_to_binary_classes(self, labels):
#         # Map original labels to binary classes
#         binary_labels = np.zeros_like(labels)
#         binary_labels[np.isin(labels, self.binary_classes)] = 1  # Mark chosen classes as 1
#         return binary_labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.data[idx], self.labels[idx]
