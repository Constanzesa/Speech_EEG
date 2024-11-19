import numpy as np
from pathlib import Path
from typing import List, Literal
import torch
import os

# class Dataset_Large:
#     def __init__(self, data_dir: Path, label: Literal["labels"], special: str = None):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.data = []
#         self.labels = []

#         # Load all session data
#         trials = sorted(data_dir.glob("S*"))
#         print(f"Found sessions: {len(trials)}")

#         for file_path in trials:
#             data_path = file_path / "data.npy"
#             label_path = file_path / "labels.npy"

#             # Load session data and labels
#             data_sessions = np.load(data_path, allow_pickle=True)
#             label_sessions = np.load(label_path, allow_pickle=True)

#             # Permute data to [trials, n_channels, time_samples]
#             data_sessions = data_sessions.transpose(0, 2, 1)  # [trials, time_samples, n_channels] -> [trials, n_channels, time_samples]

#             self.data.append(torch.from_numpy(data_sessions).float().to(self.device))
#             self.labels.append(torch.from_numpy(label_sessions).long().to(self.device))

#         print(f"Loaded {len(self.data)} sessions with corresponding labels.")

#     def __len__(self):
#         # Total number of trials across all sessions
#         return sum(len(session_data) for session_data in self.data)

#     def __getitem__(self, idx):
#         # Determine which session the index belongs to
#         session_idx, trial_idx = self._get_session_and_trial_idx(idx)
#         data = self.data[session_idx][trial_idx].unsqueeze(0)  # Add singleton in_channels dimension: [1, n_channels, time_samples]
#         label = self.labels[session_idx][trial_idx]
#         return data, label

#     def _get_session_and_trial_idx(self, idx):
#         cumulative_idx = 0
#         for session_idx, session_data in enumerate(self.data):
#             if idx < cumulative_idx + len(session_data):
#                 trial_idx = idx - cumulative_idx
#                 return session_idx, trial_idx
#             cumulative_idx += len(session_data)
#         raise IndexError("Index out of range")





# class Dataset_Large:
#     def __init__(self, data_dir: Path, label: Literal["labels"], special: str = None, min_samples: int = 64):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.data = []
#         self.labels = []
#         self.min_samples = min_samples  # Minimum required time samples


#         # Load all session data
#         trials = sorted(data_dir.glob("S*"))
#         print(f"Found sessions: {len(trials)}")

#         for file_path in trials:
#             data_path = file_path / "data.npy"
#             label_path = file_path / "labels.npy"

#             # Load session data and labels
#             data_sessions = np.load(data_path, allow_pickle=True)
#             label_sessions = np.load(label_path, allow_pickle=True)

#             # Adjust the time dimension to match the model's expected input window size
#             data_sessions = self._ensure_min_samples(data_sessions)

#             self.data.append(torch.from_numpy(data_sessions).float().to(self.device))
#             self.labels.append(torch.from_numpy(label_sessions).long().to(self.device))

#         print(f"Loaded {len(self.data)} sessions with corresponding labels.")

#     def _ensure_min_samples(self, data):
#         target_samples = 64  # Match the model's expected input window size
#         if data.shape[2] < target_samples:
#             padding = target_samples - data.shape[2]
#             data = np.pad(data, ((0, 0), (0, 0), (0, padding)), mode="constant")
#         elif data.shape[2] > target_samples:
#             data = data[:, :, :target_samples]
#         return data

#     def __len__(self):
#         return sum(len(session_data) for session_data in self.data)
    
#     def __getitem__(self, idx):
#         session_idx, trial_idx = self._get_session_and_trial_idx(idx)
#         data = self.data[session_idx][trial_idx]  # Original shape: [n_channels, time_samples]
#         label = self.labels[session_idx][trial_idx]

#         print(f"Shape of data before reshaping: {data.shape}")

#         # Reshape to match model input: [1, height=n_channels, width=time_samples]
#         data = data.unsqueeze(0)  # Add channel dimension: [1, n_channels, time_samples]
#         data = data.permute(0, 2, 1)  # Swap dimensions to [1, height=n_channels, width=time_samples]

#         print(f"Shape of data after reshaping: {data.shape}")

#         return data, label


class Dataset_Small:
    def __init__(self, data_dir: Path, label: Literal["labels"], train: bool = True):
        self.label_names = "labels" 
        self.data = []
        self.labels = []
        trials = sorted(data_dir.glob("S*"))
        print(f"Found sessions: {len(trials)}")

        for file_path in trials:
            data_path = file_path / "data.npy"
            _labels_path = file_path / "labels.npy"  # Path to the labels file
            _labels = np.load(_labels_path, allow_pickle=True)  # Load the labels into a NumPy array
            
            if train:
                selection = np.concatenate([np.argwhere(_labels == label_id).flatten()[:-5] for label_id in np.unique(_labels)])
            else:
                selection = np.concatenate([np.argwhere(_labels == label_id).flatten()[-5:] for label_id in np.unique(_labels)])
            
            # Append each session's data and labels as separate arrays
            
            # Load and filter data
            selected_data = np.load(data_path, allow_pickle=True)[selection]
            selected_labels = _labels[selection]
            
            # Append each session's data and labels to the list
            self.data.append(selected_data)
            self.labels.append(selected_labels)
        self.max_trial_length = max([self.data[d].shape[2] for d in range(len(self.data))])
        for d in range(len(self.data)):
            # Pad each trial to the maximum length
            padding = self.max_trial_length - self.data[d].shape[2]
            self.data[d] = np.pad(self.data[d], ((0, 0), (0, 0), (0, padding)), mode="constant")

        # Concatenate the list of NumPy arrays into single arrays
        self.data = np.concatenate(self.data, axis=0)  # Combine trials into one array
        self.labels = np.concatenate(self.labels, axis=0)  # Combine labels into one array
        
        # # Convert labels to torch tensors
        # self.labels = torch.tensor(self.labels, dtype=torch.long)

        # for file_path in trials:
        #     data_path = file_path / "data.npy"
        #     _labels_path = file_path / "labels.npy"  # Path to the labels file
        #     _labels = np.load(_labels_path, allow_pickle=True)  # Load the labels into a NumPy array

        #     if train:
        #         # Select all but the last 5 samples for each label
        #         selection = np.concatenate([np.argwhere(_labels == label_id).flatten()[:-5] for label_id in np.unique(_labels)])
        #     else:
        #         # Select the last 5 samples for each label
        #         selection = np.concatenate([np.argwhere(_labels == label_id).flatten()[-5:] for label_id in np.unique(_labels)])

        #     # Append the selected data and labels
        #     self.data.append(np.load(data_path, allow_pickle=True)[selection])
        #     self.labels.append(_labels[selection])

       
        # self.data = np.concatenate(self.data)
        # self.labels = np.concatenate(self.labels)

        self.data = torch.from_numpy(self.data).float() #swap axes to get (n_trials, channels, samples) 
        self.labels = torch.from_numpy(self.labels).long() #turn into one-hot encoding torch.nn.functional.one_hot( , num_classes = -1)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

   

class Dataset_Large:
    def __init__(
                self, 
                data_dir: Path, 
                label: Literal["labels"], 
                train: bool = True, 
                val_run: str = None,
                special: str = None
                ):
        
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.label_names = "labels" 
            self.data = []
            self.labels = []

            # load all sessions: 
            trials = sorted(data_dir.glob("S*"))
            print(f"Found sessions: {len(trials)}")

            for file_path in trials:
                data_path = file_path / "data.npy"
                label_path = file_path / "labels.npy"

                # Load session data and labels
                data_sessions = np.load(data_path, allow_pickle=True)
                label_sessions = np.load(label_path, allow_pickle=True)

                self.data.append(data_sessions)
                self.labels.append(label_sessions)

                self.labels = np.concatenate(self.labels)
                self.data = np.concatenate(self.data)

                self.data = torch.from_numpy(self.data.swapaxes(1,2)).to(self.device) #swap axes to get (n_trials, channels, samples) 
                self.labels = torch.from_numpy(self.labels).long().to(self.device) #turn into one-hot encoding torch.nn.functional.one_hot( , num_classes = -1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]