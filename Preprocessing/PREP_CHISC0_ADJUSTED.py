"Chisco_prp"

# !pip install mne
# !pip install pyxdf
# !pip install PyWavelets
# !pip install pandas 
# !pip install pyprep
# !pip install numpy==1.21.6

from tqdm import tqdm
import pyxdf
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from copy import deepcopy
from matplotlib import pyplot
import matplotlib.pyplot as plt 
from mne.decoding import Scaler
import copy
import pathlib
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm.auto import tqdm
# import plotly.graph_objects as go
# import plotly.subplots as sp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import argparse
from utils_clean import show_streams, find_stream, get_duration, get_time_series, set_channel_names, plot_channel_correlation,detect_bad_channels,plot_bads, ica_analysis,remove_breaks,build_class_epochs_mne,create_dataset,return_dataset,plot_topo,plot_eeg,plot_evoked,standartization

##ADDED:
import os
import mne
import pickle
import pyprep
from pyprep.find_noisy_channels import NoisyChannels
from pyprep.prep_pipeline import PrepPipeline
import pandas as pd 
import argparse

import argparse

# session_name = "S01"
# base_path = "/Users/arnavkapur/Desktop/EEG_Speech"
# data_path = os.path.join(base_path, "DATA","RAW")

# xdf_file_path = os.path.join(data_path, f"{session_name}.xdf")
xdf_file_path = "C:\\Users\\msi\\Desktop\\Constanze\\Docs\\DATA\\RAW\\S01.xdf"
# Load the .xdf file
data, header = pyxdf.load_xdf(xdf_file_path)
print(f"Successfully loaded data from {xdf_file_path}")

csv_file = "C:\\Users\\msi\\Desktop\\Constanze\\Docs\\DATA\\marker\\S01.csv"
df = pd.read_csv(csv_file, header=None)

sample_rate = 500

show_streams(data)
eeg_stream = find_stream('eeg', data)
marker_stream = find_stream('marker', data)
df_marker = get_time_series(marker_stream)
sfreq = float(eeg_stream["info"]["nominal_srate"][0])

channels_info = eeg_stream['info']['desc'][0]['channels'][0]['channel']

# # Extract labels from each channel
ch_names = [channel['label'][0] for channel in channels_info]
ch_names = ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2', 'ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL']



"""Create MNE file raw containing the EEG Stream"""

eeg_data = eeg_stream["time_series"].T
eeg_data = eeg_data[:64]
print(eeg_data.shape)
sfreq = float(eeg_stream["info"]["nominal_srate"][0])

eeg_info = mne.create_info(64, sfreq, ["eeg"]*64)
raw = mne.io.RawArray(eeg_data, eeg_info)

# review information
ssp_projectors = raw.info["projs"]
raw.del_proj()
# Calculate the duration in seconds and convert to minutes
duration_minutes = (raw.n_times / raw.info['sfreq']) / 60

# Print the duration in minutes
print(f"Duration of the recording: {duration_minutes:.2f} minutes")
useless_channels = ['ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL']

#ADDED:
raw.resample(sample_rate)
raw.set_montage('standard_1020', on_missing='warn')   
set_channel_names(raw,ch_names)
raw.drop_channels(useless_channels)  
ch_names_new = ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2']



from pyprep.prep_pipeline import PrepPipeline

# Define preprocessing parameters
prep_params = {
    "ref_chs": "eeg",
    "reref_chs": "eeg",
    "line_freqs": np.arange(60, sample_rate / 2, 60),
}

# Assign a valid montage (standard 10-20 system)
montage = 'standard_1020'  # Use the name of the standard montage

RANSAC = False
print("Running pyprep")

prep = PrepPipeline(
    raw,
    prep_params,
    montage=montage,  # Pass the montage name
    ransac=RANSAC
)
prep.fit()
raw_new = prep.raw
print("Preprocessing completed.")
print("Still bad channels: ", raw_new.info['bads'])

# High-pass filter
raw_new.filter(l_freq=1, h_freq=None)




eeg_start = eeg_stream['time_stamps'][0]  # EEG start time, 679.891
marker_start = marker_stream['time_stamps'][0]  # Marker start time, 2311465.920826529

time_offset = eeg_stream['clock_times'][0] - data[1]['clock_times'][0]
aligned_marker_relative = [(ts + time_offset - marker_start + eeg_start) for ts in data[1]['time_stamps']]

aligned_pairs = []
for marker_time in aligned_marker_relative:
    closest_idx = np.argmin(np.abs(np.array(eeg_stream['time_stamps']) - marker_time))
    aligned_pairs.append((marker_time, eeg_stream['time_stamps'][closest_idx]))
df_marker = pd.DataFrame(marker_stream['time_series'], aligned_marker_relative,columns=['marker'])

# Extract the sentences column
sentences = df.iloc[2:, 0].values  # Ensure this is a numpy array or list for proper assignment
labels = df.iloc[2:, 1].values  # Ensure this is a numpy array or list for proper assignment

# Filter rows with marker == 1 and create a new column in df_marker
df_marker.loc[df_marker['marker'] == 1, 'sentences'] = sentences
df_marker.loc[df_marker['marker'] == 1, 'labels'] = labels
metadata = df_marker.loc[df_marker['marker'] == 1]
metadata = metadata.iloc[:-1]
metadata



sf = 500
eeg_time_series = raw_new.get_data()
eeg_time_series = eeg_time_series.T
eeg_timestamps = eeg_stream["time_stamps"]
event_time_series= marker_stream['time_series']
event_time_stamps= aligned_marker_relative

event_time_series_onset = df_marker[df_marker['marker'] == 1].reset_index()
event_time_series_offset = df_marker[df_marker['marker'] == 0].reset_index()
event_time_series_onset['numb'] = range(len(event_time_series_onset))
event_time_series_offset['numb'] = range(len(event_time_series_offset))
print("EVNETTIMESERIES", event_time_series_onset.shape)
print("EVENTTIMESERIESOFFSET", event_time_series_offset.shape)

eeg_closest_timestamps = []  

for event_time in event_time_stamps:
    closest_index = np.argmin(np.abs(eeg_timestamps - event_time))
    closest_timestamp = eeg_timestamps[closest_index]  
    eeg_closest_timestamps.append(closest_timestamp)


eeg_indeces = np.array(eeg_closest_timestamps)

labels = np.empty(eeg_timestamps.shape, dtype=int)
labels.fill(900) #Fill with 900 (break)
group_labels = np.copy(labels)
trials = np.copy(labels)

start = eeg_indeces[0]

eeg_data = eeg_time_series[:, :59]
out = pd.DataFrame(eeg_data, columns=ch_names_new)
print("OUT",out.shape)    


"Define the labels for the 5 sentences"

trial_label = df.iloc[2:-1, 1]
nan_count = trial_label.isna().sum()
trial_label = trial_label.dropna().astype(int)
print("Trial label shape", trial_label.shape)
print("Trial Labels:",trial_label)  

trial_label.index = range(len(trial_label))
event_time_series_offset['label'] = trial_label.astype(int)
event_time_series_offset = event_time_series_offset[:-1]
event_time_series_onset['label'] = trial_label.astype(int)  
event_time_series_onset = event_time_series_onset[:-1]


# Build the dataset
start_index = eeg_indeces[::2]
end_index = eeg_indeces[1::2]
start_index = start_index[:-1]
end_index = end_index[:-1]
print("START AND END", start_index.shape, end_index.shape, event_time_series_onset['label'].shape) #(461,) (461,) (461,)



durations = end_index - start_index
labels = np.full(len(eeg_timestamps), 900)

sf = sample_rate
for i in range(len(start_index)):
    start_idx = np.where(eeg_timestamps == start_index[i])[0]
    if len(start_idx) == 0:
        print(f"Start index {start_index[i]} not found in eeg_timestamps.")
        continue
    start_idx = start_idx[0]
        
    num_timestamps = int(durations[i] * sf)
    end_idx = min(start_idx + num_timestamps, len(eeg_timestamps))
    labels[start_idx:end_idx] = event_time_series_onset['label'][i]


# df = pd.DataFrame({'time': eeg_timestamps, 'label': labels})


dataset = []
df = pd.concat([pd.DataFrame({'time': eeg_timestamps, 'label': labels}), out], axis=1)
dataset.append(df)
dataset


def print_sample_distribution(labels):
    label_count = pd.Series(labels).value_counts().sort_index()

    print("Time sample distribution among different trials:")
    for label, count in label_count.items():
        print(f"Label {label}: {count} time samples")


label_count = df['label'].value_counts()
dataset[0]['label'].value_counts()

"Delete the breaks from the dataset"
#  using .query() to maintain the order of the columns removing th
dataset[0] = dataset[0].query("label != 900").reset_index(drop=True)

# using this to remove breaks without maining   the order of the columns
# dataset[0] = dataset[0].loc[dataset[0]['label'] != 900].reset_index(drop=True)
dataset[0]['label'].value_counts()

# Split the dataset based on the durations into trials
split_datasets = []
start_idx = 0
d =dataset[0]    

for duration in durations:
    end_idx = start_idx + int(duration * sf)
    split_df = d.iloc[start_idx:end_idx].copy()
    split_datasets.append(split_df)
    start_idx = end_idx

# # Verify the splits
# for i, split_df in enumerate(split_datasets):
#     print(f"Split {i}: {split_df.shape}")
# for dataset in split_datasets:
#     print(dataset['label'].value_counts())


d['time'].iloc[-1] - d['time'].iloc[0]

# Padding data to max timestamp
max_duration = 0
longest_array = None

for dataset in split_datasets:
    duration = d['time'].iloc[-1] - d['time'].iloc[0]
    if duration > max_duration:
        max_duration = duration
        longest_array = dataset

print("MAX Duration", max_duration)

max_length = max([dataset.shape[0] for dataset in split_datasets])

padded_dataset = []
for dataset in split_datasets:
    num_samples_to_pad = max_length - dataset.shape[0]
    
    if num_samples_to_pad > 0:
        label = d['label'].iloc[0]  
        padding_df = pd.DataFrame(0, index=range(num_samples_to_pad), columns=dataset.columns)
        padding_df['label'] = label 
        padded_dataset.append(pd.concat([dataset, padding_df], ignore_index=True))
    else:
        padded_dataset.append(dataset)

dataset_lengths = [dataset.shape[0] for dataset in padded_dataset]
print(f"Lengths after padding: {set(dataset_lengths)}")  
# for dataset in padded_dataset:
#     print(dataset['label'].value_counts())

dataset_lengths = [dataset.shape[0] for dataset in padded_dataset]

min_length = min(dataset_lengths)
max_length = max(dataset_lengths)

print(f"Minimum length: {min_length}")
print(f"Maximum length: {max_length}")


sfreq = sf
# Convert aligned marker timestamps to sample indices relative to the EEG data
marker_sample_indices = [
    int((marker_time - eeg_stream['time_stamps'][0]) * sfreq) 
    for marker_time in aligned_marker_relative
]

# Create MNE-compatible events array
# Assuming `df_marker` contains a 'marker' column with the event codes
events = np.array([
    [sample_idx, 0, int(marker)] 
    for sample_idx, marker in zip(marker_sample_indices, df_marker['marker'])
])

# Filter out invalid events (e.g., negative sample indices)
events = events[events[:, 0] > 0]

# Add the events to the Raw object
print(f"Created {len(events)} events")
mne.viz.plot_events(events, sfreq=sfreq)

events = events[events[:, 2] == 1]


tmax = max_duration/sample_rate +0.1
epochs = mne.Epochs(
    raw, 
    events, 
    event_id=None, 
    tmin=0, 
    tmax= tmax, 
    metadata=metadata, 
    baseline=None,  # No baseline correction
    preload=True, 
    reject= None, 
    reject_by_annotation = False, 
    on_missing='ignore'  # Allow shorter epochs
)

print(f"Metadata shape: {epochs.metadata.shape}")


import os
import pickle

# File path to save the pickle file
file_path = "C:\\Users\\msi\\Desktop\\Constanze\\Docs\\DATA\\PREPROCESSED\\PREP_CH\\data.pkl"

# # Drop bad epochs if they haven't been dropped yet
# if not epochs._bad_dropped:
#     epochs.drop_bad()

# Prepare the list to save data
data_to_save = []
for epoch_idx in range(len(epochs)):
    # Extract sentence (or None if metadata is unavailable)
    sentence = (
        epochs.metadata.iloc[epoch_idx]['sentences'] 
        if epochs.metadata is not None else None
    )

    # Extract epoch data without copying
    data = epochs[epoch_idx].get_data(copy=True)
    print(f"Epoch {epoch_idx} data shape: {data.shape}")

    # Append data and metadata to the list
    data_to_save.append({"text": sentence, "input_features": data})

# Open the file in binary write mode and save the data
with open(file_path, 'wb') as file:
    pickle.dump(data_to_save, file)

print(f"Data successfully saved to {file_path}")
