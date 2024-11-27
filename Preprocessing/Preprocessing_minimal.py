"short preprocessing"

# !pip install mne
# !pip install pyxdf
# !pip install PyWavelets
# !pip install pandas 
# !pip install seaborn

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
# from preprocessing_1 import Preprocessing

# # Parse command-line arguments
# parser = argparse.ArgumentParser(description="Process EEG session data.")
# parser.add_argument("session_name", type=str, help="Name of the session (e.g., S04)")
# args = parser.parse_args()

# Parse command-line arguments
# Argument Parsing
parser = argparse.ArgumentParser(description="Process EEG session data.")
parser.add_argument("session_name", type=str, help="Name of the session (e.g., S01)")
args = parser.parse_args()



# Input parameters
session_name = args.session_name
# bad_visual = args.bad_visual

# Use the session_name from the command-line argument
session_name = args.session_name

base_path = "/Users/arnavkapur/Desktop/EEG_Speech"
data_path = os.path.join(base_path, "DATA","RAW")

xdf_file_path = os.path.join(data_path, f"{session_name}.xdf")

# Load the .xdf file
data, header = pyxdf.load_xdf(xdf_file_path)
# print(f"Successfully loaded data from {xdf_file_path}")

mark_path = ("/Users/arnavkapur/Desktop/EEG_Speech/DATA/marker/")
mark_session = os.path.join(mark_path, f"{session_name}.csv")
mark = pd.read_csv(mark_session)


name = session_name
show_streams(data)
eeg_stream = find_stream('eeg', data)
marker_stream = find_stream('marker', data)
df_marker = get_time_series(marker_stream)
sfreq = float(eeg_stream["info"]["nominal_srate"][0])

## Identify channel position for Neurable headset

# channels_info = data[3]['info']['desc'][0]['channels'][0]['channel']

# # Extract labels from each channel
# ch_names = [channel['label'][0] for channel in channels_info]
# # print("Channel names", ch_names)
ch_names = ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2', 'ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL']



"""Create MNE file raw containing the EEG Stream"""

eeg_data = eeg_stream["time_series"].T
eeg_data = eeg_data[:64]
# print(eeg_data.shape)
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


"""Filtering the EEG data"""

raw_highpass = raw.copy().filter(l_freq=1, h_freq=None)
raw_lowpass = raw_highpass.filter(l_freq=None, h_freq=100)
raw_notch = raw_lowpass.notch_filter(freqs=[60, 120,180])
raw_filtered = raw_notch.copy()

print("RAW Shape", raw_filtered.get_data().shape) 



# Plot Raw EEG Data
set_channel_names(raw,ch_names)
raw.plot(start=0, n_channels=64, scalings = {"eeg": 50})
set_channel_names(raw,ch_names)
set_channel_names(raw_filtered,ch_names)

raw_filtered.set_montage('standard_1020', on_missing='warn')

selected_channels = raw.copy().pick_channels(raw.ch_names[60:62])
selected_channels = raw_filtered.copy().pick_channels(raw.ch_names[50:59])
channel_correlation = plot_channel_correlation(raw_filtered.get_data(), ch_names)
bad_channels = channel_correlation[1]
bad_channels_2 = detect_bad_channels(raw_filtered.get_data(), ch_names)

# # Input for bad_visual
# print("Please visually inspect the raw plot and provide additional bad channels (comma-separated):")
# bad_visual_input = input("Enter channels (e.g., 'T8, C6, P7'): ").strip()
# bad_visual = [ch.strip() for ch in bad_visual_input.split(",") if ch.strip()]

# # Combine all bad channels

# bad_channels.extend(bad_visual)
bads = ['ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL']
bad_channels.extend(bads)
bad_channels.extend(bad_channels_2)

raw_removed = raw_filtered.copy()

raw_removed.info['bads'] = bad_channels
# raw_removed.info['bads'].extend(['VEOU', 'VEOL'])

#raw_removed.set_montage('standard_1020', on_missing='warn')
print("bads", raw_removed.info['bads'])

# montage = raw_removed.get_montage()

# if montage is not None:
#     # Get the channel positions from the montage
#     positions = montage.get_positions()['ch_pos']

#     # Adjust positions for overlapping channels
#     if 'VEOU' in positions and 'VEOL' in positions:
#         positions['VEOU'][0] += 0.01  # Adjust x-coordinate for VEOU
#         positions['VEOL'][0] -= 0.01  # Adjust x-coordinate for VEOL

#         # Create a new montage with the adjusted positions
#         new_montage = mne.channels.make_dig_montage(
#             ch_pos=positions,
#             coord_frame='head'
#         )

#         # Apply the modified montage to the raw data
#         raw_removed.set_montage(new_montage)

set_channel_names(raw_removed, ch_names)

print("START WITH DATA")

# Retrieve and print necessary values
# eeg_start = data[3]['time_stamps'][0]  # EEG start time
eeg_start = eeg_stream['time_stamps'][0]  # EEG start time

print("EEG START", eeg_start)
# marker_start = data[1]['time_stamps'][0]  # Marker start time
marker_start = marker_stream['time_stamps'][0]  # Marker start time

time_offset = eeg_stream['clock_times'][0] - marker_stream['clock_times'][0]
print("TIME OFFSET", time_offset)

# Use NumPy arrays for efficiency, avoid copying unnecessarily
marker_times = np.asarray(marker_stream['time_stamps'])  # Convert to NumPy array directly
eeg_time = np.asarray(eeg_stream['time_stamps'])      # Convert to NumPy array directly

# Precompute aligned marker times
aligned_marker_relative = marker_times + time_offset - marker_start + eeg_start

# Use a more memory-efficient method for finding closest indices
# Instead of creating the full difference matrix, iterate over aligned_marker_relative
closest_indices = np.empty(len(aligned_marker_relative), dtype=np.int64)

for i, marker_time in enumerate(aligned_marker_relative):
    closest_indices[i] = np.abs(eeg_time - marker_time).argmin()

# Generate aligned pairs (optional, use generator if possible to avoid memory cost)
aligned_pairs = np.stack((aligned_marker_relative, eeg_time[closest_indices]), axis=1)

print("END ALIGNMENT")

df_marker = pd.DataFrame(marker_stream['time_series'], aligned_marker_relative,columns=['marker'])

print(df_marker.head())
print("MARKEr", df_marker.shape)
eeg_time_series = raw_removed.get_data()
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

eeg_data = eeg_time_series[:, :64]

out = pd.DataFrame(eeg_data, columns=ch_names)
print("OUT",out.shape)    


"Define the labels for the 5 sentences"

trial_label = mark['labels']
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



sf = raw.info['sfreq']

durations = end_index - start_index
labels = np.full(len(eeg_timestamps), 900)


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



# Epoching Data into (n_trials, n_channels, n_samples)
epoch =[]
labels = []
eeg_data_list = []

for dataset in padded_dataset:
    labels.append(dataset['label'].iloc[0]) 
    eeg_data = dataset.drop(columns=['time', 'label']).values.T
    eeg_data_list.append(eeg_data)

labels = np.array(labels)
epoch = [labels] + eeg_data_list

eeg_data_stacked = np.stack(eeg_data_list)
epoch[1:] = [eeg_data_stacked]  

print("Labels shape:", epoch[0].shape) 
print("EEG data shape:", epoch[1].shape)  

eeg_data = epoch[1]
labels = epoch[0]

import os

print("Saving preprocessed data...")

data_path = os.path.join(base_path, "DATA")
folder_path = os.path.join(data_path, 'PREPROCESSED','PREP_MINIMAL', name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

eeg_path = os.path.join(folder_path, 'data.npy')
np.save(eeg_path, eeg_data)
label_path = os.path.join(folder_path, 'labels.npy')
np.save(label_path,labels)
