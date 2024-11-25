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

##ADDED:
import os
import mne
import numpy as np
import pickle
import pyprep
from pyprep.find_noisy_channels import NoisyChannels
from pyprep.prep_pipeline import PrepPipeline
# from mne_icalabel import label_components
import pandas as pd 
import argparse

import argparse



# If you DO NOT use any of the following arguments, the preprocessing will run with default settings.
# usage: preprocessing.py [-h] [--id ID] [--test] [--not_prep] [--not_reject] [--count_limit COUNT_LIMIT] [--method_str METHOD_STR] [--ica] [--ransac] [--step]
# options:
#   -h, --help            show this help message and exit
#   --id ID, -i ID        id attribute, default is A
#   --test, -t            Run test, only process part of the code
#   --not_prep, -p        Do not run prep
#   --not_reject, -a      Do not perform bad channel rejection
#   --count_limit COUNT_LIMIT, -c COUNT_LIMIT
#                         Maximum number of files to process before stopping, default is 10086
#   --method_str METHOD_STR, -m METHOD_STR
#                         Identifier for distinguishing different processing settings, default is undefined
#   --ica, -I             Run ICA
#   --ransac, -R          Use RANSAC in PREP
#   --step, -s            Save processed data after each step

parser = argparse.ArgumentParser(description='Options for preprocessing', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--session_name", type=str, help="Name of the session (e.g., S01)")

parser.add_argument('--id', '-i', help='id attribute, default is A', default='A')
parser.add_argument('--test', '-t', help='Run test, only process part of the code', action='store_true')
parser.add_argument('--not_prep', '-p', help='Do not run prep', action='store_false')
parser.add_argument('--not_reject', '-a', help='Do not perform bad channel rejection', action='store_false')
parser.add_argument('--count_limit', '-c', help='Maximum number of files to process before stopping, default is 10086', default=10086, type=int)
parser.add_argument('--method_str', '-m', help='Identifier for distinguishing different processing settings, default is undefined', default='undefined')
parser.add_argument('--ica', '-I', help='Run ICA', action='store_true')
parser.add_argument('--ransac', '-R', help='Use RANSAC in PREP', action='store_true')
parser.add_argument('--step', '-s', help='Save processed data after each step', action='store_true')
args = parser.parse_args()
print(args)



session_name = args.session_name
base_path = "C:\\Users\\msi\\Desktop\\Constanze\\Docs\\"
data_path = os.path.join(base_path, "DATA","RAW")
xdf_file_path = os.path.join(data_path, f"{session_name}.xdf")

data, header = pyxdf.load_xdf(xdf_file_path)

mark_path = ("C:\\Users\\msi\\Desktop\\Constanze\\Docs\\DATA\\marker")
mark_session = os.path.join(mark_path, f"{session_name}.csv")
mark = pd.read_csv(mark_session)


IC_NUM = 30
PATCH = False 


subject_id = args.id 
PREP = args.not_prep
AUTO_REJECT = args.not_reject 
method_str = args.method_str 
ICA = args.ica 
RANSAC = args.ransac 
count_limit = args.count_limit 
TEST = args.test 
STEP = args.step

name = session_name
show_streams(data)
eeg_stream = find_stream('eeg', data)
marker_stream = find_stream('marker', data)
df_marker = get_time_series(marker_stream)
sfreq = float(eeg_stream["info"]["nominal_srate"][0])
print(sfreq)
print(df_marker)
print("df shape", df_marker.shape)
## Identify channel position for Neurable headset

channels_info = eeg_stream['info']['desc'][0]['channels'][0]['channel']

# # Extract labels from each channel
ch_names = [channel['label'][0] for channel in channels_info]
ch_names = ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2', 'ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL']



"""Create MNE file raw containing the EEG Stream"""

eeg_data = eeg_stream["time_series"].T
eeg_data = eeg_data[:64]
print(eeg_data.shape)
sfreq = float(eeg_stream["info"]["nominal_srate"][0])
sample_rate = 500

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

##
montage=  raw.set_montage('standard_1020', on_missing='warn')   

if PREP:
    # run pyprep
    print("Running pyprep")
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(50, sample_rate / 2, 50),
    }

    prep = PrepPipeline(raw, prep_params, montage, ransac=RANSAC)
    prep.fit()

    raw_new = prep.raw 

else:
    print("Not running pyprep")
    raw_new = raw.copy()

print("Still bad channels: ", raw_new.info['bads'])

# High-pass filter
raw_new.filter(l_freq=1, h_freq=None)


eeg_start = eeg_stream['time_stamps'][0]  # EEG start time, 679.891
marker_start = marker_stream['time_stamps'][0]  # Marker start time, 2311465.920826529

time_offset = eeg_stream['clock_times'][0] - eeg_stream['clock_times'][0]
aligned_marker_relative = [(ts + time_offset - marker_start + eeg_start) for ts in eeg_stream['time_stamps']]

aligned_pairs = []
for marker_time in aligned_marker_relative:
    closest_idx = np.argmin(np.abs(np.array(eeg_stream['time_stamps']) - marker_time))
    aligned_pairs.append((marker_time, eeg_stream['time_stamps'][closest_idx]))
df_marker = pd.DataFrame(marker_stream['time_series'], aligned_marker_relative,columns=['marker'])


## Create MNE Events

# Sampling frequency from the EEG stream
sfreq = float(eeg_stream["info"]["nominal_srate"][0])

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

csv_file = "C:\\Users\\msi\\Desktop\\Constanze\\Docs\\DATA\\marker\\S01.csv"


# Load the CSV
df = pd.read_csv(csv_file, header=None)

# Extract sentences (first column)
sentences = df.iloc[:, 0].tolist()

# Extract timing information (assume column 6 has onset times, adjust index as needed)
onset_times = df.iloc[:, 6].tolist()

# Ensure the number of sentences matches the number of events
n_events = len(events)
if len(sentences) > n_events:
    sentences = sentences[:n_events]  # Truncate if too long
    print(f"Warning: Sentence list truncated to match {n_events} events.")
elif len(sentences) < n_events:
    print(f"Warning: Fewer sentences ({len(sentences)}) than events ({n_events}).")

# Create metadata DataFrame
metadata = pd.DataFrame({'Sentence': sentences, 'OnsetTime': onset_times[:n_events]})

# Attach metadata to epochs
epochs = mne.Epochs(raw_new, events, event_id=None, tmin=0, tmax= 6,baseline = None,  metadata=metadata, preload=True)

# View metadata
print("Metadata:")
print(epochs.metadata)


if AUTO_REJECT:
    from autoreject import AutoReject
    ar = AutoReject()
    epochs_clean_r, reject_log_r = ar.fit_transform(epochs, return_log=True)
    # epochs_clean_i, reject_log_i = ar.fit_transform(epochs_i, return_log=True)
else:
    print("Not running autoreject")
    epochs_clean_r = epochs
    epochs_clean_i = epochs

# if STEP:
#     # Save the data after bad segment rejection as FIF files
#     for (epoch, type_str) in [(epochs_clean_r, 'read'), (epochs_clean_i, 'imagine')]:
#         save_epochs_to_fif(epoch, edf_file, type_str, '_rej')

#     # Save the data after bad segment rejection as pickle files
#     for (epoch, type_str) in [(epochs_clean_r, 'read'), (epochs_clean_i, 'imagine')]:
#         save_epochs_to_pickle(epoch, edf_file, type_str, '_rej')

# Start ICA artifact removal

if ICA:
    ica_r = mne.preprocessing.ICA(n_components=30, random_state=97, max_iter="auto",method='infomax', fit_params=dict(extended=True)) # 使用extended-infomax算法
    ica_r.fit(epochs_clean_r)
    ic_labels_r = label_components(epochs_clean_r, ica_r, method="iclabel")
    labels_r = ic_labels_r["labels"]
    exclude_idx_r = [
        idx for idx, label in enumerate(labels_r) if label not in ["brain", "other"]
    ]
    print(f"Reading Epochs Excluding these ICA components: {exclude_idx_r}")
    epochs_clean_r_reconstructed = epochs_clean_r.copy()
    ica_r.apply(epochs_clean_r_reconstructed, exclude=exclude_idx_r)

    ica_i = mne.preprocessing.ICA(n_components=30, random_state=97, max_iter="auto",method='infomax', fit_params=dict(extended=True))
    ica_i.fit(epochs_clean_i)
    ic_labels_i = label_components(epochs_clean_i, ica_i, method="iclabel")
    labels_i = ic_labels_i["labels"]
    exclude_idx_i = [
        idx for idx, label in enumerate(labels_i) if label not in ["brain", "other"]
    ]
    print(f"Imagine Epochs Excluding these ICA components: {exclude_idx_i}")
    epochs_clean_i_reconstructed = epochs_clean_i.copy()
    ica_i.apply(epochs_clean_i_reconstructed, exclude=exclude_idx_i)
    
    # if STEP:
    #     # Save the data after artifact removal as FIF files
    #     for (epoch, type_str) in [(epochs_clean_r_reconstructed, 'read'), (epochs_clean_i_reconstructed, 'imagine')]:
    #         save_epochs_to_fif(epoch, edf_file, type_str, '_rej_ica')

    #     # Save the data after artifact removal as pickle files
    #     for (epoch, type_str) in [(epochs_clean_r_reconstructed, 'read'), (epochs_clean_i_reconstructed, 'imagine')]:
    #         save_epochs_to_pickle(epoch, edf_file, type_str, '_rej_ica')

      
def save_epochs_to_fif(epochs, edf_file, type_str, extra_str=''):

    if not os.path.exists(os.path.join(output_folder,f'fif-epo{extra_str}')):
        os.makedirs(os.path.join(output_folder,f'fif-epo{extra_str}'))

    epochs.save(os.path.join(output_folder,f'fif-epo{extra_str}',f"{os.path.basename(edf_file).replace('.edf', '')}_{subject_id}_{type_str}_{method_str}{extra_str}-epo.fif"), overwrite=True)

