"CONCATENATE"

import os
import json
import numpy as np

def calculate_max_length(base_path, session_range, skip_session="S02"):
    """
    Reads all session files, calculates the maximum data length, and returns the overall maximum length.
    
    Args:
        base_path (str): The base directory where session files are stored.
        session_range (range): Range of session numbers (e.g., range(1, 7) for S01 to S06).
        skip_session (str): The session name to skip (e.g., "S02").
        
    Returns:
        int: The maximum length across all sessions.
    """
    max_lengths = []  # List to store the max length of each session

    for session_num in session_range:
        session_name = f"S{session_num:02d}"
        if session_name == skip_session:
            print(f"Skipping session: {session_name}")
            continue

        # Path to the JSON file for the current session
        json_path = os.path.join(base_path, f"{session_name}.json")

        if not os.path.exists(json_path):
            print(f"Session file not found: {json_path}")
            continue

        # Load session data
        try:
            with open(json_path, "r") as file:
                session_data = json.load(file)
                print(f"Loaded session: {session_name}, Number of entries: {len(session_data)}")

                # Calculate lengths (assume session_data is a dict of lists or arrays)
                session_lengths = [len(entry) for entry in session_data.values()]
                max_lengths.append(max(session_lengths))
        except Exception as e:
            print(f"Error loading session {session_name}: {e}")
            continue

    # Determine the overall maximum length
    overall_max_length = max(max_lengths) if max_lengths else 0
    return overall_max_length


# Define base path and session range
base_path = "C:\\Users\\msi\\Desktop\\Constanze\\Docs\\DATA\\marker\\textmaps"
session_range = range(1, 7)  # Sessions S01 to S06

# Calculate the max length
overall_max_length = calculate_max_length(base_path, session_range)

print(f"Overall maximum length across all sessions: {overall_max_length}")


import os
import pickle
import numpy as np

def get_imagine(base_path: str, session_name: str):
    """
    Load and validate data for a given session.

    Args:
        base_path (str): Base directory containing session folders.
        session_name (str): Name of the session (e.g., "S01").

    Returns:
        list: List of trials for the session.
    """
    paths = os.path.join(base_path, session_name, "data.pkl")

    if not os.path.exists(paths):
        print(f"Session {session_name} not found at {paths}. Skipping.")
        return None

    with open(paths, "rb") as file:
        pickles = pickle.load(file)

    print(f"Loaded data for {session_name}: {len(pickles)} trials.")
    
    # # Validate the data
    # for trial in pickles:
    #     assert isinstance(trial['input_features'], np.ndarray)
    #     assert trial['input_features'].dtype == np.float64
    #     assert trial['input_features'].shape == (1, 59, 3344)

    return pickles


def get_dataset(base_path: str):
    """
    Load and process data from all sessions, skipping S02.

    Args:
        base_path (str): Base directory containing session folders.

    Returns:
        dict: Dataset with input features and labels.
    """
    dsplit = {"input_features": [], "labels": []}

    for epoch in range(1, 10):  # Sessions S01 to S09
        session_name = f"S{epoch:02d}"
        if session_name == "S02":  # Skip S02
            print(f"Skipping session {session_name}.")
            continue

        pickles = get_imagine(base_path=base_path, session_name=session_name)
        if pickles is None:
            continue

        for trial in pickles:
            input_features = trial['input_features'][0, :59, :]  # Extract (59, samples)
            input_ids = trial['text'].strip()  # Assuming 'text' exists in the trial

            # Convert to numpy.float32
            input_features = np.float32(input_features)

            # Append to dataset
            dsplit["input_features"].append(input_features)
            dsplit["labels"].append(input_ids)

    return dsplit


def process_and_pad_data(base_path: str):
    """
    Process data from all sessions, determine max sample length,
    pad data to max length, and concatenate.

    Args:
        base_path (str): Base directory containing session folders.

    Returns:
        np.ndarray: Concatenated padded data of shape (n_trials, n_channels, max_samples).
        list: Concatenated labels for all trials.
    """
    # Load dataset
    data = get_dataset(base_path)

    # Determine max sample length
    max_samples = max(features.shape[1] for features in data['input_features'])
    print(f"Maximum sample length: {max_samples}")

    # Stack and pad data
    all_data = []
    for features in data['input_features']:
        padded_features = np.zeros((features.shape[0], max_samples), dtype=np.float32)
        padded_features[:, :features.shape[1]] = features
        all_data.append(padded_features)

    # Concatenate data and labels
    concatenated_data = np.stack(all_data, axis=0)  # Shape: (n_trials, n_channels, max_samples)
    concatenated_labels = np.array(data['labels'])

    print(f"Concatenated data shape: {concatenated_data.shape}")
    print(f"Concatenated labels shape: {concatenated_labels.shape}")

    return concatenated_data, concatenated_labels


# Base path to the data directory
base_path = "C:\\Users\\msi\\Desktop\\Constanze\\Docs\\DATA\\PREPROCESSED\\PREP_CH"

# Process and pad data
try:
    data, labels = process_and_pad_data(base_path)
except ValueError as e:
    print(f"Error: {e}")


def save_processed_data(base_path: str, save_path: str):
    """
    Process data from all sessions, pad them to the maximum length, 
    and save each trial in the same structure as original .pkl files.

    Args:
        base_path (str): Base directory containing session folders.
        save_path (str): Path to save the processed .pkl file.
    """
    # Process data
    concatenated_data, concatenated_labels = process_and_pad_data(base_path)

    # Create the trial structure similar to the original data
    processed_trials = []
    for idx in range(len(concatenated_data)):
        trial = {
            "input_features": concatenated_data[idx][np.newaxis, :, :],  # Shape: (1, 59, max_samples)
            "text": concatenated_labels[idx],
        }
        processed_trials.append(trial)

    # Save the processed trials to a new .pkl file
    save_file_path = os.path.join(save_path, "processed_data.pkl")
    with open(save_file_path, "wb") as file:
        pickle.dump(processed_trials, file)

    print(f"Processed data saved to {save_file_path}. Total trials: {len(processed_trials)}")


# Define paths
save_path = "C:\\Users\\msi\\Desktop\\Constanze\\Docs\\DATA\\PROCESSED\\PREP_CH\\"

# Ensure the save directory exists
os.makedirs(save_path, exist_ok=True)

# Save the processed data
save_processed_data(base_path, save_path)
