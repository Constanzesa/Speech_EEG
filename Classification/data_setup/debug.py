import torch
from pathlib import Path
from Dataset import Dataset_Large, Dataset_Small

# Set the path to your data directory
data_dir = Path("/Users/arnavkapur/Desktop/EEG_Speech/DATA/PREPROCESSED/")

# Initialize the dataset
print("Initializing Dataset...")
dataset = Dataset_Small(data_dir=data_dir, label="labels")

# Print dataset information
print(f"Total trials in dataset: {len(dataset)}")
print(f"Shape of data: {dataset.data.shape}")
print(f"Shape of labels: {dataset.labels.shape}")

# Print individual samples
for idx in range(min(10, len(dataset))):  # Limit to the first 10 samples
    data, label = dataset[idx]
    print(f"Sample {idx} - Data shape: {data.shape}, Label: {label}")

# Iterate through all trials (optional, remove limit if needed)
for idx, (data, label) in enumerate(dataset):
    print(f"Trial {idx}: Data shape: {data.shape}, Label: {label}")
    if idx == 10:  # Stop after 10 samples for debugging
        break
