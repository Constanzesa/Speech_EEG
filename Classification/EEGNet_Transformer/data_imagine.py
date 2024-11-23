import torch
import pickle
import numpy as np
# Load the preprocessed data
from joblib import load
import tqdm


def get_imagine(sub: str, epoch: int):
    paths = "C:\\Users\\msi\\Desktop\\Constanze\\Docs\\DATA\\PREPROCESSED\\PREP_CH\\data.pkl"
    # with open(paths, "rb") as file:
    #         data = pickle.load(file)
    #         print( len(data))
    pickles = load(paths)
    pickles = pickle.load(open(paths, "rb"))
    print(sub, epoch, len(pickles))
    
    for idx, trial in enumerate(pickles):
        assert isinstance(trial['input_features'], np.ndarray)
        assert trial['input_features'].dtype == np.float64
        assert trial['input_features'].shape == (1, 59, 3344)
        input_features = trial['input_features'][0, :59, :]*1000000
        mean = np.absolute(np.mean(input_features, axis=1))
        stds = np.std(input_features, axis=1)
        assert isinstance(input_features, np.ndarray)
        assert input_features.dtype == np.float64
        assert input_features.shape == (59, 3344)
        # assert (mean > 0).all() and (mean < 10000).all()
        # assert (stds > 0).all() and (stds < 10000).all()
    return pickles

def get_dataset(sub: str):
    print("HERE")
    dsplit = {"input_features": [], "labels": []}
    for epoch in range(1, 46):
        pickles = get_imagine(sub=sub, epoch=epoch)

        for trial in pickles:
            input_features = trial['input_features'][0, :59, :]*1000000
            input_ids = trial['text'].strip()

            input_features = np.float32(input_features)
            input_features = torch.tensor(input_features)
            dsplit["input_features"].append(input_features)
            dsplit["labels"].append(input_ids)
    return dsplit


