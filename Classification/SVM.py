# import argparse
# import numpy as np
# import pandas as pd
# from scipy import stats, signal
# from sklearn import svm
# from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, classification_report
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# import re
# import os

# # Time-Domain Features
# def extract_time_domain_features(epoch):
#     mean = np.mean(epoch, axis=-1)
#     std = np.std(epoch, axis=-1)
#     skewness = stats.skew(epoch, axis=-1)
#     kurtosis = stats.kurtosis(epoch, axis=-1)
    
#     features = {
#         'Mean': [mean],
#         'StdDev': [std],
#         'Skewness': [skewness],
#         'Kurtosis': [kurtosis],
#     }
    
#     return pd.DataFrame(features)

# # Frequency-Domain Features
# def extract_frequency_domain_features(epoch, sample_rate):
#     freqs, psd = signal.welch(epoch, fs=sample_rate, axis=-1)
    
#     theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)], axis=-1)
#     alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)], axis=-1)
#     beta_power = np.sum(psd[(freqs >= 13) & (freqs <= 30)], axis=-1)
    
#     mean_freq = np.sum(freqs * psd, axis=-1) / np.sum(psd, axis=-1)
#     peak_freq = freqs[np.argmax(psd, axis=-1)]
    
#     features = {
#         'ThetaPower': [theta_power],
#         'AlphaPower': [alpha_power],
#         'BetaPower': [beta_power],
#         'MeanFrequency': [mean_freq],
#         'PeakFrequency': [peak_freq]
#     }
    
#     return pd.DataFrame(features)

# def extract_alpha_asymmetry(epoch, sample_rate, left_channel, right_channel):
#     freqs, psd_left = signal.welch(epoch[left_channel], fs=sample_rate)
#     _, psd_right = signal.welch(epoch[right_channel], fs=sample_rate)
    
#     alpha_power_left = np.sum(psd_left[(freqs >= 8) & (freqs <= 13)])
#     alpha_power_right = np.sum(psd_right[(freqs >= 8) & (freqs <= 13)])
    
#     alpha_asymmetry = np.log(alpha_power_left + 1e-10) - np.log(alpha_power_right + 1e-10)
    
#     return alpha_asymmetry

# def extract_features_from_eeg(eeg_data, 
#                               sample_rate, 
#                               ch_names, 
#                               bad_channels,
#                               tmax=None,
#                               ch_inclusion_re='.*',
#                               left_right_pairs=None):
#     if tmax:
#         eeg_data = eeg_data[:, :, :int(tmax*sample_rate)]
    
#     features = []
#     channes_used = set([])
    
#     for i in range(eeg_data.shape[0]):
#         trial_features = []
        
#         for j in range(eeg_data.shape[1]):
#             if ch_names[j] in bad_channels or not re.match(ch_inclusion_re, ch_names[j]):
#                 continue
#             else:
#                 channes_used.add(ch_names[j])

#             epoch = eeg_data[i, j, :]

#             time_features = extract_time_domain_features(epoch)
#             freq_features = extract_frequency_domain_features(epoch, sample_rate)
            
#             channel_features = pd.concat([time_features, freq_features], axis=1)
#             channel_features = channel_features.add_suffix(f'_{ch_names[j]}')
#             trial_features.append(channel_features)
        
#         if left_right_pairs:
#             for left_channel_name, right_channel_name in left_right_pairs:
#                 if left_channel_name in ch_names and right_channel_name in ch_names:
#                     left_idx = ch_names.index(left_channel_name)
#                     right_idx = ch_names.index(right_channel_name)
#                     alpha_asymmetry = extract_alpha_asymmetry(eeg_data[i], sample_rate, left_idx, right_idx)
#                     asymmetry_feature = pd.DataFrame({'AlphaAsymmetry': [alpha_asymmetry]})
#                     asymmetry_feature = asymmetry_feature.add_suffix(f'_{left_channel_name}-{right_channel_name}')
#                     trial_features.append(asymmetry_feature)
    
#         features.append(pd.concat(trial_features, axis=1))

#     features = pd.concat(features, axis=0)
    
#     return features, channes_used


# def run_classification(session_number, sample_rate, bad_channels, tmax, ch_inclusion_re, left_right_pairs):
#     # Define the base path for data storage
#     base_data_path = "C:\\Users\\msi\\Desktop\\Constanze\\Docs\\DATA\\PREPROCESSED"
    
#     # Define the list of channels, excluding 'HEOR', 'HEOL', 'VEOU', 'VEOL'
#     ch_names = ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FCz', 
#                 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 
#                 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POz', 'PO3', 
#                 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2', 'ECG','HEOR', 'HEOL', 'VEOU', 'VEOL']  # Excluded channels are not listed here.

#     # Add the channels you want to exclude to the bad_channels list
#     bad_channels += ['HEOR', 'HEOL', 'VEOU', 'VEOL']
    
#     # Construct the data and label file paths dynamically
#     data_path = os.path.join(base_data_path, f"S0{session_number}", "data.npy")
#     labels_path = os.path.join(base_data_path, f"S0{session_number}", "labels.npy")
    
#     # Load data and labels
#     data = np.load(data_path)
#     labels = np.load(labels_path)

#     # Ensure that the data shape matches the expected number of channels
#     if data.shape[1] != len(ch_names):
#         print(f"Warning: Data shape mismatch. Expected {len(ch_names)} channels, but got {data.shape[1]} channels.")
#         return
    
#     # Extract features from EEG data
#     final_features, channels_used = extract_features_from_eeg(
#         eeg_data=data,
#         sample_rate=sample_rate,
#         ch_names=ch_names,
#         bad_channels=bad_channels,
#         tmax=tmax,
#         ch_inclusion_re=ch_inclusion_re,
#         left_right_pairs=left_right_pairs
#     )

#     print(final_features.head())
#     print('Channels used:', channels_used)

#     # SVM Classifier
#     X_data = final_features
#     y_labels = labels

#     X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.2, random_state=42, stratify=y_labels)

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     param_grid = {
#         'C': [0.01, 0.1, 1, 10, 100],
#         'kernel': ['linear', 'rbf'],
#         'gamma': ['scale', 'auto']
#     }

#     grid_search = GridSearchCV(svm.SVC(probability=True, class_weight='balanced'), param_grid, cv=5, scoring='accuracy')
#     grid_search.fit(X_train, y_train)
#     print("Best parameters:", grid_search.best_params_)

#     model = grid_search.best_estimator_

#     y_pred = model.predict(X_test)
#     y_pred_proba = model.predict_proba(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {accuracy * 100:.2f}%")

#     loss = log_loss(y_test, y_pred_proba)
#     print(f"Loss: {loss:.6f}")

#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))

#     print("Classification Report:")
#     print(classification_report(y_test, y_pred))



# # Command-line argument parsing
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="EEG Feature Extraction and SVM Classification")

#     parser.add_argument('--session_number', type=int, required=True, help='Session number (e.g., S1, S2, etc.)')
#     parser.add_argument('--sample_rate', type=int, default=1000, help='Sample rate (Hz)')
#     parser.add_argument('--bad_channels', type=str, nargs='+', default= ['ECG'], help='List of bad channels to exclude')
#     parser.add_argument('--tmax', type=int, default=5, help='Max time for data segment (optional)')
#     parser.add_argument('--ch_inclusion_re', type=str, default='.*', help='Regex for channel inclusion')
#     parser.add_argument('--left_right_pairs', type=str, nargs='+', default=None, help='Left-right channel pairs for asymmetry')

#     args = parser.parse_args()

#     run_classification(
#         session_number=args.session_number,
#         sample_rate=args.sample_rate,
#         bad_channels=args.bad_channels,
#         tmax=args.tmax,
#         ch_inclusion_re=args.ch_inclusion_re,
#         left_right_pairs=args.left_right_pairs
#     )


import argparse
import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import re
import os

# Time-Domain Features
def extract_time_domain_features(epoch):
    mean = np.mean(epoch, axis=-1)
    std = np.std(epoch, axis=-1)
    skewness = stats.skew(epoch, axis=-1)
    kurtosis = stats.kurtosis(epoch, axis=-1)
    
    features = {
        'Mean': [mean],
        'StdDev': [std],
        'Skewness': [skewness],
        'Kurtosis': [kurtosis],
    }
    
    return pd.DataFrame(features)

# Frequency-Domain Features
def extract_frequency_domain_features(epoch, sample_rate):
    freqs, psd = signal.welch(epoch, fs=sample_rate, axis=-1)
    
    theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)], axis=-1)
    alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)], axis=-1)
    beta_power = np.sum(psd[(freqs >= 13) & (freqs <= 30)], axis=-1)
    
    mean_freq = np.sum(freqs * psd, axis=-1) / np.sum(psd, axis=-1)
    peak_freq = freqs[np.argmax(psd, axis=-1)]
    
    features = {
        'ThetaPower': [theta_power],
        'AlphaPower': [alpha_power],
        'BetaPower': [beta_power],
        'MeanFrequency': [mean_freq],
        'PeakFrequency': [peak_freq]
    }
    
    return pd.DataFrame(features)

def extract_alpha_asymmetry(epoch, sample_rate, left_channel, right_channel):
    freqs, psd_left = signal.welch(epoch[left_channel], fs=sample_rate)
    _, psd_right = signal.welch(epoch[right_channel], fs=sample_rate)
    
    alpha_power_left = np.sum(psd_left[(freqs >= 8) & (freqs <= 13)])
    alpha_power_right = np.sum(psd_right[(freqs >= 8) & (freqs <= 13)])
    
    alpha_asymmetry = np.log(alpha_power_left + 1e-10) - np.log(alpha_power_right + 1e-10)
    
    return alpha_asymmetry

def extract_features_from_eeg(eeg_data, 
                              sample_rate, 
                              ch_names, 
                              bad_channels,
                              tmax=None,
                              ch_inclusion_re='.*',
                              left_right_pairs=None):
    if tmax:
        eeg_data = eeg_data[:, :, :int(tmax*sample_rate)]
    
    features = []
    channes_used = set([])
    
    for i in range(eeg_data.shape[0]):
        trial_features = []
        
        for j in range(eeg_data.shape[1]):
            if ch_names[j] in bad_channels or not re.match(ch_inclusion_re, ch_names[j]):
                continue
            else:
                channes_used.add(ch_names[j])

            epoch = eeg_data[i, j, :]

            time_features = extract_time_domain_features(epoch)
            freq_features = extract_frequency_domain_features(epoch, sample_rate)
            
            channel_features = pd.concat([time_features, freq_features], axis=1)
            channel_features = channel_features.add_suffix(f'_{ch_names[j]}')
            trial_features.append(channel_features)
        
        if left_right_pairs:
            for left_channel_name, right_channel_name in left_right_pairs:
                if left_channel_name in ch_names and right_channel_name in ch_names:
                    left_idx = ch_names.index(left_channel_name)
                    right_idx = ch_names.index(right_channel_name)
                    alpha_asymmetry = extract_alpha_asymmetry(eeg_data[i], sample_rate, left_idx, right_idx)
                    asymmetry_feature = pd.DataFrame({'AlphaAsymmetry': [alpha_asymmetry]})
                    asymmetry_feature = asymmetry_feature.add_suffix(f'_{left_channel_name}-{right_channel_name}')
                    trial_features.append(asymmetry_feature)
    
        features.append(pd.concat(trial_features, axis=1))

    features = pd.concat(features, axis=0)
    
    return features, channes_used


# Main function to run classification
def run_classification(session_number, sample_rate, bad_channels, tmax, ch_inclusion_re, left_right_pairs):
    # Define the base path for data storage
    base_data_path = "C:\\Users\\msi\\Desktop\\Constanze\\Docs\\DATA\\PREPROCESSED"
    ch_names = ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2', 'ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL']
    bad_channels = ['ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL']
    

    # Construct the data and label file paths dynamically
    data_path = os.path.join(base_data_path, f"S0{session_number}", "eeg.npy")
    labels_path = os.path.join(base_data_path, f"S0{session_number}", "labels.npy")
    
    # Load data and labels
    data = np.load(data_path)
    labels = np.load(labels_path)

    final_features, channels_used = extract_features_from_eeg(
        eeg_data=data,
        sample_rate=sample_rate,
        ch_names=ch_names,
        bad_channels=bad_channels,
        tmax=tmax,
        ch_inclusion_re=ch_inclusion_re,
        left_right_pairs=left_right_pairs
    )

    print(final_features.head())
    print('Channels used:', channels_used)

    # SVM Classifier
    X_data = final_features
    y_labels = labels

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.2, random_state=33, stratify=y_labels)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(svm.SVC(probability=True, class_weight='balanced'), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)

    model = grid_search.best_estimator_

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    loss = log_loss(y_test, y_pred_proba)
    print(f"Loss: {loss:.6f}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report:")
    print(classification_report(y_test, y_pred))


# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Feature Extraction and SVM Classification")

    parser.add_argument('--session_number', type=int, required=True, help='Session number (e.g., S1, S2, etc.)')
    parser.add_argument('--sample_rate', type=int, default=1000, help='Sample rate (Hz)')
    parser.add_argument('--bad_channels', type=str, nargs='+', default=['EOG'], help='List of bad channels to exclude')
    parser.add_argument('--tmax', type=int, default=None, help='Max time for data segment (optional)')
    parser.add_argument('--ch_inclusion_re', type=str, default='.*', help='Regex for channel inclusion')
    parser.add_argument('--left_right_pairs', type=str, nargs='+', default=None, help='Left-right channel pairs for asymmetry')

    args = parser.parse_args()

    run_classification(
        session_number=args.session_number,
        sample_rate=args.sample_rate,
        bad_channels=args.bad_channels,
        tmax=args.tmax,
        ch_inclusion_re=args.ch_inclusion_re,
        left_right_pairs=args.left_right_pairs
    )