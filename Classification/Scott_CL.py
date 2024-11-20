import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Literal
from pytorch_lightning import LightningDataModule
import tqdm
import logging
import torch.nn.functional as F

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# CNN Model
class CNN(nn.Module):
    def __init__(self, input_channels=64, num_classes=5):
        super(CNN, self).__init__()
        self.resblock1 = ResidualBlock(input_channels, 16, stride=1)
        self.resblock2 = ResidualBlock(16, 32, stride=1)
        self.resblock3 = ResidualBlock(32, 64, stride=1)

        self.fc1 = nn.Linear(64 * 1874, 128)  # Adjust based on sequence length
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        # x = x.transpose(1, 2)  # Change shape to [batch, channels, sequence_length]
        x = self.resblock1(x)
        x = self.pool(x)
        x = self.resblock2(x)
        x = self.pool(x)
        x = self.resblock3(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Padded Dataset
class PaddedDataset(Dataset):
    def __init__(self, data_dir: Path, label: Literal["labels"], train: bool = True):
        self.data = []
        self.labels = []
        trials = sorted(data_dir.glob("S*"))

        print(f"Found sessions: {len(trials)}")
        for file_path in trials:
            data_path = file_path / "data.npy"
            label_path = file_path / "labels.npy"
            _data = np.load(data_path, allow_pickle=True)
            _labels = np.load(label_path, allow_pickle=True)

            if train:
                selection = np.concatenate([np.argwhere(_labels == label_id).flatten()[:-10] for label_id in np.unique(_labels)])
            # elif test:
            #     selection = np.concatenate([np.argwhere(_labels == label_id).flatten()[-5:] for label_id in np.unique(_labels)])
            else:
                selection = np.concatenate([np.argwhere(_labels == label_id).flatten()[-10:] for label_id in np.unique(_labels)])


            selected_data = _data[selection]
            selected_labels = _labels[selection]

            self.data.append(selected_data)
            self.labels.append(selected_labels)

        self.max_length = max([data.shape[2] for data in self.data])
        print(f"Max Trial Length: {self.max_length}")

        for i in range(len(self.data)):
            padding = self.max_length - self.data[i].shape[2]
            self.data[i] = np.pad(self.data[i], ((0, 0), (0, 0), (0, padding)), mode="constant")

        self.data = np.concatenate(self.data, axis=0)  # Combine all sessions
        self.labels = np.concatenate(self.labels, axis=0)

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Data Module
class EEGDataModule(LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Literal["fit", "test"] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = PaddedDataset(data_dir=self.data_dir, label="labels", train=True)
            self.val_dataset = PaddedDataset(data_dir=self.data_dir, label="labels", train=False)

        if stage == "test" or stage is None:
            self.test_dataset = PaddedDataset(data_dir=self.data_dir, label="labels", train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


# Training Function
def train_model(train_loader, val_loader, device, n_epochs=25):
    model = CNN(input_channels=64, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch_idx in range(n_epochs):
        model.train()
        losses = []

        for batch_idx, (X, y) in enumerate(tqdm.tqdm(train_loader, desc='Train step')):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = sum(losses) / len(losses)
        print(f'Epoch [{epoch_idx+1}/{n_epochs}], Loss: {train_loss:.4f}')

        val_loss, val_accuracy = test_model(val_loader, model, device)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    return model


# Testing Function
def test_model(loader, model, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    losses = []
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in tqdm.tqdm(loader, desc='Test step'):
            X, y = X.to(device), y.to(device)
            pred = model(X)

            loss = criterion(pred, y)
            losses.append(loss.item())

            _, predicted = torch.max(pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    test_loss = sum(losses) / len(losses)
    accuracy = 100 * correct / total
    # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

    return test_loss, accuracy


# Main Execution
if __name__ == '__main__':
    data_dir = Path("C:\\Users\\msi\\Desktop\\Constanze\\Docs\\DATA\\PREPROCESSED")  # Adjust the path
    batch_size = 8
    num_workers = 4

    datamodule = EEGDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
    datamodule.setup(stage="fit")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    model = train_model(datamodule.train_dataloader(), datamodule.val_dataloader(), device, n_epochs=25)

    # # Test the model
    # datamodule.setup(stage="test")
    # test_loss, test_accuracy = test_model(datamodule.test_dataloader(), model, device)


# import tqdm
# import logging
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# from data_setup.DataModule import DataModule

# # Residual Block
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(out_channels)

#         # Downsample if in_channels != out_channels or stride != 1
#         self.downsample = None
#         if stride != 1 or in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(out_channels)
#             )

#     def forward(self, x):
#         identity = x
#         out = self.conv1(x)
#         out = self.relu(out)
#         out = self.conv2(out)
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         out += identity
#         out = self.relu(out)
#         return out

# # CNN Model
# class CNN(nn.Module):
#     def __init__(self, input_channels=64, num_classes=5):  # Adjust input_channels to your dataset
#         super(CNN, self).__init__()
#         self.resblock1 = ResidualBlock(input_channels, 16, stride=1)
#         self.resblock2 = ResidualBlock(16, 32, stride=1)
#         self.resblock3 = ResidualBlock(32, 64, stride=1)

#         self.fc1 = nn.Linear(64 * 899, 128)  # Adjust based on sequence length --
#         "way of adjusting; output length = input length/ 2`n with n = n of pooling layers ie.7196/2`3 "
#         self.fc2 = nn.Linear(128, num_classes)

#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.dropout = nn.Dropout(p=0.4)

#     def forward(self, x):
#         x = x.transpose(1, 2)  # Change shape to [batch, channels, sequence_length]

#         x = self.resblock1(x)
#         x = self.pool(x)

#         x = self.resblock2(x)
#         x = self.pool(x)

#         x = self.resblock3(x)
#         x = self.pool(x)

#         x = self.dropout(x)

#         # Flatten for fully connected layers
#         x = x.view(x.size(0), -1)

#         # Fully connected layers
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)

#         return x

# # Training Function
# def train_model(datamodule, device, n_epochs=150):
#     datamodule.setup(stage='fit')
#     train_loader = datamodule.train_dataloader()
#     val_loader = datamodule.val_dataloader()

#     model = CNN(input_channels=64, num_classes=5).to(device)  # Adjust input_channels and num_classes as needed
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#     for epoch_idx in range(n_epochs):
#         model.train()
#         losses = []

#         for batch_idx, (X, y) in enumerate(tqdm.tqdm(train_loader, desc='Train step')):
#             X, y = X.to(device), y.to(device)

#             pred = model(X)
#             loss = criterion(pred, y)
#             losses.append(loss.item())

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         train_loss = sum(losses) / len(losses)
#         print(f'Epoch [{epoch_idx+1}/{n_epochs}], Loss: {train_loss:.4f}')

#         val_loss, val_accuracy = test_model(val_loader, model, device)
#         print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

#     return model

# # Testing Function
# def test_model(loader, model, device):
#     criterion = nn.CrossEntropyLoss()
#     model.eval()

#     losses = []
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for X, y in tqdm.tqdm(loader, desc='Test step'):
#             X, y = X.to(device), y.to(device)
#             pred = model(X)

#             loss = criterion(pred, y)
#             losses.append(loss.item())

#             _, predicted = torch.max(pred, 1)
#             total += y.size(0)
#             correct += (predicted == y).sum().item()

#     test_loss = sum(losses) / len(losses)
#     accuracy = 100 * correct / total
#     print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

#     return test_loss, accuracy

# # Main Execution
# if __name__ == '__main__':
#     data_dir = 'C:\\Users\\msi\\Desktop\\Constanze\\Docs\\DATA\\PREPROCESSED\\S01\\data.npy'
#     batch_size = 8
#     num_workers = 4

#     datamodule = DataModule(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     model = train_model(datamodule, device, n_epochs=25)

#     datamodule.setup(stage='test')
#     test_loader = datamodule.test_dataloader()
#     test_loss, test_accuracy = test_model(test_loader, model, device)



# import tqdm

# import logging

# import torch

# from torch import nn

# from torch.utils.data import DataLoader

# from data_setup.DataModule import DataModule
# import torch.nn.functional as F


# class ResidualBlock(nn.Module):

#     def __init__(self, in_channels, out_channels, stride=1):

#         super(ResidualBlock, self).__init__()

#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

#         self.bn1 = nn.BatchNorm1d(out_channels)

#         self.relu = nn.ReLU(inplace=True)

#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

#         self.bn2 = nn.BatchNorm1d(out_channels)


#         # Downsample if in_channels != out_channels or stride != 1

#         self.downsample = None

#         if stride != 1 or in_channels != out_channels:

#             self.downsample = nn.Sequential(

#                 nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),

#                 nn.BatchNorm1d(out_channels)

#             )


#     def forward(self, x):

#         identity = x


#         out = self.conv1(x)

#         # out = self.bn1(out) # BatchNorm1d

#         out = self.relu(out)


#         out = self.conv2(out)

#         # out = self.bn2(out)


#         if self.downsample is not None:

#             identity = self.downsample(x)


#         out += identity

#         out = self.relu(out)

#         return out


# class CNN(nn.Module):
#     def __init__(self, num_classes=10):

#         super(CNN, self).__init__()


#         self.resblock1 = ResidualBlock(8, 16, stride=1)

#         self.resblock2 = ResidualBlock(16, 32, stride=1)

#         self.resblock3 = ResidualBlock(32, 64, stride=1)


#         self.fc1 = nn.Linear(12800, 128)

#         self.fc2 = nn.Linear(128, num_classes)


#         self.relu = nn.ReLU()

#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

#         self.dropout = nn.Dropout(p=0.4)


#     def forward(self, x):

#         x = x.transpose(1, 2)


#         x = self.resblock1(x)

#         x = self.pool(x)

#         # x = self.dropout(x)


#         x = self.resblock2(x)

#         x = self.pool(x)

#         # x = self.dropout(x)


#         x = self.resblock3(x)

#         x = self.pool(x)

#         x = self.dropout(x)


#         # Flatten for fully connected layers

#         x = x.view(x.size(0), -1)


#         # Fully connected layers

#         x = self.relu(self.fc1(x))

#         x = self.dropout(x)

#         x = self.fc2(x)


#         return x


# def train_model(trainset, devset, device, n_epochs=150):

#     # Needs a data loader


#     model = CNN(num_classes=5).to(device)


#     criterion = nn.CrossEntropyLoss()

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#     lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


#     for epoch_idx in range(n_epochs):

#         model.train()

#         losses = []

#         for batch_idx, example in enumerate(tqdm.tqdm(dataloader, ‘Train step’, disable=None)):

#             X = combine_fixed_length(example[‘raw_emg’], 200 * 8).to(device) # This is for adjusting the input size to the same length

#             y = torch.cat(example[‘text_int’]).to(device) # This is for the target labels


#             pred = model(X)


#             if pred.shape[0] != y.shape[0]:

#                 continue


#             loss = criterion(pred, y)

#             losses.append(loss.item())


#             optimizer.zero_grad()

#             loss.backward()

#             optimizer.step()


#         train_loss = sum(losses) / len(losses)

#         print(f’Epoch [{epoch_idx+1}/{n_epochs}], Loss: {train_loss:.4f}, LR: {optimizer.param_groups[0][“lr”]:.6f}‘)


#         val_loss, val_accuracy = test_model(devset, model, device)

#         logging.info(f’finished epoch {epoch_idx + 1} - training loss: {train_loss:.4f} validation loss: {val_loss:.4f} validation accuracy: {val_accuracy:.2f}%‘)


#         model_save_path = “”

#         torch.save(model.state_dict(), model_save_path)


#     print(f’Model saved to {model_save_path}’)


#     return model


# def test_model(testset, model, device):

#     # Needs a data loader


#     criterion = nn.CrossEntropyLoss()

#     model.eval()


#     losses = []

#     correct = 0

#     total = 0


#     with torch.no_grad():

#         for example in tqdm.tqdm(dataloader, ‘Test step’, disable=None):

#             X = combine_fixed_length(example[‘raw_emg’], 200 * 8).to(device)

#             y = torch.cat(example[‘text_int’]).to(device)


#             pred = model(X)


#             if pred.shape[0] != y.shape[0]:

#                 continue


#             loss = criterion(pred, y)

#             losses.append(loss.item())


#             _, predicted = torch.max(pred, 1)

#             total += y.size(0)

#             correct += (predicted == y).sum().item()


#             with open(‘’) as f: # Path to the file containing the labels

#                 sentences = f.readlines()

#             sentences = [sentence.strip() for sentence in sentences]


#             predicted_sentences = [sentences[idx] for idx in predicted]

#             actual_sentences = [sentences[idx] for idx in y]


#             print(“predicted: “, predicted_sentences[0])

#             print(“y: “, actual_sentences[0])


#     test_loss = sum(losses) / len(losses)

#     accuracy = 100 * correct / total

#     print(f’Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%’)


#     return test_loss, accuracy


# if __name__ == ‘__main__‘:


#     # Load the dataset

#     trainset = train_dataloader(dev=False, test=False)

#     devset = val_dataloader(dev=True)

#     testset = test_dataloader(test=True)


#     device = torch.device(‘cuda’ if torch.cuda.is_available() else ‘cpu’)


#     model = train_model(trainset, devset, device, n_epochs=25)

#     test_loss, test_accuracy = test_model(testset, model, device)
