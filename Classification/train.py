import numpy as np
import torch
import pytorch_lightning as pl
# from torch.optim.lr_scheduler import LRScheduler

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import wandb
import yaml
from data_setup.DataModule import DataModule
from models.EEGNet import EEGNetv4
from models.EEGNet_Embedding_version import EEGNet_Embedding
from pytorch_lightning.callbacks import EarlyStopping  
from data_setup.Dataset import Dataset_Small

def read_config(config_path: str):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def merge_dicts(dict1, dict2):
    """Recursively merge dict2 into dict1."""
    for key in dict2:
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                merge_dicts(dict1[key], dict2[key])
            elif key == "name" and isinstance(dict1[key], str) and isinstance(dict2[key], str):
                dict1[key] += dict2[key]
            else:
                dict1[key] = dict2[key]
        else:
            dict1[key] = dict2[key]
    return dict1

def combine_configs(head_config):
    assert "combine" in head_config, "Config is no head config!"
    head_config = head_config.pop("combine")
    for yaml in head_config:
        if "combined_config" in locals():
            combined_config = merge_dicts(combined_config, read_config(yaml))
        else:
            combined_config = read_config(yaml)
    return combined_config



def main(config = None):
    wandb.init(config=config, project = "SPEECH_EEG")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # binary_classes = wandb.config["datamodule"].get("binary_classes", None)  #ADD_CL: Get two classes defined in ME.yaml
    # dm = DataModule(**wandb.config["datamodule"], binary_classes=binary_classes) ##ADD_CL; Use this only for 2 class classification

    dm = DataModule(**wandb.config["datamodule"])
    print(f"DataModule Loaded: {dm}")    
    
    # wandb.config["model"]["n_classes"] = 2  #ADD_CL: Ensure n_classes is set to 2 for binary classification in EEGNet.yaml

    if wandb.config["model_name"] == "EEGNET":
        model = EEGNetv4(**wandb.config["model"], epochs = wandb.config.trainer["max_epochs"]) 
    elif wandb.config["model_name"] == "EEGNET_Embedding":
        model = EEGNet_Embedding(**wandb.config["model"], epochs = wandb.config.trainer["max_epochs"]) 
    
    # print(f"Binary Classification with classes: {binary_classes}") ##ADD_CL
    print("Wandb Config: ", wandb.config)
    print(f"Model Loaded: {model}")
    model = model.to(device)

    # Create a ModelCheckpoint callback
    if wandb.config["final_model"] == False:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            filename="best-model-{epoch:02d}-{val_acc:.2f}",
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        )
    else:
        print("FINAL MODEL - saving weights only after last epoch")
        checkpoint_callback = ModelCheckpoint(
            monitor=None,
            filename="final-model",
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    early_stopping_callback = EarlyStopping(
            monitor="val_acc",  # You can also monitor "val_loss"
            patience=10,  # Number of epochs to wait for improvement
            mode="max",  # "min" for loss, "max" for accuracy or metrics to be maximized
            verbose=True,
        )
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs = wandb.config.trainer["max_epochs"],
        logger = pl.loggers.WandbLogger(save_dir="./wandb"),
        callbacks = [checkpoint_callback, lr_monitor,early_stopping_callback],
        default_root_dir="./checkpoints", 
        #pl.callbacks.EarlyStopping(monitor="val_acc")
        accelerator="auto"
    )

    # Train model 
    trainer.fit(model = model, datamodule = dm)

    # if wandb.config["fine_tuning"]:
    #     trainer.fit(model=model)

    # Test model
    #Note: this should only be used once!
    if wandb.config["final_model"] == True:
        trainer.test(datamodule = dm) #test_dataset is integrated in datamodule

    wandb.run.finish()






# Initialize new sweep (Change subject and model name to run different models)
# sweep_config = read_config(config_path = "./pytorch/configs/final/P001/EEGNET_P001.yaml")

sweep_config = read_config(config_path = "./configs/final/EEGNET.yaml")

sweep_config = combine_configs(sweep_config) #Comment out for test run
# sweep_id = wandb.sweep(sweep_config, project=sweep_config["name"])
sweep_id = wandb.sweep(sweep_config, project="SPEECH_EEG")

# Run sweep
wandb.agent(sweep_id, function=main)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wandb sweeps for hyperparameter optimization")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="./configs/final/EEGNET.yaml",
        # default="./pytorch/configs/final/P001/EEGNET_P001.yaml",
        help="Path to config file")
    args = parser.parse_args()

    sweep_config = read_config(config_path = args.config_path) # Read config file
    print("Using config: {}".format(args.config_path))
    sweep_config = combine_configs(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project=sweep_config["SPEECH_EEG"]) # Init sweep
    wandb.agent(sweep_id, function=main) # Run the sweep

