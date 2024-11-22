if __name__ == '__main__':
    from pathlib import Path
    from data_setup.Dataset_Chisco import Dataset_Small
    from data_setup.DataModule_Chisco import DataModule
    from EEGclassify import EEGclassification, train
    from collections import Counter
    import torch
    import os

    # Hardcoded path to your dataset
    data_dir = "C:\\Users\\msi\\Desktop\\Constanze\\Docs\\DATA\\PREPROCESSED"  # Replace with the actual path to your dataset
    batch_size = 16
    num_workers = 0 ## Changed from 4
    num_classes = 5
    segment_length = 647  # Adjusted time dimension for segments

    # Dataset setup
    trainset = Dataset_Small(data_dir=Path(data_dir), label="labels", train=True)
    validset = Dataset_Small(data_dir=Path(data_dir), label="labels", train=False)

    print(f"Train dataset: {len(trainset)} samples")
    print(f"Validation dataset: {len(validset)} samples")
    print(f"Sample from trainset: {trainset[0]}")

    # Initialize DataModule
    data_module = DataModule(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
    data_module.train_dataset = trainset
    data_module.val_dataset = validset
    data_module.setup(stage="fit")

    # Get data loaders
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    # Initialize model
    model = EEGclassification(
        chans=64, timestamp=segment_length, cls=num_classes, dropout1=0.1, dropout2=0.1, layer=0, pooling="mean", size1=8, size2=8, feel1=125, feel2=25
    )

    # Define training arguments
    class Args:
        seed = 42
        lr1 = 1e-3
        wd1 = 0.01
        tau = 0.0
        warmup_ratio = 0.1
        epoch = 10
        train_log = 10
        evals_log = 100
        checkpoint_log = 100
        checkpoint_path = "./checkpoints"

    args = Args()

    # # Calculate label frequencies for loss adjustment
    # label_freqs = [0.0 for _ in range(num_classes)]
    # label_count = Counter([label.item() for _, label in train_dataloader.dataset])
    # for i in label_count:
    #     label_freqs[i] = label_count[i] / len(train_dataloader.dataset)
    # label_freqs = torch.tensor(label_freqs)

    # Calculate label frequencies for loss adjustment
    label_count = Counter([label.item() for _, label, _ in train_dataloader.dataset])  # Adjusted to unpack 3 values
    num_classes = max(label_count.keys()) + 1  # Dynamically calculate the number of classes
    label_freqs = [0.0 for _ in range(num_classes)]

    # Assign frequencies
    for i in label_count:
        label_freqs[i] = label_count[i] / len(train_dataloader.dataset)
    label_freqs = torch.tensor(label_freqs)

    # Debugging outputs
    print(f"Label counts: {label_count}")
    print(f"Label frequencies: {label_freqs}")


    # Ensure checkpoint directory exists
    if not os.path.exists(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)

    # Train model
    train(train_dataloader, val_dataloader, model, args, label_freqs)
