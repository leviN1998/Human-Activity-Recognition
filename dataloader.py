"""
Dataloader class for the UCI HAR dataset.

provides a PyTorch Dataset and Dataloader implementation for loading and preprocessing the UCI HAR dataset.

Author: Levin Kolmar
Date: 27.1.2026
"""

import os 
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class UCIHARDataset(Dataset):
    """
    Loads UCI HAR "Inertial Signals" time series

    Expected folder structure:
    root/
        UCI HAR Dataset/
            train/
                Inertial Signals/
                    body_acc_x_train.txt
                    body_acc_y_train.txt
                    body_acc_z_train.txt
                    body_gyro_x_train.txt
                    body_gyro_y_train.txt
                    body_gyro_z_train.txt
                    total_acc_x_train.txt
                    total_acc_y_train.txt
                    total_acc_z_train.txt
                y_train.txt
            test/
                ...


    Returns:
        x: torch.float32 (C, T)   ->    (batch, channels, time)
        y: torch.long in [0..5]
    """

    CHANNEL_SETS = {
        "6ch": [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
        ],
        "9ch": [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_",
        ],
    }

    def __init__(self, root: str | Path, split: str = "train", channels: str = "6ch", cache: bool = True):
        self.root = Path(root)
        self.split = split.lower()
        assert self.split in ["train", "test"], "split must be 'train' or 'test'"

        assert channels in self.CHANNEL_SETS, f"channels must be one of {list(self.CHANNEL_SETS.keys())}"
        self.channel_names = self.CHANNEL_SETS[channels]

        self.base_dir = self._resolve_base_dir(self.root)
        self.split_dir = self.base_dir / self.split
        self.signals_dir = self.split_dir / "Inertial Signals"
        self.y_path = self.split_dir / f"y_{self.split}.txt"

        if not self.signals_dir.exists():
            raise FileNotFoundError(f"Signals directory not found: {self.signals_dir}")
        if not self.y_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.y_path}")
        
        self.cache = cache
        cache_name = f"uci_har_{self.split}_{channels}_cached.npz"
        self.cache_path = self.split_dir / cache_name

        if self.cache and self.cache_path.exists():
            data = np.load(self.cache_path)
            self.X = data["X"]
            self.y = data["y"]
        else:
            self.X, self.y = self._load_raw()
            if self.cache:
                np.savez_compressed(self.cache_path, X=self.X, y=self.y)

    
    def _resolve_base_dir(self, root: Path) -> Path:
        """
        Allow passing either:
          - path to extracted folder that contains "UCI HAR Dataset"
          - path directly to "UCI HAR Dataset"
        
        :param self: Description
        :param root: Description
        :type root: Path
        :return: Description
        :rtype: Path
        """
        if (root / "UCI HAR Dataset").exists():
            return root / "UCI HAR Dataset"
        if (root / "train").exists() and (root / "test").exists():
            return root
        raise FileNotFoundError(
            "Root must point to either the folder containing 'UCI HAR Dataset' "
            "or directly to the 'UCI HAR Dataset' folder."
        )
    

    def _read_signal_file(self, path: Path) -> np.ndarray:
        """
        Each file is shape (N, 128) stored as whitespace separated floats per line..
    
        :param self: Description
        :param path: Description
        :type path: Path
        :return: Description
        :rtype: ndarray
        """
        return np.loadtxt(path, dtype=np.float32)
    

    def _load_raw(self) -> tuple[np.ndarray, np.ndarray]:
        y = np.loadtxt(self.y_path, dtype=np.int64) - 1  # Convert to 0-based labels

        signals = []
        for base in self.channel_names:
            file_path = self.signals_dir / f"{base}{self.split}.txt"
            if not file_path.exists():
                raise FileNotFoundError(f"Signal file not found: {file_path}")
            sig = self._read_signal_file(file_path)
            signals.append(sig)

        # Stack to (N, 128, C)
        X = np.stack(signals, axis=-1)  # (N, T, C)

        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatch between number of samples in X and y")
        
        return X, y
    

    def __len__(self):
        return self.X.shape[0]
    

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]) # (T, C)
        x = x.transpose(0, 1).contiguous()  # (C, T)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
    

def make_har_loaders(
        root: str | Path,
        channels: str = "6ch",
        batch_size: int = 128,
        num_workers: int = 2,
        pin_memory: bool = True,
):
    train_ds = UCIHARDataset(root=root, split="train", channels=channels, cache=True)
    test_ds = UCIHARDataset(root=root, split="test", channels=channels, cache=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    root_dir = "./UCI HAR Dataset"
    train_loader, test_loader = make_har_loaders(root=root_dir, channels="9ch", batch_size=64, num_workers=2)

    x, y = next(iter(train_loader))
    print(f"Batch x shape: {x.shape}")
    print(f"Batch y shape: {y.shape}")
    print(f"y min/max: {y.min().item()}/{y.max().item()}")