import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from blind_robot.data import CalvinDataset

ROOT = "/Users/emrecanacikgoz/Desktop/rl-project/calvin_debug_dataset/training"
THRESHOLD = 100 # crop data for debugging
data = CalvinDataset(root=ROOT, threshold=THRESHOLD)
train_dataloader = DataLoader(data, batch_size=3, shuffle=False)
train_features = next(iter(train_dataloader))

breakpoint()
