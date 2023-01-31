import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class CalvinDataset(Dataset):
    def __init__(self, root, **kwargs):
        super().__init__()
        self.root = root
        self.threshold = kwargs["threshold"] # DEBUG
        self._load_data()
    
    def _load_data(self):
        
        print("Getting Episodes...")
        episodes = [f for f in tqdm(os.listdir(self.root)) if f.endswith(".npz")]
        print("Getting Episode Paths...")
        episode_paths = [os.path.join(self.root, episode) for episode in tqdm(episodes[:self.threshold])]
        print("Getting Actions...")
        actions = [np.load(episode_path)["actions"] for episode_path in tqdm(episode_paths)]
        print("Getting Relative Actions...")
        relative_actions = [np.load(episode_path)["rel_actions"] for episode_path in tqdm(episode_paths)]
        print("Getting Robot States...")
        robot_states = [np.load(episode_path)["robot_obs"] for episode_path in tqdm(episode_paths)]

        assert len(episode_paths) == len(actions) == len(relative_actions) == len(robot_states), "Lengths of episode_paths-actions-relative_actions-robot_states should be same!"

        self.items = list()
        for action, relative_action, robot_state in zip(actions, relative_actions, robot_states):
            self.items.append({
                'action_xyz': action[:3],
                'action_ExEyEz': action[3:6],
                'action_gripper_oc': action[6],
                'relative_action_xyz': relative_action[:3],
                'relative_action_ExEyEz': relative_action[3:6],
                'relative_action_gripper_oc': relative_action[6],
                'robot_state_xyz': robot_state[:3],
                'robot_state_ExEyEz': robot_state[3:6],
                'robot_state_gripper_width': robot_state[6],
                'robot_state_arm_joints': robot_state[7:14],
                'robot_state_gripper_oc': robot_state[14],
            })

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]


class CalvinDataModule(pl.LightningDataModule):
    def __init__(self, train_dir=None, val_dir=None, batch_size=8, num_workers=0, pin_memory=False):
        super().__init__()

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.train_data = self.load_split(split='train')
        if stage == 'val' or stage == 'predict':
            self.val_data = self.load_split(split='val')
    
    def load_split(self, split):
        if split == 'train':
            root = self.train_dir
        elif split == 'val':
            root = self.val_dir
        return CalvinDataset(root=root)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )




