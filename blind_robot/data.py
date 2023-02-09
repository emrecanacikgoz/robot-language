import os, hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
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
        
        annotations = np.load(f"{self.root}/lang_annotations/auto_lang_ann.npy", allow_pickle=True).item()
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"]))

        self.items = list()
        episode_lens = []
        for annotation in tqdm(annotations):
            indices = list(range(annotation[0][0], annotation[0][1] + 1))
            episode = list()
            for idx, _ in enumerate(indices):
                state = np.load(f"{self.root}/episode_{indices[idx]:07d}.npz", allow_pickle=True)
                episode.append({
                'action_xyz': state['actions'][:3],
                'action_ExEyEz': state['actions'][3:6],
                'action_gripper_oc': state['actions'][6],
                'relative_action_xyz': state['rel_actions'][:3],
                'relative_action_ExEyEz': state['rel_actions'][3:6],
                'relative_action_gripper_oc': state['rel_actions'][6],
                'robot_state_xyz': state['robot_obs'][:3],
                'robot_state_ExEyEz': state['robot_obs'][3:6],
                'robot_state_gripper_width': state['robot_obs'][6],
                'robot_state_arm_joints': state['robot_obs'][7:14],
                'robot_state_gripper_oc': state['robot_obs'][14],
                'language_annotation': annotation[1]
                })
            self.items.append(episode)
            episode_lens.append(len(episode))
        self.max_episode_len = max(episode_lens)

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        
        episode = self.items[index]


        source = [np.concatenate((state['action_xyz'], 
                                 state['action_ExEyEz'], 
                                 state['action_gripper_oc'], 
                                 state['robot_state_xyz'], 
                                 state['robot_state_ExEyEz'], 
                                 state['robot_state_gripper_width'],
                                 state['robot_state_arm_joints'], 
                                 state['robot_state_gripper_oc'], 
                                ), axis=None) 
                                for state in episode]
        source = self._check_padding(source)                 
        return source
    
    def _check_padding(self, source):
        pad = abs(source[0]*0)
        if len(source) < self.max_episode_len:
            pads = [pad for _ in range(self.max_episode_len-len(source))]
            source = source + pads
            return torch.tensor(source)
        else:
            return torch.tensor(source)


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



