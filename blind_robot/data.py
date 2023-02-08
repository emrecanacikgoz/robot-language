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
        
        annotations = np.load(f"{self.root}/lang_annotations/auto_lang_ann.npy", allow_pickle=True).item()
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"]))


        self.items = list()
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
            collate_fn=calvin_collate_fn(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=calvin_collate_fn(),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=calvin_collate_fn(),
        )


def calvin_collate_fn(pad_token_id=0):
    def _collate_fn(batch):
        _helper = lambda key: [j[key] for i in batch for j in i]
        B = len(batch)
        T = max([len(x) for x in batch])
        input_ids = torch.empty((B, T), dtype=torch.long)
        input_ids.fill_(pad_token_id)
        attention_mask = torch.empty((B, T), dtype=torch.long)
        attention_mask.fill_(0)
        breakpoint()
        for i, item in enumerate(batch):
            L = item['prompt_ids'].numel()
            input_ids[i, :L] = item['prompt_ids']
            attention_mask[i, :L] = item['prompt_mask']
        
        pixel_values = torch.cat(_helper('pixel_values'), dim=0)
        labels = torch.tensor(_helper('label')).view(1, -1)

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'indexes': _helper('index'),
            'raw_words': _helper('raw_word'),
            'raw_contexts': _helper('raw_context'),
            'image_files': _helper('image_files'),
        }

    return _collate_fn

