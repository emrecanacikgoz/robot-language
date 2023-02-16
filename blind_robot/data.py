import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class CalvinDataset(Dataset):
    def __init__(
        self, 
        root_data_dir=None, 
        keys=["actions", "rel_actions", "robot_obs"]
        ):
        super().__init__()
        
        self.root_data_dir = root_data_dir
        self.keys=keys
        self._load_data()
    
    def _load_data(self):
        annotations = np.load(f"{self.root_data_dir}/lang_annotations/auto_lang_ann.npy", allow_pickle=True).item()
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"]))

        self.items = list()
        self.max_episode_len = -1
        for annotation in tqdm(annotations):
            indices = list(range(annotation[0][0], annotation[0][1] + 1))
            episode = list()
            for idx, _ in enumerate(indices):
                state = np.load(f"{self.root_data_dir}/episode_{indices[idx]:07d}.npz", allow_pickle=True)
                episode.append({"actions": state['actions'],
                                "rel_actions": state['rel_actions'],
                                "robot_obs": state['robot_obs'],
                                "robot_obs_tcp_position": state['robot_obs'][:3],
                                "robot_obs_tcp_orientation": state['robot_obs'][3:6],
                                "robot_obs_gripper_opening_width": state['robot_obs'][6:7],
                                "robot_obs_arm_joint_states": state['robot_obs'][7:14],
                                "gripper_action": state['robot_obs'][14:],
                                "language_annotation": annotation[1],
                                })
            self.items.append(episode)
            if len(episode) > self.max_episode_len:
                self.max_episode_len = len(episode)

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        episode = self.items[index]

        source = []
        for state in episode:
            state_actions = [state[key] for key in self.keys]
            source.append(np.concatenate(state_actions, axis=0))

        pad_size = self._get_pad_size(source)
        source = self._pad_with_zeros(source, pad_size)
        return source
    
    def _get_pad_size(self, source):
        return self.max_episode_len-len(source)
    
    def _pad_with_zeros(self, source, pad_size):
        pad = abs(source[0]*0)
        if len(source) < self.max_episode_len:
            pads = [pad for _ in range(pad_size)]
            source = source + pads
            return torch.tensor(np.array(source), dtype=torch.float)
        else:
            return torch.tensor(np.array(source), dtype=torch.float)
    
    def _quantize(self, source):
        pass
