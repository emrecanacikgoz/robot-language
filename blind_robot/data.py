import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

class CalvinDataset(Dataset):
    def __init__(
        self, 
        data=None,
        data_format=None,
        max_length=None,
        keys=["actions", "rel_actions", "robot_obs"]
        ):
        super().__init__()
        self.data=data
        self.keys=keys
        self.max_length=max_length
        self.data_format=data_format
        if data_format == "tsv":
            self._load_data_tsv()
        elif data_format == "npy":
            self._load_data_npy()
        else:
            raise NotImplementedError
    
    def _load_data_npy(self):
        annotations = np.load(f"{self.data}/lang_annotations/auto_lang_ann.npy", allow_pickle=True).item()
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"], annotations["language"]["task"]))

        self.items = list()
        self.max_episode_len = -1
        for annotation in tqdm(annotations):
            indices = list(range(annotation[0][0], annotation[0][1] + 1))
            episode = list()
            for idx, _ in enumerate(indices):
                state = np.load(f"{self.data}/episode_{indices[idx]:07d}.npz", allow_pickle=True)
                episode.append({"actions": state['actions'],
                                "rel_actions": state['rel_actions'],
                                "robot_obs_tcp_position": state['robot_obs'][:3],
                                "robot_obs_tcp_orientation": state['robot_obs'][3:6],
                                "robot_obs_gripper_opening_width": state['robot_obs'][6:7],
                                "robot_obs_arm_joint_states": state['robot_obs'][7:14],
                                "gripper_action": state['robot_obs'][14:],
                                "scene_obs": state['scene_obs'],
                                "language": annotation[1],
                                "task": annotation[2],
                                })
            self.items.append(episode)
            if len(episode) > self.max_episode_len:
                self.max_episode_len = len(episode)

    def _load_data_tsv(self):
        states = list()
        print(f"Loading the {self.data.split('/')[-1].split('-')[-1]} data from path {self.data}:")
        with open(self.data, "r") as f:
            for line in tqdm(f):
                l=line.rstrip("\n").split('\t')
                id = l[0]
                state = [float(i) for i in l[1:]]
                k = {"actions": state[0:7],
                    "rel_actions": state[7:14],
                    "robot_obs_tcp_position": state[14:17],
                    "robot_obs_tcp_orientation": state[17:20],
                    "robot_obs_gripper_opening_width": state[20:21],
                    "robot_obs_arm_joint_states": state[21:28],
                    "gripper_action": state[28:29],
                    "scene_obs": state[29:],
                    }
                state = list()
                for key in self.keys:
                    state.extend(k[key])
                states.append(state)
        self.items = self._split_equal_lengths(states)

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        episode = self.items[index]

        if self.data_format == "tsv":
            data = torch.tensor(np.array(episode), dtype=torch.float)
            x, y = data[:self.max_length], data[1:self.max_length+1]
            return (x, y)
            
        elif self.data_format == "npy":
            source = []
            for state in episode:
                state_actions = [state[key] for key in self.keys]
                source.append(np.concatenate(state_actions, axis=0))
            print(source)
            pad_size = self._get_pad_size(source)
            return self._pad_with_zeros(source, pad_size)

        else:
            source = None
            raise NotImplementedError
    
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
    
    def _split_equal_lengths(self, states):
        print(f"Splitting states to {self.max_length} equal episodes:")
        episodes = [states[i:i+(self.max_length+1)] for i in tqdm(range(0, len(states), (self.max_length+1)))]
        episodes.pop()
        return episodes

    def _quantize(self, source):
        raise NotImplementedError


class CalvinDataset_MLP(Dataset):
    def __init__(self, data=None, data_format=None, max_length=None, keys=None):
        super().__init__()
        self.data=data
        self.keys=keys
        self.max_length=max_length
        self.data_format=data_format
        self._load_data_npy()
    
    def _load_data_npy(self):
        
        # load annotated %1 data
        annotations = np.load(f"{self.data}/lang_annotations/auto_lang_ann.npy", allow_pickle=True).item()
        
        # create labels
        labels = sorted(set(annotations["language"]["task"]))
        self.stoi = { ch:i for i,ch in enumerate(labels) }
        self.itos = { i:ch for i,ch in enumerate(labels) }

        # create datamodule
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"], annotations["language"]["task"]))
        self.items = list()
        self.max_episode_len = -1
        for annotation in tqdm(annotations):
            indices = list(range(annotation[0][0], annotation[0][1] + 1))
            episode = list()
            for idx, _ in enumerate(indices):
                state = np.load(f"{self.data}/episode_{indices[idx]:07d}.npz", allow_pickle=True)
                episode.append({"actions": state['actions'],
                                "rel_actions": state['rel_actions'],
                                "robot_obs_tcp_position": state['robot_obs'][:3],
                                "robot_obs_tcp_orientation": state['robot_obs'][3:6],
                                "robot_obs_gripper_opening_width": state['robot_obs'][6:7],
                                "robot_obs_arm_joint_states": state['robot_obs'][7:14],
                                "gripper_action": state['robot_obs'][14:],
                                "scene_obs": state['scene_obs'],
                                "language": annotation[1],
                                "task": annotation[2],
                                "language": annotation[1],
                                "task": self._encode(annotation[2]),
                                })
            if len(episode) == 65:
                self.items.append(episode)

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        episode = self.items[index]
        x, y = [], []
        for state in episode:
            state_actions = [state[key] for key in self.keys]
            x.append(np.concatenate(state_actions, axis=0))
            y.append(state["task"])
        return torch.tensor(np.array(x), dtype=torch.float), torch.tensor(np.array(y[0]), dtype=torch.long)
    
    def _encode(self, string):
        return self.stoi[string]

    def _decode(self, id):
        return self.itos[id]




