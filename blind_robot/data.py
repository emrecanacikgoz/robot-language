import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

class CalvinDataset_GPT(Dataset):
    def __init__(self, data=None, max_length=None, keys=None):
        super().__init__()
        self.data=data
        self.keys=keys
        self.max_length=max_length
        self._load_data()

    def _load_data(self):
        print(f"Loading the {self.data.split('/')[-1].split('-')[-1]} data from path {self.data}:")

        # read data
        states = list()
        with open(self.data, "r") as f:
            for line in tqdm(f):

                # set data fields
                l = line.rstrip("\n").split('\t')
                _stateID, state = l[0], [float(i) for i in l[1:]]
                fields = {"actions": state[0:7],
                          "rel_actions": state[7:14],
                          "robot_obs_tcp_position": state[14:17],
                          "robot_obs_tcp_orientation": state[17:20],
                          "robot_obs_gripper_opening_width": state[20:21],
                          "robot_obs_arm_joint_states": state[21:28],
                          "gripper_action": state[28:29],
                          "scene_obs": state[29:],
                         }
                
                # get only the desired fields in config.yaml
                desired_state = list()
                for desired_field in self.keys:
                    desired_state.extend(fields[desired_field])
                states.append(desired_state)

        # set context length of gpt to max_length
        self.items = self._split_equal_lengths(states)

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        episode = self.items[index]
        # convert tensor
        data = torch.tensor(np.array(episode), dtype=torch.float)
        # set source and target for next-token prediction; i.e., auto-regressive decoder training
        x, y = data[:self.max_length], data[1:self.max_length+1]
        return (x, y)
    
    def _split_equal_lengths(self, states):
        print(f"Splitting states to {self.max_length} equal episodes:")
        episodes = [states[i:i+(self.max_length+1)] for i in tqdm(range(0, len(states), (self.max_length+1)))]
        episodes.pop()
        return episodes

    def _quantize(self, source):
        raise NotImplementedError


class CalvinDataset_MLP(Dataset):
    def __init__(self, np_data=None, tsv_data=None, keys=None):
        super().__init__()
        self.np_data=np_data
        self.tsv_data=tsv_data
        self.keys=keys
        self._load_data()
    
    def _load_data(self):
        
        # load full data
        data = pd.read_csv(self.tsv_data, sep='\t', header=None)

        # load annotated %1 data
        annotations = np.load(f"{self.np_data}/lang_annotations/auto_lang_ann.npy", allow_pickle=True).item()
        
        # create labels
        labels = sorted(set(annotations["language"]["task"]))
        self.stoi = { ch:i for i,ch in enumerate(labels) }
        self.itos = { i:ch for i,ch in enumerate(labels) }

        # create datamodule
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"], annotations["language"]["task"]))
        self.items = list()
        for annotation in tqdm(annotations):
            indices = list(range(annotation[0][0], annotation[0][1] + 1))
            episode = list()
            for _idx, index in enumerate(indices):
                state = data.loc[data[0] == index]
                state = state.to_numpy().squeeze().tolist() 
                episode.append({"actions": state[1:8],
                                "rel_actions": state[8:15],
                                "robot_obs_tcp_position": state[15:18],
                                "robot_obs_tcp_orientation": state[18:21],
                                "robot_obs_gripper_opening_width": state[21:22],
                                "robot_obs_arm_joint_states": state[22:29],
                                "gripper_action": state[29:30],
                                "scene_obs": state[30:],
                                "language": annotation[1],
                                "task": self._encode(annotation[2]),
                                })
            
            # drop episodes shorter than 65 to avoid padding
            if len(episode) == 65:
                self.items.append(episode)

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        episode = self.items[index]
        x, y = [], []

        # get only the desired fields in config.yaml
        for state in episode:
            state_actions = [state[key] for key in self.keys]
            x.append(np.concatenate(state_actions, axis=0))
            y.append(state["task"])

        return torch.tensor(np.array(x), dtype=torch.float), torch.tensor(np.array(y[0]), dtype=torch.long)
    
    def _encode(self, string):
        return self.stoi[string]

    def _decode(self, id):
        return self.itos[id]




