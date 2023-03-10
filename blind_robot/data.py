import sys

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

from blind_robot.utils.data_utils import dtype_lang
from blind_robot.utils.data_utils import int2task


class CalvinDataset(Dataset):
    def __init__(self, path, config):
        super().__init__()
        self.path = path
        self.keys = config.data["keys"]
        self._load_data(path=path, window=config.data["window"])

    def _load_state(self, path=None):
        print(f"Loading {path}...", file=sys.stderr)

        # load base data (53-dimensional)
        with open(path + '.tsv', 'rt') as f:
            data = np.loadtxt(f, delimiter='\t', dtype='float32')

        # load controller data (12-dimensional)
        with open(path + '-controllers.tsv', 'rt') as f:
            cont = np.loadtxt(f, delimiter='\t', dtype='float32')
            assert np.array_equal(data[:,0], cont[:,0]), 'cont indices do not match'

        # load tactile data (8-dimensional)
        with open(path + '-tactile2.tsv', 'rt') as f:
            tact = np.loadtxt(f, delimiter='\t', dtype='float32')
            assert np.array_equal(data[:,0], tact[:,0]), 'tact indices do not match'

        # create 73-dimensional instances and normalize
        data = np.concatenate((data, cont[:,1:], tact[:,1:]), axis=1)
        data = self._normalize(data)

        pos2id = data[:,0].astype(int)
        id2pos = np.full(1+max(pos2id), -1)
        for (pos, id) in enumerate(pos2id):
            id2pos[id] = pos

        return data, pos2id, id2pos

    def _load_language(self, path=None):
        print(f"Loading {path}-lang...", file=sys.stderr)
        with open(path + '-lang.tsv', 'rt') as f:
            lang = np.loadtxt(f, delimiter='\t', dtype=dtype_lang)
        lang.sort(order = 'end')

        task2int = {ch: i for i, ch in enumerate(int2task)}
        print(f"task2int: {task2int}")
        for task in lang['task']:
            assert task in task2int, 'task not found'
        return lang, task2int, int2task

    def _load_data(self, path, window=64, features=range(1,74)):

        # get action and language data
        data, _pos2id, id2pos = self._load_state(path=path)
        lang, task2int, _int2task = self._load_language(path=path)

        # create instances
        self.items = []
        for (_, idx, task, _annot) in tqdm(lang):
            taskID = task2int[task]
            pos = id2pos[idx]
            sample = {
                "source": np.ravel(data[np.ix_(range(pos-window+1, pos+1), features)]),
                "target": taskID,
                "idx": idx
            }
            self.items.append(sample)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        episode = self.items[index]

        # convert narray2tensor
        source = torch.tensor(episode["source"], dtype=torch.float)
        target = torch.tensor(episode["target"], dtype=torch.long)

        return source, target

    def _normalize(self, data):
        # button and switch
        data[:, 32:34] = data[:, 32:34] * 10.0
        # gripper opening
        data[:, 21] = data[:, 21] * 10.0

        return data


class CalvinDatasetGPT(Dataset):
    def __init__(self, data=None, max_length=None, keys=None):
        super().__init__()
        self.data = data
        self.keys = keys
        self.max_length = max_length
        self._load_data()

    def _load_data(self):
        print(
            f"Loading the {self.data.split('/')[-1].split('-')[-1]} data from path"
            f" {self.data}:"
        )

        # read data
        states = []
        with open(self.data, "r", encoding="utf-8") as f:
            for line in tqdm(f):

                # set data fields
                l = line.rstrip("\n").split("\t")
                _, state = l[0], [float(i) for i in l[1:]]
                fields = {
                    "actions": state[0:7],
                    "rel_actions": state[7:14],
                    "robot_obs_tcp_position": state[14:17],
                    "robot_obs_tcp_orientation": state[17:20],
                    "robot_obs_gripper_opening_width": state[20:21],
                    "robot_obs_arm_joint_states": state[21:28],
                    "robot_obs_gripper_action": state[28:29],
                    "scene_obs": state[29:],
                }

                # get only the desired fields in config.yaml
                desired_state = []
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
        # set source and target for next-token prediction; i.e.,
        # auto-regressive decoder training
        x, y = data[: self.max_length], data[1 : self.max_length + 1]
        return (x, y)

    def _split_equal_lengths(self, states):
        print(f"Splitting states to {self.max_length} equal episodes:")
        episodes = [
            states[i : i + (self.max_length + 1)]
            for i in tqdm(range(0, len(states), (self.max_length + 1)))
        ]
        episodes.pop()
        return episodes

    def _quantize(self, source):
        raise NotImplementedError




