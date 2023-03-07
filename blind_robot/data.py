import sys
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from blind_robot.data_utils import dtype_lang
from blind_robot.data_utils import int2task


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


class CalvinDatasetMLP(Dataset):
    def __init__(self, np_data=None, tsv_data=None, keys=None):
        super().__init__()
        self.np_data = np_data
        self.tsv_data = tsv_data
        self.keys = keys
        self._load_data()

    def _load_data(self):

        # load full data
        data = pd.read_csv(self.tsv_data, sep="\t", header=None)

        # load annotated %1 data
        annotations = np.load(
            f"{self.np_data}/lang_annotations/auto_lang_ann.npy", allow_pickle=True
        ).item()

        # create labels
        labels = sorted(set(annotations["language"]["task"]))
        self.stoi = {ch: i for i, ch in enumerate(labels)}
        self.itos = {i: ch for i, ch in enumerate(labels)}  # pylint: disable=R1721
        print(f"stoi: {self.stoi}")
        print(f"itos: {self.itos}")

        # create datamodule
        annotations = list(
            zip(
                annotations["info"]["indx"],
                annotations["language"]["ann"],
                annotations["language"]["task"],
            )
        )
        self.items = []
        for annotation in tqdm(annotations):
            if (annotation[0][1] - annotation[0][0]) < 64:
                continue
            else:
                indices = list(range(annotation[0][0], annotation[0][1] + 1))
                episode = []
                for _, index in enumerate(indices):
                    state = data.loc[data[0] == index]
                    state = state.to_numpy().squeeze().tolist()
                    episode.append(
                        {
                            "actions": state[1:8],
                            "rel_actions": state[8:15],
                            "robot_obs_tcp_position": state[15:18],
                            "robot_obs_tcp_orientation": state[18:21],
                            "robot_obs_gripper_opening_width": state[21:22],
                            "robot_obs_arm_joint_states": state[22:29],
                            "robot_obs_gripper_action": state[29:30],
                            "scene_obs": state[30:],
                            "language": annotation[1],
                            "task": self._encode(annotation[2]),
                        }
                    )
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

    def _decode(self, idx):
        return self.itos[idx]


class CalvinDatasetMLP2(Dataset):
    def __init__(self, path, config):
        super().__init__()
        self.path = path
        self.keys = config.data["keys"]
        self._load_data(path=path, window=config.data["window"])

    def _load_state(self, path=None):
        print(f"Loading {path}...", file=sys.stderr)
        with open(path + '.tsv', 'rt') as f:
            data = np.loadtxt(f, delimiter='\t', dtype='float32')

        with open(path + '-controllers.tsv', 'rt') as f:
            cont = np.loadtxt(f, delimiter='\t', dtype='float32')
            assert np.array_equal(data[:,0], cont[:,0]), 'cont indices do not match'

        with open(path + '-tactile2.tsv', 'rt') as f:
            tact = np.loadtxt(f, delimiter='\t', dtype='float32')
            assert np.array_equal(data[:,0], tact[:,0]), 'tact indices do not match'

        data = np.concatenate((data, cont[:,1:], tact[:,1:]), axis=1)
        data = self._normalize(data)

        pos2id = data[:,0].astype(int)
        id2pos = np.full(1+max(pos2id), -1)

        for (pos, id) in enumerate(pos2id):
            id2pos[id] = pos

        return data, pos2id, id2pos

    def _load_lang(self, path=None):
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
        data, pos2id, id2pos = self._load_state(path=path)
        lang, task2int, int2task = self._load_lang(path=path)

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

        source = torch.tensor(episode["source"], dtype=torch.float)
        target = torch.tensor(episode["target"], dtype=torch.long)

        return source, target

    def _normalize(self, data):
        data[:, 32:34] = data[:, 32:34] * 10.0  # button and switch
        data[:, 21] = data[:, 21] * 10.0        # gripper opening
        return data

