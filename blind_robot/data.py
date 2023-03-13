import pickle
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CalvinDataset(Dataset):
    def __init__(self, path, config):
        super().__init__()
        self.path = path
        self.keys = config.data["keys"]
        self._load_data(path=path, window=config.data["window"])

    def _load_state(self, path=None):
        print(f"Loading {path}...", file=sys.stderr)

        # load data
        with open(path, "rb") as handle:
            data = pickle.load(handle)

        # normalize features
        data["features"] = self._normalize(data["features"])

        frame_ids = data["frame_ids"]

        frame_id_to_index = np.full(1 + max(frame_ids), -1)

        frame_id_to_index[frame_ids] = np.arange(len(frame_ids))

        return data, frame_id_to_index

    def _load_language(self, data):
        vocabulary = {label: index for index, label in enumerate(data["task_names"])}

        print(f"vocabulary: {vocabulary}")

        language_data = data["language"]

        for label in language_data:
            assert label[2] in vocabulary, "task not found"

        return language_data, vocabulary

    def _load_data(self, path, window=64, features=range(1, 98)):
        # load data
        data, frame_id_to_index = self._load_state(path=path)

        feature_data = data["features"]

        language_data, vocabulary = self._load_language(data)

        # TODO(ekin): assert max episode of language_data and feature_data

        self.items = []
        for index in tqdm(range(len(language_data))):
            start_episode_id = language_data[index][0]
            stop_episode_id = language_data[index][1]
            task_label = language_data[index][2]
            instruction = language_data[index][3]

            start_index = frame_id_to_index[start_episode_id]

            stop_index = frame_id_to_index[stop_episode_id]

            start_index = min(stop_index - window + 1, start_index)

            context_idx = range(start_index, stop_index + 1)

            if len(context_idx) == window + 1:
                current_data = feature_data[np.ix_(context_idx, features)]

                sample = {
                    "source": np.ravel(current_data),
                    "target": vocabulary[task_label],
                    "index": index,
                    "instruction": instruction,
                    "start_end_ids": (start_episode_id, stop_episode_id),
                }

                self.items.append(sample)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        episode = self.items[index]

        # convert numpy arrays to torch tensors
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
