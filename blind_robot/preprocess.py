from concurrent.futures import ProcessPoolExecutor
import functools
import logging
import os
import pickle
import re
from typing import Any, AnyStr, Iterable, List, Mapping, Pattern

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from blind_robot.utils.data_utils import fieldnames as FIELD_NAMES
from blind_robot.utils.data_utils import int2task as TASK_LABELS


class CalvinPreprocessor:
    """Preprocessing utility the Calvin dataset.
    Reference: github.com/denizyuret/calvin-scripts
    """

    directory: str = "/raid/lingo/data/calvin/"
    episode_regex: Pattern[AnyStr] = re.compile(r"episode_(\d{7})\.npz")
    max_workers: int = 16

    def extract_language(self, subdirectory: str = "D", split: str = "validation"):
        """Extract language data from the annotation files."""
        file = os.path.join(
            self.directory, subdirectory, split, "lang_annotations", "auto_lang_ann.npy"
        )
        logging.info("Loading language data from: %s", file)
        annotations = np.load(file, allow_pickle=True).item()

        assert len(annotations["info"]["indx"]) == len(annotations["language"]["task"])
        assert len(annotations["info"]["indx"]) == len(annotations["language"]["ann"])

        for index, task_id, task_str in zip(
            annotations["info"]["indx"],
            annotations["language"]["task"],
            annotations["language"]["ann"],
        ):
            yield (*index, task_id, task_str)

    def _swap(
        self, features: List, metadata: Mapping[str, Any], subdirectory: str, split: str
    ):
        """Patch1: Swap the object locations for bug fix in the original data."""
        if split == "training" and "ABC" in subdirectory:
            episode_id = features[0]
            c_start, c_stop = metadata["scene_info"]["calvin_scene_C"]
            a_start, a_stop = metadata["scene_info"]["calvin_scene_A"]
            if c_start <= episode_id <= c_stop:
                temp = features[36:42].copy()
                features[36:42] = features[42:48]
                features[42:48] = temp
            elif a_start <= episode_id <= a_stop:
                temp = features[36:42].copy()
                features[36:42] = features[48:54]
                features[48:54] = temp

        return features

    def _coordinates(
        self, features: List, metadata: Mapping[str, Any], subdirectory: str, split: str
    ):
        """Patch2: Add the object coordinates to the original data."""
        episode_id = features[0]
        scene = "D"

        if split == "training" and subdirectory != "D":
            for name in ["A", "B", "C", "D"]:
                interval = metadata["scene_info"].get(f"calvin_scene_{name}")
                if interval:
                    start, stop = interval
                    if start <= episode_id <= stop:
                        scene = name

        slider, drawer, button, switch = features[30:34]

        if scene == "A":
            slider_xyz = [0.04 - slider, 0.00, 0.53]
            drawer_xyz = [0.10, -0.20 - drawer, 0.36]
            button_xyz = [-0.28, -0.10, 0.5158 - 1.7591 * button]
            switch_xyz = [0.30, 0.3413 * switch + 0.0211, 0.5470 * switch + 0.5410]
        elif scene == "B":
            slider_xyz = [0.23 - slider, 0.00, 0.53]
            drawer_xyz = [0.18, -0.20 - drawer, 0.36]
            button_xyz = [0.28, -0.12, 0.5158 - 1.7591 * button]
            switch_xyz = [-0.32, 0.3413 * switch + 0.0211, 0.5470 * switch + 0.5410]
        elif scene == "C":
            slider_xyz = [0.20 - slider, 0.00, 0.53]
            drawer_xyz = [0.10, -0.20 - drawer, 0.36]
            button_xyz = [-0.12, -0.12, 0.5158 - 1.7591 * button]
            switch_xyz = [-0.32, 0.3413 * switch + 0.0211, 0.5470 * switch + 0.5410]
        elif scene == "D":
            slider_xyz = [0.04 - slider, 0.00, 0.53]
            drawer_xyz = [0.18, -0.20 - drawer, 0.36]
            button_xyz = [-0.12, -0.12, 0.5158 - 1.7591 * button]
            switch_xyz = [0.30, 0.3413 * switch + 0.0211, 0.5470 * switch + 0.5410]

        return features + slider_xyz + drawer_xyz + button_xyz + switch_xyz

    def _post_process(
        self, features: List, metadata: Mapping[str, Any], subdirectory: str, split: str
    ):
        """Postprocess the features. Fixes swapping bug, and adds coordinates."""
        features = self._swap(features, metadata, subdirectory, split)
        features = self._coordinates(features, metadata, subdirectory, split)
        return features

    def extract_scene_info(self, subdirectory: str, split: str):
        """Extract scene info from the annotation files."""
        info = {}
        directory = os.path.join(self.directory, subdirectory, split)
        for file in ["scene_info.npy", "ep_lens.npy", "ep_start_end_ids.npy"]:
            path = os.path.join(directory, file)
            feature = file.replace(".npy", "")
            if os.path.isfile(path):
                data = np.load(path, allow_pickle=True)
                info[feature] = data.tolist()
        return info

    def _read_features(self, idx: int, path: str):
        """Read the features from the npy file."""
        data = np.load(path, allow_pickle=True, mmap_mode="r")
        actions = data["actions"]  # 1-7(7)
        rel_actions = data["rel_actions"]  # 8-14(7)
        robot_obs = data["robot_obs"]  # 15-29(15)
        scene_obs = data["scene_obs"]  # 30-53(24)
        depth_tactile = data["depth_tactile"].mean(axis=(0, 1)) * 100.0
        rgb_tactile = data["rgb_tactile"].mean(axis=(0, 1)) / 255.0
        features = [
            idx,
            *actions,
            *rel_actions,
            *robot_obs,
            *scene_obs,
            *depth_tactile,
            *rgb_tactile,
        ]
        return features

    def extract_features(self, subdirectory: str = "debug", split: str = "validation"):
        """Extract numbers from the annotation files."""
        directory = os.path.join(self.directory, subdirectory, split)
        metadata = self.extract_scene_info(subdirectory, split)
        process = functools.partial(
            self._post_process,
            metadata=metadata,
            subdirectory=subdirectory,
            split=split,
        )

        buffer = []

        executer = ProcessPoolExecutor(max_workers=self.max_workers)

        for file in tqdm(sorted(os.listdir(directory))):
            match = self.episode_regex.match(file)
            if match is not None:
                idx = int(match.group(1))
                path = os.path.join(directory, file)
                future = executer.submit(self._read_features, idx, path)
                buffer.append(future)

                if len(buffer) >= 10 * self.max_workers:
                    for future in buffer:
                        yield process(future.result())
                    buffer = []

        if buffer:
            for future in buffer:
                yield process(future.result())

    def get_intervals(self, data: Iterable):
        interval_start = -1
        last_index = -1
        for fields in data:
            index = int(fields[0])
            if interval_start < 0:
                interval_start = last_index = index
            elif index == last_index + 1:
                last_index = index
            else:
                yield (interval_start, last_index)
                interval_start = last_index = index

        yield (interval_start, last_index)

    def get_controller_coordinates(
        self,
        data: Iterable,
        field: int = 30,
        threshold: float = 0.0001,
        buffer_size: int = 10,
    ):
        previous_value = None
        buffer = []

        for fields in data:
            value = fields[field]
            if not previous_value:
                previous_value = value
                continue
            elif abs(value - previous_value) > threshold:
                buffer.append((fields[0], value, fields[15], fields[16], fields[17]))
                previous_value = value
            else:
                if len(buffer) > buffer_size:
                    for d in buffer:
                        yield d
                buffer = []

        if len(buffer) > buffer_size:
            for d in buffer:
                yield d
                buffer = []

    def _normalize(self, diff):
        meter_fields = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 18, 19, 20]
        radian_fields = [9, 10, 11, 15, 16, 17, 21, 22, 23]
        # boolean_fields = [4, 5]
        for i in meter_fields:
            diff[i] = np.clip(diff[i], -0.02, 0.02) / 0.02  # 0.02m = 2cm = 1
            if abs(diff[i]) < 1e-3:  # clean up noise < 0.02 mm
                diff[i] = 0
        for i in radian_fields:
            diff[i] = (
                np.clip((diff[i] + np.pi) % (2 * np.pi) - np.pi, -0.05, 0.05) / 0.05
            )  # 0.05 rad = 2.86 deg = 1
            if abs(diff[i]) < 1e-2:  # clean up noise < 0.02 degrees
                diff[i] = 0
        return diff

    def get_scene_differences(self, data: np.ndarray, metadata: Mapping[str, Any]):
        episode_start_ends = metadata["ep_start_end_ids"]
        episode_ends = set()
        for _, end in episode_start_ends:
            episode_ends.add(end)

        for i in tqdm(range(data.shape[0])):
            curr_frame = data[i, 0]
            curr_scene = data[i, 30:54]
            if curr_frame in episode_ends:
                diff = np.zeros(len(curr_scene), dtype="float32")
            else:
                next_scene = data[i + 1, 30:54]
                diff = self._normalize(next_scene - curr_scene)

            yield diff

    def regression_for_controller(self, data: Iterable):
        # Load data from output of calvin_controller_coordinates
        data = pd.DataFrame(list(data))

        # Separate the independent and dependent variables
        inputs = data.iloc[:, 1:2].values
        x = data.iloc[:, 2].values
        y = data.iloc[:, 3].values
        z = data.iloc[:, 4].values

        # Create a linear regression model and fit the data
        x_model = LinearRegression().fit(inputs, x)
        y_model = LinearRegression().fit(inputs, y)
        z_model = LinearRegression().fit(inputs, z)

        # Print the coefficients of the linear regression model
        print(f"x = {x_model.coef_[0]}*u {x_model.intercept_}")
        print(f"y = {y_model.coef_[0]}*u {y_model.intercept_}")
        print(f"z = {z_model.coef_[0]}*u {z_model.intercept_}")

    def pipeline(self, subdirectory: str, split: str):
        metadata = self.extract_scene_info(subdirectory, split)

        language_data = list(self.extract_language(subdirectory, split))

        language_start_ids = np.array([d[0] for d in language_data], dtype=int)
        language_end_ids = np.array([d[1] for d in language_data], dtype=int)
        language_task_labels = np.array([d[2] for d in language_data], dtype=str)
        language_instructions = np.array([d[3] for d in language_data], dtype=str)
        language = np.concatenate(
            (
                language_start_ids,
                language_end_ids,
                language_task_labels,
                language_instructions,
            ),
            axis=1,
        )

        features = np.array(
            list(self.extract_features(subdirectory, split)), dtype=np.float32
        )
        scene_differences = np.array(
            list(self.get_scene_differences(features, metadata))
        )

        features = np.concatenate((features, scene_differences), axis=1)

        frame_ids, features = features[:, 0].astype(int), features[:, 1:]

        data = {
            "features": features,
            "language": language,
            "frame_ids": frame_ids,
            "task_names": TASK_LABELS,
            "field_names": FIELD_NAMES,
            "metadata": metadata,
        }

        return data


if __name__ == "__main__":
    import itertools

    def test():
        preprocessor = CalvinPreprocessor()

        subdirectory = "ABC"
        split = "training"

        data = preprocessor.extract_features(subdirectory, split)

        for d in itertools.islice(data, 10):
            print(d, sep="\t")

        info = preprocessor.extract_scene_info(subdirectory, split)

        print(info)

        data = preprocessor.extract_language(subdirectory, split)
        for d in itertools.islice(data, 10):
            print(*d, sep="\t")

        data = preprocessor.extract_features(subdirectory, split)
        data = itertools.islice(data, 10)
        for d in preprocessor.get_intervals(data=data):
            print(*d, sep="\t")

    def pipeline():
        preprocessor = CalvinPreprocessor()
        for subdirectory in ["D", "ABC", "ABCD"]:
            for split in [
                "training",
                "validation",
            ]:
                data = preprocessor.pipeline(subdirectory=subdirectory, split=split)
                with open(file=f"data/{subdirectory}_{split}.pkl", mode="wb") as f:
                    pickle.dump(data, f)

    test()
    # pipeline()
