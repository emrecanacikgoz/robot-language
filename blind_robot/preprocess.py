import glob
import logging
import os
import re
from typing import AnyStr, Iterable, Pattern

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


class CalvinPreprocessor:
    """Abstract class for the Calvin dataset."""

    directory: str = "/raid/lingo/data/calvin/"
    episode_regex: Pattern[AnyStr] = re.compile(r"episode_(\d{7})\.npz")

    def extract_language(
        self, subdirectory: str = "task_D_D/", split: str = "validation"
    ):
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

    def extract_scene_info(self):
        """Extract scene info from the annotation files."""
        splits = glob.glob(os.path.join(self.directory, "**/training/"))
        splits += glob.glob(os.path.join(self.directory, "**/validation/"))
        for directory in splits:
            for file in ["scene_info.npy", "ep_lens.npy", "ep_start_end_ids.npy"]:
                path = os.path.join(directory, file)
                if os.path.isfile(path):
                    yield path
                    data = np.load(path, allow_pickle=True)
                    yield data

    def extract_features(self, subdirectory: str = "debug", split: str = "validation"):
        """Extract numbers from the annotation files."""
        directory = os.path.join(self.directory, subdirectory, split)
        for file in tqdm(sorted(os.listdir(directory))):
            match = self.episode_regex.match(file)
            if match is not None:
                idx = match.group(1)
                path = os.path.join(directory, file)
                data = np.load(path, allow_pickle=True, mmap_mode="r")
                actions = data["actions"]  # 1-7(7)
                rel_actions = data["rel_actions"]  # 8-14(7)
                robot_obs = data["robot_obs"]  # 15-29(15)
                scene_obs = data["scene_obs"]  # 30-53(24)
                depth_tactile = data["depth_tactile"].mean(axis=(0, 1)) * 100.0
                rgb_tactile = data["rgb_tactile"].mean(axis=(0, 1)) / 255.0
                yield (
                    idx,
                    *np.concatenate(
                        (
                            actions,
                            rel_actions,
                            robot_obs,
                            scene_obs,
                            depth_tactile,
                            rgb_tactile,
                        )
                    ),
                )

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

    def regression_for_controller(self, data: Iterable):
        # Load data from output of calvin_controller_coordinates
        data = pd.DataFrame(list(data))

        # Output some stats:
        print(data.describe().to_string())

        # Separate the independent and dependent variables
        inputs = data.iloc[:, 1:2].values
        x = data.iloc[:, 2].values
        y = data.iloc[:, 3].values
        z = data.iloc[:, 4].values

        # Create a linear regression model and fit the data
        xmodel = LinearRegression().fit(inputs, x)
        ymodel = LinearRegression().fit(inputs, y)
        zmodel = LinearRegression().fit(inputs, z)

        # Print the coefficients of the linear regression model
        print(f"x = {xmodel.coef_[0]}*input {xmodel.intercept_:+.6f}")
        print(f"y = {ymodel.coef_[0]}*input {ymodel.intercept_:+.6f}")
        print(f"z = {zmodel.coef_[0]}*input {zmodel.intercept_:+.6f}")


if __name__ == "__main__":
    import itertools

    def test():
        preprocessor = CalvinPreprocessor()

        subdirectory = "task_D_D"
        split = "training"

        data = preprocessor.extract_features(subdirectory=subdirectory, split=split)
        for d in itertools.islice(data, 10):
            print(d, sep="\t")

        data = preprocessor.extract_scene_info()
        for info in data:
            print(info)

        data = preprocessor.extract_language(subdirectory=subdirectory, split=split)
        for d in itertools.islice(data, 10):
            print(*d, sep="\t")

        data = preprocessor.extract_features(subdirectory=subdirectory, split=split)
        data = itertools.islice(data, 10)
        for d in preprocessor.get_intervals(data=data):
            print(*d, sep="\t")

        data = preprocessor.extract_features(subdirectory=subdirectory, split=split)
        data = list(itertools.islice(data, 1000))

        preprocessor.regression_for_controller(
            preprocessor.get_controller_coordinates(data, field=30)
        )

        preprocessor.regression_for_controller(
            preprocessor.get_controller_coordinates(data, field=31)
        )

        preprocessor.regression_for_controller(
            preprocessor.get_controller_coordinates(data, field=32)
        )

        preprocessor.regression_for_controller(
            preprocessor.get_controller_coordinates(data, field=33)
        )

    test()
