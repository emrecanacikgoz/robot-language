from argparse import ArgumentParser
import gzip
import os
import re
import sys

import gradio as gr
import numpy as np
from PIL import Image


def float2uint8(a: np.ndarray) -> np.ndarray:
    return (255 * (a - a.min()) / (a.max() - a.min() + sys.float_info.epsilon)).astype(
        np.uint8
    )


def array2image(a: np.ndarray):
    if np.issubdtype(a.dtype, np.floating):
        a = float2uint8(a)
    return Image.fromarray(a)


def numeric_fields(npz, idnum: int):
    """Format numerical fields in npz and text as a single string"""
    index = int(idnum)
    scene = ""
    if scenes is not None:
        for key, (low, high) in scenes.items():
            if low <= index <= high:
                scene = key[-1]
                break
    ep_start = ""
    ep_end = ""
    if episodes is not None:
        for low, high in episodes:
            if low <= index <= high:
                ep_start = low
                ep_end = high
                break
    indexstr = f"frame: {idnum} ({scene}:{ep_start}:{ep_end})"

    a = npz["actions"]
    actions = (
        "act: "
        f" x:{a[0]: 6.3f} y:{a[1]: 6.3f} z:{a[2]: 6.3f} a:{a[3]: 6.3f} b:{a[4]: 6.3f} "
        f"c:{a[5]: 6.3f} grp:{a[6]: 6.3f}"
    )
    b = npz["rel_actions"]
    rel_actions = (
        "rel: "
        f" x:{b[0]: 6.3f} y:{b[1]: 6.3f} z:{b[2]: 6.3f} a:{b[3]: 6.3f} b:{b[4]: 6.3f} "
        f"c:{b[5]: 6.3f} grp:{b[6]: 6.3f}"
    )
    c = npz["robot_obs"]
    robot_obs = (
        "tcp: "
        f" x:{c[0]: 6.3f} y:{c[1]: 6.3f} z:{c[2]: 6.3f} a:{c[3]: 6.3f} b:{c[4]: 6.3f} "
        f"c:{c[5]: 6.3f} grp:{c[6]*100: 6.3f}"
    )
    robot_arm = (
        "arm: "
        f" a:{c[7]: 6.3f} b:{c[8]: 6.3f} c:{c[9]: 6.3f} d:{c[10]: 6.3f} e:{c[11]: 6.3f}"
        f" f:{c[12]: 6.3f} g:{c[13]: 6.3f} grp:{c[14]: 6.3f}"
    )
    d = npz["scene_obs"]
    red = (
        "red: "
        f" x:{d[6]: 6.3f} y:{d[7]: 6.3f} z:{d[8]: 6.3f} a:{d[9]: 6.3f} b:{d[10]: 6.3f} "
        f"c:{d[11]: 6.3f}"
    )
    blue = (
        "blue:"
        f" x:{d[12]: 6.3f} y:{d[13]: 6.3f} z:{d[14]: 6.3f} a:{d[15]: 6.3f} "
        f"b:{d[16]: 6.3f} c:{d[17]: 6.3f}"
    )
    pink = (
        "pink:"
        f" x:{d[18]: 6.3f} y:{d[19]: 6.3f} z:{d[20]: 6.3f} a:{d[21]: 6.3f} "
        f"b:{d[22]: 6.3f} c:{d[23]: 6.3f}"
    )
    desk = (
        f"door:{d[0]: 6.3f} drawer:{d[1]: 6.3f} button:{d[2]: 6.3f} switch:{d[3]: 6.3f}"
        f" bulb:{d[4]: 6.3f} green:{d[5]: 6.3f}"
    )
    ann = []
    prev = ""
    if annotations is not None:
        curr_tasks = {}
        for _, ((low, high), t, s) in enumerate(annotations):
            if index > high:
                prev = f"<{low}:{high}:{t}: {s}"
            elif low <= index <= high:
                if not ann and prev:
                    ann.append(prev)
                if t not in curr_tasks:
                    ann.append(f"={low}:{high}:{t}: {s}")
                    curr_tasks[t] = True
            elif index < low:
                if not ann and prev:
                    ann.append(prev)
                ann.append(f">{low}:{high}:{t}: {s}")
                break

    if text is not None and index in text:
        ann.extend(text[index])

    for i in range(len(ann)):
        if len(ann[i]) > 78:
            ann[i] = ann[i][:78]

    return "\n".join(
        (
            indexstr,
            actions,
            rel_actions,
            robot_obs,
            robot_arm,
            red,
            blue,
            pink,
            desk,
            *ann,
        )
    )


def read_annotations(args):
    if args.lang is None:
        annotfile = args.dir + "/lang_annotations/auto_lang_ann.npy"
    else:
        annotfile = args.lang
    if os.path.exists(annotfile):
        print(f"Reading {annotfile}", file=sys.stderr)
        if annotfile.endswith(".npy"):
            annotations = np.load(annotfile, allow_pickle=True).item()
            annotations = sorted(
                list(
                    zip(
                        annotations["info"]["indx"],
                        annotations["language"]["task"],
                        annotations["language"]["ann"],
                    )
                )
            )
        elif annotfile.endswith(".tsv.gz"):
            annotations = []
            with gzip.open(annotfile, "rt") as f:
                for line in f:
                    x = line.strip().split("\t")
                    annotations.append(((int(x[0]), int(x[1])), x[2], x[3]))
        else:
            os.error(f"Unknown extension: {annotfile}")
        print(f"Found {len(annotations)} annotations", file=sys.stderr)
    else:
        print(
            f"{annotfile} does not exist, annotations will not be displayed",
            file=sys.stderr,
        )
        annotations = None
    return annotations


def read_episodes(directory: str):
    """Read episode boundaries"""
    episodefile = directory + "/ep_start_end_ids.npy"
    if os.path.exists(episodefile):
        print(f"Reading {episodefile}", file=sys.stderr)
        episodes = sorted(np.load(episodefile, allow_pickle=True).tolist())
        print(f"Found {len(episodes)} episodes")
    else:
        print(
            f"{episodefile} does not exist, episode boundaries will not be displayed",
            file=sys.stderr,
        )
        episodes = None
    return episodes


def read_scenes(directory: str):
    """Read scene info"""
    scenefile = directory + "/scene_info.npy"
    if os.path.exists(scenefile):
        print(f"Reading {scenefile}", file=sys.stderr)
        scenes = np.load(scenefile, allow_pickle=True).item()
    else:
        print(
            f"{scenefile} does not exist, scene ids will not be displayed",
            file=sys.stderr,
        )
        scenes = None
    return scenes


def read_text(path):
    """Read additional text to display, e.g. predictions. Should be tsv with first
    col = frame id
    """
    if os.path.exists(path):
        print(f"Reading {path}", file=sys.stderr)
        text = {}
        with open(path, "rt", encoding="utf-8") as f:
            for line in f:
                cols = line.split("\t")
                idx = int(cols[0])
                if idx in text:
                    text[idx].extend(cols[1:])
                else:
                    text[idx] = cols[1:]
    else:
        print(
            f"{path} does not exist, predictions will not be displayed", file=sys.stderr
        )
        text = None
    return text


def read_dir(directory: str):
    """Find episode-XXXXXXX.npz files in dir and return their ids"""
    idnums = []
    iddict = {}
    print(f"Reading directory {directory}", file=sys.stderr)
    for f in sorted(os.listdir(directory)):
        m = re.match(r"episode_(\d{7})\.npz", f)
        if m is not None:
            idnum = m.group(1)
            iddict[idnum] = len(idnums)
            idnums.append(idnum)
    print(f"Found {len(idnums)} frames.", file=sys.stderr)
    return idnums, iddict


if __name__ == "__main__":
    parser = ArgumentParser(description="Interactive visualization of CALVIN dataset")
    parser.add_argument(
        "-d",
        "--dir",
        default=".",
        type=str,
        help="Path to dir containing episode_XXXXXXX.npz files",
    )
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        help="Path to tsv file containing additional text, (1st column=frame id).",
    )
    parser.add_argument(
        "-l",
        "--lang",
        type=str,
        help="(Optional) path to language file, default is to read it from --dir.",
    )
    args = parser.parse_args()

    idnums, iddict = read_dir(args.dir)
    if len(idnums) == 0:
        sys.exit(f"Error: Could not find any episode files in {args.dir}.")
    annotations = read_annotations(args)
    episodes = read_episodes(args.dir)
    scenes = read_scenes(args.dir)
    if args.text is not None:
        text = read_text(args.text)
    else:
        text = None

    with gr.Blocks() as demo:
        with gr.Row():
            rgb_static = gr.Image(interactive=False, label="rgb_static")
            depth_static = gr.Image(interactive=False, label="depth_static")
            rgb_gripper = gr.Image(interactive=False, label="rgb_gripper")
            depth_gripper = gr.Image(interactive=False, label="depth_gripper")

        with gr.Row():
            rgb_tactile1 = gr.Image(interactive=False, label="rgb_tactile1")
            rgb_tactile2 = gr.Image(interactive=False, label="rgb_tactile2")
            depth_tactile1 = gr.Image(interactive=False, label="depth_tactile1")
            depth_tactile2 = gr.Image(interactive=False, label="depth_tactile2")

        with gr.Row():
            text_info = gr.Text(label="text")

        with gr.Row():
            slider = gr.Slider(
                label="slider", minimum=0, maximum=len(idnums) - 1, step=1, value=0
            )

        def update_frame(value):
            index = int(value)
            idnum = idnums[index]
            npz = np.load(f"{args.dir}/episode_{idnum}.npz", allow_pickle=True)
            return (
                array2image(npz["rgb_static"]),
                array2image(npz["depth_static"]),
                array2image(npz["rgb_gripper"]),
                array2image(npz["depth_gripper"]),
                array2image(npz["rgb_tactile"][:, :, 0:3]),
                array2image(npz["rgb_tactile"][:, :, 3:6]),
                array2image(npz["depth_tactile"][:, :, 0]),
                array2image(npz["depth_tactile"][:, :, 1]),
                numeric_fields(npz, idnum),
            )

        slider.change(
            fn=update_frame,
            inputs=[slider],
            outputs=[
                rgb_static,
                depth_static,
                rgb_gripper,
                depth_gripper,
                rgb_tactile1,
                rgb_tactile2,
                depth_tactile1,
                depth_tactile2,
                text_info,
            ],
        )
        demo.load(lambda: 1, inputs=None, outputs=slider)

    demo.launch(server_name="0.0.0.0")
