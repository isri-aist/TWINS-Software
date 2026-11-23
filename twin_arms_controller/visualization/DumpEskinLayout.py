#!/usr/bin/env python3

import argparse
import os

import yaml

from twin_arms_controller.envs.common import TwinArmsEskinReceiver
from twin_arms_controller.visualization.EskinLayout import save_eskin_layout


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--patch_names", nargs="+", default=None)
    parser.add_argument(
        "--config",
        default=None,
        help="Env YAML file. eskin_patch_names is read from this file.",
    )
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument(
        "--output",
        required=True,
        help="Output layout JSON path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    patch_names = get_patch_names(args)
    receiver = TwinArmsEskinReceiver(args.endpoint, patch_names, args.timeout)
    try:
        output = os.path.expanduser(args.output)
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        save_eskin_layout(output, receiver.patch_names, receiver.latest_result)
        print(f"Saved eSkin layout: {output}")
    finally:
        receiver.close()


def get_patch_names(args):
    if args.patch_names is not None:
        return args.patch_names
    if args.config is not None:
        with open(os.path.expanduser(args.config), "r") as f:
            config = yaml.safe_load(f)
        patch_names = config.get("eskin_patch_names")
        if patch_names:
            return patch_names
    raise ValueError("Specify --patch_names or --config with eskin_patch_names.")


if __name__ == "__main__":
    main()
