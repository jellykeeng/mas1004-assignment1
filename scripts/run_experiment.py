"""Run the supplied experiment with readable Korean plots on macOS."""

import argparse
import sys
import unicodedata
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fonts
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Matplotlib needs a Korean font; filesystem names on macOS are decomposed.
font_path = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
if font_path.exists():
    fonts.fontManager.addfont(str(font_path))
    matplotlib.rcParams["font.family"] = fonts.FontProperties(fname=str(font_path)).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

import evaluate

original_confusion = evaluate.plot_confusion
original_worst = evaluate.plot_worst


def plot_confusion(matrix, class_names, out_path):
    names = [unicodedata.normalize("NFC", name) for name in class_names]
    return original_confusion(matrix, names, out_path)


def plot_worst(mistakes, class_names, out_path, ncols=5):
    names = [unicodedata.normalize("NFC", name) for name in class_names]
    return original_worst(mistakes, names, out_path, ncols=ncols)


evaluate.plot_confusion = plot_confusion
evaluate.plot_worst = plot_worst

from run import main


if __name__ == "__main__":
    # The given --seed already controls the split; seed model initialization too.
    seed_parser = argparse.ArgumentParser(add_help=False)
    seed_parser.add_argument("--seed", type=int, default=0)
    seed_args, _ = seed_parser.parse_known_args()
    torch.manual_seed(seed_args.seed)
    main()
