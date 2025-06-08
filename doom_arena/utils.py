from typing import Dict
import os
import sys
import re
import numpy as np
import vizdoom as vzd
import torch
import torch.nn.functional as F
from contextlib import contextmanager

from doom_arena.player import player_setup


CHANNELS_FORMAT = {
    vzd.ScreenFormat.CBCGCR: 3,
    vzd.ScreenFormat.CRCGCB: 3,
    vzd.ScreenFormat.GRAY8: 1,
}


def get_screen_shape(
    screen_format,
    screen_resolution,
    labels: bool = False,
    depth: bool = False,
    automap: bool = False,
):
    ch = CHANNELS_FORMAT[screen_format] + int(labels) + int(depth) + 3 * int(automap)
    match = re.search(r"RES_(\d+)X(\d+)", str(screen_resolution))
    return ch, int(match.group(2)), int(match.group(1))


def get_doom_buttons(player_cfg):
    g = vzd.DoomGame()
    g = player_setup(g, player_cfg)
    return g.get_available_buttons()


@contextmanager
def suppress_stdout(verbose):
    if not verbose:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
    else:
        yield


def to_tensor(raw: Dict[str, np.ndarray]):
    def _to_tensor(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x

    return {k: _to_tensor(v) for k, v in raw.items()}


def resize(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    def _resize(x):
        has_time = x.ndim == 4
        if has_time:
            x = x.permute(1, 0, 2, 3)  # (c, t, h, w) -> (t, c, h, w)
        else:
            x = x.unsqueeze(0)  # (c, h, w) -> (1, c, h, w)
        # downsample resolution to fixed size
        x = F.interpolate(x, (128, 128))
        if has_time:
            return x.permute(1, 0, 2, 3)  # (t, c, h, w) -> (c, t, h, w)
        else:
            return x.squeeze(0)  # (1, c, h, w) -> (c, h, w)

    return {k: _resize(v) for k, v in raw.items()}


def minmax(raw: Dict[str, torch.Tensor]):
    def _minmax(x, key):
        if key == "screen":
            # rgb / grayscale (fixed ranges)
            x = x / 255
        else:
            # other
            x = x / (x.max() + 1e-8)
        return torch.nan_to_num(x)

    return {k: _minmax(v, k) for k, v in raw.items()}


def cat_dict(raw: Dict[str, torch.Tensor]):
    # concat along channels
    buffers = ["screen", "labels", "depth", "automap"]
    return torch.cat([raw[k] for k in buffers if k in raw], 0)
