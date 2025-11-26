"""
Created on Mon Nov 17 22:20:42 2025.

@author: Alexandros Papagiannakis, HHMI @Stanford University, 2025
"""

import pickle

import numpy as np


def load_drift_statistics(drift_save_path):
    """Get pre-computed drift statistics from UnDrift."""
    with open(drift_save_path+'/drift_statistics', 'rb') as handle:
        rect_crop, cum_drifts = pickle.load(handle)
    return rect_crop, cum_drifts



def apply_drift_correction(masks_dict, cum_drift_x, cum_drift_y):
    """Apply drift correction from UnDrift to segmentations.

    You can find UnDrift at github.com/alexSysBio/UnDrift.

    """
    aligned = {}

    keys = sorted(masks_dict.keys())
    shapes = [masks_dict[k].shape for k in keys]
    H, W = shapes[0][0], shapes[0][1]

    iy = cum_drift_y
    ix = cum_drift_x

    y0 = int(np.max(iy))
    y1 = int(H + np.min(iy))
    x0 = int(np.max(ix))
    x1 = int(W + np.min(ix))

    for i, k in enumerate(keys):
        mask = masks_dict[k]
        sy0 = y0 - iy[i]
        sy1 = y1 - iy[i]
        sx0 = x0 - ix[i]
        sx1 = x1 - ix[i]
        aligned[k] = mask[sy0:sy1, sx0:sx1]

    return aligned


