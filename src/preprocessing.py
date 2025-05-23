import numpy as np
import scipy.ndimage as ndimage

def normalize(volume):
    volume = np.clip(volume, -1000, 400)  # HU windowing
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    return volume

def resample(volume, old_spacing, new_spacing=[1, 1, 1]):
    resize_factor = np.array(old_spacing) / np.array(new_spacing)
    new_shape = np.round(volume.shape * resize_factor)
    return ndimage.zoom(volume, resize_factor, order=1)
