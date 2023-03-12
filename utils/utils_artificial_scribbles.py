import numpy as np
from scipy import ndimage
from skimage import morphology

def generate_scribble_fn(lab, num_classes, ignored_index):
    """

    :param lab: np.array, (H, W);  thelabel that is to be transformed to artificial scribbles
    :param num_classes: int, the number of classes (including the background class and the foreground class(es))
    :param ignored_index: int, the index that indicates unlabeled regions
    :return: np.array, (H, W), the artificial scribbles
    """
    h, w = lab.shape
    lab_oh = np.zeros((num_classes, h, w))
    scb_oh = np.zeros_like(lab_oh)
    for c in range(num_classes):
        lab_oh[c][lab == c] = 1
        ske = morphology.skeletonize(lab_oh[c])
        # Dilation makes the task too easy
        # ske = morphology.dilation(ske)
        ske = ske * lab_oh[c]
        scb_oh[c] = ske
    ignored_region = 1 - np.sum(scb_oh, axis=0, keepdims=True)
    scb_oh = np.concatenate([scb_oh, ignored_region], axis=0)

    # Given the operation above, the aritifical scribble of an images containing only background is a point.
    # The following operation extends the point to a line, which is more similar to a scribble.
    scb_bg = scb_oh[0]
    if set(np.unique(np.argmax(scb_oh, axis=0))) == {0, ignored_index}:
        scb_bg = ndimage.binary_dilation(scb_oh[0], np.eye(3)[::-1], iterations=40, mask=lab_oh[0])
        scb_bg = morphology.skeletonize(scb_bg)
    scb_oh[0] = scb_bg

    scb = np.argmax(scb_oh, axis=0)
    return scb