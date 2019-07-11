import numpy as np


class Frame:
    def __init(self, img: np.ndarray, number: np.uint64, new_map=False, weight_mask=None, binary_map=None):
        self.img = img
        self.num = number
        self.new_map = new_map
        self.wm = weight_mask
        self.bm = binary_map
        self.ROI = None
