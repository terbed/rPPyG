import numpy as np
from threading import Lock
from src.frame import Frame
from PyQt5.QtCore import QThread, pyqtSignal


class Buffer:
    def __init__(self):
        self.container = []
        self.lock = Lock()

        self.__lock = Lock()
        self.__frame_dict: {int: np.ndarray}
        self.H: np.ndarray
        self.H_rppg: np.ndarray
        self.H_tracker: np.ndarray

