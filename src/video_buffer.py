import numpy as np
from threading import Lock


class VideoBuffer:
    def __init__(self):
        self.lock = Lock()
        self.container = []
