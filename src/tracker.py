import time
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QMutex
import numpy as np

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}


class RoiTracker(QThread):

    frame_tracked = pyqtSignal(np.ndarray)

    def __init__(self, tracker_type="kcf"):
        """
        Track selected ROI

        :param buffer: buffer of image container
        :param tracker_type:     "csrt": cv2.TrackerCSRT_create,
                                 "kcf": cv2.TrackerKCF_create,
                                 "boosting": cv2.TrackerBoosting_create,
                                 "mil": cv2.TrackerMIL_create,
                                 "tld": cv2.TrackerTLD_create,
                                 "medianflow": cv2.TrackerMedianFlow_create,
                                 "mosse": cv2.TrackerMOSSE_create
                            default: "mosse"
        :param disp: Display tracking? True/False
        """
        QThread.__init__(self)

        self.track_type = tracker_type
        self.frame = None
        self.mtx = QMutex()

    @pyqtSlot(np.ndarray, tuple)
    def init_tracker(self, frame, ROI):
        """
        Initialize tracking ROI

        :param ROI: A tuple containing ROI parameters: (offset_x, offset_y, width, height)
        :return: starts tracking
        """
        self.tracker = OPENCV_OBJECT_TRACKERS[self.track_type]()
        is_inted = self.tracker.init(np.uint8(frame), ROI)
        (x, y, w, h) = [int(i) for i in ROI]

        tracked_frame = frame.copy()
        if is_inted:
            cv2.rectangle(tracked_frame, (x, y), (x + w, y + h), (0, 200, 200), 3)
            self.frame_tracked.emit(tracked_frame)
        else:
            print("\n\nWARNING!!! Initialization of tracker was not successful!")
            cv2.rectangle(tracked_frame, (x, y), (x + w, y + h), (0, 0, 256), 3)
            self.frame_tracked.emit(tracked_frame)

    @pyqtSlot(np.ndarray)
    def on_new_frame(self, src: np.ndarray):
        self.mtx.lock()
        self.frame = src.copy()
        self.mtx.unlock()

        self.start()

    def run(self):
        (success, box) = self.tracker.update(np.uint8(self.frame))

        if success:
            self.mtx.lock()
            tracked_frame = self.frame.copy()
            self.mtx.unlock()

            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(tracked_frame, (x, y), (x + w, y + h), (209, 0, 206), 3)

            self.frame_tracked.emit(tracked_frame)
            #print("Tracker is running, signal emitted! ========================================================")
        else:
            print("\n\nTracker lost the object!")
