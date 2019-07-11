import time
import cv2
from threading import Thread
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


class RoiTracker(Thread):
    def __init__(self, buffer, tracker_type="mosse", disp=False):
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
        Thread.__init__(self)

        self.tracker = OPENCV_OBJECT_TRACKERS[tracker_type]()
        self.buffer = buffer
        self.disp = disp

    def init_tracker(self, ROI=None):
        """
        Initialize tracking ROI

        :param ROI: A tuple containing ROI parameters: (offset_x, offset_y, width, height)
        :return: starts tracking
        """
        if ROI is None:
            time.sleep(5)
            # select by hand
            if len(self.buffer.container) > 0:
                ROI = cv2.selectROI("Frame", self.buffer.container[0], fromCenter=False, showCrosshair=True)
                print(f"ROI: {ROI}")

                self.tracker.init(self.buffer.container[0], ROI)

                self.start()
            else:
                print("Container does not have frame in it!")

    def run(self, ROI=None):
        if ROI is None:
            time.sleep(1)
            # select by hand
            if len(self.buffer.container) > 0:
                ROI = cv2.selectROI("Frame", self.buffer.container[0], fromCenter=False, showCrosshair=True)
                print(f"ROI: {ROI}")

                self.tracker.init(np.uint8(self.buffer.container[0]), ROI)
            else:
                print("Container does not have frame in it!")
        else:
            self.tracker.init(np.uint8(self.buffer.container[0]), ROI)

        while True:
            if len(self.buffer.container) > 0:
                # grab the new bounding box coordinates of the object
                (success, box) = self.tracker.update(np.uint8(self.buffer.container[0]))

                if self.disp:
                    # check to see if the tracking was a success
                    if success:
                        (x, y, w, h) = [int(v) for v in box]
                        cv2.rectangle(self.buffer.container[0], (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # show the output frame
                    cv2.imshow("Frame", self.buffer.container[0])
                    key = cv2.waitKey(1) & 0xFF

                del self.buffer.container[0]
            else:
                print("Container i empty, waiting for 20 ms")
                time.sleep(20/1000.)
