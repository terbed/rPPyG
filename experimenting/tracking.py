import time
import cv2
from src.load import FrameLoader
from src.video_buffer import VideoBuffer

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

tracker = OPENCV_OBJECT_TRACKERS["mosse"]()

# initialize the bounding box coordinates of the object we are going
# to track
ROI = None

buffer = VideoBuffer()
loader = FrameLoader("/media/terbe/sztaki/rPPG_on_TV/youtuber/", buffer, "youtuber", "", starting_frame_num=2, zero_padding_num=4)

if __name__ == "__main__":
    loader.start()

time.sleep(5)

for frame in buffer.container[1:]:

    scale_percent = 50  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


    # check to see if we are currently tracking an object
    if ROI is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)

        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    else:
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track

        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        ROI = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        print(f"ROI: {ROI}")

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, ROI)

"""
RESULT:
The best tracking seems to bee the "mosse" algotithm along with "kcf", but "mosse" is faster
"""