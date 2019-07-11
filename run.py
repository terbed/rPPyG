from src.cam import *
from src.load import *
from src.algo import FVP
from src.video_buffer import Buffer
from src.tracker import RoiTracker

# buffer for the images
buffer = Buffer()
#tracker = RoiTracker(buffer, disp=True, tracker_type="mosse")

algo = FVP(buffer, 20, [60/60., 250/60], 20, 256)
# ---------------------- ONLINE/OFFLINE
#cam = Camera(buffer)
#cam = BuiltinCam(buffer)
#loader = FrameLoader("/media/terbe/sztaki/incubator-records/PIC-2019y1m11d_7h12m2s/", buffer, pre="video_30.bin_", post="_proc", disp_fact=15)
loader = VideoLoader(buffer, "/media/terbe/sztaki/NADAM/segmented/output_red.mp4", disp=True)

if __name__ == "__main__":
    loader.start()
    algo.start()

    # tracker.start()
    # cam.start()

