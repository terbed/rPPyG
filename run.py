from src.cam import *
from src.load import FrameLoader
from src.main import FVP
from src.video_buffer import VideoBuffer

# buffer for the images
buffer = VideoBuffer()


algo = FVP(buffer, 20, [40/60., 120/60], 40, 256)
# ---------------------- ONLINE/OFFLINE
# cam = Camera(buffer)
loader = FrameLoader("/media/terbe/sztaki/incubator-records/PIC-2019y1m11d_7h12m2s/", buffer, pre="video_30.bin_", post="_proc", disp_fact=15)

if __name__ == "__main__":
    algo.start()
    loader.start()

