from src.cam import *
from src.load import *
from src.disp import *
from src.algo import Hybrid
from src.tracker import RoiTracker
from PyQt5.QtWidgets import QApplication

app = QApplication([])

# SET UP ALGORITHM ------------------------------------------------------
# tracker = RoiTracker(buffer, disp=True, tracker_type="mosse")
#algo = Hybrid(width=1502, height=1002, patch_size=100, l1=20)
#algo = Hybrid()
algo = Hybrid(width=1040, height=1392, patch_size=50)

# SET UP DISPLAY -------------------------------------------------------
frame_disp = ImgDisp("Frame", disp_fact=1, width=1040/2, height=1392/2)
pruned_disp = ImgDisp("Selected regions", width=1040/2, height=1392/2)
# frame_disp = ImgDisp("Frame", disp_fact=16)
# pruned_disp = ImgDisp("Selected regions")
pr_disp = NumericDisp("Pulse Rate")

# SET UP VIDEO SOURCE ---------------------------------------------------
#cam = Camera(buffer)
#cam = BuiltinCam(buffer)
#loader = FrameLoader("/media/terbe/sztaki/incubator-records/PIC-2019y1m11d_7h12m2s/", pre="video_30.bin_", post="_proc")
# loader = FrameLoader("/media/terbe/sztaki/incubator-records/2018-06-12-baba-plexi-20fps/",
#                      pre="Basler acA1600-20uc (21993673)_20180612_111836210_", post="", fmt=".tiff", zero_padding_num=4,
#                      starting_frame_num=1)
loader = FrameLoader("/media/terbe/sztaki/MMSE-HR/T1/", pre="", post="", fmt=".jpg", zero_padding_num=3)

#loader = VideoLoader(buffer, "/media/terbe/sztaki/NADAM/segmented/output_red.mp4", disp=True)

# CONNECT SIGNALS --------------------------------------------------------
loader.new_frame.connect(algo.on_new_frame)
loader.new_frame.connect(frame_disp.set_image)

algo.pulse_estimated.connect(pr_disp.set_value)
algo.pruned.connect(pruned_disp.set_image)

if __name__ == "__main__":
    loader.start()


    # tracker.start()
    # cam.start()

app.exec_()
