from src.load import *
from src.disp import *
from src.algo import Hybrid
from PyQt5.QtWidgets import QApplication

app = QApplication([])

algo = Hybrid()

frame_disp = ImgDisp("Frame", disp_fact=16)
pruned_disp = ImgDisp("Selected regions", heat_map=True)
weightmap_disp = ImgDisp("Weight map from similarity mat", heat_map=True)

pr_disp = NumericDisp("Pulse Rate")
pca_signal_disp = SignalDisp("First principal component")

loader = FrameLoader("/media/terbe/sztaki/incubator-records/PIC-2019y1m11d_7h12m2s/", pre="video_30.bin_", post="_proc")

# CONNECT SIGNALS --------------------------------------------------------
loader.new_frame.connect(algo.on_new_frame)
loader.new_frame.connect(frame_disp.set_image)

algo.pulse_estimated.connect(pr_disp.set_value)
algo.pruned.connect(pruned_disp.set_image)
algo.pca_computed.connect(pca_signal_disp.set_data)
algo.weight_map_calculated.connect(weightmap_disp.set_image)

if __name__ == "__main__":
    loader.start()

app.exec_()