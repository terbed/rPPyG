"""
Modularized program flow example optimized for Qt's signal-slot mechanism
"""
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QMutex
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import pyqtgraph as pg


class Cam(QThread):
    """
    Read camera and emit frame to worker threads
    """

    # Init signals:
    # frame emitted to worker threads when new frame is obtained:
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self):
        QThread.__init__(self)

    def run(self):
        while True:
            # catch a frame from camera
            self.usleep(20000)  # simulate 20ms exposure time
            frame = np.random.randint(0, 255, size=(500, 500, 3), dtype=np.uint8)

            self.new_frame.emit(frame)
            print("\nFrame emitted to different worker threads!")


class Tracker(QThread):
    tracked_frame = pyqtSignal(np.ndarray)

    def __init__(self):
        QThread.__init__(self)
        self.current_frame = None
        self.num_of_calls = 0

    @pyqtSlot(np.ndarray)
    def on_new_frame(self, src: np.ndarray):
        self.num_of_calls += 1
        print(f"1) Tracker thread got {self.num_of_calls}th frame signal!")
        self.current_frame = src.copy()
        self.start()

    def run(self):
        tmp = self.num_of_calls
        self.usleep(10000)
        print(f"1) Average: {np.mean(self.current_frame)}")
        print(f"1) {tmp}th iter finished!")

        # emit processed frame
        self.tracked_frame.emit(self.current_frame/2)


class Rppg(QThread):
    pulse_signal_computed = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self):
        QThread.__init__(self)
        self.current_frame = None
        self.buffer = []
        self.num_of_calls = 0

        self.mtx = QMutex()

    @pyqtSlot(np.ndarray)
    def on_new_frame(self, src: np.ndarray):
        self.num_of_calls += 1
        print(f"2) Rppg thread got {self.num_of_calls}th frame signal!")

        if not self.isRunning():
            self.mtx.lock()
            self.current_frame = src.copy()
            self.mtx.unlock()
            self.start()
        else:
            print("Buffer is has to be used in rPPG thread, because it is not finished the processing yet!")
            self.mtx.lock()
            self.buffer.append(src.copy())
            self.mtx.unlock()

    def run(self):
        tmp = self.num_of_calls
        if len(self.buffer) == 0:
            self.usleep(20000)

            self.mtx.lock()
            print(f"2) Std: {np.std(self.current_frame)}")
            print(f"2) {tmp}th iter finished!")
            # emit computed signal
            S = np.mean(self.current_frame, (0, 2))
            self.mtx.unlock()

            x = np.linspace(0, 20, S.size)
            self.pulse_signal_computed.emit(x, S)

            if len(self.buffer) != 0:
                self.start()
        else:
            while len(self.buffer) != 0:

                self.mtx.lock()
                print(f"2) Std: {np.std(self.buffer[0])}")
                # emit computed signal
                S = np.mean(self.buffer[0], (0, 2))
                self.mtx.unlock()

                x = np.linspace(0, 20, S.size)
                self.pulse_signal_computed.emit(x, S)

                del self.buffer[0]
                print(f"Buffer size: {len(self.buffer)}")


class ImgDisp(QWidget):
    """
    A class to display images
    """
    def __init__(self, title: str, offset_x=0, offset_y=0, width=500, height=500):
        QWidget.__init__(self)
        self.setWindowTitle(title)
        self.setGeometry(offset_x, offset_y, width, height)
        # self.resize(500, 500)
        # create a label
        self.label = QLabel(self)
        # self.label.move(280, 120)
        self.label.resize(width, height)
        self.show()

    @staticmethod
    def get_qimage(src: np.ndarray) -> QImage:
        height, width, colors = src.shape
        bytes_per_line = colors * width       # If the bit depth is 8 bit (if 16 then multiply it with 2)

        out = QImage(src.data, width, height, bytes_per_line, QImage.Format_RGB888)

        out = out.rgbSwapped()
        return out

    @pyqtSlot(np.ndarray)
    def set_image(self, src):
        qimg = self.get_qimage(src)
        self.label.setPixmap(QPixmap.fromImage(qimg))


class SignalDisp(pg.GraphicsWindow):
    def __init__(self, title: str, offset_x=0, offset_y=0, width=800, height=300, color='r', parent=None):
        """
        Display signal in a window
        :param title: title of the plot
        :param width: width of the window
        :param height: height of the window
        :param color: plot color; chose from: 'r', 'g','b' or custom in form: (255, 124, 0)
        :param parent: None
        """
        pg.GraphicsWindow.__init__(self, parent=parent)

        if color == 'r':
            c = (255, 0, 0)
        elif color == 'g':
            c = (0, 255, 0)
        elif color == 'b':
            c = (0, 0, 255)
        else:
            c = color

        self.setWindowTitle(title)
        self.setGeometry(offset_x, offset_y, width, height)
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.plotItem = self.addPlot()
        self.plotDataItem = self.plotItem.plot([], pen=1, symbolBrush=c, symbolSize=5, symbolPen=None)

        self.resize(width, height)
        self.raise_()
        self.show()

    @pyqtSlot(np.ndarray, np.ndarray)
    def set_data(self, x, y):
        self.plotDataItem.setData(x, y)


# -------------------------------- MAIN THREAD -----------------------------------------------------------------
app = QApplication([])

# emitter thread
cam = Cam()

# worker threads
workers = []
tracker = Tracker()
workers.append(tracker)

rppg = Rppg()
workers.append(rppg)

# connect workers to emitter thread --------------------------------------------------------------------
for worker in workers:
    cam.new_frame.connect(worker.on_new_frame)

# Create and connect displays ---------------------------------------------------------------------------
raw_disp = ImgDisp("Raw Frame")
cam.new_frame.connect(raw_disp.set_image)

tracker_disp = ImgDisp("Tracker Disp", offset_x=510)
tracker.tracked_frame.connect(tracker_disp.set_image)

# Create and connect signal plots
pulse_signal_disp = SignalDisp("Pulse signal plot", 0, 510, 1200, color='b')
rppg.pulse_signal_computed.connect(pulse_signal_disp.set_data)

if __name__ == "__main__":
    cam.start()

app.exec_()
