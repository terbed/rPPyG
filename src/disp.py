from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QMutex, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QMainWindow, QLCDNumber
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg
import time
import numpy as np


class ImgDisp(QWidget):
    """
    A class to display images
    """
    def __init__(self, title: str, offset_x=0, offset_y=0, width=500, height=500, disp_fact=1):
        """
        :param title:
        :param offset_x:
        :param offset_y:
        :param width:
        :param height:
        :param disp_fact: Display factor, set it to "16" if 16 bit image...
        """
        QWidget.__init__(self)
        self.setWindowTitle(title)
        self.setGeometry(offset_x, offset_y, width, height)
        self.disp_fact = disp_fact
        # self.resize(500, 500)
        # create a label
        self.label = QLabel(self)
        # self.label.move(280, 120)
        self.label_h = height
        self.label_w = width
        self.label.resize(width, height)
        self.show()

    def get_qimage(self, src: np.ndarray) -> QImage:
        imh, imw, colors = src.shape
        tmp = src.copy()

        if tmp.dtype == np.uint16:
            tmp = (tmp/256).astype(np.uint8)

        bytes_per_line = colors * imw    # If the bit depth is 8 bit (if 16 then multiply it with 2)
        if self.disp_fact != 1:
            tmp = tmp*self.disp_fact

        if colors == 1:
            fmt = QImage.Format_Grayscale8
        else:
            fmt = QImage.Format_RGB888

        out = QImage(tmp.data, imw, imh, bytes_per_line, fmt)

        if colors == 3:
            out = out.rgbSwapped()

        return out

    @pyqtSlot(np.ndarray)
    def set_image(self, src):
        qimg = self.get_qimage(src)
        p = QPixmap.fromImage(qimg)
        self.label.setPixmap(p.scaled(self.label_w, self.label_h, Qt.KeepAspectRatio))


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


class NumericDisp(QMainWindow):
    def __init__(self, title: str):
        """
        :param title: The title of the display
        """
        QMainWindow.__init__(self)
        self.title = title

        self.lcd = QLCDNumber(self)
        self.lcd.display(0)
        self.setCentralWidget(self.lcd)
        self.setGeometry(300, 300, 250, 100)
        self.setWindowTitle(self.title)
        self.show()

    @pyqtSlot(int)
    def set_value(self, val):
        self.lcd.display(val)
