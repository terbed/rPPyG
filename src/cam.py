from pypylon import pylon
from pypylon import genicam
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QMutex
import numpy as np
import time
import cv2


class Camera(QThread):
    def __init__(self, buffer, disp=False):
        QThread.__init__(self)

        self.buffer = buffer
        self.disp = disp
        # self.img_width = img_width
        # self.img_height = img_height
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        # Print the model name of the camera.
        print("Using device ", self.camera.GetDeviceInfo().GetModelName())
        self.camera.Open()

    def __del__(self):
        self.camera.StopGrabbing()
        self.camera.Close()

    def run(self):
        try:
            # Set up self.camera parameters first in PylonViewer!!!
            # self.camera.Width.Value = self.img_width
            # self.camera.Height.Value = self.img_height
            # self.camera.OffsetX.Value = 200
            # self.camera.OffsetY.Value = 100
            # self.camera.PixelFormat = "BayerBG12"

            # Grabing Continusely (video) with minimal delay
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            converter = pylon.ImageFormatConverter()

            # converting to opencv bgr format
            converter.OutputPixelFormat = pylon.PixelType_RGB16packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_LsbAligned
            #bgr_img = frame = np.ndarray(shape=(self.camera.Height.Value, self.camera.Width.Value, 3), dtype=np.uint16)

            while self.camera.IsGrabbing():
                start_t = time.time()
                grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                if grab_result.GrabSucceeded():

                    # Access the image from data
                    image = converter.Convert(grab_result)
                    frame = np.ndarray(buffer=image.GetBuffer(), shape=(image.GetHeight(), image.GetWidth(), 3),
                                       dtype=np.uint16)
                    # frame[:, :, 0] = frame[:, :, 2]
                    # frame[:, :, 1] = frame[:, :, 1]
                    # frame[:, :, 2] = frame[:, :, 0]

                    # Save image to buffer
                    with self.buffer.lock:
                        self.buffer.container.append(frame)
                        print(f"Buffer size: {len(self.buffer.container)}")

                    if self.disp:
                        # Display video if needed
                        cv2.namedWindow('Camera video', cv2.WINDOW_AUTOSIZE)
                        cv2.imshow('Camera video', frame*16)
                        k = cv2.waitKey(1)
                        if k == 27:
                            break

                    running_time = (time.time() - start_t)
                    fps = 1.0 / running_time
                    print(f"Camera speed: {fps}  FPS")
                    #print(f"Buffer size: {len(self.buffer)}")

                grab_result.Release()
            self.camera.StopGrabbing()
        except genicam.GenericException as e:
            print("ERROR! An exception occurred.")
            print(str(e))


class BuiltinCam(QThread):
    def __init__(self, buffer, disp=False):
        QThread.__init__(self)

        self.buffer = buffer
        self.disp = disp

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 20)
        print(f"Built in Camera FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
        print(f"Width: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"Width: {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    def __del__(self):
        # When everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        while (True):
            start = time.time()
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            self.buffer.container.append(np.copy(frame))

            if self.disp:
                # Display the resulting frame
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            print(f"camera fps: {1. / (time.time() - start)}")
