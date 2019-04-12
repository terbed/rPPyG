from threading import Thread
import numpy as np
import time
import skimage.io as sio
import glob
import cv2


class FrameLoader(Thread):
    def __init__(self, path, fmt, buffer, disp_fact=1):
        """

        :param path: path to directory containing the images
        :param fmt: Extension of the images
        :param buffer: buffer list in which the images will be loaded
        :param disp_fact: The image will be multiplied with this number before display (scale=15 or 16 if 16bit image)
        """
        Thread.__init__(self)

        self.buffer = buffer
        self.disp_scale = disp_fact
        self.filenames = glob.glob(f"{path}/*.{fmt}")
        self.filenames.sort()

    def run(self):
        for idx, frame_path in enumerate(self.filenames):

            if len(self.buffer) < 500:
                start = time.time()
                rgb_img = sio.imread(frame_path)

                # Convert to BGR
                bgr_img = np.ndarray(shape=(rgb_img.shape[0], int(rgb_img.shape[1]), rgb_img.shape[2]), dtype=rgb_img.dtype)
                bgr_img[:, :, 0] = rgb_img[:, :, 2]
                bgr_img[:, :, 1] = rgb_img[:, :, 1]
                bgr_img[:, :, 2] = rgb_img[:, :, 0]

                if bgr_img.size != 0:
                    # Access the image data
                    print(f"{idx}th frame is loaded successfully!")

                    # Load into buffer
                    self.buffer.append(bgr_img)

                    # Preview of the video
                    cv2.namedWindow('Loaded image', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('Loaded image', bgr_img * self.disp_scale)
                    cv2.waitKey(1)
                else:
                    print("ERROR!!! The loaded image is empty!")
                    break

                fps = 1./(time.time()-start)
                print(f"Loading framerate: {fps} FPS")
            else:
                # Terminate loading thread for 5 sec if lots of data in buffer
                time.sleep(5)
