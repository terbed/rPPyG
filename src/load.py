from threading import Thread
import numpy as np
import time
import skimage.io as sio
import glob
import cv2


class FrameLoader(Thread):
    def __init__(self, path,  buffer, pre, post, fmt=".png", starting_frame_num=0, zero_padding_num=1, disp_fact=1):
        """

        :param path: path to directory containing the images WITH "/" at the end
        :param fmt: Extension of the images e.g. fmt=".jpg"
        :param buffer: buffer list in which the images will be loaded
        :param pre: pre term befor numbering
        :param post: post term after numbering and before dot in fmt
        :param starting_frame_num: the number of frame at which to start the loading
        :param zero_padding_num: it is general to pad the numbering with zeros. e.g.: 001 -> zero_padding_num = 3
        :param disp_fact: The image will be multiplied with this number before display (scale=15 or 16 if 16bit image)
        """
        Thread.__init__(self)

        self.rootpath = path
        self.buffer = buffer
        self.disp_scale = disp_fact

        self.fs = glob.glob(f"{path}/*{fmt}")
        self.n = len(self.fs)

        self.start_frame_num = starting_frame_num
        self.preterm = pre
        self.postterm = post
        self.fmt = fmt

        self.padding_code = "{:0>" + str(zero_padding_num) + "d}"
        self.filenames = []
        self.__create_pathlist()

    def __create_pathlist(self):
        for frame_num in range(self.start_frame_num, self.n):
            self.filenames.append((self.rootpath + self.preterm + self.padding_code + self.postterm + self.fmt).format(frame_num))
        ### DEBUG
        print(self.filenames[10])

    def run(self):
        for idx, frame_path in enumerate(self.filenames):

            if len(self.buffer.container) < 500:
                start = time.time()
                rgb_img = sio.imread(frame_path)

                # Convert to BGR
                bgr_img = np.ndarray(shape=(rgb_img.shape[0], int(rgb_img.shape[1]), rgb_img.shape[2]), dtype=rgb_img.dtype)
                bgr_img[:, :, 0] = rgb_img[:, :, 2]
                bgr_img[:, :, 1] = rgb_img[:, :, 1]
                bgr_img[:, :, 2] = rgb_img[:, :, 0]

                if bgr_img.size != 0:
                    # Access the image data
                    print(f"{idx}th frame is loaded successfully from {frame_path}")

                    # Load into buffer
                    with self.buffer.lock:
                        self.buffer.container.append(bgr_img)

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
