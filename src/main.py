import numpy as np
from threading import Thread
import time
import src.core as core
import logging
import cv2


class FVP(Thread):
    def __init__(self, buffer, frame_rate, hr_band, L1, L2):
        Thread.__init__(self)
        self.is_running = True
        self.buffer = buffer
        self.frame_rate = frame_rate
        self.hr_band = hr_band
        self.L1 = L1
        self.L2 = L2

    def run(self):
        # Initialize FVP method
        K = 6  # number of top ranked eigenvectors
        patch_size = 25
        L1 = self.L1
        u0 = 1
        L2 = self.L2  # window length in frame

        l1 = float(L1) / self.frame_rate
        Fb1 = 1. / l1
        l2 = float(L2) / self.frame_rate  # window length in seconds
        Fb2 = 1. / l2

        f1 = np.linspace(0, (L1 - 1) * Fb1, L1, dtype=np.double)  # frequency bin in Hz
        f2 = np.linspace(0, (L2 - 1) * Fb2, L2, dtype=np.double)  # frequency vector in Hz
        t = np.linspace(0, (L2 - 1) * 1. / self.frame_rate, L2)
        t = t.reshape((1, len(t)))
        hr_min_idx1 = np.argmin(np.abs(f1 - self.hr_band[0]))
        hr_max_idx1 = np.argmin(np.abs(f1 - self.hr_band[1]))
        B1 = [hr_min_idx1, hr_max_idx1]  # HR range ~ [50, 220] bpm

        hr_min_idx2 = np.argmin(np.abs(f2 - self.hr_band[0]))
        hr_max_idx2 = np.argmin(np.abs(f2 - self.hr_band[1]))
        B2 = [hr_min_idx2, hr_max_idx2]

        # Channel ordering for BGR
        # The largest pulsatile strength is in G then B then R
        channel_ordering = [1, 0, 2]

        Jt = []

        add_row = np.zeros(shape=(1, K * 4), dtype=np.double)
        Pt = np.zeros(shape=(L2, K * 4), dtype=np.double)
        Zt = np.zeros(shape=(L2, K * 4), dtype=np.double)

        # Container for the overlap-added signal
        H = np.zeros(shape=(1, L2), dtype=np.double)
        H_RAW = np.zeros(shape=(1, L2), dtype=np.double)

        shift_idx = 0
        heart_rates = []
        quality_measures = []
        final_HRs = []
        final_QMs = []
        timestamp_nums = []
        first_run = True

        while self.is_running:
            print(f"--------------------> Buffer size: {len(self.buffer.container)}")
            if len(self.buffer.container) > 0:
                start_time = time.time()
                # ----------------------------------------------------------------- Image processing with FVP algorithm
                with self.buffer.lock:
                    Jt.append(core.fvp(self.buffer.container[0], patch_size, K))
                    del self.buffer.container[0]
                
                # Extract the PPG signal
                if len(Jt) == L1:
                    C = np.array(Jt)
                    del Jt[0]  # delete the first element

                    # --------------------------------------------------------------------- Pulse extraction algorithm
                    P, Z = core.pos(C, channel_ordering)

                    if shift_idx + L1 - 1 < L2:
                        Pt[shift_idx:shift_idx + L1, :] = Pt[shift_idx:shift_idx + L1, :] + P
                        Zt[shift_idx:shift_idx + L1, :] = Zt[shift_idx:shift_idx + L1, :] + Z

                        # average add, not overlap add
                        Pt[shift_idx:shift_idx + L1 - 1, :] = Pt[shift_idx:shift_idx + L1 - 1, :] / 2
                        Zt[shift_idx:shift_idx + L1 - 1, :] = Zt[shift_idx:shift_idx + L1 - 1, :] / 2

                        shift_idx = shift_idx + 1
                    else:  # In this case the L2 length is fully loaded, we have to remove the first element and add a new one at the end
                        # overlap add result
                        Pt = np.delete(Pt, 0, 0)  # delete first row (last frame)
                        Pt = np.append(Pt, add_row, 0)  # append zeros for the new frame point

                        Zt = np.delete(Zt, 0, 0)  # delete first row (last frame)
                        Zt = np.append(Zt, add_row, 0)  # append zeros for the new frame point

                        Pt[shift_idx - 1:shift_idx + L1 - 1, :] = Pt[shift_idx - 1:shift_idx + L1 - 1, :] + P
                        Zt[shift_idx - 1:shift_idx + L1 - 1, :] = Zt[shift_idx - 1:shift_idx + L1 - 1, :] + Z

                        # overlap average
                        Pt[shift_idx - 1:shift_idx + L1 - 2, :] = Pt[shift_idx - 1:shift_idx + L1 - 2, :] / 2
                        Zt[shift_idx - 1:shift_idx + L1 - 2, :] = Zt[shift_idx - 1:shift_idx + L1 - 2, :] / 2

                        computing = True

                if shift_idx == L2 - L1 + 1:
                    # now we can also calculate fourier and signal combination
                    h, h_raw, hr_est, q_meas = core.signal_combination(Pt, Zt, L2, B2, f2)
                    heart_rates.append(hr_est)
                    quality_measures.append(q_meas)

                    H = np.delete(H, 0, 0)
                    H_RAW = np.delete(H_RAW, 0, 0)
                    H = np.append(H, 0.)
                    H_RAW = np.append(H_RAW, 0.)

                    # overlap add
                    H = H + h
                    H_RAW = H_RAW + h_raw

                    # overlap average
                    H[0:L2 - 1] = H[0:L2 - 1] / 2
                    H_RAW[0:L2 - 1] = H_RAW[0:L2 - 1] / 2

                if len(heart_rates) == self.frame_rate * 1:
                    # Display HR estimate and signals
                    best_idx = np.argmax(quality_measures)
                    estimated_HR = heart_rates[best_idx]
                    final_HRs.append(estimated_HR)
                    final_QMs.append(quality_measures[best_idx])
                    heart_rates = []
                    quality_measures = []
                    
                    print(f"----------------------------------------> Estimated HR: {estimated_HR}")

                running_time = (time.time() - start_time)
                fps = 1.0 / running_time
                print(f"---> Algorithm speed: {fps}  FPS")
            else:
                time.sleep(1)


class RoiBased(Thread):
    def __init__(self, buffer, frame_rate=20, hr_band=(40/60., 240/60.), width=500, height=500, patch_size=25):
        Thread.__init__(self)
        self.logger = logging.getLogger("RoiBased")
        self.buffer = buffer
        self.Fs = frame_rate
        self.hr_band = hr_band

        self.patch_size = patch_size
        self.w = width
        self.h = height
        self.rows = int(self.h / self.patch_size)
        self.cols = int(self.w / self.patch_size)
        self.subregs = self.rows * self.cols

        self.__init_logger()
        self.logger.info(f"Number of subregions: {self.subregs}")

    def __init_logger(self):
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('RoiBased.log', mode='w')
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s | %(levelname)s | %(message)s')
        f_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

        self.logger.debug('Logger is initialized!')

    @staticmethod
    def __gaussian_blur(img):
        """
        Calculates and return gaussian smoothed image

        :param img: the source image to be smoothed
        :return: blurred image
        """
        return cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=5)

    def __downscale_image(self, img):
        """
        Averages subregions

        :param img: the source image to be downscaled
        :return: the sub-regions in rows with 3 color channel columns (shape: sub-regions x color channels)
        """

        Id = np.empty(shape=(self.rows, self.cols, 3), dtype=np.float)
        for i in range(self.rows):
            for j in range(self.cols):
                Id[i, j, :] = np.mean(img[i * self.patch_size:i * self.patch_size + self.patch_size - 1,
                                          j * self.patch_size:j * self.patch_size + self.patch_size - 1, :], axis=(0, 1))

        Id = np.reshape(Id, (Id.shape[0] * Id.shape[1], 3))

        return Id