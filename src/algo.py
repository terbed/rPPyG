import numpy as np
from statistics import mode
import time
import src.core as core
import logging
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QMutex

logging.basicConfig(level=logging.INFO)


class Hybrid(QThread):
    """
    Hybrid solution for minimal motion cases. Rigid rPPG estimation and simultaneously ROI tracking. In case of motion
    the tracked ROI is used for signal extraction, otherwise the more accurate rigid rPPG
    """

    pruned = pyqtSignal(np.ndarray)
    pulse_estimated = pyqtSignal(int)

    def __init__(self, frame_rate=20, l1=32, l2=256, hr_band=(40/60., 250/60.), width=500, height=500, patch_size=25):
        QThread.__init__(self)
        self.current_frame = None
        self.buffer = []
        self.mtx = QMutex()

        self.Fs = frame_rate
        self.T = 1./frame_rate
        self.L1 = l1
        self.L2 = l2
        self.Len = l2*self.T
        self.Fb = 1/self.Len

        self.hr_band = hr_band
        self.t = np.linspace(0, (l2-1)*self.T, l2)
        self.f = np.linspace(0, self.Fs/2., int(l2/2)+1)
        self.hr_min_idx = np.argmin(np.abs(self.f - self.hr_band[0]))
        self.hr_max_idx = np.argmin(np.abs(self.f - self.hr_band[1]))
        self.pulse_template = np.zeros(shape=self.f.shape)
        self.pulse_template[self.hr_min_idx:self.hr_max_idx+1] = 1

        # Init down-sampling
        self.patch_size = patch_size
        self.w = width
        self.h = height
        self.n_rows = int(self.h / self.patch_size)
        self.n_cols = int(self.w / self.patch_size)
        self.n_color = 3
        self.n_subregs = self.n_rows * self.n_cols

        # algo variables/ containers
        self.Id_t = []
        self.zero_row = np.zeros(shape=(1, self.n_subregs), dtype=np.double)
        self.Pt = np.zeros(shape=(self.L2, self.n_subregs), dtype=np.double)
        self.Zt = np.zeros(shape=(self.L2, self.n_subregs), dtype=np.double)
        self.shift_idx = 0
        self.counter = frame_rate*2

        # logger for the class
        self.logger = logging.getLogger("Hybrid")
        self.__init_logger()
        self.logger.info(f"Number of subregions: {self.n_subregs}\nFrame rate: {self.Fs} Hz\n"
                         f"Freq. bin: {self.Fb*60} BPM\nWindow length: {self.Len} sec")

    def __init_logger(self):
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('run.log', mode='a')
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s | %(levelname)s | %(message)s')
        f_format = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

        self.logger.info('Logger is initialized!')

    @staticmethod
    def __gaussian_blur(img: np.ndarray) -> np.ndarray:
        """
        Calculates and return gaussian smoothed image

        :param img: the source image to be smoothed
        :return: blurred image
        """
        return cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=5)

    def __downscale_image(self, img: np.ndarray) -> np.ndarray:
        """
        Averages subregions

        :param img: the source image to be downscaled
        :return: the sub-regions in rows with 3 color channel columns (shape: sub-regions x color channels)
        """

        Id = np.empty(shape=(self.n_rows, self.n_cols, self.n_color), dtype=np.float)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                Id[i, j, :] = np.mean(img[i * self.patch_size:i * self.patch_size + self.patch_size - 1,
                                          j * self.patch_size:j * self.patch_size + self.patch_size - 1, :], axis=(0, 1))

        Id = np.reshape(Id, (self.n_subregs, self.n_color))

        return Id

    @pyqtSlot(np.ndarray)
    def on_new_frame(self, src: np.ndarray):
        self.logger.debug("Rppg thread got frame signal!")

        self.mtx.lock()
        self.buffer.append(src.copy())
        print(len(self.buffer))
        self.mtx.unlock()

        self.start()

    def run(self):
        while len(self.buffer) != 0:
            # 1) ---> Down-sample image and append to container list
            self.mtx.lock()
            self.Id_t.append(self.__downscale_image(self.__gaussian_blur(self.buffer[0])))
            self.mtx.unlock()

            # Remove processed frame from buffer
            del self.buffer[0]
            # print(len(self.buffer))
            if len(self.buffer) > 50:
                self.logger.warning(f"Large buffer size! Decrease frame loading speed! Buffer size: {len(self.buffer)}")

            # 2) ---> Extract the PPG signal
            if len(self.Id_t) == self.L1:
                C = np.array(self.Id_t).copy()
                del self.Id_t[0]
                # print(f"Id_t len: {len(self.Id_t)}")

                # Pulse extraction algorithm
                P, Z = core.pos(C)

                if self.shift_idx + self.L1 <= self.L2:                     # L2 length is not reached yet
                    # Fill @param self.Pt with calculated P
                    self.__fill_add(P, Z)
                else:                                                       # In this case the L2 length is fully loaded
                    # Chuck the beginning and extend with new data
                    self.__chuck_add(P, Z)

                    # 3) ---> Calculate similarity matrix and combine signals every 2 second
                    if self.counter == self.Fs*2:
                        self.logger.debug("Calculating similarity matrix and combining pulse signals...")
                        self.counter = 0

                        # TODO: implement pruning, similarity matrix and stuff
                        self.__pruning()

                    self.counter += 1

    def __chuck_add(self, P, Z):
        n = self.shift_idx - 1
        # delete first row (last frame)
        self.Pt = np.delete(self.Pt, 0, 0)
        self.Zt = np.delete(self.Zt, 0, 0)
        # append zeros for the new frame point
        self.Pt = np.append(self.Pt, self.zero_row, 0)
        self.Zt = np.append(self.Zt, self.zero_row, 0)
        # overlap add result
        self.Pt[n:n + self.L1, :] = self.Pt[n:n + self.L1, :] + P
        self.Zt[n:n + self.L1, :] = self.Zt[n:n + self.L1, :] + Z

    def __fill_add(self, P, Z):
        """
        Fills L2 with calculated pulse segments

        :param P: shape: (pulse_sig, subregions)
        :param Z: shape: (intensity_sig, subregions)
        :return: Fills L2
        """
        self.Pt[self.shift_idx:self.shift_idx + self.L1, :] = self.Pt[self.shift_idx:self.shift_idx + self.L1, :] + P
        self.Zt[self.shift_idx:self.shift_idx + self.L1, :] = self.Zt[self.shift_idx:self.shift_idx + self.L1, :] + Z
        self.shift_idx = self.shift_idx + 1

    @staticmethod
    def __get_largest_idxs(l: np.ndarray, n: int):
        """
        :param l: list of numbers
        :param n: number of largest element
        :return: the list with number of n maximum indices
        """

        out = l.argsort()[::-1][:n]

        return out

    def __pruning(self):
        """
        Prune regions does not containing pulse information
        :return:
        """

        # (1) - Calculate the spectrum of the signals corresponding each sub-region
        fft_input = core.windowing_on_cols(self.Pt, type="hanning")
        fft_out = np.fft.rfft(fft_input)/self.L2

        spect = np.multiply(fft_out, np.conj(fft_out))
        # shape: (subreg, samples) -> spectrum in rows

        # (2) - Calculating SNR for each subreg
        # normalize spectrum
        normed_spect = np.divide(spect.transpose(), spect.max(axis=1)).transpose()
        # shape: (subreg, samples) -> normed spectrum in rows

        SNRs = np.empty((self.n_subregs,))  # holds the SNR value for each sub-region
        freq_argmaxs = np.empty((self.n_subregs,)) # holds the index of the max frequency component for each sub-region
        for i in range(normed_spect.shape[0]):
            # work only in the pulse band range, zero out the remaining parts
            y = np.multiply(normed_spect[i, :], self.pulse_template)
            max_idx = int(np.argmax(y))
            freq_argmaxs[i] = max_idx

            template = core.snr_binary_template(len(y), max_idx)
            SNR = core.calc_snr(normed_spec=y, template=template)
            SNRs[i] = SNR

        self.logger.info(f"The max SNR value: {max(SNRs)} dB")

        # Select the first 50 signals with highest SNR
        selected_idxs = self.__get_largest_idxs(SNRs, 50)
        print(f"Selected indexes: {selected_idxs}")
        # calculate a pulse rate estimate: the mode of the selected signals
        PR_est_idx = int(np.median(freq_argmaxs[selected_idxs]))
        print(f"Estimated pulse rate idx: {PR_est_idx}")
        PR_est = self.f[PR_est_idx]*60
        self.logger.info(f"Pulse rate is estimated to be: {PR_est} BPM")
        self.pulse_estimated.emit(int(round(PR_est)))

        selected_regions = [1 if (item >= PR_est_idx-1 and item <= PR_est_idx+1) else 0 for item in freq_argmaxs]
        print(selected_regions)
        s = np.reshape(selected_regions, (self.n_rows, self.n_cols, 1))*255
        print(s.shape)
        self.pruned.emit(s.astype(np.uint8))
