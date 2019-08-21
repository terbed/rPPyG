import numpy as np
import time
import src.core as core
import logging
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QMutex
from sklearn.cluster import AffinityPropagation

logging.basicConfig(level=logging.INFO)


class Hybrid(QThread):
    """
    Hybrid solution for minimal motion cases. Rigid rPPG estimation and simultaneously ROI tracking. In case of motion
    the tracked ROI is used for signal extraction, otherwise the more accurate rigid rPPG
    """

    pruned = pyqtSignal(np.ndarray)
    pulse_estimated = pyqtSignal(int)
    pca_computed = pyqtSignal(np.ndarray, np.ndarray)
    weight_map_calculated = pyqtSignal(np.ndarray)
    bin_map_calculated = pyqtSignal(np.ndarray)

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

        # set up threshold values
        self.thrs_nclust = 0.05 * self.n_subregs
        self.thrs_SNR = 4.
        self.thrs_sparsity = 0.75

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
        # Check the size of buffer, and WARN if too big
        if len(self.buffer) > 50:
            self.logger.warning(f"Large buffer size! Decrease frame loading speed! Buffer size: {len(self.buffer)}")
        self.mtx.unlock()

        self.start()

    def run(self):
        while len(self.buffer) != 0:
            # 1) ---> Down-sample image and append to container list
            self.mtx.lock()
            self.Id_t.append(self.__downscale_image(self.__gaussian_blur(self.buffer[0])))
            self.mtx.unlock()

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

                        self.__pruning()

                    self.counter += 1

            # Remove processed frame from buffer
            del self.buffer[0]

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

    def __pruning(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Prune regions does not containing pulse information
        :return: selected sub-region indexes, SNR weight map of subregions, where zeros are pruned, the output of fft
        """

        # (A1) Calculate PCA on signals ------------------------------------------------------------------------------
        principal_comps = core.pca(self.Pt, n_max_comp=5)
        princ_freqdom = np.fft.rfft(principal_comps, axis=0)
        princ_freqdom_argmaxs = np.argmax(np.abs(np.multiply(princ_freqdom.T, self.pulse_template).T), axis=0)
        self.logger.info(f"The spectrum peaks of the main 5 principal components in order: {self.f[princ_freqdom_argmaxs]*60} BPM")
        x = principal_comps[:, 0].T
        y = np.linspace(0, x.size-1, x.size)
        self.pca_computed.emit(y, x)

        # (B1) - Calculate the spectrum of the signals corresponding each sub-region ---------------------------------
        fft_input = core.windowing_on_cols(self.Pt, type="hanning")
        # shape: (samples, sub-regs)

        fft_out = np.fft.rfft(fft_input, axis=0)/self.L2
        # shape: (samples, sub-regs)

        # work only in the pulse band range, zero out the remaining parts
        fft_out = np.multiply(fft_out.T, self.pulse_template).T
        # shape: (samples, sub-regs)

        spect = np.multiply(fft_out, np.conj(fft_out))
        # shape: (samples, sub-regs) -> spectrum in cols for each sub-reg

        # (B2) - Calculating SNR for each subreg and estimate pulse rate
        # normalize spectrum
        spect_maxs = spect.max(axis=0)
        normed_spect = np.divide(spect, spect_maxs)
        # shape: (samples, sub-regs) -> normed spectrum in rows

        SNRs = np.empty((self.n_subregs,))  # holds the SNR value for each sub-region
        freq_argmaxs = np.empty((self.n_subregs,)) # holds the index of the max frequency component for each sub-region
        for i in range(normed_spect.shape[1]):
            y = normed_spect[:, i]
            max_idx = int(np.argmax(y))
            freq_argmaxs[i] = max_idx

            template = core.snr_binary_template(len(y), max_idx)
            SNR = core.calc_snr(normed_spec=y, template=template)
            SNRs[i] = SNR

        self.logger.info(f"The max SNR value: {max(SNRs)} dB")

        # calculate a pulse rate estimate: the median of the selected (high SNR) signals
        PR_est_idx = int(np.median(freq_argmaxs))
        PR_est = self.f[PR_est_idx]*60
        self.logger.info(f"Pulse rate is estimated to be: {PR_est} BPM")
        self.pulse_estimated.emit(int(round(PR_est)))

        # Prune regions
        regions_wmap = [SNRs[idx] if (item >= PR_est_idx-1 and item <= PR_est_idx+1) else 0 for idx, item in enumerate(freq_argmaxs)]
        regions_wmap = regions_wmap/np.max(regions_wmap)
        regions_wmap = np.array(regions_wmap)
        # Display pruned binary map
        s = np.reshape(regions_wmap, (self.n_rows, self.n_cols, 1))*255
        self.pruned.emit(s.astype(np.uint8))

        #  CALCULATE SIMILARITY MATRIX =============================================================================
        # recreate selected_idxs based on region weightmaps
        accepted_idxs = []
        for idx, item in enumerate(regions_wmap):
            if item > 0:
                accepted_idxs.append(idx)
        accepted_idxs = np.array(accepted_idxs)

        n = accepted_idxs.size

        spect_CC = np.empty(shape=(n, n, self.f.size), dtype=np.complex64)
        spect_NCC = np.empty(shape=(n, n, self.f.size), dtype=np.complex64)
        self.logger.info(f"Calculate similarity matrix for {n} selected sub-regions...")

        # 1. Spectrum peak amplitude
        F = np.empty(shape=(n, n))
        # The diagonal is already almost computed (before in the pruning step)
        for i in range(n):
            idx = accepted_idxs[i]
            spect_CC[i, i, :] = spect[:, idx]*(SNRs[idx]**2)
            F[i, i] = np.abs(spect_maxs[idx])
        # compute the cross-diagonal elements
        for i in range(n):
            for j in range(n):
                if i != j:
                    # select sub-regions those of which are not pruned
                    row = accepted_idxs[i]
                    col = accepted_idxs[j]
                    spect_CC[i, j, :] = np.multiply(fft_out[:, row]*SNRs[row], fft_out[:, col].conj()*SNRs[col])
                    F[i, j] = np.max(np.abs(spect_CC[i, j, :]))

        # 2. Spectrum phase
        P = np.empty(shape=(n, n))

        for i in range(n):
            # In the diagonal there is only real part
            norm = np.sqrt(np.sum(np.square(np.abs(spect_CC[i, i, :]))))
            spect_NCC[i, i, :] = spect_CC[i, i, :]/norm
            P[i, i] = np.max(np.fft.irfft(spect_NCC[i, i, :]))
        for i in range(n):
            for j in range(n):
                if i != j:
                    norm = np.sqrt(np.sum(np.abs(np.multiply(spect_CC[i, j, :], spect_CC[i, j, :].conj()))))
                    spect_NCC[i, j, :] = spect_CC[i, j, :]/norm
                    P[i, j] = np.max(np.fft.irfft(spect_NCC[i, j, :]))

        # 3. Calculate spectrum entropy
        E = np.empty(shape=(n, n))
        norm = np.log(self.hr_band[1]-self.hr_band[0])
        for i in range(n):
            E[i, i] = np.sum(
                        np.multiply(
                                np.abs(spect_NCC[i, i, self.hr_min_idx:self.hr_max_idx+1]),
                                np.log(np.abs(spect_NCC[i, i, self.hr_min_idx:self.hr_max_idx+1]))
                                    )
                            ) / norm
        for i in range(n):
            for j in range(n):
                if i != j:
                    E[i, j] = np.sum(
                        np.multiply(
                            np.abs(spect_NCC[i, j, self.hr_min_idx:self.hr_max_idx + 1]),
                            np.log(np.abs(spect_NCC[i, j, self.hr_min_idx:self.hr_max_idx + 1]))
                        )
                    ) / norm

        # 4. Calculate Inner product
        I = np.empty(shape=(n, n))
        for i in range(n):
            idx = accepted_idxs[i]
            norm = np.sqrt(np.sum(np.square(self.Pt[:, idx])))
            I[i, i] = (self.Pt[:, idx]/norm).dot((self.Pt[:, idx]/norm).T)
        # Fill cross elements of a symmetric matrix
        for i in range(n - 1):
            for j in range(i + 1, n):
                row = accepted_idxs[i]
                col = accepted_idxs[j]
                norm_r = np.sqrt(np.sum(np.square(self.Pt[:, row])))
                norm_c = np.sqrt(np.sum(np.square(self.Pt[:, col])))
                tmp = (self.Pt[:, row]/norm_r).dot((self.Pt[:, col]/norm_c).T)
                I[i, j] = tmp
                I[j, i] = tmp

        # Construct similarity matrix from previous elements
        tmp = np.empty(shape=(n, n, 4))
        tmp[:, :, 0] = F
        tmp[:, :, 1] = P
        tmp[:, :, 2] = E
        tmp[:, :, 3] = I
        sigma = np.squeeze(np.std(tmp, axis=2))

        # Multiply different features
        mult_feat = np.ones(shape=(n, n))
        for idx in range(4):
            mult_feat = np.multiply(mult_feat, np.squeeze(tmp[:, :, idx]))

        sim_mat = 1 - np.exp(-np.divide(np.square(mult_feat), 2*np.square(sigma)))

        # Compute the eigenvectors
        u, s_vals, _ = np.linalg.svd(sim_mat, compute_uv=True)

        # weights will be the largest eigenvector
        w = u[:, 0]

        # ensure positive sign
        if w[0] < 0:
            w = -1*w

        # shift to be positive
        w = w - np.min(w)

        # norm with max
        w = w / np.max(w)

        # remap weight
        weight_map = np.zeros(shape=regions_wmap.shape)
        for counter, idx in enumerate(accepted_idxs):
            weight_map[idx] = w[counter]

        weight_map = np.reshape(weight_map, (self.n_rows, self.n_cols, 1))
        self.weight_map_calculated.emit(weight_map)

        # TODO: combine the result of PCA and avrg PR

        # Create binary mask from weight_map with automated threshold algorithm --------------------------------------
        #bin_map = cv2.adaptiveThreshold(np.round(weight_map*255).astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, int(self.n_rows/2), int(self.n_cols/2))
        ret, bin_map = cv2.threshold(np.round(weight_map*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.bin_map_calculated.emit(np.reshape(bin_map, (bin_map.shape[0], bin_map.shape[1], 1)))

        # TODO: confidence metric -----------------------------------------------------------------------------------
        # 1.) SNR criteria, default value: 4dB
        # The average SNR of the sub-regions where binary mask is 1
        binary_map_SNR = []
        for idx, item in enumerate(bin_map.flatten()):
            if item != 0:
                binary_map_SNR.append(SNRs[idx])

        avrg_SNR = np.average(binary_map_SNR)
        crit_SNR = avrg_SNR >= self.thrs_SNR
        self.logger.info(f"Average SNR on segmented skin region: {avrg_SNR} [dB]")
        self.logger.info(f"SNR criteria fulfilled? -> {crit_SNR}\n")

        # TODO: clustering the binary map
        # #############################################################################
        # Compute Affinity Propagation
        # af = AffinityPropagation().fit(bin_map)
        # cluster_centers_indices = af.cluster_centers_indices_
        # labels = af.labels_
        # n_clusters_ = len(cluster_centers_indices)
        # self.logger.info(f"Number of clusters in binary map: {n_clusters_}")
        #
        # # Select larges cluster
        # label_count = np.zeros(shape=(n_clusters_,))
        # for item in labels:
        #     label_count[item] += 1
        #
        # n_largest_cluster = np.max(label_count)

        # 2.) Reasonable number of sub-region, default: 0.0075*N
        n_largest_cluster = len(binary_map_SNR)
        crit_clust_size = n_largest_cluster >= self.thrs_nclust
        self.logger.info(f"Number of region in largest cluster: {n_largest_cluster}")
        self.logger.info(f"Cluster size criteria fulfilled? -> {crit_clust_size}\n")

        # 3.) Sparsity criteria of the binary mask
        n_bin = len(binary_map_SNR)
        crit_clust_spars = n_bin/n_largest_cluster >= self.thrs_sparsity
        self.logger.info(f"Cluster sparsity criteria fulfilled? -> {crit_clust_spars}\n")

        # OVERALL
        self.logger.info(f"ALL criteria fulfilled? -> {crit_clust_spars and crit_clust_size and crit_SNR}\n")

        # TODO: Initialize tracker
