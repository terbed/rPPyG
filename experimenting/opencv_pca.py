import numpy as np
import cv2
from matplotlib import pyplot as plt
from src import core

# (1) Generate synthetic data
# Create synthetic data
Fs = 20.
T = 1/Fs
N = 512         # aprrx. 25 sec if Fs is 20 Hz
t = np.linspace(0, (N-1)*T, N)

T_pulse = 1.
T_resp = 3.
T_i = 12.

pulse_signal = np.sin(2*np.pi/T_pulse*t)
resp_signal = 0.5*np.sin(2*np.pi/T_resp*t)
intensity_variation = 2*np.sin(2*np.pi/T_i*t)

data = pulse_signal + resp_signal

plt.figure(figsize=(12, 6))
plt.plot(t, data + intensity_variation + 12, 'k', label="Synthetic data")
plt.plot(t, pulse_signal+8, 'g', label="60 BPM pulse source signal")
plt.plot(t, resp_signal + 4, 'b', label="20 RPM respiration source signal")
plt.plot(t, intensity_variation, 'r', label="intensity variation with long trend")
plt.title("Constructing signal from sources")
plt.xlabel("Time [s]")
plt.legend(loc="upper right")

# Generate 50 from this
S = np.zeros(shape=(N, 100))    # signals ordered in columns

# 50 subregion containing pulse information
for i in range(100):
    A = np.random.uniform(1, 4, 1)
    B = np.random.uniform(0, 2, 1)
    C = np.random.uniform(0, 1, 1)

    intensity_variation = A * np.sin(2 * np.pi / T_i * t)
    pulse_signal = B * np.sin(2 * np.pi / T_pulse * t)
    resp_signal = C * np.sin(2 * np.pi / T_resp * t)
    S[:, i] = (intensity_variation + pulse_signal + resp_signal + np.random.normal(scale=0.7, size=data.size))*30


# Calculate PCA
def pca(X: np.ndarray, n_max_comp: int) -> np.ndarray:
    """
    Computes and returns the first n_max principal component
    :param X: Features in cols
    :param n_max_comp: Number of principal components to return
    :return:
    """
    means, eigen_vecs = cv2.PCACompute(X, mean=None, maxComponents=n_max_comp)
    print(f"Eigenvecs.shape: {eigen_vecs.shape}")
    print(f"Mean.shape: {means.shape}")

    # Subtract the mean from data to be zero centered
    X_cent = np.subtract(X, means)

    # Project X onto PC space
    X_pca = X_cent @ eigen_vecs.T

    return X_pca


principal_comps = pca(S, 5)
print(principal_comps.shape)

# Plot the five principal component
for i in range(5):
    plt.figure(figsize=(12, 6))
    plt.plot(t, principal_comps[:, i])
    plt.title(f"{i+1}th independent component")
    plt.xlabel("Time [s]")
plt.show()