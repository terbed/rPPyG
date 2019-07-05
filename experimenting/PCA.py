from sklearn.decomposition import PCA, FastICA
import numpy as np
from matplotlib import pyplot as plt

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


noise = np.random.normal(scale=0.7, size=data.size)
synthetic_data = data + noise

plt.figure(figsize=(12, 6))
plt.plot(t, synthetic_data)
plt.title("synthetic data")
plt.xlabel("Time [s]")


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
    S[:, i] = data + np.random.normal(scale=0.7, size=data.size)

S_fq = np.fft.fft(S, axis=0)

# PCA ----------------------------------------------------------------------
pca = PCA(n_components=3)
principal_components = pca.fit_transform(np.abs(S_fq))
principal_components = np.fft.ifft(principal_components, axis=0)

# Plot the five principal component
for i in range(3):
    plt.figure(figsize=(12, 6))
    plt.plot(t, principal_components[:, i])
    plt.title(f"{i}th principal component")
    plt.xlabel("Time [s]")


# ICA ----------------------------------------------------------------------
ica = FastICA(n_components=3)
independent_components = ica.fit_transform(np.abs(S_fq))
independent_components = np.fft.ifft(independent_components, axis=0)

# Plot the five principal component
for i in range(3):
    plt.figure(figsize=(12, 6))
    plt.plot(t, independent_components[:, i])
    plt.title(f"{i}th independent component")
    plt.xlabel("Time [s]")

plt.show()