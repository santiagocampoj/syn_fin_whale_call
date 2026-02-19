import numpy as np
from scipy.signal import chirp, spectrogram
import matplotlib.pyplot as plt



def plot_spectrogram(y, fs):
    # f, t, Sxx = spectrogram(y, fs=fs, nperseg=256, noverlap=128)
    nperseg = min(128, len(y))
    noverlap = min(96, nperseg - 1)
    f, t, Sxx = spectrogram(y, fs=fs, nperseg=nperseg, noverlap=noverlap)


    # plotting
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud')
    plt.colorbar(label='Intensity [dB]')
    plt.ylim(0, 100)
    plt.show()