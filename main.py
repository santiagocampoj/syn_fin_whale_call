import numpy as np
from scipy.signal import chirp, spectrogram
import matplotlib.pyplot as plt
import soundfile as sf
from utils import *



def fin_whale_downsweep(fs,dur,f0,f1,tau,harmonics=False):
    t= np.linspace(0, dur, int(fs * dur), endpoint=False)
    #phase in degrees
    phi =np.random.uniform(0, 360)

    #downsweep signal
    y = chirp(t, f0=f0, f1=f1, t1=dur, method="logarithmic", phi=phi)

    # windowed envelope
    # env = np.hanning(len(y))
    # exponential decay, the bigger the tau the slower the decay
    env= np.exp(-t/tau)
    fade_n = max(1, int(0.02 * len(t)))
    env[:fade_n] *= np.linspace(0, 1, fade_n)
    # combined envelope
    y = y * env

    #harmonics ???
    if harmonics:
        y += 0.4 * chirp(t, f0=2*f0, f1=2*f1, t1=dur, method="logarithmic", phi=phi) * env
        y += 0.2 * chirp(t, f0=3*f0, f1=3*f1, t1=dur, method="logarithmic", phi=phi) * env
    # normalizing to -1 to 1
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.95
    return y.astype(np.float32), fs



def main():
    audio, fs = fin_whale_downsweep(fs=20, dur=0.8, f0=110.0, f1=40.0, tau=0.25,harmonics=True)
    plot_spectrogram(audio, fs)
    exit()
    sf.write("fin_whale_downsweep.wav", audio, fs)

    

    print("Wrote fin_whale_downsweep.wav")


if __name__ == "__main__":
    main()