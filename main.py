import numpy as np
import soundfile as sf
from scipy.signal import chirp




def fin_whale_call(fs=200,dur=1.2,f0=26.0,f1=18.0,harmonics=True):
    t= np.linspace(0, dur, int(fs * dur), endpoint=False)
    phi =np.random.uniform(0, 360)
    y = chirp(t, f0=f0, f1=f1, t1=dur, method="linear", phi=phi)

    # windowed envelope
    env = np.hanning(len(y))
    y = y * env

    #harmonics ???
    if harmonics:
        y += 0.4 * chirp(t, f0=2*f0, f1=2*f1, t1=dur, method="linear", phi=phi) * env
        y += 0.2 * chirp(t, f0=3*f0, f1=3*f1, t1=dur, method="linear", phi=phi) * env

    # normalizing to -1 to 1
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.95
    return y.astype(np.float32), fs



def plot_spectrogram(y, fs):
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    f, t, Sxx = spectrogram(y, fs=fs, nperseg=256, noverlap=128)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud')
    plt.colorbar(label='Intensity [dB]')
    plt.ylim(0, 100)
    plt.show()


def main():
    audio, fs = fin_whale_call(fs=200, dur=1.2, f0=26, f1=18, harmonics=True)
    # sf.write("synthetic_fin_whale_call.wav", audio, fs)
    

    print("Wrote synthetic_fin_whale_call.wav")



if __name__ == "__main__":
    main()