from class_params import DownsweepParams, PulseParams
import numpy as np
from scipy.signal import chirp, tukey, hann



def fin_whale_downsweep(p: DownsweepParams) -> np.ndarray:
    n = int(p.fs * p.dur)
    t = np.linspace(0, p.dur, n, endpoint=False)

    # phi=0 instead of random — random phi caused apparent upsweep in STFT
    y = chirp(t, f0=p.f0, f1=p.f1, t1=p.dur, method="logarithmic", phi=0)

    # adding the tukey window (smooth fade-in AND fade-out) + exponential decay
    # replaces the old sharp 2%-sample fade-in which left a hard edge at the end
    win = tukey(n, alpha=p.alpha)
    decay = np.exp(-t / p.tau)
    env = win * decay
    env /= (env.max() + 1e-12)  # keep envelope peak at 1

    y = y * env

    # harmonics with the same envelope
    if p.harmonics:
        y += 0.4 * chirp(t, f0=2*p.f0, f1=2*p.f1, t1=p.dur,
                         method="logarithmic", phi=0) * env
        y += 0.2 * chirp(t, f0=3*p.f0, f1=3*p.f1, t1=p.dur,
                         method="logarithmic", phi=0) * env

    y = y / (np.max(np.abs(y)) + 1e-12) * p.amplitude
    return y.astype(np.float32)




def fin_whale_pulse(p: PulseParams) -> np.ndarray:
    def _single_pulse(pulse_dur: float, f0: float, f1: float, fs: float) -> np.ndarray:
        n = int(fs * pulse_dur)
        t = np.linspace(0, pulse_dur, n, endpoint=False)
        # FIX: phi=0 for consistent sweep direction
        y = chirp(t, f0=f0, f1=f1, t1=pulse_dur, method="linear", phi=0)
        # FIX: Hann window — smooth bell shape, zero at both edges, no clicks
        env = hann(n)
        return (y * env).astype(np.float32)

    pulse1 = _single_pulse(p.pulse_dur, p.f0, p.f1, p.fs)

    # Second pulse: slightly quieter (natural variation)
    amp2 = np.random.uniform(0.85, 1.0)
    pulse2 = _single_pulse(p.pulse_dur, p.f0, p.f1, p.fs) * amp2

    gap = np.zeros(int(p.fs * p.inter_pulse_gap), dtype=np.float32)

    y = np.concatenate([pulse1, gap, pulse2]).astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-12) * p.amplitude
    return y