from class_params import DownsweepParams, PulseParams
import numpy as np
from scipy.signal import chirp
from scipy.signal.windows import tukey, hann
from randomizer import randomize_downsweep, randomize_pulse
from plots import plot_spec





########################
# DOWNSWEEP
#######################
def fin_whale_downsweep(p: DownsweepParams) -> np.ndarray:
    # logarithmic sweep from f0 to f1 over dur seconds, sampled at fs
    n = int(p.fs * p.dur)
    t = np.linspace(0, p.dur, n, endpoint=False)

    # phi=0 instead of random — random phi caused apparent upsweep in STFT
    y = chirp(t, f0=p.f0, f1=p.f1, t1=p.dur, method="logarithmic", phi=0)

    # adding the tukey window (smooth fade-in AND fade-out) + exponential decay
    # replaces the old sharp 2%-sample fade-in which left a hard edge at the end
    # tau controls the decay rate — higher tau = slower decay, more natural
    win = tukey(n, alpha=p.alpha)
    decay = np.exp(-t / p.tau)
    env = win * decay
    # keep envelope peak at 1
    env /= (env.max() + 1e-12)

    y = y * env

    # harmonics with the same envelope
    if p.harmonics:
        y += 0.4 * chirp(t, f0=2*p.f0, f1=2*p.f1, t1=p.dur,method="logarithmic", phi=0) * env
        y += 0.2 * chirp(t, f0=3*p.f0, f1=3*p.f1, t1=p.dur,method="logarithmic", phi=0) * env

    y = y / (np.max(np.abs(y)) + 1e-12) * p.amplitude
    return y.astype(np.float32)





#####################
# 20 Hz PULSE
#####################
def _single_pulse(pulse_dur: float, f0: float, f1: float, fs: float) -> np.ndarray:
    # linear sweep from f0 to f1 over pulse_dur seconds, sampled at fs
    n = int(fs * pulse_dur)
    t = np.linspace(0, pulse_dur, n, endpoint=False)
    # phi=0 for consistent sweep direction
    y = chirp(t, f0=f0, f1=f1, t1=pulse_dur, method="linear", phi=0)
    # Hann window — smooth bell shape, zero at both edges, no clicks
    env = hann(n)
    return (y * env).astype(np.float32)



def fin_whale_pulse(p: PulseParams) -> np.ndarray:
    pulse1 = _single_pulse(p.pulse_dur, p.f0, p.f1, p.fs)

    # Second pulse: slightly quieter (natural variation)
    amp2 = np.random.uniform(0.85, 1.0)
    pulse2 = _single_pulse(p.pulse_dur, p.f0, p.f1, p.fs) * amp2

    gap = np.zeros(int(p.fs * p.inter_pulse_gap), dtype=np.float32)

    y = np.concatenate([pulse1, gap, pulse2]).astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-12) * p.amplitude
    return y




########## test
def test_synth(logger=None) -> None:
    """Test the downsweep and pulse synthesis by generating a sample call and plotting its spectrogram"""

    logger.info("TESTING --> Generating a sample downsweep call")
    dp = DownsweepParams()
    rp_ds = randomize_downsweep(dp)
    ds_audio = fin_whale_downsweep(rp_ds)
    logger.info(f"Downsweep: f0={rp_ds.f0:.1f} Hz -> f1={rp_ds.f1:.1f} Hz, "
          f"dur={rp_ds.dur:.3f}s, tau={rp_ds.tau:.3f}")
    plot_spec(ds_audio, int(rp_ds.fs), "Fin Whale Downsweep for 50 Hz component")


    logger.info("TESTING --> Generating a sample 20 Hz pulse call")
    pp = PulseParams()
    rp_p = randomize_pulse(pp)
    pulse_audio = fin_whale_pulse(rp_p)
    logger.info(f"Pulse: f0={rp_p.f0:.1f} Hz -> f1={rp_p.f1:.1f} Hz, "
          f"pulse_dur={rp_p.pulse_dur:.3f}s, gap={rp_p.inter_pulse_gap:.3f}s")
    plot_spec(pulse_audio, int(rp_p.fs), "Fin Whale 20 Hz Pulse (doublet)")
