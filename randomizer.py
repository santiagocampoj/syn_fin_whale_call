from class_params import DownsweepParams, PulseParams
import numpy as np


# ?????
np.random.seed(0)


def randomize(value: float, delta: float = 1.0, rel: bool = False) -> float:
    if rel:
        noise = np.random.uniform(-delta, delta) * value
    else:
        noise = np.random.uniform(-delta, delta)
    return value + noise


def randomize_downsweep(p: DownsweepParams, delta: float = 1.0) -> DownsweepParams:
    return DownsweepParams(
        fs=p.fs,
        dur=randomize(p.dur, delta=delta * 0.05, rel=True), # +-5% duration
        f0=randomize(p.f0, delta=delta * 2.0), # +-2 Hz start freq
        f1=randomize(p.f1, delta=delta * 1.0), # +-1 Hz end freq
        tau=randomize(p.tau, delta=delta * 0.03, rel=True), # +-3% decay
        alpha=p.alpha,
        harmonics=p.harmonics,
        amplitude=randomize(p.amplitude, delta=delta * 0.03, rel=True),
    )


# def randomize_pulse(p: PulseParams, delta: float = 1.0) -> PulseParams:
#     return PulseParams(
#         fs=p.fs,
#         pulse_dur=randomize(p.pulse_dur, delta=delta * 0.05, rel=True),# +-5% pulse duration
#         f0=randomize(p.f0, delta=delta * 0.5), # +-0.5 Hz start freq
#         f1=randomize(p.f1, delta=delta * 0.5), # +-0.5 Hz end freq
#         inter_pulse_gap=randomize(p.inter_pulse_gap, delta=delta * 0.01),# +-10 ms gap
#         amplitude=randomize(p.amplitude,delta=delta * 0.03, rel=True),# +-3% amplitude
#     )


def randomize_pulse(p: PulseParams, delta: float = 1.0) -> PulseParams:
    new_f0 = randomize(p.f0, delta=delta * 0.8) # ±0.8 Hz
    new_f1 = randomize(p.f1, delta=delta * 0.8) # ±0.8 Hz

    # ensure f0 > f1 always
    if new_f0 <= new_f1:
        new_f0, new_f1 = new_f1 + 0.5, new_f1

    return PulseParams(
        fs=p.fs,
        pulse_dur=randomize(p.pulse_dur, delta=delta * 0.03, rel=True), # ±3%
        f0=new_f0,
        f1=new_f1,
        inter_pulse_gap=randomize(p.inter_pulse_gap, delta=delta * 0.02), # ±0.02s
        amplitude=randomize(p.amplitude,delta=delta * 0.03, rel=True), # ±3%
        
        
        # ADSR: small perturbations so each call sounds slightly different
        attack_s=randomize(p.attack_s, delta=delta * 0.01), # ±0.01s
        decay_s=randomize(p.decay_s, delta=delta * 0.02), # ±0.02s
        sustain_level=randomize(p.sustain_level, delta=delta * 0.05), # ±0.05
        release_s=randomize(p.release_s, delta=delta * 0.03), # ±0.03s
    )