"""
Synthetic fin whale call generator for database creation.

Two call types:
  - Downsweep (~50 Hz component): logarithmic chirp with exponential decay
  - 20 Hz pulse: short tonal call with slight downsweep, characteristic double pulse

Parameter randomization is applied per call to simulate natural biological variability.
A batch generator creates a labelled database of .wav files + a metadata CSV.
"""

import argparse

from class_params import DownsweepParams, PulseParams
from randomizer import randomize_downsweep, randomize_pulse
from synthesize_signal import fin_whale_downsweep, fin_whale_pulse
from utils_plot import plot_spec
from database_generator import generate_database
from logging_config import setup_logging



def argument_parser():
    parser = argparse.ArgumentParser(description="Process XML transcript pipeline")
    parser.add_argument("-p", "--path", required=False, help="Path to input XML file")
    return parser.parse_args()


def main():
    # initializing
    logger = setup_logging()
    args = argument_parser()
    logger.info(f"Starting fin whale call synthesis")


    ##########################
    # Downsweep
    dp = DownsweepParams()
    rp_ds = randomize_downsweep(dp)
    ds_audio = fin_whale_downsweep(rp_ds)
    logger.info(f"Downsweep: f0={rp_ds.f0:.1f} Hz -> f1={rp_ds.f1:.1f} Hz, "
          f"dur={rp_ds.dur:.3f}s, tau={rp_ds.tau:.3f}")
    plot_spec(ds_audio, int(rp_ds.fs), "Fin Whale Downsweep for 50 Hz component")


    ##########################
    #20 Hz pulse preview
    pp = PulseParams()
    rp_p = randomize_pulse(pp)
    pulse_audio = fin_whale_pulse(rp_p)
    logger.info(f"Pulse: f0={rp_p.f0:.1f} Hz -> f1={rp_p.f1:.1f} Hz, "
          f"pulse_dur={rp_p.pulse_dur:.3f}s, gap={rp_p.inter_pulse_gap:.3f}s")
    plot_spec(pulse_audio, int(rp_p.fs), "Fin Whale 20 Hz Pulse (doublet)")



    ##########################
    #test database
    generate_database(
        out_dir="fin_whale_db",
        n_downsweep=10,
        n_pulse=10,
        delta=1.0,
        seed=42,
        logger=logger,
    )


if __name__ == "__main__":
    main()