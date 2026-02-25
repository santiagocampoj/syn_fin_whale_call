"""
Synthetic fin whale call generator for database creation.

Two call types:
  - Downsweep (~50 Hz component): logarithmic chirp with exponential decay
  - 20 Hz pulse: short tonal call with slight downsweep, characteristic double pulse

Parameter randomization is applied per call to simulate natural biological variability.
A batch generator creates a labelled database of .wav files + a metadata CSV.
"""

import argparse

from class_params import *
from randomizer import *
from synthesize_signal import *
from plots import plot_spec
from database_generator import *
from logging_config import setup_logging
from call_injection import *
from config import *




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
    # TEST
    test_synth(logger=logger)



    ##########################
    # database
    logger.info("")
    logger.info("Generating a test database with 10 downsweep and 10 pulse calls")
    generate_database(
        out_dir="fin_whale_db",
        n_downsweep=10,
        n_pulse=10,
        delta=1.0,
        seed=42,
        logger=logger,
    )



    ###########################
    # TODO:
    #injection dataset into a real sea noise recording
    logger.info("")
    logger.info("Injecting synthetic calls into a real sea noise recording")
    exit()

    background, fs = load_background(
        wav_path=SEA_RECORDING,
        fs_target=int(dp.fs),
        chunk_start_s=0.0,
        chunk_dur_s=60.0,
        logger=logger,
    )

    # play the background audio to listen to the output
    





if __name__ == "__main__":
    main()