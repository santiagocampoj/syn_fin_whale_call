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
from snr_databse import *
import soundfile as sf
import glob




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
    # test_synth(logger=logger)



    ##########################
    # database
    logger.info("")
    logger.info("Generating a test database with 10 downsweep and 10 pulse calls")
    generate_database(
        out_dir="fin_whale_db",
        n_downsweep=10,
        n_pulse=10,
        delta=1.0, # parameter randomization range (0 = no randomization, 1 = full range)
        seed=42,
        logger=logger,
    )



    ###########################
    # TODO:
    #injection dataset into a real sea noise recording
    logger.info("")
    logger.info(f"Loading background recording from {SEA_RECORDING}")
    dp = DownsweepParams()
    background, fs = load_background(
        wav_path=SEA_RECORDING,
        fs_target=int(dp.fs),
        chunk_start_s=0.0,
        chunk_dur_s=120.0,
        logger=logger,
    )



    ###########
    logger.info("Loading calls from synthetic database for injection")

    calls_to_inject = []
    for fpath in sorted(glob.glob("fin_whale_db/downsweep/*.wav")):
        audio, _ = sf.read(fpath, dtype="float32")
        calls_to_inject.append(("downsweep", audio))

    for fpath in sorted(glob.glob("fin_whale_db/pulse/*.wav")):
        audio, _ = sf.read(fpath, dtype="float32")
        calls_to_inject.append(("pulse", audio))
    logger.info(f"Loaded {len(calls_to_inject)} calls from fin_whale_db/")




    logger.info("")
    logger.info("Injecting calls into ocean recording")
    mixed, events = inject_calls_into_recording(
        background=background,
        fs=fs,
        calls=calls_to_inject,
        snr_db_range=(5.0, 20.0), # random SNR between 5 and 20 dB per call
        min_gap_s=2.0, # at least 2 s between injected calls
        seed=42,
        logger=logger,
    )

    wav_path, csv_path = save_injection_results(
        mixed=mixed,
        fs=fs,
        events=events,
        out_dir="injection_test",
        out_name="mixed_60s",
        logger=logger,
    )




    logger.info("")
    logger.info("Building SNR database from mixed recording")
    build_snr_database(
        mixed_wav=wav_path,
        labels_csv=csv_path,
        fs=fs,
        out_dir="snr_database",
        padding_s=0.5, # add x s of real ocean noise before/after each clip
        logger=logger,
    )

    logger.info("")
    logger.info("Execution complete")



if __name__ == "__main__":
    main()