import argparse
import soundfile as sf
import glob


from class_params import *
from randomizer import *
from synthesize_signal import *
from plots import plot_spec, plot_spec_file
from database_generator import *
from logging_config import setup_logging
from call_injection import *
from config import *
from snr_databse import *




def argument_parser():
    parser = argparse.ArgumentParser(description="Process XML transcript pipeline")
    parser.add_argument("-p", "--path", required=False, help="Path to input XML file")
    parser.add_argument("--create_db", action="store_true", help="Whether to create the synthetic database")
    parser.add_argument("--downsweep", action="store_true", help="Whether to synthesize downsweep calls")
    parser.add_argument("--pulse", action="store_true", help="Whether to synthesize pulse calls")
    parser.add_argument("--background-path", action="store_true", help="Enter a different path to the background recording")
    parser.add_argument("--injection", action="store_true", help="Whether to perform the call injection step")
    return parser.parse_args()




def main():
    # initializing
    logger = setup_logging()
    logger.info(f"Starting fin whale call synthesis")

    ################
    # testing
    # args = argument_parser()
    # plot_spec_file(path=r"C:\Users\Santi\OneDrive - UPV\TFM - Santiago Campo Jurado\datasets\AcousticTrends_BlueFinLibrary\BallenyIslands2015\pulses_20hz\sel0002_20150309_160000.wav", 
    #                title="20 Hz pulse", 
    #                save=False, 
    #                pad_s=1.0,
    #                logger=logger
    #                )
    ################

    args = argument_parser()
    create_database = args.create_db
    use_downsweep = args.downsweep or not args.pulse
    use_pulse = args.pulse or not args.downsweep
    background_path = args.background_path if args.background_path else SEA_RECORDING
    injection_enabled = args.injection or False



    ##########################
    # [1] GENERATE SYNTHETIC DATABASE
    if create_database:
        logger.info("")
        logger.info("Generating a test database with 10 downsweep and 10 pulse calls")
        generate_database(
            out_dir="fin_whale_db",
            n_downsweep=10 if use_downsweep else 0,
            n_pulse=10 if use_pulse else 0,
            delta=1.0, # parameter randomization range (0 = no randomization, 1 = full range)
            seed=42,
            logger=logger,
        )
    else:
        logger.info("")
        logger.info("Skipping database generation step (use --create_db to enable)")



    ###########################
    if not injection_enabled:
        logger.info("")
        logger.info("Skipping call injection step (use --injection to enable)")
        return

    else:
        # [2] INJECT CALLS INTO REAL OCEAN RECORDING
        logger.info("")
        logger.info(f"Loading background recording from {SEA_RECORDING}")
        dp = DownsweepParams()
        background, fs = load_background(
            wav_path=background_path,
            fs_target=int(dp.fs),
            chunk_start_s=0.0,
            chunk_dur_s=120.0,
            logger=logger,
        )



        ###########
        # [3] LOAD CALLS FROM SYNTHETIC DATABASE
        logger.info("Loading calls from synthetic database for injection")
        calls_to_inject = []
        if use_downsweep:
            for fpath in sorted(glob.glob("fin_whale_db/downsweep/*.wav")):
                audio, _ = sf.read(fpath, dtype="float32")
                calls_to_inject.append(("downsweep", audio))
        if use_pulse:
            for fpath in sorted(glob.glob("fin_whale_db/pulse/*.wav")):
                audio, _ = sf.read(fpath, dtype="float32")
                calls_to_inject.append(("pulse", audio))
        logger.info(f"Loaded {len(calls_to_inject)} calls from fin_whale_db/")




        # [4] INJECT CALLS INTO BACKGROUND RECORDING
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



        # [5] SAVE MIXED RECORDING + METADATA
        wav_path, csv_path = save_injection_results(
            mixed=mixed,
            fs=fs,
            events=events,
            out_dir="injection_test",
            out_name="mixed_60s",
            logger=logger,
        )



        #######################
        # [6] BUILD SNR DATABASE
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