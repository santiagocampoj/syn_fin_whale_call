import os
import argparse
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from logging_config import setup_logging
from config import ABS_PATH_BallenyIslands2015,NFFT, OVERLAP, COLORMAP, FREQ_MIN, FREQ_MAX



def generate_spectrograms(clips_dir: str, output_dir: str, single_file: str = None, logger=None) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if single_file:
        wav_files = [single_file]
        logger.info(f"Single clip mode: {single_file}")
    else:
        wav_files = sorted([f for f in os.listdir(clips_dir) if f.endswith('.wav')])
        logger.info(f"Found {len(wav_files)} clips to visualize")

    pdf_path = os.path.join(output_dir, "all_spectrograms.pdf")

    with PdfPages(pdf_path) as pdf:
        for wav_file in wav_files:
            wav_path = os.path.join(clips_dir, wav_file)

            if not os.path.isfile(wav_path):
                logger.warning(f"File not found: {wav_path}")
                continue

            logger.info(f"Processing: {wav_file}")
            clip, sr = sf.read(wav_path)

            if clip.ndim > 1:
                clip = clip[:, 0]

            fig, ax = plt.subplots(figsize=(12, 4))

            im = ax.specgram(
                clip,
                NFFT=NFFT,
                Fs=sr,
                noverlap=int(NFFT * OVERLAP),
                window=np.hamming(NFFT),   # Hamming window
                cmap=COLORMAP,
                scale='dB',
                vmin=-80,                  # Range (dB) = 80
                vmax=20,                   # Gain (dB)  = 20
                pad_to=NFFT * 2,           # Zero padding factor = 2
            )[3]

            ax.set_ylim(FREQ_MIN, FREQ_MAX)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title(wav_file, fontsize=9)

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Amplitude (dB)")

            plt.tight_layout()

            png_path = os.path.join(output_dir, wav_file.replace('.wav', '.png'))
            fig.savefig(png_path, dpi=150)
            logger.info(f"Saved PNG: {png_path}")

            pdf.savefig(fig)
            plt.close(fig)

    logger.info(f"Saved PDF: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cetacean spectrogram visualizer")
    parser.add_argument(
        '--clip',
        type=str,
        default=None,
        help='Single WAV filename to process (e.g. sel0001_20150309_160000.wav)'
    )
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("")
    logger.info("Starting spectrogram visualization")
    logger.info("")

    clips_dir  = os.path.join(ABS_PATH_BallenyIslands2015, 'pulses_20hz')
    output_dir = os.path.join(ABS_PATH_BallenyIslands2015, 'spectrograms')

    generate_spectrograms(clips_dir, output_dir, single_file=args.clip, logger=logger)

    logger.info("")
    logger.info("Done")


if __name__ == "__main__":
    main()