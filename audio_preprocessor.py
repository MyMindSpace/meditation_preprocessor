import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import librosa
    import librosa.display  # noqa: F401
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "librosa is required. Install with: pip install librosa soundfile matplotlib numpy"
    ) from e

try:
    import soundfile as sf
except Exception:
    sf = None  # optional, librosa can load audio via audioread

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # spectrogram image saving becomes optional


def load_audio(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return y, sr


def reduce_noise_simple(y: np.ndarray, sr: int, noise_ms: int = 300) -> np.ndarray:
    """
    Very simple spectral gating using a noise profile estimated from the first noise_ms.
    This is lightweight and dependency-free; for production, prefer specialized libraries.
    """
    n_noise = max(1, int(sr * noise_ms / 1000))
    noise_clip = y[:n_noise]

    # STFT
    hop_length = 512
    win_length = 1024
    Y = librosa.stft(y, n_fft=win_length, hop_length=hop_length, win_length=win_length)
    N = librosa.stft(noise_clip, n_fft=win_length, hop_length=hop_length, win_length=win_length)

    Y_mag, Y_phase = np.abs(Y), np.angle(Y)
    N_mag = np.abs(N)
    noise_profile = np.median(N_mag, axis=1, keepdims=True)

    # Gate: subtract noise floor with a safety factor, floor at small epsilon
    gate_strength = 1.0
    cleaned_mag = np.maximum(Y_mag - gate_strength * noise_profile, 1e-6)

    # Reconstruct
    Y_clean = cleaned_mag * np.exp(1j * Y_phase)
    y_clean = librosa.istft(Y_clean, hop_length=hop_length, win_length=win_length, length=len(y))
    return y_clean.astype(np.float32)


def compute_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.astype(np.float32)


def compute_mel_spectrogram(y: np.ndarray, sr: int, n_mels: int = 80) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)


def save_spectrogram_png(S_db: np.ndarray, out_path: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 3))
    librosa.display.specshow(S_db, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def process_file(
    in_path: Path,
    out_dir: Path,
    sr: int,
    n_mfcc: int,
    n_mels: int,
    apply_denoise: bool,
    save_spec_png: bool,
) -> None:
    y, sr = load_audio(in_path, sr)
    if apply_denoise:
        y = reduce_noise_simple(y, sr)

    mfcc = compute_mfcc(y, sr, n_mfcc=n_mfcc)
    mel = compute_mel_spectrogram(y, sr, n_mels=n_mels)

    rel = in_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{rel}_mfcc.npy", mfcc)
    np.save(out_dir / f"{rel}_mel.npy", mel)

    if save_spec_png:
        save_spectrogram_png(mel, out_dir / f"{rel}_mel.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio Preprocessor (AP) for MFCCs, spectrograms, noise reduction")
    parser.add_argument("--input", type=str, default="librispeech", help="Input folder containing LibriSpeech-style WAVs")
    parser.add_argument("--output", type=str, default="audio_features", help="Output folder for features")
    parser.add_argument("--sr", type=int, default=16000, help="Target sampling rate")
    parser.add_argument("--n_mfcc", type=int, default=13, help="Number of MFCC coefficients")
    parser.add_argument("--n_mels", type=int, default=80, help="Number of mel bins for spectrogram")
    parser.add_argument("--denoise", action="store_true", help="Apply simple noise reduction")
    parser.add_argument("--png", action="store_true", help="Also save mel spectrogram PNGs")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    audio_paths = [
        Path(root) / f
        for root, _, files in os.walk(input_dir)
        for f in files
        if f.lower().endswith((".wav", ".flac"))
    ]

    if not audio_paths:
        print(f"No WAV/FLAC files found in {input_dir}")
        return

    print(f"Found {len(audio_paths)} audio files (wav/flac). Processing...")
    for wav in audio_paths:
        # mirror directory structure under output
        rel_parent = wav.parent.relative_to(input_dir)
        out_dir = output_dir / rel_parent
        process_file(
            wav,
            out_dir,
            sr=args.sr,
            n_mfcc=args.n_mfcc,
            n_mels=args.n_mels,
            apply_denoise=args.denoise,
            save_spec_png=args.png,
        )

    print(f"Done. Features saved under {output_dir}")


if __name__ == "__main__":
    main()


