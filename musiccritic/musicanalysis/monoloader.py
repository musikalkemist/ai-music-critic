from pathlib import Path

from essentia.standard import MonoLoader


def load_mono_audio(
    song_path: Path, sample_rate: int = 16000, resample_quality: int = 4
):
    return MonoLoader(
        filename=str(song_path),
        sampleRate=sample_rate,
        resampleQuality=resample_quality,
    )()
