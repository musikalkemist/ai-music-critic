from pathlib import Path

models_dir = Path(__file__).parents[1] / "models"


class Configs:
    """
    A class that contains paths and configurations for the music analysis
    models.
    """

    GENRES_MODEL_WEIGHTS_PATH = (
        models_dir / "mtg_jamendo_genre-discogs-effnet-1.pb"
    )
    GENRES_MODEL_METADATA_PATH = (
        models_dir / "mtg_jamendo_genre-discogs-effnet-1.json"
    )
    GENRES_EMBEDDING_MODEL_PATH = models_dir / "discogs-effnet-bs64-1.pb"
    GENRES_TOP_N_LABELS = 3

    MOODS_MODEL_WEIGHTS_PATH = (
        models_dir / "mtg_jamendo_moodtheme-discogs-effnet-1.pb"
    )
    MOODS_MODEL_METADATA_PATH = (
        models_dir / "mtg_jamendo_moodtheme-discogs-effnet-1.json"
    )
    MOODS_EMBEDDING_MODEL_PATH = models_dir / "discogs-effnet-bs64-1.pb"
    MOODS_TOP_N_LABELS = 4

    INSTRUMENTS_MODEL_WEIGHTS_PATH = (
        models_dir / "mtg_jamendo_instrument-discogs-effnet-1.pb"
    )
    INSTRUMENTS_MODEL_METADATA_PATH = (
        models_dir / "mtg_jamendo_instrument-discogs-effnet-1.json"
    )
    INSTRUMENTS_EMBEDDING_MODEL_PATH = models_dir / "discogs-effnet-bs64-1.pb"
    INSTRUMENTS_TOP_N_LABELS = 6

    VOICE_MODEL_WEIGHTS_PATH = models_dir / "gender-audioset-vggish-1.pb"
    VOICE_MODEL_METADATA_PATH = models_dir / "gender-audioset-vggish-1.json"
    VOICE_EMBEDDING_MODEL_PATH = models_dir / "audioset-vggish-3.pb"
    VOICE_TOP_N_LABELS = 1

    TEMPO_MODEL_WEIGHTS_PATH = models_dir / "deepsquare-k16-3.pb"
