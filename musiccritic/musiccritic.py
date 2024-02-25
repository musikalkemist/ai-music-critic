"""
This module is the entry point of the Critic application, which
critiques a song based on its music features and lyrics.
"""

import argparse
import os
from pathlib import Path

from musiccritic import Configs, configs
from musiccritic.chatgpt import ChatGPT
from musiccritic.chatgptpromptpreparer import ChatGPTPromptPreparer
from musiccritic.critic import Critic
from musiccritic.musicanalysis.essentiaembeddinganalyzer import (
    create_essentia_jamendo_analyzer,
    create_voice_gender_analyzer,
)
from musiccritic.musicanalysis.musicanalyzers import MusicAnalyzers
from musiccritic.musicanalysis.tempoanalyzer import TempoAnalyzer
from musiccritic.prompt import chat_gpt_messages
from musiccritic.whisper import Whisper


def main():
    """Main function that runs the Critic application."""

    command_line_args = _parse_command_line_args()
    song_path = Path(command_line_args.song_path)
    if not song_path.exists():
        print(f"The file {song_path} does not exist.")
        return

    music_analyzers = _create_music_analyzers(configs)
    lyrics_transcriber = Whisper(os.getenv("OPENAI_API_KEY"))
    prompt_preparer = ChatGPTPromptPreparer(chat_gpt_messages)
    text_generator = ChatGPT(os.getenv("OPENAI_API_KEY"))

    music_critic = Critic(
        music_analyzers, lyrics_transcriber, prompt_preparer, text_generator
    )
    critique = music_critic.critique(song_path)
    print(f"Here's the critique for your song:\n\n{critique}")


def _parse_command_line_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generates music critiques for a given song."
    )
    parser.add_argument(
        "song_path",
        type=str,
        help="The path to the audio file of the song to critique.",
    )
    return parser.parse_args()


def _create_music_analyzers(configs: Configs) -> MusicAnalyzers:
    """
    Initializes music analyzers based on provided configurations.

    Args:
        configs (Configs): Configuration settings for the analyzers.

    Returns:
        MusicAnalyzers: A collection of initialized music analyzers.
    """
    genres_analyzer = create_essentia_jamendo_analyzer(
        configs.GENRES_EMBEDDING_MODEL_PATH,
        configs.GENRES_MODEL_WEIGHTS_PATH,
        configs.GENRES_MODEL_METADATA_PATH,
        configs.GENRES_TOP_N_LABELS,
        "genres",
    )
    moods_analyzer = create_essentia_jamendo_analyzer(
        configs.MOODS_EMBEDDING_MODEL_PATH,
        configs.MOODS_MODEL_WEIGHTS_PATH,
        configs.MOODS_MODEL_METADATA_PATH,
        configs.MOODS_TOP_N_LABELS,
        "moods",
    )
    instruments_analyzer = create_essentia_jamendo_analyzer(
        configs.INSTRUMENTS_EMBEDDING_MODEL_PATH,
        configs.INSTRUMENTS_MODEL_WEIGHTS_PATH,
        configs.INSTRUMENTS_MODEL_METADATA_PATH,
        configs.INSTRUMENTS_TOP_N_LABELS,
        "instruments",
    )
    voice_analyzer = create_voice_gender_analyzer(
        configs.VOICE_EMBEDDING_MODEL_PATH,
        configs.VOICE_MODEL_WEIGHTS_PATH,
        configs.VOICE_MODEL_METADATA_PATH,
        "voice",
    )
    tempo_analyzer = TempoAnalyzer(configs.TEMPO_MODEL_WEIGHTS_PATH)
    analyzers = [
        genres_analyzer,
        moods_analyzer,
        instruments_analyzer,
        voice_analyzer,
        tempo_analyzer,
    ]
    return MusicAnalyzers(analyzers)


if __name__ == "__main__":
    main()
