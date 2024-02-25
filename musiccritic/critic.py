"""
This module provides the Critic class, integrating various components
for analyzing music, transcribing lyrics, preparing prompts, and generating
text-based critiques using ChatGPT.
"""

from pathlib import Path

from musiccritic.chatgpt import ChatGPT
from musiccritic.chatgptpromptpreparer import ChatGPTPromptPreparer
from musiccritic.musicanalysis.musicanalyzers import MusicAnalyzers
from musiccritic.whisper import Whisper


class Critic:
    """
    Facilitates comprehensive music critique by combining music analysis,
    lyrics transcription, prompt preparation, and critique generation.

    Attributes:
        music_analyzers (MusicAnalyzers): For analyzing music attributes.
        lyrics_transcriber (Whisper): For transcribing song lyrics.
        prompt_preparer (ChatGPTPromptPreparer): For preparing prompts for ChatGPT.
        text_generator (ChatGPT): For generating text-based critiques.
    """

    def __init__(
        self,
        music_analyzers: MusicAnalyzers,
        lyrics_transcriber: Whisper,
        prompt_preparer: ChatGPTPromptPreparer,
        text_generator: ChatGPT,
    ) -> None:
        self.music_analyzers = music_analyzers
        self.lyrics_transcriber = lyrics_transcriber
        self.prompt_preparer = prompt_preparer
        self.text_generator = text_generator

    def critique(self, song_path: Path) -> str:
        """
        Generates a text-based critique for a given song.

        Args:
            song_path (Path): The path to the audio file of the song.

        Returns:
            str: A text-based critique of the song combining its analysis
            and transcribed lyrics.
        """
        music_analysis = self.music_analyzers.analyze(song_path)
        lyrics = self.lyrics_transcriber.transcribe(song_path)
        prompt = self.prompt_preparer.prepare(music_analysis, lyrics)
        critique = self.text_generator.generate(prompt)
        return critique
