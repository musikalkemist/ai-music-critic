"""
This module provides a wrapper class to utilize multiple music analysis models
simultaneously on a given audio track. It facilitates running a collection of
analyses and aggregating their results.
"""

from pathlib import Path
from typing import Dict, List

from musiccritic.musicanalysis.monoloader import load_mono_audio
from musiccritic.musicanalysis.musicanalyzer import MusicAnalyzer


class MusicAnalyzers:
    """
    A class that aggregates multiple music analyzer instances to perform a
    comprehensive analysis on audio tracks.

    Attributes:
        analyzers (List[MusicAnalyzer]): A list of music analyzer instances.
    """

    def __init__(self, analyzers: List[MusicAnalyzer]):
        """
        Initializes the MusicAnalyzers class with a list of music analyzer
        instances.

        Args:
            analyzers (List[MusicAnalyzer]): Music analyzer instances for
                performing various analyses.
        """
        self.analyzers = analyzers

    def analyze(self, song_path: Path) -> Dict[str, any]:
        """
        Analyzes an audio track using all configured music analyzers and
        aggregates their results.

        Args:
            song_path (Path): The file path to the audio track to be analyzed.

        Returns:
            Dict[str, any]: A dictionary containing analysis results, with
                analyzer names as keys and their analysis outputs as values.
        """
        audio = load_mono_audio(song_path)
        analysis = {}
        for analyzer in self.analyzers:
            analysis[analyzer.analyzer_name] = analyzer.analyze(audio)
        return analysis
