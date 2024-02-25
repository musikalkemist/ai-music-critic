"""
This module integrates the TempoCNN model from Essentia for tempo analysis,
providing functionality to analyze audio signals and estimate their tempo.
"""

from pathlib import Path

import numpy as np
from essentia.standard import TempoCNN

from musiccritic import logger
from musiccritic.musicanalysis.musicanalyzer import MusicAnalyzer


class TempoAnalyzer(MusicAnalyzer):
    """
    A class that utilizes the TempoCNN model to analyze audio signals and
    estimate their tempo.

    Attributes:
        model (TempoCNN): The loaded TempoCNN model for tempo estimation.
    """

    def __init__(self, model_weights_path: Path) -> None:
        """
        Initializes the TempoAnalyzer with the TempoCNN model.

        Args:
            model_weights_path (Path): Path to the TempoCNN model's weights.
        """
        self.model = TempoCNN(graphFilename=str(model_weights_path))
        super().__init__("tempo")

    def analyze(self, audio: np.ndarray) -> int:
        """
        Analyzes an audio signal and estimates its global tempo.

        Args:
            audio (np.ndarray): The audio signal to be analyzed.

        Returns:
            float: The estimated global tempo of the audio signal.
        """
        global_tempo, _, _ = self.model(audio)
        logger.info(f"Predicted tempo: {global_tempo}")
        return round(global_tempo)
