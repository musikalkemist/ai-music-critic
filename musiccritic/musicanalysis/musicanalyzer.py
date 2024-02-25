from abc import ABC, abstractmethod


import numpy as np


class MusicAnalyzer(ABC):
    """
    MusicAnalyzer is an abstract for analyzing music files and
    extracting tags.
    """

    def __init__(self, analyzer_name: str) -> None:
        self.analyzer_name = analyzer_name

    @abstractmethod
    def analyze(self, audio: np.ndarray):
        pass
