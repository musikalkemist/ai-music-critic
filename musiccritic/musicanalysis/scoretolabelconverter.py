"""
This module provides a class for converting numerical prediction scores
into corresponding class labels based on model metadata. It selects the
top N scores and maps them to their respective labels.
"""

import json
from pathlib import Path
from typing import List

import numpy as np


class ScoreToLabelConverter:
    """
    A class that converts model prediction scores into class labels.

    Attributes:
        labels (List[str]): A list of class labels loaded from model metadata.
        top_n (int): Number of top scores to convert into labels.

    Methods:
        convert(scores: np.ndarray) -> List[str]:
            Converts an array of scores into class labels.
    """

    def __init__(self, model_metadata_path: Path, top_n: int = 5):
        """
        Initializes the ScoreToLabelConverter with model metadata and top_n.

        Args:
            model_metadata_path (Path): Path to the model metadata JSON file,
                which includes class labels.
            top_n (int): Number of top prediction scores to convert to labels.
        """
        with open(model_metadata_path, "r") as f:
            self.labels = json.load(f)["classes"]
        self.top_n = top_n

    def convert_top_n(self, scores: np.ndarray) -> List[str]:
        """
        Converts an array of scores into a list of class labels.

        Args:
            scores (np.ndarray): An array of prediction scores.

        Returns:
            List[str]: A list of class labels corresponding to the top N scores.
        """
        top_n_indices = self._get_top_n_indices(scores)
        return [self.labels[i] for i in top_n_indices]

    def _get_top_n_indices(self, scores: np.ndarray) -> List[int]:
        """
        Identifies the indices of the top N scores.

        Args:
            scores (np.ndarray): An array of prediction scores.

        Returns:
            List[int]: Indices of the top N scores.
        """
        sorted_indices = scores.argsort()
        top_n_indices = sorted_indices[-self.top_n :]
        top_n_indices_descending = top_n_indices[::-1]
        return top_n_indices_descending
