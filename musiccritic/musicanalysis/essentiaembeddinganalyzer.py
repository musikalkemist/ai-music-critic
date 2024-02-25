"""
This module integrates various Essentia TensorFlow models for audio analysis,
allowing for the extraction of embeddings and subsequent classification into
categories such as music genres, moods, or vocal characteristics.

Classes:
    EssentiaEmbeddingAnalyzer: Analyzes audio using specified Essentia models.

Functions:
    create_essentia_jamendo_analyzer: Initializes an analyzer for Jamendo dataset.
    create_voice_gender_analyzer: Initializes an analyzer for voice gender.
"""

from pathlib import Path
from typing import List

import numpy as np
from essentia.standard import (
    TensorflowPredict2D,
    TensorflowPredictEffnetDiscogs,
    TensorflowPredictVGGish,
)

from musiccritic import logger
from musiccritic.musicanalysis.musicanalyzer import MusicAnalyzer
from musiccritic.musicanalysis.scoretolabelconverter import (
    ScoreToLabelConverter,
)


class EssentiaEmbeddingAnalyzer(MusicAnalyzer):
    """
    A class to analyze audio using embedding and classification models from
    Essentia's TensorFlow implementation.

    Attributes:
        embedding_model: Model to generate audio embeddings.
        model: Classification model for prediction based on embeddings.
        score_to_label_converter (ScoreToLabelConverter): Converts prediction
            scores to meaningful labels.
    """

    def __init__(
        self,
        embedding_model,
        model,
        score_to_label_converter: ScoreToLabelConverter,
        analyzer_name: str,
    ):
        """
        Initializes the analyzer with models and a converter.

        Args:
            embedding_model: Essentia embedding model instance.
            model: Essentia classification model instance.
            score_to_label_converter (ScoreToLabelConverter): Instance for
                converting scores to labels.
        """
        self.embedding_model = embedding_model
        self.model = model
        self.score_to_label_converter = score_to_label_converter
        super().__init__(analyzer_name)

    def analyze(self, audio: np.ndarray) -> List[str]:
        """
        Analyzes an audio signal and returns predicted labels.

        Args:
            audio (np.ndarray): The audio signal to analyze.

        Returns:
            List[str]: Predicted labels for the audio signal.
        """
        embeddings = self.embedding_model(audio)
        prediction_scores = self.model(embeddings)
        flattened_prediction_scores = np.sum(prediction_scores, axis=0)
        labels = self.score_to_label_converter.convert_top_n(
            flattened_prediction_scores
        )
        logger.info(f"Predicted labels: {labels}")
        return labels


def create_essentia_jamendo_analyzer(
    embedding_model_path: Path,
    model_weights_path: Path,
    model_metadata_path: Path,
    top_n: int,
    analyzer_name: str,
) -> EssentiaEmbeddingAnalyzer:
    """
    Factory function to create an analyzer for music using the Jamendo dataset.

    Args:
        embedding_model_path (Path): Path to the embedding model's graph file.
        model_weights_path (Path): Path to the model weights graph file.
        model_metadata_path (Path): Path to the model's metadata file.
        top_n (int): Number of top predictions to convert to labels.

    Returns:
        EssentiaEmbeddingAnalyzer: Configured music analyzer instance.
    """
    score_to_label_converter = ScoreToLabelConverter(
        model_metadata_path, top_n=top_n
    )
    embedding_model = TensorflowPredictEffnetDiscogs(
        graphFilename=str(embedding_model_path), output="PartitionedCall:1"
    )
    model = TensorflowPredict2D(graphFilename=str(model_weights_path))

    return EssentiaEmbeddingAnalyzer(
        embedding_model, model, score_to_label_converter, analyzer_name
    )


def create_voice_gender_analyzer(
    embedding_model_path: Path,
    model_weights_path: Path,
    model_metadata_path: Path,
    analyzer_name: str,
) -> EssentiaEmbeddingAnalyzer:
    """
    Factory function to create an analyzer for detecting voice gender.

    Args:
        embedding_model_path (Path): Path to the VGGish embedding model's graph.
        model_weights_path (Path): Path to the classification model's graph.
        model_metadata_path (Path): Path to the model's metadata file.

    Returns:
        EssentiaEmbeddingAnalyzer: Configured voice gender analyzer instance.
    """
    score_to_label_converter = ScoreToLabelConverter(
        model_metadata_path, top_n=1
    )
    embedding_model = TensorflowPredictVGGish(
        graphFilename=str(embedding_model_path),
        output="model/vggish/embeddings",
    )
    model = TensorflowPredict2D(
        graphFilename=str(model_weights_path), output="model/Softmax"
    )

    return EssentiaEmbeddingAnalyzer(
        embedding_model, model, score_to_label_converter, analyzer_name
    )
