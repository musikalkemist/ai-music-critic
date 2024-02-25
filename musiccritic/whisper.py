from pathlib import Path

from openai import OpenAI

from musiccritic import logger


class Whisper:
    """
    A class for transcribing speech to text using OpenAI's Whisper API.

    This class provides an interface for transcribing audio files into text
    by leveraging OpenAI's Whisper model. It requires an OpenAI API key for
    authentication and allows specifying a model version for transcription.

    Attributes:
        openai_api_key (str): The API key for authenticating requests to OpenAI.
        model (str): The version of the Whisper model to use for transcription.
            Defaults to 'whisper-1'.
    """

    def __init__(
        self,
        openai_api_key: str,
        model: str = "whisper-1",
    ) -> None:
        """
        Initializes the Whisper class with the necessary authentication details
        and model version.

        Args:
            openai_api_key: The API key for authenticating requests to
                OpenAI.
            model: The version of the Whisper model to use for
                transcription. Defaults to 'whisper-1'.
        """
        self.openai_api_key = openai_api_key
        self.model = model
        self._client = OpenAI(api_key=openai_api_key)

    def transcribe(self, audio_file_path: Path) -> str:
        """
        Transcribes the audio content of the given file into text using
        OpenAI's Whisper model.

        This method reads the specified audio file, sends it to the Whisper
        API for transcription, and returns the transcribed text.

        Args:
            audio_file_path: The file path of the audio file to transcribe.

        Returns:
            The transcribed text of the audio file.
        """
        logger.info(
            "Transcribing audio file '%s' with Whisper API.", audio_file_path
        )
        with open(audio_file_path, "rb") as audio_file:
            transcription = self._client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
            )
        logger.info(
            "Finished transcribing audio file '%s' with Whisper API.",
            audio_file_path,
        )
        return transcription.text
