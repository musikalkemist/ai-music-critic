"""
This module provides a class for preparing prompts for the ChatGPT model,
incorporating music analysis results and lyrics into a structured format
that can be utilized for generating music critiques or analyses.
"""

import copy
from typing import Any, Dict, List

from musiccritic import logger


class ChatGPTPromptPreparer:
    """
    A class for preparing structured prompts for the ChatGPT model by
    incorporating music analysis results and lyrics.

    Attributes:
        chat_gpt_messages (List[Dict[str, Any]]): A template or base structure
            for the prompts to be sent to ChatGPT.
    """

    def __init__(self, chat_gpt_messages: List[Dict[str, Any]]) -> None:
        self.chat_gpt_messages = chat_gpt_messages

    def prepare(self, music_analysis: dict, lyrics: str) -> List[Dict]:
        """
        Prepares a prompt for the ChatGPT model using the given music analysis
        and lyrics.

        Args:
            music_analysis (dict): The music analysis results.
            lyrics (str): The lyrics of the song to be critiqued.

        Returns:
            str: A prepared prompt for the ChatGPT model.
        """
        chat_gpt_messages = copy.deepcopy(self.chat_gpt_messages)
        filled_user_prompt = chat_gpt_messages[1]["content"].substitute(
            moods=music_analysis["moods"],
            genres=music_analysis["genres"],
            instruments=music_analysis["instruments"],
            voice=music_analysis["voice"],
            tempo=music_analysis["tempo"],
            lyrics=lyrics,
        )
        chat_gpt_messages[1]["content"] = filled_user_prompt
        logger.info("Prepared prompt for ChatGPT.")
        return chat_gpt_messages
