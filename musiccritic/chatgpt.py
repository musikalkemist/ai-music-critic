from typing import List

from openai import OpenAI

from musiccritic import logger


class ChatGPT:
    """
    A class for generating text using ChatGPT provided by OpenAI.

    This class encapsulates the functionality to interact with the OpenAI API
    for generating text responses based on a series of input messages.

    Attributes:
        openai_api_key (str): The API key required to authenticate requests to
            the OpenAI service.
        max_tokens (int): The maximum number of tokens to generate in the
            response. Defaults to 1000.
        temperature (float): Controls the randomness of the generated text.
            Higher values lead to more unpredictable outputs. Value must be
            between 0 and 2. Defaults to 0.7.
        model (str): The model identifier to use for text generation.
            Defaults to "gpt4".
    """

    def __init__(
        self,
        openai_api_key: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: str = "gpt-4",
    ) -> None:
        """Initializes with an API key.

        Args:
            openai_api_key: OpenAI API key.
            max_tokens: Maximum number of tokens to generate.
            temperature: The higher the value, the more random the generated
                text. Must be between 0 and 2.
            model: The model to use for generating text.
        """

        self.openai_api_key = openai_api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = model
        self._client = OpenAI(api_key=openai_api_key)

    def generate(self, messages: List) -> str:
        """Generates text using ChatGPT.

        Args:
            messages: A list of messages to feed to ChatGPT.

        Returns:
            The generated text.
        """
        logger.info("Generating text with '%s'", self.model)
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        generated_text = completion.choices[0].message.content
        logger.info("Generated text with '%s'", self.model)
        return generated_text
