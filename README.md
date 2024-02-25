# AI Music Critic

MusicCritic is an AI-powered application designed to analyze 
music tracks, transcribe lyrics, and generate critiques 
leveraging machine learning models, including 
Essentia for audio analysis and OpenAI's GPT for text generation. 

## YouTube Video
This repo contains the code for the *I coded an AI that roasts pop songs* 
video on *The Sound of AI* YouTube channel. You can watch the video to 
learn more about the project and see the AI in action.

## Features
- **Music Analysis**: Analyzes music tracks for genres, moods, instruments, 
voice genders, and tempos using the Essentia audio analysis library.
- **Lyrics Transcription**: Utilizes OpenAI's Whisper model for 
  speech-to-text transcription of song lyrics. 
- **AI-Generated Critiques**: Generates detailed and engaging music critiques 
  based on the analysis and transcribed lyrics, powered by ChatGPT.

## Prerequisites
- Python 3.10.13 
- Poetry for dependency management

## Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/yourgithub/musiccritic.git 
   cd musiccritic
   ```
2. Install the dependencies using Poetry:
   ```bash
    poetry install
    ```

## Environment Variables
Create a `.env` file in the root of the project and add the following environment variables:
```bash
OPENAI_API_KEY=your_openai_api_key
```

## Usage
1. Download the pre-trained Essentia models and the relative metadata JSON 
   files from [Essentia's website](https://essentia.upf.edu/models.html) and place them in the `models` 
   directory. Check the `configs.py` file for the expected model names. If 
   you want to change the Essentia models you use, remember to change them in 
   the config file as well.
2. Obtain an API key from OpenAI and set it as an environment variable.
2. After installation, run the application launching the `musiccritic` script:
   ```bash
   cd musiccritic
   poetry run musiccritic path/to/music/file/to/analyse.mp3
   ```
   Alternatively, if you've activated the virtual environment, you can run:
   ```bash
   musiccritic path/to/music/file/to/analyse.wav
   ```    

## Dependencies
The application relies on the following libraries and APIs:
- [Essentia ML Models](https://essentia.upf.edu/models.html) for music analysis
- [OpenAI's Whisper](https://platform.openai.com/docs/guides/speech-to-text) for speech-to-text transcription
- [OpenAI's GPT-4](https://platform.openai.com/docs/guides/text-generation/chat-completions-api) for text generation

## License
This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.