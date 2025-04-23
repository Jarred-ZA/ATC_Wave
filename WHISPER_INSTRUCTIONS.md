# Using OpenAI Whisper for ATC Transcription

This document provides instructions on how to use OpenAI's Whisper API or the local Whisper CLI tool to transcribe your ATC radio recordings.

## Option 1: Using OpenAI Whisper API (Recommended)

### 1. Get an OpenAI API Key

1. Create an account on [OpenAI](https://platform.openai.com/) if you don't have one
2. Navigate to [API Keys](https://platform.openai.com/api-keys)
3. Create a new secret key
4. Copy the key (you won't be able to see it again after closing the page)

### 2. Set up your API Key

1. In the project directory, edit the `.env` file
2. Add your API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
3. Save the file

### 3. Run the ATC Monitor

Now when you run the ATC Monitor, it will automatically use the Whisper API for transcription when voice activity is detected:

```bash
python atc_monitor.py 60  # Run for 60 seconds
```

## Option 2: Using Local Whisper CLI Tool

If you prefer to manually transcribe your recordings using the local Whisper CLI tool:

### 1. Install Whisper Locally

```bash
# Install Whisper
pip install -U openai-whisper

# Install FFmpeg
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# Download from ffmpeg.org and add to PATH
```

### 2. Transcribe an Audio File

```bash
# Basic transcription
whisper audio_archive/your_recording.wav --language en

# With more options (higher accuracy)
whisper audio_archive/your_recording.wav --language en --model medium --condition_on_previous_text False
```

### 3. Get Help with Whisper CLI

```bash
whisper --help
```

## Option 3: Using Our Manual Transcribe Script

We've included a manual transcription script that can be used with the OpenAI API key:

```bash
python manual_transcribe.py audio_archive/your_recording.wav
```

This will transcribe the file and save the result to a text file with the same name as the audio file.

## Tips for Better Transcriptions

1. Use longer recordings (60+ seconds) to capture more context
2. Look for recordings with higher energy levels (shown in the monitoring results)
3. Try the `medium` or `large` models with local Whisper for better accuracy with ATC terminology
4. For online API transcription, our system already includes aviation context prompting