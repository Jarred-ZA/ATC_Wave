# ATC Radio Monitor & Transcriber: Frontend Guide

This guide provides detailed information about using the interactive frontend for the ATC Radio Monitor & Transcriber.

## Getting Started

### Installation

1. Install all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure the OpenAI API key (for transcription):
   ```bash
   # Copy the sample environment file
   cp .env.sample .env
   
   # Edit the .env file and add your API key
   nano .env
   ```

### Launch the Frontend

Run the frontend using either:

```bash
# Using the convenience script
./run_frontend.sh

# Or directly with Streamlit
streamlit run atc_frontend.py
```

The frontend will open in your default web browser at `http://localhost:8501`.

## Features

### Audio Library

The Audio Library tab provides a complete overview of all recorded ATC communications:

- **Audio Files Table**: Lists all recordings with important details like tower, duration, recording date, and analysis/transcription status
- **Selection**: Choose any recording to view its details
- **Audio Player**: Listen to the selected recording directly in your browser
- **Advanced Waveform Display**: View the interactive audio waveform with voice regions highlighted
- **Energy Level Analysis**: See real-time energy levels with voice activity threshold indicators
- **Analysis Results**: Explore detailed analysis visualizations including spectrograms in collapsible sections
- **Intelligent Transcription**: System automatically determines if audio contains voice before offering transcription
- **Transcription Management**: Read existing transcriptions or create new ones with a single click

### Downloading New Samples

You can download new ATC recordings without leaving the frontend:

1. Set the desired duration (in seconds) using the sidebar control
2. Click the "Download Sample" button
3. Watch the progress in real-time
4. The new recording will automatically appear in your library when complete

### Transcribing Audio

To transcribe a recording with OpenAI Whisper:

1. Select the recording from the Audio Library
2. Scroll to the Transcription section
3. Click the "Transcribe with Whisper" button
4. Wait for the process to complete
5. The transcription will be displayed and saved for future reference

## Understanding the Interface

### Main Sections

- **App Controls** (Sidebar): Contains controls for downloading new samples and displays API status
- **Audio Library** (Tab): Browse and interact with all recorded ATC communications
- **Documentation** (Tab): Provides usage instructions directly in the app

### Status Indicators

- **Voice Activity**: 🔊 Shows if voice was detected in the recording
- **API Status**: Indicates if the OpenAI API key is configured correctly
- **Download Progress**: Displays real-time progress when downloading new samples

## Troubleshooting

### Missing API Key

If you see "OpenAI API Key not found" in the sidebar:

1. Ensure you've created a `.env` file in the project root directory
2. Add your OpenAI API key in the format: `OPENAI_API_KEY=your_key_here`
3. Restart the frontend application

### Transcription Errors

If transcription fails:

1. Check your OpenAI API key validity and quota
2. Ensure the audio file is in a valid format
3. Try a recording with clearer voice activity

### Other Issues

If the application crashes or exhibits strange behavior:

1. Check the terminal output for error messages
2. Ensure all dependencies are installed correctly
3. Restart the application