# ATC Radio Monitor & Transcriber: Frontend Architecture

This document provides a technical overview of the frontend architecture for the ATC Radio Monitor & Transcriber application.

## Technology Stack

The frontend is built using the following technologies:

- **Streamlit**: A Python framework for building interactive web applications
- **Plotly**: Used for interactive visualizations and waveform displays
- **Pandas**: For data manipulation and presentation
- **OpenAI API**: For Whisper-based speech recognition
- **Librosa**: For audio analysis and processing
- **Python-dotenv**: For environment variable management

## Component Architecture

```
┌─────────────────────────────────┐
│        Frontend Application      │
│  ┌─────────────┐ ┌─────────────┐ │
│  │ Audio Library│ │Documentation│ │
│  └─────────────┘ └─────────────┘ │
│  ┌─────────────┐ ┌─────────────┐ │
│  │ Controls    │ │ API Status  │ │
│  └─────────────┘ └─────────────┘ │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│     Core Processing Modules      │
│  ┌─────────────┐ ┌─────────────┐ │
│  │ File Manager│ │Transcription│ │
│  └─────────────┘ └─────────────┘ │
│  ┌─────────────┐ ┌─────────────┐ │
│  │ Audio Player│ │Visualization│ │
│  └─────────────┘ └─────────────┘ │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│        Backend Integration       │
│  ┌─────────────┐ ┌─────────────┐ │
│  │ ATC Monitor │ │ OpenAI API  │ │
│  └─────────────┘ └─────────────┘ │
└─────────────────────────────────┘
```

## Key Components

### 1. Main Application (`main()`)

The entry point for the Streamlit application that sets up the UI, controls the flow, and manages the application state.

### 2. Audio File Management

- **`get_audio_files()`**: Scans the audio archive for recordings and builds metadata for each file
- **`get_audio_duration()`**: Determines the duration of audio files using Librosa
- **`get_transcription()`**: Retrieves existing transcriptions for audio files

### 3. Visualization Components

- **`display_waveform()`**: Creates interactive waveform visualizations using Plotly
- **`display_visualization()`**: Renders saved analysis visualizations (spectrograms, energy plots)

### 4. Transcription Engine

- **`transcribe_with_whisper()`**: Interfaces with OpenAI's API to transcribe audio files
- **`save_transcription()`**: Saves transcription results to text files for future reference

### 5. Sample Acquisition

- **`download_new_sample()`**: Interfaces with the core ATC monitor to download new samples

## Data Flow

1. **Initialization**: The application loads environment variables and creates necessary directories
2. **File Discovery**: The system scans for existing recordings and analysis results
3. **User Interaction**: The user selects files or initiates actions (download, transcribe)
4. **Processing**: The system processes requests by calling appropriate functions
5. **Visualization**: Results are displayed in the UI with interactive elements
6. **Persistence**: New data (transcriptions, downloads) is saved to the file system

## File System Integration

The frontend interacts with several directories:

- **audio_archive/**: Contains all recorded WAV files
- **analysis_results/**: Holds visualization images generated during analysis
- **monitoring_results/**: Stores text reports from monitoring sessions
- **audio_segments/**: Contains segmented voice transmissions for focused analysis

## API Integration

The frontend integrates with the OpenAI API for transcription services:

1. API key is loaded from environment variables
2. Audio files are prepared and submitted to the Whisper API
3. Responses are processed and displayed in the UI
4. Transcriptions are saved for future reference

## Error Handling

The application implements comprehensive error handling:

- **API Errors**: Detected and displayed with helpful messages
- **File Operation Errors**: Managed with appropriate fallbacks
- **UI Feedback**: Visual indicators for success, warnings, and errors

## Performance Considerations

- **Lazy Loading**: Audio files are analyzed only when selected
- **Caching**: Transcriptions and analysis results are cached to disk
- **Progress Indicators**: Real-time feedback for long-running operations