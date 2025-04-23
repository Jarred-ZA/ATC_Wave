# ATC Radio Monitor & Transcriber

A sophisticated tool to monitor air traffic control radio transmissions, detect voice activity with visual feedback, and transcribe communications with speech recognition technology.

![ATC Radio Monitor Banner](https://via.placeholder.com/800x200?text=ATC+Radio+Monitor)

## 🌟 Features

- **Live ATC Monitoring**: Download and analyze real-time air traffic control radio feeds
- **Voice Activity Detection**: Automatically detect presence of voice transmissions
- **Advanced Speech Recognition**: Transcribe ATC communications using OpenAI's Whisper AI model
- **Aviation Context Awareness**: Specialized prompts for accurate ATC terminology recognition
- **Audio Segmentation**: Identify and isolate individual voice transmissions
- **Visual Analytics**: Generate waveform, energy, and spectrogram visualizations
- **Detailed Reporting**: Save comprehensive analysis results to text files
- **Progress Tracking**: Real-time progress indicators for all operations
- **Command-line Interface**: Simple CLI with duration parameter support
- **Audio Archive**: Save all monitored transmissions for later analysis

## 📋 Prerequisites

- Python 3.7+
- ffmpeg (for audio conversion)
- curl (for downloading streams)
- OpenAI API key (for Whisper transcription, optional)

## 🔧 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Jarred-ZA/ATC_Wave.git
   cd ATC_Wave
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure ffmpeg and curl are installed on your system:
   - **macOS**: `brew install ffmpeg curl`
   - **Ubuntu/Debian**: `sudo apt install ffmpeg curl`
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/) and [curl.se](https://curl.se/)

4. Set up OpenAI API key (optional but recommended):
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file
   - Get your API key from [OpenAI](https://platform.openai.com/api-keys)

## 🚀 Usage

### Command-Line Usage

Run the monitor with default settings (60-second samples):

```bash
python atc_monitor.py
```

Specify a custom sampling duration in seconds:

```bash
python atc_monitor.py 120  # Sample for 2 minutes
```

### 📱 Interactive Frontend

The application includes a user-friendly web interface for managing recordings:

```bash
# Start the web interface
./run_frontend.sh

# Or run directly with Streamlit
streamlit run atc_frontend.py
```

The frontend provides:
- An audio library to browse, play, and manage recordings
- Visualizations and waveform displays
- Transcription capabilities with a single click
- Easy downloading of new samples
- Complete documentation

![ATC Frontend Screenshot](https://via.placeholder.com/800x400?text=ATC+Frontend+Screenshot)

### Using OpenAI Whisper

For better transcription results:

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
2. Add it to your `.env` file
3. Run the monitor or frontend as usual - it will automatically use Whisper for transcription

For detailed instructions on using Whisper (both API and CLI), see [WHISPER_INSTRUCTIONS.md](WHISPER_INSTRUCTIONS.md)

## 📊 Output Files

The program generates several types of output files:

### Analysis Results

Located in `analysis_results/` directory:
- PNG images containing waveform, energy level, and spectrogram visualizations
- Filename format: `analysis_YYYYMMDD_HHMMSS.png`

### Monitoring Results

Located in `monitoring_results/` directory:
- Text files containing detailed monitoring information
- Includes energy levels, zero crossing rates, and transcriptions
- Filename format: `Tower_Name_YYYYMMDD_HHMMSS.txt`

### Audio Archive

Located in `audio_archive/` directory:
- Full WAV recordings of monitored feeds
- Preserved for later analysis or verification
- Filename format: `Tower_Name_YYYYMMDD_HHMMSS.wav`

### Audio Segments

Located in `audio_segments/` directory:
- Individual segments containing isolated voice transmissions
- Each segment is transcribed separately
- Filename format: `Tower_Name_YYYYMMDD_HHMMSS_segment_N.wav`

## 🎯 How It Works

1. **Download**: Captures live streaming audio from ATC feeds
2. **Convert**: Transforms MP3 stream to WAV format for analysis
3. **Archive**: Saves a copy of the audio file for reference
4. **Analyze**: 
   - Calculates audio energy (RMS)
   - Determines zero crossing rate
   - Generates spectrogram
   - Applies threshold to detect voice activity
5. **Segment**: Identifies individual voice transmissions within the audio
6. **Transcribe**: Uses speech recognition to convert audio to text
   - Transcribes the complete audio
   - Also transcribes individual segments separately
7. **Visualize**: Creates graphs for waveform, energy, and frequency content
8. **Report**: Saves detailed results including transcriptions

## 📡 Supported ATC Feeds

Currently monitored towers:

| Tower | Location | Feed URL |
|-------|----------|----------|
| Osaka Tower | Osaka, Japan | https://s1-bos.liveatc.net/rjoo1 |
| Boston Tower | Boston, MA, USA | https://s1-bos.liveatc.net/kbos_twr |

## 🔧 Technical Details

### Voice Activity Detection

The system uses a threshold-based approach for voice activity detection:

- **Energy (RMS)**: Measures the overall volume level of the audio
- **Zero Crossing Rate**: Indicates frequency content and helps differentiate noise from speech

### Speech Recognition

For transcription, the system employs several techniques:

- **OpenAI Whisper API**: State-of-the-art model for speech recognition with high accuracy
- **Aviation Context Prompting**: Provides context about ATC terminology to improve results
- **Google Speech Recognition API**: Used as fallback when OpenAI API key is not available
- **Noise Reduction**: Pre-processes audio to enhance recognition quality
- **Segmentation**: Identifies distinct voice transmissions for targeted transcription
- **Energy Thresholding**: Focuses on high-energy segments likely to contain speech

## 📝 License

[MIT License](LICENSE)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🔄 Development Branches

### Main Branch
The main branch contains the core ATC monitoring and transcription functionality.

### Feature Branches

- **feature/frontend**: Interactive web interface for the ATC Monitor
  - Streamlit-based UI for managing recordings
  - Audio visualization and playback
  - Integrated transcription capabilities
  - Download controls for new samples
  - Detailed documentation in [FRONTEND_GUIDE.md](FRONTEND_GUIDE.md)

## 🔄 Future Improvements

- Add more sophisticated voice activity detection algorithms
- Improve transcription accuracy with aviation-specific language models
- Support for more ATC feeds worldwide
- Real-time alerting for detected transmissions
- Historical data analysis and pattern recognition
- Airport-specific callsign and terminology recognition
- Multi-language support for international towers
- Speech analytics to identify emergency situations