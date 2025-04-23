# ATC Radio Monitor

A sophisticated tool to monitor air traffic control radio transmissions and detect voice activity with visual feedback and detailed analytics.

![ATC Radio Monitor Banner](https://via.placeholder.com/800x200?text=ATC+Radio+Monitor)

## 🌟 Features

- **Live ATC Monitoring**: Download and analyze real-time air traffic control radio feeds
- **Voice Activity Detection**: Automatically detect presence of voice transmissions
- **Visual Analytics**: Generate waveform, energy, and spectrogram visualizations
- **Detailed Reporting**: Save comprehensive analysis results to text files
- **Progress Tracking**: Real-time progress indicators for all operations
- **Command-line Interface**: Simple CLI with duration parameter support

## 📋 Prerequisites

- Python 3.7+
- ffmpeg (for audio conversion)
- curl (for downloading streams)

## 🔧 Installation

1. Clone this repository:
   ```bash
   git clone [repository-url]
   cd atc-radio-monitor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure ffmpeg and curl are installed on your system:
   - **macOS**: `brew install ffmpeg curl`
   - **Ubuntu/Debian**: `sudo apt install ffmpeg curl`
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/) and [curl.se](https://curl.se/)

## 🚀 Usage

### Basic Usage

Run the monitor with default settings (30-second samples):

```bash
python atc_monitor.py
```

### Custom Duration

Specify a custom sampling duration in seconds:

```bash
python atc_monitor.py 60  # Sample for 60 seconds
```

## 📊 Output Files

The program generates two types of output files:

### Analysis Results

Located in `analysis_results/` directory:
- PNG images containing waveform, energy level, and spectrogram visualizations
- Filename format: `analysis_YYYYMMDD_HHMMSS.png`

### Monitoring Results

Located in `monitoring_results/` directory:
- Text files containing detailed monitoring information
- Includes energy levels, zero crossing rates, and analysis timestamps
- Filename format: `Tower_Name_YYYYMMDD_HHMMSS.txt`

## 🎯 How It Works

1. **Download**: Captures live streaming audio from ATC feeds
2. **Convert**: Transforms MP3 stream to WAV format for analysis
3. **Analyze**: 
   - Calculates audio energy (RMS)
   - Determines zero crossing rate
   - Generates spectrogram
   - Applies threshold to detect voice activity
4. **Visualize**: Creates graphs for waveform, energy, and frequency content
5. **Report**: Saves results and generates summary

## 📡 Supported ATC Feeds

Currently monitored towers:

| Tower | Location | Feed URL |
|-------|----------|----------|
| Osaka Tower | Osaka, Japan | https://s1-bos.liveatc.net/rjoo1 |
| Boston Tower | Boston, MA, USA | https://s1-bos.liveatc.net/kbos_twr |

## 🔧 Technical Details

The voice activity detection uses a simple threshold-based approach, analyzing:

- **Energy (RMS)**: Measures the overall volume level of the audio
- **Zero Crossing Rate**: Indicates frequency content and helps differentiate noise from speech

## 📝 License

[MIT License](LICENSE)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🔄 Future Improvements

- Add more sophisticated voice activity detection algorithms
- Support for more ATC feeds worldwide
- Web interface for monitoring and visualization
- Real-time alerting for detected transmissions
- Historical data analysis and pattern recognition