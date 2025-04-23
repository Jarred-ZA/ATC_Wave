# DBrain ATC Monitor

A simple program to monitor air traffic control radio transmissions and detect voice activity.

## Features

- Downloads live ATC radio streams from LiveATC.net
- Analyzes audio for voice activity using energy and zero-crossing rate
- Reports if transmissions are detected

## Prerequisites

- Python 3.7+
- ffmpeg (for audio conversion)
- curl (for downloading streams)

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the script to monitor the configured ATC feeds:

```
python atc_monitor.py
```

The script will:
1. Download 2 minutes of audio from each configured ATC feed
2. Analyze the audio for voice activity
3. Report whether transmissions were detected

## Supported ATC Feeds

- Osaka Tower (RJOO)
- Boston Tower (KBOS)