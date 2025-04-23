#!/usr/bin/env python3
import os
import sys
import time
import urllib.request
import tempfile
import subprocess
import numpy as np
from scipy.io import wavfile
import librosa

def download_stream(url, duration=120):
    """
    Download audio stream from LiveATC for specified duration
    """
    print(f"Downloading {duration}s from {url}")
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
        temp_path = temp_file.name
    
    # Use curl to download the stream for a specific duration
    command = f"curl -s {url} > {temp_path} & pid=$!; sleep {duration}; kill $pid"
    subprocess.run(command, shell=True)
    
    return temp_path

def convert_to_wav(mp3_path):
    """
    Convert mp3 to wav using ffmpeg
    """
    wav_path = mp3_path.replace('.mp3', '.wav')
    subprocess.run(['ffmpeg', '-i', mp3_path, '-ar', '16000', '-ac', '1', wav_path, '-y', '-loglevel', 'error'], check=True)
    return wav_path

def detect_voice_activity(wav_path):
    """
    Analyze audio file for voice activity
    """
    # Load audio file
    sample_rate, audio = wavfile.read(wav_path)
    
    # Convert to float and normalize
    audio = audio.astype(np.float32) / np.iinfo(np.int16).max
    
    # Calculate energy
    energy = np.mean(librosa.feature.rms(y=audio)[0])
    
    # Calculate zero crossing rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio)[0])
    
    # Basic voice activity detection
    has_activity = energy > 0.015  # Threshold may need adjustment
    
    return {
        'has_activity': has_activity,
        'energy': energy,
        'zero_crossing_rate': zcr
    }

def monitor_atc_feed(url, name):
    """
    Download and analyze an ATC feed
    """
    print(f"\nMonitoring {name} at {url}")
    
    try:
        # Download the stream
        mp3_path = download_stream(url)
        
        # Convert to WAV
        wav_path = convert_to_wav(mp3_path)
        
        # Analyze for voice activity
        result = detect_voice_activity(wav_path)
        
        # Clean up temporary files
        os.remove(mp3_path)
        os.remove(wav_path)
        
        # Output results
        if result['has_activity']:
            print(f"✓ Voice activity detected on {name}")
            print(f"  Energy: {result['energy']:.6f}")
            print(f"  Zero crossing rate: {result['zero_crossing_rate']:.6f}")
        else:
            print(f"✗ No voice activity detected on {name}")
            print(f"  Energy: {result['energy']:.6f}")
            print(f"  Zero crossing rate: {result['zero_crossing_rate']:.6f}")
            
        return result['has_activity']
        
    except Exception as e:
        print(f"Error monitoring {name}: {str(e)}")
        return False

def main():
    feeds = [
        ("https://s1-bos.liveatc.net/rjoo1", "Osaka Tower"),
        ("https://s1-bos.liveatc.net/kbos_twr", "Boston Tower"),
    ]
    
    print("ATC Radio Transmission Monitor")
    print("==============================")
    
    for url, name in feeds:
        monitor_atc_feed(url, name)

if __name__ == "__main__":
    main()