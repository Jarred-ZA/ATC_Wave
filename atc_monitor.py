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
import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

def download_stream(url, duration=120):
    """
    Download audio stream from LiveATC for specified duration
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📡 Downloading {duration}s from {url}")
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
        temp_path = temp_file.name
    
    # Create a progress bar for the download
    progress_bar = tqdm.tqdm(total=duration, unit='s', desc="Download Progress", 
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} seconds')
    
    # Use curl to download the stream for a specific duration with progress updates
    start_time = time.time()
    process = subprocess.Popen(f"curl -s {url} > {temp_path}", shell=True)
    
    # Update progress bar while downloading
    while process.poll() is None and time.time() - start_time < duration:
        elapsed = min(int(time.time() - start_time), duration)
        progress_bar.update(elapsed - progress_bar.n)
        time.sleep(0.1)
    
    # Kill the process if it's still running
    if process.poll() is None:
        process.terminate()
        process.wait()
    
    progress_bar.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Download complete")
    
    return temp_path

def convert_to_wav(mp3_path):
    """
    Convert mp3 to wav using ffmpeg
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Converting audio to WAV format...")
    wav_path = mp3_path.replace('.mp3', '.wav')
    
    try:
        subprocess.run(['ffmpeg', '-i', mp3_path, '-ar', '16000', '-ac', '1', wav_path, '-y', '-loglevel', 'error'], check=True)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Conversion complete")
        return wav_path
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Conversion failed: {str(e)}")
        raise

def detect_voice_activity(wav_path):
    """
    Analyze audio file for voice activity
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Analyzing audio for voice activity...")
    
    # Load audio file
    sample_rate, audio = wavfile.read(wav_path)
    
    # Convert to float and normalize
    audio = audio.astype(np.float32) / np.iinfo(np.int16).max
    
    # Calculate energy and display visual feedback
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Calculating audio energy...")
    rms = librosa.feature.rms(y=audio)[0]
    energy = np.mean(rms)
    
    # Calculate zero crossing rate
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Calculating zero crossing rate...")
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio)[0])
    
    # Basic voice activity detection
    threshold = 0.015  # Threshold may need adjustment
    has_activity = energy > threshold
    
    # Generate and save visualization
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Plot the waveform and energy
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(3, 1, 1)
    plt.plot(audio)
    plt.title('Waveform')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    
    # Plot energy (RMS)
    plt.subplot(3, 1, 2)
    frames = np.arange(len(rms))
    plt.plot(frames, rms)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.title(f'Energy (RMS) - Mean: {energy:.4f}')
    plt.xlabel('Frames')
    plt.ylabel('Energy')
    plt.legend()
    
    # Plot spectrogram
    plt.subplot(3, 1, 3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    plt.imshow(D, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'analysis_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📈 Generated visualization: {plot_path}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Analysis complete")
    
    return {
        'has_activity': has_activity,
        'energy': energy,
        'zero_crossing_rate': zcr,
        'plot_path': plot_path
    }

def monitor_atc_feed(url, name, duration=30):
    """
    Download and analyze an ATC feed
    """
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 Monitoring {name} at {url}")
    print(f"{'='*80}")
    
    results_dir = 'monitoring_results'
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(results_dir, f'{name.replace(" ", "_")}_{timestamp}.txt')
    
    with open(result_file, 'w') as f:
        f.write(f"ATC Monitoring Results: {name}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"URL: {url}\n")
        f.write(f"Duration: {duration} seconds\n\n")
    
    try:
        # Download the stream
        mp3_path = download_stream(url, duration)
        
        # Convert to WAV
        wav_path = convert_to_wav(mp3_path)
        
        # Analyze for voice activity
        result = detect_voice_activity(wav_path)
        
        # Clean up temporary files
        os.remove(mp3_path)
        os.remove(wav_path)
        
        # Output results with visual indicators
        status_icon = "🔊" if result['has_activity'] else "🔇"
        status_text = "DETECTED" if result['has_activity'] else "NOT DETECTED"
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {status_icon} Voice activity {status_text} on {name}")
        print(f"  📊 Energy: {result['energy']:.6f}")
        print(f"  📊 Zero crossing rate: {result['zero_crossing_rate']:.6f}")
        print(f"  📈 Visualization saved to: {result['plot_path']}")
        
        # Save detailed results to file
        with open(result_file, 'a') as f:
            f.write(f"Results Summary:\n")
            f.write(f"Voice Activity: {status_text}\n")
            f.write(f"Energy: {result['energy']:.6f}\n")
            f.write(f"Zero Crossing Rate: {result['zero_crossing_rate']:.6f}\n")
            f.write(f"Visualization: {result['plot_path']}\n")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📝 Detailed results saved to: {result_file}")
        return result['has_activity']
        
    except Exception as e:
        error_message = f"Error monitoring {name}: {str(e)}"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ {error_message}")
        
        # Save error to file
        with open(result_file, 'a') as f:
            f.write(f"ERROR: {error_message}\n")
        
        return False

def print_banner():
    """Print a fancy banner for the application"""
    banner = """
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃                                         ┃
    ┃        🛫 ATC RADIO MONITOR 🛬           ┃
    ┃                                         ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    
    Monitoring air traffic control transmissions
    Looking for voice activity in real-time feeds
    """
    print(banner)

def main():
    """Main application entry point"""
    print_banner()
    
    # Get duration from command line argument or use default
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid duration specified. Using default.")
            duration = 30
    else:
        duration = 30  # Default duration of 30 seconds
    
    # Display configuration details
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️  Configuration:")
    print(f"  📊 Sample duration: {duration} seconds")
    print(f"  📂 Results directory: monitoring_results/")
    print(f"  📊 Visualizations directory: analysis_results/")
    
    # Create required directories
    os.makedirs('monitoring_results', exist_ok=True)
    os.makedirs('analysis_results', exist_ok=True)
    
    # List of ATC feeds to monitor
    feeds = [
        ("https://s1-bos.liveatc.net/rjoo1", "Osaka Tower"),
        ("https://s1-bos.liveatc.net/kbos_twr", "Boston Tower"),
    ]
    
    # Start monitoring process
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting monitoring session")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📡 {len(feeds)} feeds will be monitored")
    
    # Track results for summary
    results = []
    
    # Monitor each feed
    for url, name in feeds:
        has_activity = monitor_atc_feed(url, name, duration)
        results.append((name, has_activity))
    
    # Print summary report
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📋 MONITORING SESSION SUMMARY")
    print(f"{'='*80}")
    for name, has_activity in results:
        status = "🔊 ACTIVE" if has_activity else "🔇 SILENT"
        print(f"  {name}: {status}")
    print(f"{'='*80}")
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✅ Monitoring session completed")

if __name__ == "__main__":
    main()