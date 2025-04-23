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
import speech_recognition as sr
from pydub import AudioSegment
import re
import openai
import json
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

def transcribe_with_whisper(audio_path):
    """
    Transcribe speech using OpenAI's Whisper API
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎙️ Transcribing with Whisper...")
    
    if not OPENAI_API_KEY:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ OpenAI API key not found. Falling back to Google Speech Recognition.")
        return transcribe_with_google(audio_path)
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Prepare prompt for ATC context
        prompt = "This is an air traffic control (ATC) radio transmission. It may contain standard aviation phraseology, callsigns, runway numbers, and aviation terminology."
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Uploading audio to OpenAI API...")
        
        # Open the audio file
        with open(audio_path, "rb") as audio_file:
            # Call the Whisper API
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Processing with Whisper model...")
            
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
                prompt=prompt,
                response_format="verbose_json"
            )
            
            # Extract the transcription
            if isinstance(response, dict) and 'text' in response:
                text = response['text']
            else:
                text = response.text
                
            # Clean up the transcription
            text = re.sub(r'\s+', ' ', text).strip()
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Whisper transcription complete")
            return text
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Whisper transcription error: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ️ Falling back to Google Speech Recognition.")
        return transcribe_with_google(audio_path)

def transcribe_with_google(wav_path):
    """
    Transcribe speech using Google's Speech Recognition
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎙️ Transcribing with Google Speech Recognition...")
    
    try:
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Load the audio file
        with sr.AudioFile(wav_path) as source:
            # Read the entire audio file
            audio_data = recognizer.record(source)
            
            # Apply noise reduction
            recognizer.energy_threshold = 300
            
            # Attempt to recognize speech using Google's Speech Recognition
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Processing speech recognition...")
            text = recognizer.recognize_google(audio_data)
            
            # Clean up the transcription
            text = re.sub(r'\s+', ' ', text).strip()
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Google transcription complete")
            return text
    except sr.UnknownValueError:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ️ Speech recognition could not understand audio")
        return "No clear speech detected"
    except sr.RequestError as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Speech recognition service error: {e}")
        return f"Speech recognition service error: {e}"
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Transcription error: {e}")
        return f"Transcription error: {e}"

def transcribe_audio(wav_path):
    """
    Transcribe speech in audio file using the best available method
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎙️ Transcribing audio content...")
    
    # First try with Whisper (falls back to Google if API key not available)
    return transcribe_with_whisper(wav_path)

def segment_audio(wav_path):
    """
    Segment audio to find voice activity regions
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✂️ Segmenting audio to find voice regions...")
    
    # Load audio
    sample_rate, audio = wavfile.read(wav_path)
    audio_float = audio.astype(np.float32) / np.iinfo(np.int16).max
    
    # Calculate energy
    hop_length = int(sample_rate * 0.01)  # 10ms windows
    rms = librosa.feature.rms(y=audio_float, hop_length=hop_length)[0]
    
    # Find segments with activity
    threshold = 0.015
    is_active = rms > threshold
    
    # Find segment boundaries (convert frame indices to time)
    boundaries = []
    is_in_segment = False
    segment_start = 0
    
    for i, active in enumerate(is_active):
        # Start of segment
        if active and not is_in_segment:
            segment_start = i
            is_in_segment = True
        # End of segment
        elif not active and is_in_segment:
            # Only keep segments longer than 0.5 second
            if (i - segment_start) > (0.5 * sample_rate / hop_length):
                start_time = segment_start * hop_length / sample_rate
                end_time = i * hop_length / sample_rate
                boundaries.append((start_time, end_time))
            is_in_segment = False
    
    # Handle case where the last segment extends to the end
    if is_in_segment:
        start_time = segment_start * hop_length / sample_rate
        end_time = len(audio) / sample_rate
        boundaries.append((start_time, end_time))
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Found {len(boundaries)} voice segments")
    return boundaries, sample_rate, audio

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
    
    # Perform transcription if voice activity is detected
    transcription = None
    segments = []
    
    if has_activity:
        try:
            # Attempt direct transcription on whole file
            transcription = transcribe_audio(wav_path)
            
            # Also segment the audio to find specific regions with voice
            segments, _, _ = segment_audio(wav_path)
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error during transcription: {e}")
    
    return {
        'has_activity': has_activity,
        'energy': energy,
        'zero_crossing_rate': zcr,
        'plot_path': plot_path,
        'transcription': transcription,
        'segments': segments
    }

def save_audio_segments(wav_path, segments, sample_rate, audio, name):
    """
    Save individual audio segments for detailed analysis
    """
    if not segments:
        return []
    
    # Create directory for segments
    segments_dir = 'audio_segments'
    os.makedirs(segments_dir, exist_ok=True)
    
    # Base name for segments
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"{name.replace(' ', '_')}_{timestamp}"
    
    segment_files = []
    segment_transcriptions = []
    
    # Process each segment
    for i, (start_time, end_time) in enumerate(segments):
        # Convert time to samples
        start_sample = int(start_time * sample_rate)
        end_sample = min(int(end_time * sample_rate), len(audio))
        
        # Extract segment
        segment_audio = audio[start_sample:end_sample]
        
        # Save segment to file
        segment_file = os.path.join(segments_dir, f"{base_name}_segment_{i+1}.wav")
        wavfile.write(segment_file, sample_rate, segment_audio)
        
        # Try to transcribe this segment
        try:
            segment_text = transcribe_audio(segment_file)
            segment_transcriptions.append((i+1, start_time, end_time, segment_text))
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Could not transcribe segment {i+1}: {e}")
            segment_transcriptions.append((i+1, start_time, end_time, "Transcription failed"))
        
        segment_files.append(segment_file)
    
    return segment_files, segment_transcriptions

def monitor_atc_feed(url, name, duration=30):
    """
    Download and analyze an ATC feed
    """
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 Monitoring {name} at {url}")
    print(f"{'='*80}")
    
    # Create results directory
    results_dir = 'monitoring_results'
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(results_dir, f'{name.replace(" ", "_")}_{timestamp}.txt')
    
    # Create audio archive directory
    audio_dir = 'audio_archive'
    os.makedirs(audio_dir, exist_ok=True)
    
    # Initialize result file
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
        
        # Archive a copy of the WAV file if we want to keep it
        archived_wav = os.path.join(audio_dir, f'{name.replace(" ", "_")}_{timestamp}.wav')
        subprocess.run(['cp', wav_path, archived_wav], check=True)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Audio archived to: {archived_wav}")
        
        # Analyze for voice activity
        result = detect_voice_activity(wav_path)
        
        # Process audio segments and transcriptions if activity detected
        segment_files = []
        segment_transcriptions = []
        
        if result['has_activity'] and result['segments']:
            # Get audio data for segmentation
            segments, sample_rate, audio = segment_audio(wav_path)
            # Save individual segments and get transcriptions
            segment_files, segment_transcriptions = save_audio_segments(
                wav_path, segments, sample_rate, audio, name
            )
        
        # Clean up temporary files (keeping the archived WAV)
        os.remove(mp3_path)
        os.remove(wav_path)
        
        # Output results with visual indicators
        status_icon = "🔊" if result['has_activity'] else "🔇"
        status_text = "DETECTED" if result['has_activity'] else "NOT DETECTED"
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {status_icon} Voice activity {status_text} on {name}")
        print(f"  📊 Energy: {result['energy']:.6f}")
        print(f"  📊 Zero crossing rate: {result['zero_crossing_rate']:.6f}")
        print(f"  📈 Visualization saved to: {result['plot_path']}")
        
        # Display transcription if available
        if result['has_activity'] and result['transcription']:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🎙️ TRANSCRIPTION:")
            print(f"  \"{result['transcription']}\"")
        
        # Save detailed results to file
        with open(result_file, 'a') as f:
            f.write(f"Results Summary:\n")
            f.write(f"Voice Activity: {status_text}\n")
            f.write(f"Energy: {result['energy']:.6f}\n")
            f.write(f"Zero Crossing Rate: {result['zero_crossing_rate']:.6f}\n")
            f.write(f"Visualization: {result['plot_path']}\n")
            f.write(f"Audio Archive: {archived_wav}\n\n")
            
            # Add transcription if available
            if result['has_activity'] and result['transcription']:
                f.write(f"Transcription:\n")
                f.write(f"\"{result['transcription']}\"\n\n")
            
            # Add individual segment transcriptions if available
            if segment_transcriptions:
                f.write(f"Segment Transcriptions:\n")
                for seg_id, start, end, text in segment_transcriptions:
                    f.write(f"Segment {seg_id} ({start:.2f}s - {end:.2f}s): \"{text}\"\n")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📝 Detailed results saved to: {result_file}")
        
        # Return activity status and additional data
        return {
            'has_activity': result['has_activity'],
            'transcription': result.get('transcription'),
            'result_file': result_file,
            'plot_path': result['plot_path'],
            'archived_audio': archived_wav
        }
        
    except Exception as e:
        error_message = f"Error monitoring {name}: {str(e)}"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ {error_message}")
        
        # Save error to file
        with open(result_file, 'a') as f:
            f.write(f"ERROR: {error_message}\n")
        
        return {'has_activity': False, 'error': str(e)}

def print_banner():
    """Print a fancy banner for the application"""
    banner = """
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃                                                      ┃
    ┃        🛫 ATC RADIO MONITOR & TRANSCRIBER 🛬          ┃
    ┃                                                      ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    
    Monitoring air traffic control transmissions
    Detecting voice activity and transcribing communications
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
            duration = 60
    else:
        duration = 60  # Default duration of 60 seconds (1 minute)
    
    # Display configuration details
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️  Configuration:")
    print(f"  📊 Sample duration: {duration} seconds")
    print(f"  📂 Results directory: monitoring_results/")
    print(f"  📊 Visualizations directory: analysis_results/")
    print(f"  🔊 Audio archive directory: audio_archive/")
    print(f"  🎙️ Audio segments directory: audio_segments/")
    
    # Check speech recognition configuration
    if OPENAI_API_KEY:
        print(f"  🎯 Speech recognition: OpenAI Whisper API (preferred)")
    else:
        print(f"  🎯 Speech recognition: Google Speech Recognition (fallback)")
        print(f"  ℹ️ To use OpenAI Whisper, set OPENAI_API_KEY in .env file")
    
    # Create required directories
    os.makedirs('monitoring_results', exist_ok=True)
    os.makedirs('analysis_results', exist_ok=True)
    os.makedirs('audio_archive', exist_ok=True)
    os.makedirs('audio_segments', exist_ok=True)
    
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
        result = monitor_atc_feed(url, name, duration)
        results.append((name, result))
    
    # Print summary report
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📋 MONITORING SESSION SUMMARY")
    print(f"{'='*80}")
    
    active_feeds = []
    for name, result in results:
        has_activity = result.get('has_activity', False)
        status = "🔊 ACTIVE" if has_activity else "🔇 SILENT"
        
        # Add transcription information if available
        if has_activity and 'transcription' in result and result['transcription']:
            transcription_preview = result['transcription']
            if len(transcription_preview) > 40:
                transcription_preview = transcription_preview[:37] + "..."
            status += f" - \"{transcription_preview}\""
            active_feeds.append((name, result))
            
        print(f"  {name}: {status}")
    
    print(f"{'='*80}")
    
    # Print active feed details
    if active_feeds:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🎙️ ACTIVE FEEDS DETAIL")
        print(f"{'='*80}")
        
        for name, result in active_feeds:
            print(f"  📻 {name}:")
            print(f"    📄 Report: {result['result_file']}")
            print(f"    🔊 Audio: {result['archived_audio']}")
            print(f"    📊 Visualization: {result['plot_path']}")
            
            # Display transcription if available
            if result.get('transcription'):
                print(f"    🎙️ Transcription: \"{result['transcription']}\"")
            
            print()
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✅ Monitoring session completed")

if __name__ == "__main__":
    main()