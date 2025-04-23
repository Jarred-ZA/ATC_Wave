#!/usr/bin/env python3
"""
ATC Radio Monitor & Transcriber - Frontend GUI
A Streamlit-based frontend for the ATC radio monitoring system.
"""

import os
import sys
import time
import json
import subprocess
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import wavfile
import librosa
import librosa.display
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set page configuration
st.set_page_config(
    page_title="ATC Radio Monitor & Transcriber",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Create directories if they don't exist
os.makedirs('audio_archive', exist_ok=True)
os.makedirs('analysis_results', exist_ok=True)
os.makedirs('monitoring_results', exist_ok=True)
os.makedirs('audio_segments', exist_ok=True)

# Define paths
audio_archive_path = 'audio_archive'
analysis_results_path = 'analysis_results'
monitoring_results_path = 'monitoring_results'

# Function to get list of available audio files
def get_audio_files():
    """Get a list of all available audio files in the archive"""
    files = []
    for file in os.listdir(audio_archive_path):
        if file.endswith('.wav'):
            file_path = os.path.join(audio_archive_path, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            
            # Get modification time
            mod_time = os.path.getmtime(file_path)
            mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            
            # Get duration
            try:
                duration = get_audio_duration(file_path)
            except:
                duration = 0
                
            # Check if it has a transcription
            transcription_path = os.path.join(os.getcwd(), f"{file.replace('.wav', '_transcription.txt')}")
            has_transcription = os.path.exists(transcription_path)
            
            # Get transcription content if available
            transcription_content = None
            if has_transcription:
                try:
                    transcription_content = get_transcription(file_path)
                except:
                    pass
            
            # Check if it has an analysis result
            tower_name = file.split('_')[0] + '_' + file.split('_')[1]
            timestamp = file.split('_')[2].replace('.wav', '')
            monitoring_file = os.path.join(monitoring_results_path, f"{tower_name}_{timestamp}.txt")
            has_analysis = os.path.exists(monitoring_file)
            
            # Check for visualization
            viz_file = None
            analysis_summary = {}
            if has_analysis:
                # Try to find the visualization path and other analysis details in the monitoring file
                try:
                    with open(monitoring_file, 'r') as f:
                        content = f.read()
                        # Extract visualization path
                        for line in content.split('\n'):
                            if line.startswith('Visualization:'):
                                viz_file = line.split('Visualization:')[1].strip()
                            elif line.startswith('Voice Activity:'):
                                analysis_summary['voice_activity'] = line.split('Voice Activity:')[1].strip()
                            elif line.startswith('Energy:'):
                                analysis_summary['energy'] = line.split('Energy:')[1].strip()
                            elif line.startswith('Zero Crossing Rate:'):
                                analysis_summary['zcr'] = line.split('Zero Crossing Rate:')[1].strip()
                except:
                    pass
            
            # Get tower name
            tower = ' '.join(file.split('_')[:2]).replace('_', ' ')
            
            files.append({
                'filename': file,
                'path': file_path,
                'tower': tower,
                'size_mb': file_size,
                'duration': duration,
                'modified': mod_time_str,
                'has_transcription': has_transcription,
                'transcription_content': transcription_content,
                'has_analysis': has_analysis,
                'analysis_summary': analysis_summary,
                'visualization': viz_file,
                'monitoring_file': monitoring_file if has_analysis else None,
                'transcription_file': transcription_path if has_transcription else None
            })
    
    return sorted(files, key=lambda x: x['modified'], reverse=True)

def get_audio_duration(file_path):
    """Get the duration of an audio file in seconds"""
    try:
        y, sr = librosa.load(file_path, sr=None)
        return librosa.get_duration(y=y, sr=sr)
    except Exception as e:
        st.error(f"Error getting audio duration: {e}")
        return 0

def has_voice_activity(file_info):
    """Check if the file has voice activity based on monitoring results"""
    if not file_info['has_analysis'] or not file_info['monitoring_file']:
        return False
    
    try:
        with open(file_info['monitoring_file'], 'r') as f:
            content = f.read()
            return "Voice Activity: DETECTED" in content
    except:
        return False

def get_transcription(file_path):
    """Get the transcription for an audio file"""
    transcription_path = file_path.replace('.wav', '_transcription.txt')
    
    if os.path.exists(transcription_path):
        with open(transcription_path, 'r') as f:
            content = f.read()
            # Extract just the transcription part
            if "================" in content:
                parts = content.split("================")
                if len(parts) >= 2:
                    return parts[1].strip()
            return content
    
    return None

def transcribe_with_whisper(audio_path):
    """Transcribe audio using OpenAI Whisper API"""
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found. Please add it to your .env file.")
        return None
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Prepare prompt for ATC context
        prompt = "This is an air traffic control (ATC) radio transmission. It may contain standard aviation phraseology, callsigns, runway numbers, and aviation terminology."
        
        with st.spinner("Uploading audio to OpenAI API..."):
            # Open the audio file
            with open(audio_path, "rb") as audio_file:
                # Call the Whisper API
                with st.spinner("Processing with Whisper model..."):
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
                    
                # Save the transcription
                save_transcription(audio_path, text)
                
                return text
                
    except Exception as e:
        st.error(f"Whisper transcription error: {e}")
        return None

def save_transcription(audio_path, transcription):
    """Save transcription to a text file"""
    filename = os.path.basename(audio_path).replace('.wav', '_transcription.txt')
    filepath = os.path.join(os.getcwd(), filename)
    
    with open(filepath, 'w') as f:
        f.write(f"Transcription of {audio_path}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
        f.write(transcription)
        f.write("\n" + "=" * 80 + "\n")
    
    return filepath

def download_new_sample(tower_url, tower_name, duration):
    """Download a new audio sample from a tower"""
    try:
        command = ["python", "atc_monitor.py", str(duration)]
        
        with st.spinner(f"Downloading {duration}s sample from ATC feeds..."):
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Create a placeholder for the output
            output_placeholder = st.empty()
            
            # Display output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_placeholder.text(output.strip())
            
            # Check for errors
            rc = process.poll()
            if rc != 0:
                error_output = process.stderr.read()
                st.error(f"Error downloading sample: {error_output}")
                return False
            
            return True
            
    except Exception as e:
        st.error(f"Error downloading sample: {e}")
        return False

def display_waveform(audio_path):
    """Display audio waveform using plotly"""
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Calculate RMS energy for highlighting voice sections
        hop_length = int(sr * 0.01)  # 10ms windows
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times_rms = librosa.times_like(rms, sr=sr, hop_length=hop_length)
        
        # Determine voice activity threshold
        threshold = 0.015
        is_voice = rms > threshold
        
        # Create a figure with subplots
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.7, 0.3],
                           subplot_titles=("Audio Waveform", "Energy Level"))
        
        # Add waveform to top subplot
        fig.add_trace(go.Scatter(
            y=y,
            x=np.arange(len(y)) / sr,
            name="Waveform",
            line=dict(color='royalblue', width=1),
        ), row=1, col=1)
        
        # Add energy plot to bottom subplot
        fig.add_trace(go.Scatter(
            y=rms,
            x=times_rms,
            name="Energy (RMS)",
            line=dict(color='green', width=1.5),
        ), row=2, col=1)
        
        # Add threshold line
        fig.add_trace(go.Scatter(
            y=[threshold] * len(times_rms),
            x=times_rms,
            name="Voice Threshold",
            line=dict(color='red', width=1, dash='dash'),
        ), row=2, col=1)
        
        # Highlight voice regions on the waveform
        voice_regions = []
        start_idx = None
        
        for i, active in enumerate(is_voice):
            # Start of voice segment
            if active and start_idx is None:
                start_idx = i
            # End of voice segment
            elif not active and start_idx is not None:
                # Convert frame indices to time
                start_time = times_rms[start_idx]
                end_time = times_rms[i]
                voice_regions.append((start_time, end_time))
                start_idx = None
        
        # Handle the case where the last segment extends to the end
        if start_idx is not None:
            voice_regions.append((times_rms[start_idx], times_rms[-1]))
        
        # Add voice region highlights
        for start, end in voice_regions:
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="rgba(255, 255, 0, 0.2)",
                layer="below", line_width=0,
                row=1, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Energy", row=2, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error displaying waveform: {e}")
        return None

def display_visualization(viz_path):
    """Display saved visualization image with expanded view option"""
    if not viz_path or not os.path.exists(viz_path):
        st.warning("Visualization not found.")
        return
    
    try:
        image = Image.open(viz_path)
        
        # Display image with option to expand
        col1, col2 = st.columns([4, 1])
        with col1:
            st.image(image, caption="Audio Analysis Visualization", use_column_width=True)
        
        with col2:
            # Add a button to view in full screen
            if st.button("Full Screen", key="viz_fullscreen"):
                st.session_state.show_full_viz = True
        
        # Display full-screen visualization in a dialog
        if st.session_state.get('show_full_viz', False):
            with st.expander("Full Visualization", expanded=True):
                st.image(image, caption="Full Screen Visualization", use_column_width=True)
                if st.button("Close", key="close_viz"):
                    st.session_state.show_full_viz = False
                    st.rerun()
                
    except Exception as e:
        st.error(f"Error displaying visualization: {e}")

def main():
    """Main application function"""
    # Initialize session state
    if 'show_full_viz' not in st.session_state:
        st.session_state.show_full_viz = False
        
    # Page header with logo and title
    st.write("# 🛫 ATC Radio Monitor & Transcriber 🛬")
    
    # Display app info in the sidebar
    with st.sidebar:
        st.write("## App Controls")
        
        st.write("### Download New Sample")
        col1, col2 = st.columns(2)
        with col1:
            duration = st.number_input("Duration (seconds)", min_value=10, max_value=300, value=30, step=10)
        with col2:
            if st.button("Download Sample", type="primary"):
                success = download_new_sample(None, None, duration)
                if success:
                    st.success(f"Downloaded {duration}s sample successfully!")
                    st.rerun()
        
        st.write("---")
        st.write("### About")
        st.info("""
        This application allows you to:
        - Browse and play recorded ATC communications
        - View audio visualizations and analysis
        - Transcribe ATC communications with OpenAI Whisper
        - Download new samples from ATC towers
        """)
        
        st.write("### Whisper API Status")
        if OPENAI_API_KEY:
            st.success("✅ OpenAI API Key detected")
        else:
            st.error("❌ OpenAI API Key not found")
            st.write("Add your key to the .env file:")
            st.code("OPENAI_API_KEY=your_key_here")
    
    # Main content area - tabs
    tab1, tab2 = st.tabs(["Audio Library", "Documentation"])
    
    # Tab 1: Audio Library
    with tab1:
        # Get list of available audio files
        audio_files = get_audio_files()
        
        # Display audio files in a table
        if not audio_files:
            st.info("No audio files found in the archive. Download a new sample to get started.")
        else:
            # Create a dataframe for display
            df = pd.DataFrame(audio_files)
            
            # Create a custom format for the dataframe
            def highlight_row(row):
                if row['has_transcription']:
                    return ['background-color: rgba(52, 152, 219, 0.2)']*len(row)
                return [''] * len(row)
            
            # Select columns and rename them
            display_df = df[['filename', 'tower', 'duration', 'modified', 'has_transcription', 'has_analysis']]
            display_df.columns = ['Filename', 'Tower', 'Duration (s)', 'Recorded', 'Has Transcription', 'Has Analysis']
            
            # Display table with selection
            st.write("### Available Recordings")
            
            # Group files by tower
            towers = sorted(list(set([f['tower'] for f in audio_files])))
            
            # Allow filtering by tower
            selected_tower = st.selectbox("Filter by Tower:", 
                                         options=["All Towers"] + towers,
                                         index=0)
            
            # Filter files by tower if needed
            filtered_files = audio_files
            if selected_tower != "All Towers":
                filtered_files = [f for f in audio_files if f['tower'] == selected_tower]
            
            # Create options for the selectbox
            file_options = [f['filename'] for f in filtered_files]
            
            # Format the display of each option in the dropdown
            def format_file_option(filename):
                file_info = next((f for f in filtered_files if f['filename'] == filename), None)
                transcription_indicator = "📝 " if file_info and file_info['has_transcription'] else ""
                voice_indicator = "🔊 " if file_info and file_info.get('analysis_summary', {}).get('voice_activity') == "DETECTED" else ""
                return f"{transcription_indicator}{voice_indicator}{filename} ({file_info['tower']})"
            
            # Display selection dropdown with more info
            selection = st.selectbox("Select a recording to view details:", 
                                     options=file_options,
                                     format_func=format_file_option)
            
            # Display the dataframe with styling
            st.dataframe(display_df.style.apply(highlight_row, axis=1), use_container_width=True)
            
            # Display selected file details
            if selection:
                # Get the selected file info
                file_info = next((f for f in audio_files if f['filename'] == selection), None)
                
                if file_info:
                    st.write("---")
                    
                    # Create a container for the header with title and badges
                    header_container = st.container()
                    with header_container:
                        # Create columns for title and badges
                        title_col, badge_col = st.columns([3, 1])
                        with title_col:
                            st.write(f"### Details: {file_info['tower']}")
                        with badge_col:
                            # Display badges for transcription and voice activity
                            if file_info['has_transcription']:
                                st.markdown('<span style="background-color: #3498db; color: white; padding: 4px 8px; border-radius: 4px; margin-right: 5px;">📝 Transcribed</span>', unsafe_allow_html=True)
                            
                            voice_detected = file_info.get('analysis_summary', {}).get('voice_activity') == "DETECTED"
                            if voice_detected:
                                st.markdown('<span style="background-color: #2ecc71; color: white; padding: 4px 8px; border-radius: 4px;">🔊 Voice Activity</span>', unsafe_allow_html=True)
                            
                    # File details
                    st.write("#### Recording Information")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write(f"**Date:** {file_info['modified']}")
                    with col2:
                        st.write(f"**Duration:** {file_info['duration']:.2f} seconds")
                    with col3:
                        st.write(f"**Size:** {file_info['size_mb']:.2f} MB")
                    with col4:
                        # Show analysis data if available
                        if file_info.get('analysis_summary', {}).get('energy'):
                            st.write(f"**Energy:** {file_info['analysis_summary']['energy']}")
                    
                    # Show transcription at the top if available
                    if file_info['has_transcription'] and file_info.get('transcription_content'):
                        st.write("#### Transcription")
                        transcription_container = st.container()
                        with transcription_container:
                            st.info(file_info['transcription_content'])
                    
                    # Audio player
                    st.write("### Audio Player")
                    st.audio(file_info['path'])
                    
                    # Analyze for voice activity first
                    with st.spinner("Analyzing audio for voice activity..."):
                        # Display waveform with voice activity analysis
                        st.write("### Audio Analysis")
                        waveform_fig = display_waveform(file_info['path'])
                        if waveform_fig:
                            st.plotly_chart(waveform_fig, use_container_width=True)
                            
                        # Determine if there's voice activity from the waveform analysis
                        y, sr = librosa.load(file_info['path'], sr=None)
                        rms = librosa.feature.rms(y=y)[0]
                        energy = np.mean(rms)
                        threshold = 0.015
                        has_voice = energy > threshold
                        
                        # Voice activity status with icon and sound level
                        if has_voice:
                            st.success(f"🔊 Voice activity detected (Energy: {energy:.6f})")
                        else:
                            st.warning(f"🔇 No voice activity detected (Energy: {energy:.6f})")
                    
                    # Create a tab-based interface for visualizations
                    viz_tab1, viz_tab2 = st.tabs(["Interactive Analysis", "Detailed Visualizations"])
                    
                    with viz_tab1:
                        st.write("### Audio Analysis")
                        waveform_fig = display_waveform(file_info['path'])
                        if waveform_fig:
                            st.plotly_chart(waveform_fig, use_container_width=True)
                    
                    with viz_tab2:
                        st.write("### Detailed Spectrogram Analysis")
                        
                        # Show analysis summary if available
                        if file_info.get('analysis_summary'):
                            # Create columns for analysis data
                            st.write("#### Analysis Summary")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Voice activity status
                                voice_status = file_info['analysis_summary'].get('voice_activity', 'Unknown')
                                if voice_status == "DETECTED":
                                    st.success("🔊 Voice Activity: Detected")
                                else:
                                    st.warning("🔇 Voice Activity: Not Detected")
                                
                                # Energy level
                                energy = file_info['analysis_summary'].get('energy', 'Unknown')
                                st.metric("Energy Level", energy)
                            
                            with col2:
                                # Zero crossing rate
                                zcr = file_info['analysis_summary'].get('zcr', 'Unknown')
                                st.metric("Zero Crossing Rate", zcr)
                        
                        # Display the visualization image if available
                        if file_info['visualization'] and os.path.exists(file_info['visualization']):
                            st.write("#### Analysis Visualization")
                            display_visualization(file_info['visualization'])
                        else:
                            st.warning("No detailed visualization available for this recording.")
                    
                    # Transcription options section
                    st.write("### Transcription Options")
                    
                    if has_voice:
                        # If we already displayed the transcription at the top, just show options
                        if file_info['has_transcription'] and file_info.get('transcription_content'):
                            # Option to re-transcribe
                            if st.button("Generate New Transcription", type="secondary"):
                                with st.spinner("Transcribing audio..."):
                                    new_transcription = transcribe_with_whisper(file_info['path'])
                                    if new_transcription:
                                        st.success("New transcription complete!")
                                        st.info(new_transcription)
                                        time.sleep(1)  # Short pause to let user see the success message
                                        st.rerun()  # Refresh the page to show the updated transcription
                                    else:
                                        st.error("Transcription failed. See error message above.")
                        else:
                            # No transcription yet, so offer to create one
                            st.write("No transcription available for this recording.")
                            
                            # Transcribe button - only if there's voice activity
                            if st.button("Transcribe with Whisper", type="primary"):
                                with st.spinner("Transcribing audio..."):
                                    new_transcription = transcribe_with_whisper(file_info['path'])
                                    if new_transcription:
                                        st.success("Transcription complete!")
                                        st.info(new_transcription)
                                        time.sleep(1)  # Short pause to let user see the success message
                                        st.rerun()  # Refresh the page to show the new transcription
                                    else:
                                        st.error("Transcription failed. See error message above.")
                    else:
                        st.warning("⚠️ Transcription requires voice activity. No voice detected in this recording.")
    
    # Tab 2: Documentation
    with tab2:
        st.write("### Using the ATC Radio Monitor & Transcriber")
        
        st.write("""
        #### Audio Library
        - Browse all recorded ATC communications
        - Play audio files directly in the browser
        - View waveforms and analysis visualizations
        - Transcribe audio with OpenAI Whisper
        
        #### Downloading New Samples
        1. Set the desired duration (10-300 seconds)
        2. Click "Download Sample"
        3. Wait for the download to complete
        4. The new recordings will appear in the Audio Library
        
        #### Transcribing Audio
        1. Select a recording from the Audio Library
        2. Click "Transcribe with Whisper"
        3. Wait for the transcription to complete
        4. The transcription will be displayed and saved
        
        #### API Key Configuration
        To use the OpenAI Whisper transcription:
        1. Create a file named `.env` in the project directory
        2. Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`
        3. Restart the application
        """)
        
        st.write("### About ATC Feeds")
        
        st.write("""
        This application monitors and analyzes live Air Traffic Control (ATC) radio feeds from:
        
        - **Osaka Tower (RJOO)**: Osaka International Airport, Japan
        - **Boston Tower (KBOS)**: Boston Logan International Airport, USA
        
        These feeds are provided through LiveATC.net, which streams live ATC communications from airports around the world.
        """)

if __name__ == "__main__":
    main()