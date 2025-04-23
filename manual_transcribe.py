#!/usr/bin/env python3
"""
Script to manually transcribe an audio file using OpenAI Whisper API
"""

import os
import sys
from datetime import datetime
import openai
import json
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def transcribe_with_whisper(audio_path):
    """
    Transcribe speech using OpenAI's Whisper API
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎙️ Transcribing with Whisper...")
    
    if not OPENAI_API_KEY:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ OpenAI API key not found in .env file.")
        print(f"    Please add your API key to .env file: OPENAI_API_KEY=your_key_here")
        return None
    
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
                
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Whisper transcription complete")
            return text
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Whisper transcription error: {e}")
        return None

if __name__ == "__main__":
    # Check if file path is provided
    if len(sys.argv) < 2:
        print("Usage: python manual_transcribe.py <path_to_audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    # Verify file exists
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)
    
    # Transcribe the file
    print(f"Transcribing audio file: {audio_path}")
    transcription = transcribe_with_whisper(audio_path)
    
    if transcription:
        print("\nTRANSCRIPTION RESULT:")
        print("=" * 80)
        print(transcription)
        print("=" * 80)
        
        # Save to file
        filename = os.path.basename(audio_path).replace('.wav', '_transcription.txt')
        with open(filename, 'w') as f:
            f.write(f"Transcription of {audio_path}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
            f.write(transcription)
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"\nTranscription saved to: {filename}")
    else:
        print("Transcription failed. Please check your API key and try again.")