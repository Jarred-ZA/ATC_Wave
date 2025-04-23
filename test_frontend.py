#!/usr/bin/env python3
"""
Unit tests for the ATC Radio Monitor & Transcriber Frontend
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Import the frontend module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import atc_frontend

class TestATCFrontend(unittest.TestCase):
    """Test cases for the ATC Frontend application"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.audio_archive = os.path.join(self.test_dir, "audio_archive")
        self.analysis_results = os.path.join(self.test_dir, "analysis_results")
        self.monitoring_results = os.path.join(self.test_dir, "monitoring_results")
        
        os.makedirs(self.audio_archive, exist_ok=True)
        os.makedirs(self.analysis_results, exist_ok=True)
        os.makedirs(self.monitoring_results, exist_ok=True)
        
        # Create test files
        self.create_test_files()

    def tearDown(self):
        """Tear down test fixtures"""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    def create_test_files(self):
        """Create test files for unit tests"""
        # Create sample WAV file
        with open(os.path.join(self.audio_archive, "Boston_Tower_20250423_test.wav"), "wb") as f:
            f.write(b"RIFF....WAVEfmt ")  # Minimal WAV header
        
        # Create sample analysis file
        with open(os.path.join(self.analysis_results, "analysis_test.png"), "wb") as f:
            f.write(b"PNG")  # Fake PNG file
        
        # Create sample monitoring result with voice activity
        with open(os.path.join(self.monitoring_results, "Boston_Tower_20250423_test.txt"), "w") as f:
            f.write("ATC Monitoring Results: Boston Tower\n")
            f.write("Timestamp: 2025-04-23 10:00:00\n")
            f.write("URL: https://example.com\n")
            f.write("Duration: 30 seconds\n\n")
            f.write("Results Summary:\n")
            f.write("Voice Activity: DETECTED\n")
            f.write("Energy: 0.050000\n")
            f.write("Zero Crossing Rate: 0.100000\n")
            f.write(f"Visualization: {os.path.join(self.analysis_results, 'analysis_test.png')}\n")
        
        # Create sample transcription file
        with open(os.path.join(self.test_dir, "Boston_Tower_20250423_test_transcription.txt"), "w") as f:
            f.write("Transcription of sample audio\n")
            f.write("Timestamp: 2025-04-23 10:01:00\n")
            f.write("=" * 80 + "\n")
            f.write("Test transcription content\n")
            f.write("=" * 80 + "\n")

    @patch('atc_frontend.get_audio_duration')
    def test_get_audio_files(self, mock_get_duration):
        """Test get_audio_files function"""
        # Set up mock for duration
        mock_get_duration.return_value = 30.0
        
        # Patch the necessary functions and paths
        with patch('atc_frontend.audio_archive_path', self.audio_archive):
            with patch('atc_frontend.analysis_results_path', self.analysis_results):
                with patch('atc_frontend.monitoring_results_path', self.monitoring_results):
                    with patch('os.getcwd') as mock_getcwd:
                        mock_getcwd.return_value = self.test_dir
                        
                        # Patch file operations
                        with patch('os.listdir') as mock_listdir:
                            mock_listdir.return_value = ["Boston_Tower_20250423_test.wav"]
                            
                            with patch('os.path.getsize') as mock_getsize:
                                mock_getsize.return_value = 1024 * 1024  # 1 MB
                                
                                with patch('os.path.getmtime') as mock_getmtime:
                                    mock_getmtime.return_value = 1714042800  # Fixed timestamp
                                    
                                    with patch('os.path.exists') as mock_exists:
                                        mock_exists.return_value = True
                                        
                                        # Call the function
                                        result = atc_frontend.get_audio_files()
                                        
                                        # Check the result
                                        self.assertEqual(len(result), 1)
                                        self.assertEqual(result[0]['filename'], "Boston_Tower_20250423_test.wav")
                                        self.assertEqual(result[0]['tower'], "Boston Tower")
                                        self.assertEqual(result[0]['duration'], 30.0)
                                        self.assertTrue(result[0]['has_analysis'])

    @patch('atc_frontend.OPENAI_API_KEY', 'test_key')
    @patch('openai.OpenAI')
    def test_transcribe_with_whisper(self, mock_openai):
        """Test transcribe_with_whisper function"""
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.text = "Test transcription"
        mock_client.audio.transcriptions.create.return_value = mock_response
        
        # Mock the save_transcription function
        with patch('atc_frontend.save_transcription') as mock_save:
            mock_save.return_value = "test_transcription.txt"
            
            # Mock streamlit
            with patch('streamlit.spinner'):
                with patch('builtins.open', unittest.mock.mock_open(read_data=b'test')):
                    # Call the function
                    result = atc_frontend.transcribe_with_whisper("test.wav")
                    
                    # Check the result
                    self.assertEqual(result, "Test transcription")
                    mock_client.audio.transcriptions.create.assert_called_once()
                    mock_save.assert_called_once()

    def test_has_voice_activity(self):
        """Test has_voice_activity function"""
        # Create test file info
        file_info = {
            'has_analysis': True,
            'monitoring_file': os.path.join(self.monitoring_results, "Boston_Tower_20250423_test.txt")
        }
        
        # Test with voice activity
        with patch('builtins.open', unittest.mock.mock_open(read_data="Voice Activity: DETECTED")):
            result = atc_frontend.has_voice_activity(file_info)
            self.assertTrue(result)
        
        # Test without voice activity
        with patch('builtins.open', unittest.mock.mock_open(read_data="Voice Activity: NOT DETECTED")):
            result = atc_frontend.has_voice_activity(file_info)
            self.assertFalse(result)
        
        # Test with no analysis
        file_info['has_analysis'] = False
        result = atc_frontend.has_voice_activity(file_info)
        self.assertFalse(result)

    def test_get_transcription(self):
        """Test get_transcription function"""
        # Create test path
        test_path = os.path.join(self.audio_archive, "Boston_Tower_20250423_test.wav")
        
        # Test with existing transcription
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            with patch('builtins.open', unittest.mock.mock_open(read_data="Header\n================\nTest transcription\n================")):
                result = atc_frontend.get_transcription(test_path)
                self.assertEqual(result, "Test transcription")
        
        # Test with no transcription
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            result = atc_frontend.get_transcription(test_path)
            self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()