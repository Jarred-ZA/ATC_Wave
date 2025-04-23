#!/usr/bin/env python3
import os
import unittest
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np
from scipy.io import wavfile

# Import the module we want to test
import atc_monitor

class TestATCMonitor(unittest.TestCase):
    """Test cases for the ATC Monitor application"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create test directories
        os.makedirs(os.path.join(self.test_dir, 'analysis_results'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'monitoring_results'), exist_ok=True)

    def tearDown(self):
        """Tear down test fixtures"""
        # Clean up temporary files and directories
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch('subprocess.Popen')
    def test_download_stream(self, mock_popen):
        """Test the download_stream function"""
        # Mock the subprocess.Popen
        process_mock = MagicMock()
        process_mock.poll.return_value = None
        mock_popen.return_value = process_mock
        
        # Call the function with a short duration
        with patch('time.sleep'):  # Mock time.sleep to speed up test
            with patch('tqdm.tqdm'):  # Mock the progress bar
                temp_path = atc_monitor.download_stream("https://example.com/stream", duration=1)
        
        # Assert the file was created
        self.assertTrue(os.path.exists(temp_path))
        
        # Clean up
        os.remove(temp_path)

    @patch('subprocess.run')
    def test_convert_to_wav(self, mock_run):
        """Test the convert_to_wav function"""
        # Create a temporary mp3 file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            mp3_path = temp_file.name
        
        # Mock the subprocess.run call
        mock_run.return_value = MagicMock(returncode=0)
        
        # Call the function
        wav_path = atc_monitor.convert_to_wav(mp3_path)
        
        # Assert the expected wav path is returned
        self.assertEqual(wav_path, mp3_path.replace('.mp3', '.wav'))
        
        # Clean up
        os.remove(mp3_path)

    def test_detect_voice_activity(self):
        """Test the voice activity detection function"""
        # Create a synthetic WAV file with known properties
        sample_rate = 16000
        duration = 1.0  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Generate a silent audio file (should not detect activity)
        silent_audio = np.zeros_like(t)
        silent_wav_path = os.path.join(self.test_dir, 'silent.wav')
        wavfile.write(silent_wav_path, sample_rate, (silent_audio * 32767).astype(np.int16))
        
        # Generate an audio file with a tone (should detect activity)
        freq = 440.0  # A4 note
        active_audio = np.sin(2 * np.pi * freq * t)
        active_wav_path = os.path.join(self.test_dir, 'active.wav')
        wavfile.write(active_wav_path, sample_rate, (active_audio * 32767).astype(np.int16))
        
        # Patch the plt.savefig to prevent actual file saving
        with patch('matplotlib.pyplot.savefig'):
            with patch('matplotlib.pyplot.close'):
                # Test silent audio
                result_silent = atc_monitor.detect_voice_activity(silent_wav_path)
                
                # Test active audio
                result_active = atc_monitor.detect_voice_activity(active_wav_path)
        
        # Assert the results
        self.assertFalse(result_silent['has_activity'], "Silent audio should not detect activity")
        self.assertTrue(result_active['has_activity'], "Active audio should detect activity")
        
        # Clean up
        os.remove(silent_wav_path)
        os.remove(active_wav_path)

if __name__ == '__main__':
    unittest.main()