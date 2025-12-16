"""
Audio Processor Module
Handles loading, preprocessing, and preparing audio files for transcription.
"""

import os
import librosa
import soundfile as sf
import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class AudioProcessor:
    """
    Process audio files for speech-to-text transcription.
    Supports multiple audio formats and provides preprocessing capabilities.
    """
    
    def __init__(self, sample_rate: int = 16000, max_duration: int = 300):
        """
        Initialize AudioProcessor.
        
        Args:
            sample_rate: Target sample rate for audio (Hz)
            max_duration: Maximum audio duration to process (seconds)
        """
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        
    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file from path.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio_path = str(audio_path)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"Error loading audio file: {str(e)}")
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data.
        
        Args:
            audio: Raw audio data
            
        Returns:
            Preprocessed audio data
        """
        # Trim silence from beginning and end
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Limit duration if necessary
        max_samples = self.sample_rate * self.max_duration
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        return audio
    
    def save_audio(self, audio: np.ndarray, output_path: Union[str, Path], 
                   sample_rate: Optional[int] = None) -> None:
        """
        Save audio to file.
        
        Args:
            audio: Audio data to save
            output_path: Output file path
            sample_rate: Sample rate (uses default if None)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        output_path = str(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        sf.write(output_path, audio, sample_rate)
    
    def get_audio_info(self, audio_path: Union[str, Path]) -> dict:
        """
        Get information about audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        audio, sr = self.load_audio(audio_path)
        
        duration = len(audio) / sr
        
        return {
            'path': str(audio_path),
            'sample_rate': sr,
            'duration_seconds': duration,
            'samples': len(audio),
            'channels': 1,  # Mono after loading
            'format': Path(audio_path).suffix
        }
    
    def process_audio_file(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, dict]:
        """
        Complete processing pipeline for an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (processed_audio, info_dict)
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Get info before preprocessing
        info = self.get_audio_info(audio_path)
        
        # Preprocess
        processed_audio = self.preprocess_audio(audio)
        
        # Update info with processed duration
        info['processed_duration'] = len(processed_audio) / sr
        
        return processed_audio, info
    
    @staticmethod
    def supported_formats() -> list:
        """Return list of supported audio formats."""
        return ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
    
    def validate_audio_file(self, audio_path: Union[str, Path]) -> bool:
        """
        Check if audio file is valid and supported.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if valid, False otherwise
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            return False
        
        if audio_path.suffix.lower() not in self.supported_formats():
            return False
        
        try:
            # Try to load a small portion
            librosa.load(str(audio_path), sr=self.sample_rate, duration=1.0)
            return True
        except:
            return False
