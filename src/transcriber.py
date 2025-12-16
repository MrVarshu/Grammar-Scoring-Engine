"""
Transcriber Module
Converts speech audio to text using various ASR models.
"""

import os
import torch
import warnings
from typing import Union, Optional, Dict
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')


class Transcriber:
    """
    Transcribe audio to text using speech recognition models.
    Supports multiple ASR backends including Whisper.
    """
    
    def __init__(self, model_name: str = "base", device: str = "auto", language: str = "en"):
        """
        Initialize Transcriber.
        
        Args:
            model_name: Model size for Whisper (tiny, base, small, medium, large)
            device: Device to use (auto, cpu, cuda)
            language: Language code for transcription
        """
        self.model_name = model_name
        self.language = language
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            import whisper
            print(f"Loading Whisper model '{self.model_name}' on {self.device}...")
            self.model = whisper.load_model(self.model_name, device=self.device)
            print("Model loaded successfully!")
        except ImportError:
            raise ImportError("whisper package not found. Install with: pip install openai-whisper")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
    
    def transcribe(self, audio: Union[str, Path, np.ndarray], 
                   return_segments: bool = False,
                   return_language: bool = False) -> Union[str, Dict]:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio file path or numpy array
            return_segments: Whether to return detailed segments
            return_language: Whether to return detected language
            
        Returns:
            Transcription text or dictionary with detailed results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        try:
            # Load audio with librosa if it's a file path to avoid FFmpeg dependency
            if isinstance(audio, (str, Path)):
                import librosa
                audio_array, _ = librosa.load(str(audio), sr=16000)
                audio_input = audio_array
            else:
                audio_input = audio
            
            # Transcribe using Whisper
            result = self.model.transcribe(
                audio_input,
                language=self.language,
                fp16=(self.device == "cuda")
            )
            
            # Prepare output
            if return_segments or return_language:
                output = {
                    'text': result['text'].strip(),
                    'language': result.get('language', self.language)
                }
                
                if return_segments and 'segments' in result:
                    output['segments'] = [
                        {
                            'start': seg['start'],
                            'end': seg['end'],
                            'text': seg['text'].strip()
                        }
                        for seg in result['segments']
                    ]
                
                return output
            else:
                return result['text'].strip()
                
        except Exception as e:
            raise RuntimeError(f"Transcription error: {str(e)}")
    
    def transcribe_file(self, audio_path: Union[str, Path], **kwargs) -> Dict:
        """
        Transcribe audio file and return detailed results.
        
        Args:
            audio_path: Path to audio file
            **kwargs: Additional arguments for transcribe()
            
        Returns:
            Dictionary with transcription and metadata
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Transcribe with segments and language detection
        result = self.transcribe(
            str(audio_path),
            return_segments=True,
            return_language=True
        )
        
        # Add metadata
        result['file_path'] = str(audio_path)
        result['file_name'] = audio_path.name
        result['model'] = self.model_name
        result['word_count'] = len(result['text'].split())
        
        return result
    
    def transcribe_batch(self, audio_paths: list, verbose: bool = True) -> list:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            verbose: Whether to show progress
            
        Returns:
            List of transcription results
        """
        results = []
        
        for i, audio_path in enumerate(audio_paths):
            if verbose:
                print(f"Transcribing {i+1}/{len(audio_paths)}: {Path(audio_path).name}")
            
            try:
                result = self.transcribe_file(audio_path)
                results.append(result)
            except Exception as e:
                print(f"Error transcribing {audio_path}: {str(e)}")
                results.append({
                    'file_path': str(audio_path),
                    'error': str(e),
                    'text': None
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'language': self.language,
            'model_loaded': self.model is not None
        }


class SimpleTranscriber:
    """
    Simplified transcriber for quick testing without heavy dependencies.
    Uses mock transcription for demonstration purposes.
    """
    
    def __init__(self, language: str = "en"):
        """Initialize SimpleTranscriber."""
        self.language = language
        print("Initialized SimpleTranscriber (mock mode for testing)")
    
    def transcribe(self, audio: Union[str, Path, np.ndarray], **kwargs) -> str:
        """
        Mock transcription for testing.
        
        Args:
            audio: Audio file path or array
            
        Returns:
            Mock transcription text
        """
        return "This is a sample transcription for testing the grammar scoring engine."
    
    def transcribe_file(self, audio_path: Union[str, Path], **kwargs) -> Dict:
        """
        Mock file transcription.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Mock transcription result
        """
        text = self.transcribe(audio_path)
        return {
            'text': text,
            'file_path': str(audio_path),
            'file_name': Path(audio_path).name,
            'model': 'simple_mock',
            'language': self.language,
            'word_count': len(text.split())
        }
