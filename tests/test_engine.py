"""
Unit tests for the Grammar Scoring Engine
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_processor import AudioProcessor
from src.grammar_scorer import GrammarScorer


class TestAudioProcessor:
    """Test AudioProcessor class."""
    
    def test_initialization(self):
        """Test AudioProcessor initialization."""
        processor = AudioProcessor(sample_rate=16000, max_duration=300)
        assert processor.sample_rate == 16000
        assert processor.max_duration == 300
    
    def test_supported_formats(self):
        """Test supported audio formats."""
        formats = AudioProcessor.supported_formats()
        assert '.wav' in formats
        assert '.mp3' in formats
        assert '.flac' in formats
    
    def test_preprocess_audio(self):
        """Test audio preprocessing."""
        processor = AudioProcessor()
        
        # Create dummy audio
        dummy_audio = np.random.randn(16000)  # 1 second at 16kHz
        
        # Preprocess
        processed = processor.preprocess_audio(dummy_audio)
        
        assert isinstance(processed, np.ndarray)
        assert len(processed) > 0


class TestGrammarScorer:
    """Test GrammarScorer class."""
    
    def test_initialization(self):
        """Test GrammarScorer initialization."""
        scorer = GrammarScorer(language="en-US")
        assert scorer.language == "en-US"
    
    def test_sentence_structure_analysis(self):
        """Test sentence structure analysis."""
        scorer = GrammarScorer()
        
        text = "This is a sentence. This is another sentence."
        result = scorer.analyze_sentence_structure(text)
        
        assert result['sentence_count'] == 2
        assert result['avg_sentence_length'] > 0
    
    def test_vocabulary_analysis(self):
        """Test vocabulary analysis."""
        scorer = GrammarScorer()
        
        text = "Hello world. Hello Python programming world."
        result = scorer.analyze_vocabulary(text)
        
        assert result['word_count'] == 6
        assert result['unique_words'] == 4  # hello, world, python, programming
        assert 0 <= result['lexical_diversity'] <= 1
    
    def test_score_text_basic(self):
        """Test basic text scoring."""
        scorer = GrammarScorer()
        
        text = "This is a simple sentence for testing."
        result = scorer.score_text(text)
        
        assert 'score' in result
        assert 0 <= result['score'] <= 100
        assert 'grade' in result
        assert 'error_count' in result
        assert 'component_scores' in result
    
    def test_score_empty_text(self):
        """Test scoring empty text."""
        scorer = GrammarScorer()
        
        result = scorer.score_text("")
        assert result['score'] == 0.0
        assert 'error' in result
    
    def test_custom_weights(self):
        """Test custom scoring weights."""
        scorer = GrammarScorer()
        
        text = "This is a test sentence."
        
        custom_weights = {
            'grammar_errors': 0.50,
            'sentence_structure': 0.20,
            'vocabulary_richness': 0.20,
            'readability': 0.10
        }
        
        result = scorer.score_text(text, weights=custom_weights)
        assert 'score' in result
        assert 0 <= result['score'] <= 100
    
    def test_generate_feedback(self):
        """Test feedback generation."""
        scorer = GrammarScorer()
        
        text = "This is a test sentence."
        result = scorer.score_text(text)
        feedback = scorer.generate_feedback(result)
        
        assert isinstance(feedback, str)
        assert len(feedback) > 0
        assert 'Score' in feedback
    
    def test_readability_calculation(self):
        """Test readability calculation."""
        scorer = GrammarScorer()
        
        text = "The quick brown fox jumps over the lazy dog."
        result = scorer.calculate_readability(text)
        
        assert 'flesch_reading_ease' in result
        assert 0 <= result['flesch_reading_ease'] <= 100


class TestUtils:
    """Test utility functions."""
    
    def test_format_duration(self):
        """Test duration formatting."""
        from src.utils import format_duration
        
        assert format_duration(30) == "30s"
        assert format_duration(90) == "1m 30s"
        assert format_duration(150) == "2m 30s"
    
    def test_create_timestamp(self):
        """Test timestamp creation."""
        from src.utils import create_timestamp
        
        timestamp = create_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) == 15  # YYYYMMDD_HHMMSS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
