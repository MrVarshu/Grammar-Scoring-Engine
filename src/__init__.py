"""Initialize the src package."""

__version__ = "1.0.0"
__author__ = "Grammar Scoring Engine Team"

from .audio_processor import AudioProcessor
from .transcriber import Transcriber
from .grammar_scorer import GrammarScorer

__all__ = ['AudioProcessor', 'Transcriber', 'GrammarScorer']
