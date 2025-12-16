"""
Grammar Scorer Module
Analyzes text for grammar errors and provides comprehensive scoring.
"""

import re
import string
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np


class GrammarScorer:
    """
    Analyze text for grammar quality and provide detailed scoring.
    Uses multiple metrics to evaluate grammar, spelling, and style.
    """
    
    def __init__(self, language: str = "en-US", use_language_tool: bool = True):
        """
        Initialize GrammarScorer.
        
        Args:
            language: Language code for grammar checking
            use_language_tool: Whether to use LanguageTool for checking
        """
        self.language = language
        self.use_language_tool = use_language_tool
        self.grammar_tool = None
        
        if use_language_tool:
            self._initialize_language_tool()
    
    def _initialize_language_tool(self):
        """Initialize LanguageTool for grammar checking."""
        try:
            import language_tool_python
            print(f"Initializing LanguageTool for {self.language}...")
            self.grammar_tool = language_tool_python.LanguageTool(self.language)
            print("LanguageTool initialized successfully!")
        except ImportError:
            print("Warning: language-tool-python not found. Install with: pip install language-tool-python")
            self.use_language_tool = False
        except Exception as e:
            print(f"Warning: Could not initialize LanguageTool: {str(e)}")
            self.use_language_tool = False
    
    def check_grammar(self, text: str) -> List[Dict]:
        """
        Check text for grammar errors using LanguageTool.
        
        Args:
            text: Text to check
            
        Returns:
            List of grammar error dictionaries
        """
        if not self.use_language_tool or self.grammar_tool is None:
            return []
        
        try:
            matches = self.grammar_tool.check(text)
            
            errors = []
            for match in matches:
                errors.append({
                    'message': match.message,
                    'category': match.category,
                    'rule_id': match.ruleId,
                    'suggestions': match.replacements[:3],  # Top 3 suggestions
                    'context': match.context,
                    'offset': match.offset,
                    'length': match.errorLength
                })
            
            return errors
        except Exception as e:
            print(f"Error checking grammar: {str(e)}")
            return []
    
    def analyze_sentence_structure(self, text: str) -> Dict:
        """
        Analyze sentence structure and complexity.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentence structure metrics
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'max_sentence_length': 0,
                'min_sentence_length': 0
            }
        
        # Calculate metrics
        sentence_lengths = [len(s.split()) for s in sentences]
        
        return {
            'sentence_count': len(sentences),
            'avg_sentence_length': np.mean(sentence_lengths),
            'max_sentence_length': max(sentence_lengths),
            'min_sentence_length': min(sentence_lengths),
            'sentence_length_std': np.std(sentence_lengths)
        }
    
    def analyze_vocabulary(self, text: str) -> Dict:
        """
        Analyze vocabulary richness and diversity.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with vocabulary metrics
        """
        # Tokenize and clean
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return {
                'word_count': 0,
                'unique_words': 0,
                'lexical_diversity': 0.0
            }
        
        unique_words = set(words)
        word_freq = Counter(words)
        
        # Lexical diversity (Type-Token Ratio)
        lexical_diversity = len(unique_words) / len(words) if words else 0
        
        return {
            'word_count': len(words),
            'unique_words': len(unique_words),
            'lexical_diversity': lexical_diversity,
            'avg_word_length': np.mean([len(w) for w in words]),
            'most_common_words': word_freq.most_common(10)
        }
    
    def calculate_readability(self, text: str) -> Dict:
        """
        Calculate readability scores.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with readability metrics
        """
        # Simple readability metrics
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = re.findall(r'\b\w+\b', text)
        
        if not sentences or not words:
            return {'readability_score': 0.0}
        
        # Average words per sentence
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Count syllables (simplified)
        def count_syllables(word):
            word = word.lower()
            vowels = 'aeiouy'
            count = 0
            prev_char_was_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_char_was_vowel:
                    count += 1
                prev_char_was_vowel = is_vowel
            return max(1, count)
        
        syllables = sum(count_syllables(w) for w in words)
        avg_syllables_per_word = syllables / len(words)
        
        # Simplified Flesch Reading Ease (higher is easier)
        flesch_score = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word
        flesch_score = max(0, min(100, flesch_score))
        
        return {
            'flesch_reading_ease': flesch_score,
            'avg_words_per_sentence': avg_words_per_sentence,
            'avg_syllables_per_word': avg_syllables_per_word
        }
    
    def score_text(self, text: str, weights: Optional[Dict] = None) -> Dict:
        """
        Comprehensive grammar scoring of text.
        
        Args:
            text: Text to score
            weights: Optional custom weights for scoring components
            
        Returns:
            Dictionary with detailed scoring results
        """
        if not text or not text.strip():
            return {
                'score': 0.0,
                'grade': 'N/A',
                'error': 'Empty text provided'
            }
        
        # Default weights
        if weights is None:
            weights = {
                'grammar_errors': 0.40,
                'sentence_structure': 0.20,
                'vocabulary_richness': 0.20,
                'readability': 0.20
            }
        
        # Ensure all required keys exist
        required_keys = ['grammar_errors', 'sentence_structure', 'vocabulary_richness', 'readability']
        for key in required_keys:
            if key not in weights:
                weights[key] = 0.25  # Equal weight if missing
        
        # Check grammar
        grammar_errors = self.check_grammar(text)
        
        # Analyze structure
        sentence_analysis = self.analyze_sentence_structure(text)
        
        # Analyze vocabulary
        vocab_analysis = self.analyze_vocabulary(text)
        
        # Calculate readability
        readability = self.calculate_readability(text)
        
        # Calculate component scores (0-100)
        
        # 1. Grammar score (fewer errors = higher score)
        word_count = vocab_analysis['word_count']
        error_rate = len(grammar_errors) / max(word_count, 1) * 100
        grammar_score = max(0, 100 - error_rate * 10)
        
        # 2. Sentence structure score
        avg_length = sentence_analysis['avg_sentence_length']
        structure_score = 100
        if avg_length < 5:
            structure_score = 60  # Too short
        elif avg_length > 30:
            structure_score = 70  # Too long
        
        # 3. Vocabulary richness score
        lexical_div = vocab_analysis['lexical_diversity']
        vocab_score = min(100, lexical_div * 200)  # Scale to 0-100
        
        # 4. Readability score
        readability_score = readability.get('flesch_reading_ease', 50)
        
        # Calculate weighted final score
        final_score = (
            grammar_score * weights['grammar_errors'] +
            structure_score * weights['sentence_structure'] +
            vocab_score * weights['vocabulary_richness'] +
            readability_score * weights['readability']
        )
        
        # Determine grade
        grade = self._get_grade(final_score)
        
        # Compile results
        result = {
            'score': round(final_score, 2),
            'grade': grade,
            'grammar_errors': grammar_errors,
            'error_count': len(grammar_errors),
            'sentence_analysis': sentence_analysis,
            'vocabulary_analysis': vocab_analysis,
            'readability': readability,
            'component_scores': {
                'grammar': round(grammar_score, 2),
                'structure': round(structure_score, 2),
                'vocabulary': round(vocab_score, 2),
                'readability': round(readability_score, 2)
            },
            'text_length': len(text),
            'word_count': word_count
        }
        
        return result
    
    def _get_grade(self, score: float) -> str:
        """
        Convert numerical score to letter grade.
        
        Args:
            score: Numerical score (0-100)
            
        Returns:
            Letter grade
        """
        if score >= 90:
            return 'A (Excellent)'
        elif score >= 75:
            return 'B (Good)'
        elif score >= 60:
            return 'C (Average)'
        elif score >= 40:
            return 'D (Poor)'
        else:
            return 'F (Very Poor)'
    
    def generate_feedback(self, scoring_result: Dict) -> str:
        """
        Generate human-readable feedback from scoring results.
        
        Args:
            scoring_result: Result from score_text()
            
        Returns:
            Formatted feedback string
        """
        feedback_lines = []
        
        feedback_lines.append(f"Overall Grammar Score: {scoring_result['score']}/100")
        feedback_lines.append(f"Grade: {scoring_result['grade']}")
        feedback_lines.append("")
        
        # Grammar errors
        error_count = scoring_result['error_count']
        if error_count == 0:
            feedback_lines.append("✓ No grammar errors detected!")
        else:
            feedback_lines.append(f"✗ Found {error_count} grammar issue(s):")
            for i, error in enumerate(scoring_result['grammar_errors'][:5], 1):
                feedback_lines.append(f"  {i}. {error['message']}")
                if error['suggestions']:
                    feedback_lines.append(f"     Suggestion: {error['suggestions'][0]}")
        
        feedback_lines.append("")
        
        # Sentence structure
        sent_analysis = scoring_result['sentence_analysis']
        feedback_lines.append(f"Sentence Structure:")
        feedback_lines.append(f"  - {sent_analysis['sentence_count']} sentence(s)")
        feedback_lines.append(f"  - Average length: {sent_analysis['avg_sentence_length']:.1f} words")
        
        feedback_lines.append("")
        
        # Vocabulary
        vocab = scoring_result['vocabulary_analysis']
        feedback_lines.append(f"Vocabulary:")
        feedback_lines.append(f"  - {vocab['word_count']} words ({vocab['unique_words']} unique)")
        feedback_lines.append(f"  - Lexical diversity: {vocab['lexical_diversity']:.2%}")
        
        return "\n".join(feedback_lines)
