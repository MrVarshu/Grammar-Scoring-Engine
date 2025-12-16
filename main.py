"""
Grammar Scoring Engine - Main Module
Complete pipeline for scoring grammar from voice samples.
"""

import os
import sys
from pathlib import Path
from typing import Union, List, Dict, Optional
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.audio_processor import AudioProcessor
from src.transcriber import Transcriber
from src.grammar_scorer import GrammarScorer
from src.utils import (
    load_config, save_json, save_results_to_csv, 
    get_audio_files, print_summary_statistics,
    create_detailed_report, create_timestamp
)

warnings.filterwarnings('ignore')


class GrammarScoringEngine:
    """
    Complete Grammar Scoring Engine from Voice Samples.
    Integrates audio processing, transcription, and grammar scoring.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the Grammar Scoring Engine.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        try:
            self.config = load_config(config_path)
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}. Using defaults.")
            self.config = self._get_default_config()
        
        # Initialize components
        print("Initializing Grammar Scoring Engine...")
        
        self.audio_processor = AudioProcessor(
            sample_rate=self.config['audio']['sample_rate'],
            max_duration=self.config['audio']['max_duration']
        )
        
        self.transcriber = Transcriber(
            model_name=self.config['transcription']['whisper_model_size'],
            device=self.config['transcription']['device'],
            language=self.config['transcription']['language']
        )
        
        self.grammar_scorer = GrammarScorer(
            language=self.config['grammar']['language_tool_language'],
            use_language_tool=self.config['grammar']['use_language_tool']
        )
        
        print("Grammar Scoring Engine initialized successfully!\n")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'audio': {'sample_rate': 16000, 'max_duration': 300},
            'transcription': {
                'whisper_model_size': 'base',
                'device': 'auto',
                'language': 'en'
            },
            'grammar': {
                'use_language_tool': True,
                'language_tool_language': 'en-US',
                'weights': {
                    'grammar_errors': 0.40,
                    'sentence_structure': 0.20,
                    'vocabulary_richness': 0.20,
                    'readability': 0.20
                }
            },
            'output': {
                'results_dir': './results',
                'save_transcriptions': True,
                'save_detailed_reports': True
            }
        }
    
    def score_audio(self, audio_path: Union[str, Path], 
                    save_results: bool = True) -> Dict:
        """
        Complete pipeline: Process audio file and score grammar.
        
        Args:
            audio_path: Path to audio file
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with complete results
        """
        audio_path = Path(audio_path)
        print(f"\nProcessing: {audio_path.name}")
        print("-" * 60)
        
        # Step 1: Validate audio file
        print("1. Validating audio file...")
        if not self.audio_processor.validate_audio_file(audio_path):
            raise ValueError(f"Invalid audio file: {audio_path}")
        
        # Step 2: Process audio
        print("2. Processing audio...")
        processed_audio, audio_info = self.audio_processor.process_audio_file(audio_path)
        print(f"   Duration: {audio_info['duration_seconds']:.2f}s")
        
        # Step 3: Transcribe
        print("3. Transcribing audio to text...")
        transcription_result = self.transcriber.transcribe_file(str(audio_path))
        text = transcription_result['text']
        print(f"   Transcribed {transcription_result['word_count']} words")
        
        # Step 4: Score grammar
        print("4. Analyzing grammar...")
        scoring_result = self.grammar_scorer.score_text(
            text,
            weights=self.config['grammar']['weights']
        )
        
        # Combine results
        complete_result = {
            'file_name': audio_path.name,
            'file_path': str(audio_path),
            'audio_info': audio_info,
            'transcription': transcription_result,
            'text': text,
            **scoring_result
        }
        
        # Generate feedback
        feedback = self.grammar_scorer.generate_feedback(scoring_result)
        complete_result['feedback'] = feedback
        
        print("\n" + "="*60)
        print(feedback)
        print("="*60)
        
        # Save results
        if save_results:
            self._save_results(complete_result)
        
        return complete_result
    
    def score_batch(self, audio_directory: Union[str, Path], 
                    extensions: Optional[List[str]] = None) -> List[Dict]:
        """
        Process and score multiple audio files.
        
        Args:
            audio_directory: Directory containing audio files
            extensions: List of file extensions to process
            
        Returns:
            List of result dictionaries
        """
        audio_dir = Path(audio_directory)
        
        # Get audio files
        audio_files = get_audio_files(audio_dir, extensions)
        
        if not audio_files:
            print(f"No audio files found in {audio_dir}")
            return []
        
        print(f"\nFound {len(audio_files)} audio file(s) to process\n")
        
        # Process each file
        results = []
        for i, audio_path in enumerate(audio_files, 1):
            print(f"\n{'='*60}")
            print(f"Processing file {i}/{len(audio_files)}")
            print(f"{'='*60}")
            
            try:
                result = self.score_audio(audio_path, save_results=False)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path.name}: {str(e)}")
                results.append({
                    'file_name': audio_path.name,
                    'error': str(e),
                    'score': 0
                })
        
        # Save batch results
        self._save_batch_results(results)
        
        # Print summary
        print_summary_statistics(results)
        
        return results
    
    def _save_results(self, result: Dict) -> None:
        """Save individual result to disk."""
        output_dir = Path(self.config['output']['results_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = create_timestamp()
        base_name = Path(result['file_name']).stem
        
        # Save JSON
        json_path = output_dir / f"{base_name}_{timestamp}.json"
        save_json(result, json_path)
        print(f"\nResults saved to: {json_path}")
        
        # Save detailed report
        if self.config['output']['save_detailed_reports']:
            report_path = output_dir / f"{base_name}_{timestamp}_report.txt"
            create_detailed_report(result, report_path)
            print(f"Detailed report saved to: {report_path}")
    
    def _save_batch_results(self, results: List[Dict]) -> None:
        """Save batch results to disk."""
        output_dir = Path(self.config['output']['results_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = create_timestamp()
        
        # Save JSON
        json_path = output_dir / f"batch_results_{timestamp}.json"
        save_json(results, json_path)
        
        # Save CSV summary
        csv_path = output_dir / f"batch_results_{timestamp}.csv"
        save_results_to_csv(results, csv_path)
        
        print(f"\nBatch results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Grammar Scoring Engine from Voice Samples")
    parser.add_argument('input', help='Audio file or directory path')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--batch', action='store_true', help='Process directory in batch mode')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = GrammarScoringEngine(config_path=args.config)
    
    # Process
    if args.batch:
        engine.score_batch(args.input)
    else:
        engine.score_audio(args.input)


if __name__ == "__main__":
    main()
