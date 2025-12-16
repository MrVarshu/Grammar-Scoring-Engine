"""
Utility Functions
Common utilities for the Grammar Scoring Engine.
"""

import os
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Any
from datetime import datetime


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Any, output_path: Union[str, Path]) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(json_path: Union[str, Path]) -> Any:
    """
    Load data from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results_to_csv(results: List[Dict], output_path: Union[str, Path]) -> None:
    """
    Save scoring results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_path: Output CSV path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Flatten nested dictionaries for CSV
    flattened_results = []
    for result in results:
        flat_result = {
            'file_name': result.get('file_name', ''),
            'score': result.get('score', 0),
            'grade': result.get('grade', ''),
            'error_count': result.get('error_count', 0),
            'word_count': result.get('word_count', 0),
            'text': result.get('text', '')
        }
        flattened_results.append(flat_result)
    
    df = pd.DataFrame(flattened_results)
    df.to_csv(output_path, index=False)


def get_audio_files(directory: Union[str, Path], extensions: List[str] = None) -> List[Path]:
    """
    Get all audio files from a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions (e.g., ['.wav', '.mp3'])
        
    Returns:
        List of audio file paths
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    directory = Path(directory)
    audio_files = []
    
    for ext in extensions:
        audio_files.extend(directory.glob(f'*{ext}'))
        audio_files.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(audio_files)


def create_timestamp() -> str:
    """
    Create timestamp string for filenames.
    
    Returns:
        Timestamp string (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2m 30s")
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    
    if minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def download_kaggle_dataset(dataset_name: str, download_path: Union[str, Path]) -> Path:
    """
    Download dataset from Kaggle.
    
    Args:
        dataset_name: Kaggle dataset name (format: username/dataset-name)
        download_path: Directory to download to
        
    Returns:
        Path to downloaded data
    """
    try:
        import opendatasets as od
        
        download_path = Path(download_path)
        download_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading dataset: {dataset_name}")
        od.download(f"https://www.kaggle.com/datasets/{dataset_name}", str(download_path))
        
        print(f"Dataset downloaded to: {download_path}")
        return download_path
        
    except ImportError:
        print("Error: opendatasets package not found.")
        print("Install with: pip install opendatasets")
        raise
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        raise


def print_summary_statistics(results: List[Dict]) -> None:
    """
    Print summary statistics from scoring results.
    
    Args:
        results: List of scoring result dictionaries
    """
    if not results:
        print("No results to summarize.")
        return
    
    scores = [r.get('score', 0) for r in results]
    error_counts = [r.get('error_count', 0) for r in results]
    
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Total files processed: {len(results)}")
    print(f"\nScore Statistics:")
    print(f"  Average score: {sum(scores)/len(scores):.2f}")
    print(f"  Highest score: {max(scores):.2f}")
    print(f"  Lowest score: {min(scores):.2f}")
    print(f"\nError Statistics:")
    print(f"  Average errors: {sum(error_counts)/len(error_counts):.2f}")
    print(f"  Total errors: {sum(error_counts)}")
    print("="*50 + "\n")


def create_detailed_report(result: Dict, output_path: Union[str, Path]) -> None:
    """
    Create a detailed text report for a single result.
    
    Args:
        result: Scoring result dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("GRAMMAR SCORING REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"File: {result.get('file_name', 'N/A')}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Overall Score: {result.get('score', 0):.2f}/100\n")
        f.write(f"Grade: {result.get('grade', 'N/A')}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("TRANSCRIBED TEXT\n")
        f.write("-"*70 + "\n")
        f.write(result.get('text', '') + "\n\n")
        
        f.write("-"*70 + "\n")
        f.write("DETAILED ANALYSIS\n")
        f.write("-"*70 + "\n\n")
        
        # Component scores
        if 'component_scores' in result:
            f.write("Component Scores:\n")
            for component, score in result['component_scores'].items():
                f.write(f"  - {component.capitalize()}: {score:.2f}/100\n")
            f.write("\n")
        
        # Grammar errors
        f.write(f"Grammar Errors Found: {result.get('error_count', 0)}\n")
        if result.get('grammar_errors'):
            f.write("\nDetailed Errors:\n")
            for i, error in enumerate(result['grammar_errors'], 1):
                f.write(f"\n{i}. {error.get('message', '')}\n")
                if error.get('suggestions'):
                    f.write(f"   Suggestions: {', '.join(error['suggestions'])}\n")
        
        f.write("\n" + "="*70 + "\n")
