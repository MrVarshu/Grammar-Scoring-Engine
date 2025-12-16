# Grammar Scoring Engine from Voice Samples

A comprehensive system for evaluating grammar quality from voice samples using speech-to-text transcription and NLP-based grammar analysis.

## Features

- **Audio Processing**: Load and preprocess audio files in multiple formats (WAV, MP3, FLAC, OGG, M4A)
- **Speech-to-Text**: Convert voice samples to text using OpenAI Whisper
- **Grammar Analysis**: Evaluate grammar quality using LanguageTool and custom metrics
- **Scoring Engine**: Comprehensive grammar scoring with detailed feedback
- **Batch Processing**: Process multiple audio files efficiently

## Project Structure

```
SHL/
├── data/                     # Audio files directory
├── src/
│   ├── audio_processor.py    # Audio loading and preprocessing
│   ├── transcriber.py        # Speech-to-text transcription (Whisper)
│   ├── grammar_scorer.py     # Grammar analysis and scoring
│   └── utils.py              # Utility functions
├── notebooks/
│   └── demo.ipynb            # Demo notebook
├── tests/
│   └── test_engine.py        # Unit tests
├── results/                  # Output directory for results
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── main.py                  # Main execution script
└── README.md                # This file
```

## Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd SHL
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Usage

**Process a single audio file:**
```bash
python main.py data/audio.wav
```

**Batch process all files in a directory:**
```bash
python main.py data/cat --batch
```

### Python API Usage

```python
from main import GrammarScoringEngine

# Initialize the engine
engine = GrammarScoringEngine()

# Process a single audio file
result = engine.score_audio("path/to/audio.wav")

# View results
print(f"Grammar Score: {result['score']:.2f}/100")
print(f"Grade: {result['grade']}")
print(f"Errors Found: {result['error_count']}")
print(f"Word Count: {result['word_count']}")

# Process multiple files
results = engine.score_batch("path/to/audio/directory")
```

## Scoring System

The engine evaluates text on a scale of 0-100 based on four components:

- **Grammar Errors (40%)**: Spelling, grammar, punctuation issues
- **Sentence Structure (20%)**: Sentence length and complexity
- **Vocabulary Richness (20%)**: Word diversity and lexical variety
- **Readability (20%)**: Ease of understanding (Flesch Reading Ease)

### Grade Scale
- **A (90-100)**: Excellent
- **B (75-89)**: Good
- **C (60-74)**: Average
- **D (40-59)**: Poor
- **F (0-39)**: Very Poor

## Configuration

Customize the engine by editing `config.yaml`:

```yaml
# Adjust Whisper model size
transcription:
  whisper_model_size: "base"  # Options: tiny, base, small, medium, large

# Customize scoring weights
grammar:
  weights:
    grammar_errors: 0.40
    sentence_structure: 0.20
    vocabulary_richness: 0.20
    readability: 0.20
```

## Output

Results are automatically saved to the `results/` directory in:
- **JSON format**: Complete detailed results
- **CSV format**: Summary for batch processing
- **Text reports**: Human-readable reports

## Requirements

- Python 3.8+
- FFmpeg is NOT required (uses librosa for audio loading)
- See `requirements.txt` for all Python dependencies

## Testing

Run the unit tests:
```bash
pytest tests/
```

## Example Output

```
Processing: audio_sample.wav
============================================================
Overall Grammar Score: 85.50/100
Grade: B (Good)

✓ No grammar errors detected!

Sentence Structure:
  - 3 sentence(s)
  - Average length: 12.3 words

Vocabulary:
  - 37 words (28 unique)
  - Lexical diversity: 75.68%
============================================================
```

## Troubleshooting

**Issue**: Module not found errors  
**Solution**: Make sure you're in the virtual environment and all dependencies are installed

**Issue**: Slow transcription  
**Solution**: Use a smaller Whisper model (`tiny` or `base`) or enable GPU with CUDA

**Issue**: Out of memory  
**Solution**: Process shorter audio clips or use a smaller model

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- OpenAI Whisper for speech recognition
- LanguageTool for grammar checking
- Google Speech Commands Dataset for testing
