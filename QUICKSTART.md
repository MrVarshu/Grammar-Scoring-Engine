## Quick Start Guide

### Installation

1. **Clone or download the project**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Spacy model (required for NLP):**
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Basic Usage

#### Method 1: Using the Complete Engine

```python
from main import GrammarScoringEngine

# Initialize engine
engine = GrammarScoringEngine()

# Score a single audio file
result = engine.score_audio("path/to/audio.wav")

print(f"Score: {result['score']}/100")
print(f"Grade: {result['grade']}")
```

#### Method 2: Score Text Directly (No Audio)

```python
from src.grammar_scorer import GrammarScorer

# Initialize scorer
scorer = GrammarScorer(language="en-US")

# Score text
text = "Your text to analyze here."
result = scorer.score_text(text)

print(f"Score: {result['score']}/100")
print(f"Errors: {result['error_count']}")
```

#### Method 3: Command Line Interface

```bash
# Process single file
python main.py path/to/audio.wav

# Process directory (batch mode)
python main.py path/to/audio/directory --batch
```

### Configuration

Edit `config.yaml` to customize:

- **Audio settings**: Sample rate, duration limits
- **Transcription**: Model size (tiny, base, small, medium, large)
- **Grammar scoring**: Weights for different components
- **Output**: Result format and location

Example configuration change:
```yaml
transcription:
  whisper_model_size: "small"  # Use larger model for better accuracy
  
grammar:
  weights:
    grammar_errors: 0.50      # Increase grammar weight
    vocabulary_richness: 0.30  # Increase vocabulary weight
```

### Understanding the Scoring System

The engine scores text on a scale of 0-100 based on:

1. **Grammar Errors (default 40%)**: Spelling, grammar, punctuation
2. **Sentence Structure (default 20%)**: Sentence length and variety
3. **Vocabulary Richness (default 20%)**: Word diversity and complexity
4. **Readability (default 20%)**: Ease of reading

**Grades:**
- A (90-100): Excellent
- B (75-89): Good
- C (60-74): Average
- D (40-59): Poor
- F (0-39): Very Poor

### Common Tasks

#### Process Multiple Files

```python
engine = GrammarScoringEngine()
results = engine.score_batch('./data/audio_files')
```

#### Get Detailed Feedback

```python
scorer = GrammarScorer()
result = scorer.score_text(your_text)
feedback = scorer.generate_feedback(result)
print(feedback)
```

#### Save Results

Results are automatically saved to `./results/` directory in JSON and CSV formats.

To manually save:
```python
from src.utils import save_json, create_detailed_report

save_json(result, 'my_result.json')
create_detailed_report(result, 'my_report.txt')
```

### Troubleshooting

**Issue: Whisper model download fails**
- Solution: Check internet connection, or manually download model

**Issue: LanguageTool errors**
- Solution: The tool will work without it, but with reduced accuracy
- Alternative: Install Java and language-tool-python

**Issue: Out of memory**
- Solution: Use smaller Whisper model ("tiny" or "base")
- Or process shorter audio clips

**Issue: Slow transcription**
- Solution: Use GPU if available (automatic with CUDA)
- Or use smaller model size

### Next Steps

1. Check `notebooks/demo.ipynb` for interactive examples
2. Customize weights in `config.yaml` for your use case
4. Review results in `./results/` directory

For more information, see the main README.md file.
