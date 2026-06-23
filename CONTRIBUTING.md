# Contributing to Moonshine Fine-Tuning Guide

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with a clear title and description
3. **Include:**
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (OS, Python version, GPU, etc.)
   - Error messages or logs
   - Minimal code example if applicable

### Suggesting Enhancements

For feature requests or improvements:

1. **Check the roadmap** to see if it's already planned
2. **Open an issue** with the "enhancement" label
3. **Describe:**
   - The problem you're trying to solve
   - Your proposed solution
   - Any alternatives you've considered
   - Examples of how it would be used

### Pull Requests

We welcome pull requests! Here's the process:

1. **Fork the repository**
2. **Create a new branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Follow the coding style (see below)
   - Add tests if applicable
   - Update documentation
   - Add docstrings to new functions/classes

4. **Test your changes**:
   ```bash
   # Run tests
   pytest tests/

   # Check code style
   black . --check
   isort . --check
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add support for X"
   ```

   Use conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `refactor:` for code refactoring
   - `test:` for adding tests
   - `chore:` for maintenance tasks

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**:
   - Provide a clear title and description
   - Reference any related issues
   - Add screenshots/examples if relevant

## Development Setup

### Local Development

```bash
# Clone your fork
git clone https://github.com/pierre-cheneau/finetune-moonshine-asr.git
cd finetune-moonshine-asr

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black isort flake8

# Run tests
pytest tests/
```

### Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

```bash
# Format code
black .
isort .

# Check style
flake8 .

# All at once
black . && isort . && flake8 .
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_inference.py

# Run with coverage
pytest --cov=moonshine_ft tests/
```

## Project Structure

```
finetune-moonshine-asr/
├── moonshine_ft/          # Main package
│   ├── __init__.py
│   ├── data_loader.py     # Dataset loading and preprocessing
│   ├── trainer.py         # Training logic
│   └── configs/           # Configuration utilities
├── scripts/               # Utility scripts
│   ├── inference.py       # Inference script
│   ├── evaluate.py        # Evaluation script
│   └── ...
├── docs/                  # Documentation
├── tests/                 # Test files
├── configs/               # Example configurations
└── examples/              # Example notebooks
```

## Documentation

### Adding Documentation

- Update relevant `.md` files in `docs/`
- Add docstrings to all functions and classes:

```python
def transcribe_audio(audio_path: str, model_path: str) -> str:
    """
    Transcribe audio file using Moonshine model.

    Args:
        audio_path: Path to audio file (WAV, MP3, etc.)
        model_path: Path to fine-tuned Moonshine model

    Returns:
        Transcription text

    Example:
        >>> text = transcribe_audio("sample.wav", "./model")
        >>> print(text)
        "Hello world"
    """
    pass
```

### Building Documentation

```bash
# Install sphinx
pip install sphinx sphinx-rtd-theme

# Build docs
cd docs
make html

# View docs
open _build/html/index.html
```

## Areas for Contribution

We especially welcome contributions in these areas:

### High Priority
- [ ] Multi-language examples and guides
- [ ] Quantization support (INT8, FP16)
- [ ] Streaming inference mode
- [ ] More comprehensive tests

### Medium Priority
- [ ] Speaker diarization integration
- [ ] Punctuation restoration
- [ ] Real-time factor optimization
- [ ] Docker deployment examples
- [ ] Kubernetes manifests

### Low Priority
- [ ] Gradio demo improvements
- [ ] Additional preprocessing techniques
- [ ] More example notebooks
- [ ] Performance benchmarks for different hardware

## Coding Guidelines

### Python Style

- Follow PEP 8
- Use type hints when possible
- Maximum line length: 100 characters
- Use descriptive variable names

```python
# Good
def compute_word_error_rate(predictions: List[str], references: List[str]) -> float:
    """Compute WER between predictions and references."""
    pass

# Avoid
def calc_wer(p, r):
    pass
```

### Error Handling

Always provide informative error messages:

```python
# Good
if not audio_path.exists():
    raise FileNotFoundError(
        f"Audio file not found: {audio_path}\n"
        f"Please check the path and try again."
    )

# Avoid
if not audio_path.exists():
    raise Exception("File not found")
```

### Logging

Use the logging module, not print statements:

```python
import logging

logger = logging.getLogger(__name__)

def train_model(config):
    logger.info(f"Starting training with config: {config}")
    try:
        # training code
        logger.debug("Training step completed")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
```

## Git Workflow

### Branching Strategy

- `main` - Stable, production-ready code
- `develop` - Integration branch for features
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates

### Commit Messages

Follow conventional commits:

```bash
# Feature
git commit -m "feat: add ONNX quantization support"

# Bug fix
git commit -m "fix: correct WER calculation for empty predictions"

# Documentation
git commit -m "docs: add deployment guide for AWS"

# Refactor
git commit -m "refactor: simplify data loading pipeline"

# Tests
git commit -m "test: add unit tests for inference module"
```

## Review Process

1. **Automated checks** run on every PR:
   - Code style (Black, isort)
   - Linting (flake8)
   - Tests (pytest)

2. **Manual review** by maintainers:
   - Code quality
   - Documentation
   - Test coverage
   - Design decisions

3. **Feedback and iteration**:
   - Address reviewer comments
   - Push additional commits
   - Discuss design choices if needed

4. **Merge**:
   - Squash and merge (default)
   - Or rebase and merge for clean history

## Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Credited in release notes
- Mentioned in the README

## Questions?

- Open a [GitHub Discussion](https://github.com/pierre-cheneau/finetune-moonshine-asr/discussions)
- Join our [Discord](https://discord.gg/zE4NRsTGdw) (Hogwarts Legacy Spell Recognition project's discord)

## Code of Conduct

Be respectful and constructive:
- Welcome newcomers
- Provide helpful feedback
- Respect different viewpoints
- Focus on the issue, not the person

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Moonshine Fine-Tuning Guide! 🎉
