# Installation Guide

## Overview

This document describes how to install dependencies for the Moonshine French ASR fine-tuning project.

## Quick Start

### Basic Installation (Required)

For training and evaluation:

```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### Minimal Installation

If you only need inference (no training):

```bash
# Core inference dependencies
uv pip install torch transformers torchaudio numpy

# Audio processing
uv pip install librosa soundfile

# Evaluation
uv pip install jiwer
```

## Optional Features

### Live Transcription Mode

For real-time transcription from microphone:

```bash
# Option 1: Install from requirements-live.txt
uv pip install -r requirements-live.txt

# Option 2: Install directly
uv pip install sounddevice>=0.5.0
```

**Note**: Silero VAD (Voice Activity Detection) is automatically downloaded via `torch.hub` on first use.

### ONNX Export & Deployment

For converting models to ONNX/ORT format:

```bash
uv pip install onnx onnxruntime optimum
```

### Demo Interface

For running the Gradio demo:

```bash
uv pip install gradio>=4.0.0
```

## Requirements Files

### Main Requirements (`requirements.txt`)

**Includes:**
- ✅ Core ML frameworks (PyTorch, Transformers)
- ✅ Training dependencies (schedulefree optimizer)
- ✅ Audio processing (librosa, soundfile, torchaudio)
- ✅ Evaluation (jiwer)
- ✅ Data handling (datasets, pandas, numpy)
- ✅ Optional features (marked as "Optional:")

**Install:**
```bash
uv pip install -r requirements.txt
```

### Live Mode Requirements (`requirements-live.txt`)

**Includes:**
- sounddevice (audio capture)
- Instructions for Silero VAD

**Install:**
```bash
uv pip install -r requirements-live.txt
```

## Dependency Breakdown

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0 | Deep learning framework |
| transformers | >=4.35.0 | Moonshine model & training |
| datasets | >=2.14.0 | Dataset loading & processing |
| accelerate | >=0.24.0 | Distributed training |

### Training & Evaluation

| Package | Version | Purpose |
|---------|---------|---------|
| evaluate | >=0.4.0 | Evaluation metrics |
| jiwer | >=3.0.0 | WER/CER computation |
| tensorboard | >=2.14.0 | Training visualization |
| schedulefree | >=1.4.0 | Schedule-free AdamW optimizer |

### Audio Processing

| Package | Version | Purpose |
|---------|---------|---------|
| librosa | >=0.10.0 | Audio analysis |
| soundfile | >=0.12.0 | Audio I/O |
| torchaudio | >=2.0.0 | PyTorch audio utilities |

### Live Transcription (Optional)

| Package | Version | Purpose |
|---------|---------|---------|
| sounddevice | >=0.5.0 | Real-time audio capture |
| Silero VAD | (auto) | Voice activity detection |

### ONNX Export (Optional)

| Package | Version | Purpose |
|---------|---------|---------|
| onnx | >=1.14.0 | ONNX format support |
| onnxruntime | >=1.16.0 | ONNX inference |
| optimum | >=1.14.0 | ONNX export & optimization |

## Platform-Specific Notes

### Windows (MSYS/Git Bash)

```bash
# Use uv for better dependency resolution
uv pip install -r requirements.txt
```

### Linux

```bash
# May need system dependencies for audio
sudo apt-get install libsndfile1 portaudio19-dev

# Then install Python packages
pip install -r requirements.txt
```

### macOS

```bash
# Install system dependencies via Homebrew
brew install portaudio

# Then install Python packages
pip install -r requirements.txt
```

## GPU Support

### CUDA (NVIDIA GPUs)

PyTorch with CUDA support is typically installed automatically. To explicitly install:

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Virtual Environment Setup

### Using uv (Recommended)

```bash
# Create virtual environment
uv venv

# Activate (Windows/MSYS)
source .venv/Scripts/activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Using venv

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Verification

### Check Installation

```bash
# Test core imports
python -c "import torch, transformers, datasets; print('✅ Core imports OK')"

# Test audio processing
python -c "import librosa, soundfile, torchaudio; print('✅ Audio processing OK')"

# Test evaluation
python -c "import jiwer; print('✅ Evaluation OK')"

# Test live mode (if installed)
python -c "import sounddevice; print('✅ Live mode OK')"
```

### Run Scripts

```bash
# Test inference script
uv run python inference.py --help

# Test evaluation script
uv run python evaluate.py --help

# Test conversion script
uv run python convert_for_deployment.py --help
```

## Troubleshooting

### Issue: "No module named 'sounddevice'"

**Solution:**
```bash
uv pip install sounddevice
```

### Issue: "CUDA not available"

**Solution:**
```bash
# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "ImportError: libsndfile.so.1"

**Solution (Linux):**
```bash
sudo apt-get install libsndfile1
```

**Solution (macOS):**
```bash
brew install libsndfile
```

### Issue: "Failed to load VAD model"

**Solution:**
- Check internet connection (model downloads from GitHub)
- Or use continuous mode: `--live --no-vad`

## Updating Dependencies

### Update All Packages

```bash
uv pip install -r requirements.txt --upgrade
```

### Update Specific Package

```bash
uv pip install transformers --upgrade
```

### Check for Updates

```bash
uv pip list --outdated
```

## Development Installation

For contributors:

```bash
# Install all dependencies including dev tools
uv pip install -r requirements.txt

# Install dev dependencies
uv pip install pytest black isort

# Verify installation
pytest --version
black --version
```

## Minimal Inference-Only Installation

For production deployment with minimal dependencies:

```bash
# Only what's needed for inference.py
uv pip install torch transformers torchaudio numpy soundfile jiwer

# For live mode, also add:
uv pip install sounddevice
```

**Approximate size:**
- Full installation: ~3GB
- Minimal inference: ~2GB
- With live mode: +5MB

## Docker Installation (Future)

```dockerfile
# Example Dockerfile (not yet available)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "inference.py", "--help"]
```

## Summary

**For most users:**
```bash
uv pip install -r requirements.txt
```

**For live transcription:**
```bash
uv pip install -r requirements-live.txt
```

**For inference only:**
```bash
uv pip install torch transformers torchaudio soundfile jiwer
```

---

## Next Steps

After installation:
1. ✅ Verify installation with test scripts
2. ✅ Download or train a model
3. ✅ Run inference: `python inference.py --help`
4. ✅ For live mode: `python inference.py --model ./model --live`

For detailed usage, see:
- `PUBLICATION_SCRIPTS_README.md` - General documentation
- `LIVE_MODE_GUIDE.md` - Live transcription guide
- `SCRIPTS_SUMMARY.md` - Quick reference
