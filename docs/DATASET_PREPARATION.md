# Dataset Preparation Guide

Complete guide to preparing audio datasets for Moonshine fine-tuning.

## Overview

Quality dataset preparation is crucial for successful fine-tuning. This guide covers:
- Dataset format requirements
- Audio preprocessing and segmentation
- Data quality checks
- Creating custom datasets

## Dataset Requirements

### Required Format

Moonshine expects HuggingFace Datasets format with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `audio` | Audio | Audio file (WAV, MP3, FLAC, etc.) |
| `text` or `transcription` | string | Ground truth transcription |

**Optional columns:**
- `speaker_id` - For multi-speaker datasets
- `duration` - Audio duration in seconds
- `language` - For multilingual datasets

### Audio Specifications

- **Format:** WAV, MP3, FLAC, OGG (will be converted automatically)
- **Sample Rate:** 16kHz (required)
- **Channels:** Mono (stereo will be converted)
- **Duration:** 0.5-20 seconds (recommended)
- **Quality:** Clean audio, minimal background noise

## Option 1: Use Existing HuggingFace Dataset

### Popular ASR Datasets

```python
from datasets import load_dataset

# Multilingual LibriSpeech (MLS)
dataset = load_dataset("facebook/multilingual_librispeech", "french")

# Common Voice
dataset = load_dataset("mozilla-foundation/common_voice_13_0", "fr", use_auth_token=True)

# LibriSpeech (English)
dataset = load_dataset("librispeech_asr", "clean")

# FLEURS (102 languages)
dataset = load_dataset("google/fleurs", "fr_fr")
```

### Verify Dataset Structure

```python
print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['audio', 'text', 'speaker_id', ...],
#         num_rows: 258213
#     })
#     test: Dataset({
#         features: ['audio', 'text', 'speaker_id', ...],
#         num_rows: 2426
#     })
# })

# Check audio
print(dataset['train'][0]['audio'])
# {'path': '/path/to/audio.wav', 'array': array([...]), 'sampling_rate': 16000}

# Check transcription
print(dataset['train'][0]['text'])
# "Bonjour, comment allez-vous aujourd'hui ?"
```

## Option 2: Create Custom Dataset

### From Local Audio Files

```python
from datasets import Dataset, Audio
import os

def create_dataset_from_directory(audio_dir, transcript_file):
    """
    Create dataset from directory of audio files and transcriptions.

    Args:
        audio_dir: Directory containing .wav files
        transcript_file: Text file with format: filename|transcription
    """
    # Read transcriptions
    transcriptions = {}
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line in f:
            filename, text = line.strip().split('|')
            transcriptions[filename] = text

    # Collect audio files
    data = {'audio': [], 'text': []}
    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):
            audio_path = os.path.join(audio_dir, filename)
            if filename in transcriptions:
                data['audio'].append(audio_path)
                data['text'].append(transcriptions[filename])

    # Create dataset
    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    return dataset

# Usage
dataset = create_dataset_from_directory(
    audio_dir="./my_audio_files",
    transcript_file="./transcriptions.txt"
)

# Save
dataset.save_to_disk("./my_dataset")

# Split train/test
dataset = dataset.train_test_split(test_size=0.1, seed=42)
dataset.save_to_disk("./my_dataset")
```

### Transcription File Format

```
# transcriptions.txt (pipe-separated)
audio_001.wav|Bonjour, comment allez-vous ?
audio_002.wav|Je vais très bien, merci.
audio_003.wav|Quelle belle journée aujourd'hui.
```

or

```
# transcriptions.csv
audio,text
audio_001.wav,"Bonjour, comment allez-vous ?"
audio_002.wav,"Je vais très bien, merci."
audio_003.wav,"Quelle belle journée aujourd'hui."
```

### From CSV File

```python
from datasets import Dataset, Audio
import pandas as pd

# Load CSV
df = pd.read_csv("transcriptions.csv")
# Columns: audio, text

# Create dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Split
dataset = dataset.train_test_split(test_size=0.1)
dataset.save_to_disk("./my_dataset")
```

## Audio Preprocessing

### Intelligent Segmentation

For long audio files (e.g., podcasts, audiobooks), use our segmentation script:

```bash
python scripts/intelligent_segmentation.py \
    --dataset facebook/multilingual_librispeech \
    --language french \
    --output ./data/mls_french_segmented \
    --max-duration 10.0 \
    --min-duration 1.0 \
    --use-whisper-v3 \
    --alignment-method "forced"
```

**Features:**
- Uses Whisper V3 for initial transcription
- Forced alignment for precise word timing
- Splits at natural boundaries (pauses, sentence ends)
- Maintains transcription quality

**Output:**
```python
segmented_dataset = load_from_disk("./data/mls_french_segmented")
# Each sample is 1-10 seconds with aligned transcription
```

### Manual Audio Processing

```python
from datasets import load_dataset
import librosa
import soundfile as sf

def preprocess_audio(example):
    """
    Preprocess audio: resample, convert to mono, normalize.
    """
    audio = example['audio']
    audio_array = audio['array']
    sr = audio['sampling_rate']

    # Resample to 16kHz if needed
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Convert stereo to mono
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)

    # Normalize
    audio_array = audio_array / (abs(audio_array).max() + 1e-8)

    # Update example
    example['audio']['array'] = audio_array
    example['audio']['sampling_rate'] = sr

    return example

# Apply preprocessing
dataset = load_dataset("your/dataset")
dataset = dataset.map(preprocess_audio, num_proc=4)
```

### Filter by Duration

```python
def filter_by_duration(example, min_duration=0.5, max_duration=20.0):
    """Filter audio samples by duration."""
    audio = example['audio']
    duration = len(audio['array']) / audio['sampling_rate']
    return min_duration <= duration <= max_duration

dataset = dataset.filter(filter_by_duration, num_proc=4)
```

## Data Quality Checks

### Check Audio Quality

```python
import numpy as np

def check_audio_quality(dataset, sample_size=100):
    """Check for common audio issues."""
    import random
    samples = random.sample(range(len(dataset)), min(sample_size, len(dataset)))

    issues = {
        'silent': [],
        'clipping': [],
        'low_volume': []
    }

    for idx in samples:
        audio = dataset[idx]['audio']['array']

        # Check for silence
        if abs(audio).max() < 0.01:
            issues['silent'].append(idx)

        # Check for clipping
        if abs(audio).max() > 0.95:
            issues['clipping'].append(idx)

        # Check for low volume
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.005:
            issues['low_volume'].append(idx)

    print(f"Silent samples: {len(issues['silent'])} / {sample_size}")
    print(f"Clipping samples: {len(issues['clipping'])} / {sample_size}")
    print(f"Low volume samples: {len(issues['low_volume'])} / {sample_size}")

    return issues

# Usage
issues = check_audio_quality(dataset['train'])
```

### Check Transcription Quality

```python
def check_transcription_quality(dataset):
    """Check for common transcription issues."""
    issues = {
        'empty': [],
        'too_short': [],
        'too_long': [],
        'invalid_chars': []
    }

    for idx, example in enumerate(dataset):
        text = example['text']

        # Empty transcription
        if not text or len(text.strip()) == 0:
            issues['empty'].append(idx)

        # Too short (< 3 characters)
        elif len(text) < 3:
            issues['too_short'].append(idx)

        # Too long (> 500 characters)
        elif len(text) > 500:
            issues['too_long'].append(idx)

        # Check for invalid characters (optional)
        # if any(ord(c) > 127 for c in text):
        #     issues['invalid_chars'].append(idx)

    print(f"Empty transcriptions: {len(issues['empty'])}")
    print(f"Too short: {len(issues['too_short'])}")
    print(f"Too long: {len(issues['too_long'])}")

    return issues

# Usage
issues = check_transcription_quality(dataset['train'])
```

### Compute Dataset Statistics

```python
import numpy as np

def compute_dataset_stats(dataset):
    """Compute statistics about the dataset."""
    durations = []
    text_lengths = []

    for example in dataset:
        audio = example['audio']
        duration = len(audio['array']) / audio['sampling_rate']
        durations.append(duration)

        text_length = len(example['text'])
        text_lengths.append(text_length)

    print(f"Number of samples: {len(dataset)}")
    print(f"\nAudio Duration Statistics:")
    print(f"  Mean: {np.mean(durations):.2f}s")
    print(f"  Median: {np.median(durations):.2f}s")
    print(f"  Min: {np.min(durations):.2f}s")
    print(f"  Max: {np.max(durations):.2f}s")
    print(f"  Total: {np.sum(durations)/3600:.2f} hours")

    print(f"\nTranscription Length Statistics:")
    print(f"  Mean: {np.mean(text_lengths):.1f} chars")
    print(f"  Median: {np.median(text_lengths):.1f} chars")
    print(f"  Min: {np.min(text_lengths)} chars")
    print(f"  Max: {np.max(text_lengths)} chars")

# Usage
compute_dataset_stats(dataset['train'])
```

## Advanced Preprocessing

### Noise Reduction

```python
import noisereduce as nr

def reduce_noise(example):
    """Apply noise reduction to audio."""
    audio = example['audio']['array']
    sr = example['audio']['sampling_rate']

    # Reduce noise
    reduced_audio = nr.reduce_noise(y=audio, sr=sr)

    example['audio']['array'] = reduced_audio
    return example

# Apply to dataset (optional, may reduce model robustness)
# dataset = dataset.map(reduce_noise, num_proc=4)
```

### Speed Perturbation

```python
import librosa

def speed_perturbation(example, speeds=[0.9, 1.0, 1.1]):
    """Create multiple versions with different speeds."""
    audio = example['audio']['array']
    sr = example['audio']['sampling_rate']
    text = example['text']

    augmented = []

    for speed in speeds:
        # Time stretch
        stretched = librosa.effects.time_stretch(audio, rate=speed)

        augmented.append({
            'audio': {'array': stretched, 'sampling_rate': sr},
            'text': text
        })

    return augmented

# Apply carefully - increases dataset size
```

## Best Practices

### 1. Dataset Size

- **Minimum:** 10 hours (for domain adaptation)
- **Recommended:** 50-100 hours (for good performance)
- **Ideal:** 200+ hours (for state-of-the-art)

### 2. Audio Quality

✅ **Good:**
- Clean recordings
- Minimal background noise
- Natural speech pace
- Consistent volume

❌ **Avoid:**
- Heavy noise/distortion
- Music overlays
- Extreme speed variations
- Clipping/saturation

### 3. Transcription Quality

✅ **Good:**
- Accurate word-for-word transcriptions
- Proper punctuation (optional)
- Consistent formatting
- No speaker annotations

❌ **Avoid:**
- Abbreviated transcriptions
- Phonetic spellings
- Multiple speakers without labels
- Timestamps in text

### 4. Dataset Balance

- Include variety of speakers (age, gender, accent)
- Cover different domains/topics
- Mix recording conditions (but keep quality)
- Balance duration distribution

### 5. Train/Test Split

```python
# Stratified split (recommended)
dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42,
    stratify_by_column='speaker_id'  # if available
)

# Or manual split
train_dataset = dataset.select(range(0, 9000))
test_dataset = dataset.select(range(9000, 10000))
```

## Common Issues and Solutions

### Issue 1: Audio Files Not Loading

**Error:** `ValueError: file does not start with RIFF id`

**Solution:**
```python
# Check audio file integrity
import soundfile as sf

try:
    audio, sr = sf.read("problematic_file.wav")
    print(f"Audio loaded: {len(audio)} samples at {sr}Hz")
except Exception as e:
    print(f"Error: {e}")
    # Remove or reconvert this file
```

### Issue 2: Sample Rate Mismatch

**Error:** Different sample rates in dataset

**Solution:**
```python
def ensure_16khz(example):
    audio = example['audio']
    if audio['sampling_rate'] != 16000:
        import librosa
        audio['array'] = librosa.resample(
            audio['array'],
            orig_sr=audio['sampling_rate'],
            target_sr=16000
        )
        audio['sampling_rate'] = 16000
    return example

dataset = dataset.map(ensure_16khz)
```

### Issue 3: Memory Issues with Large Datasets

**Solution:**
```python
# Use streaming mode
dataset = load_dataset("large/dataset", streaming=True)

# Or process in chunks
for i in range(0, len(dataset), 1000):
    chunk = dataset.select(range(i, min(i+1000, len(dataset))))
    # Process chunk
```

## Example Workflow

```bash
# 1. Download or prepare raw audio + transcriptions

# 2. Create dataset
python scripts/create_dataset.py \
    --audio-dir ./raw_audio \
    --transcriptions ./transcriptions.txt \
    --output ./my_dataset

# 3. Check quality
python scripts/check_dataset_quality.py --dataset ./my_dataset

# 4. Apply intelligent segmentation
python scripts/intelligent_segmentation.py \
    --dataset ./my_dataset \
    --output ./my_dataset_segmented \
    --max-duration 10.0

# 5. Compute statistics
python scripts/dataset_stats.py --dataset ./my_dataset_segmented

# 6. Ready for training!
python train.py --config configs/my_model.yaml
```

## Next Steps

- [Start training](./TRAINING_GUIDE.md)
- [Configure training](./TRAINING_GUIDE.md#hyperparameter-tuning)
- [Run evaluation](./INFERENCE_GUIDE.md)

---

**Need help?** Check the [FAQ](../README.md#troubleshooting) or open an issue.
