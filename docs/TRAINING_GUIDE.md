# Training Guide - Fine-Tuning Moonshine ASR

Complete guide to fine-tuning Moonshine models for custom languages and domains.

## Overview

This guide covers:
- Dataset preparation and formatting
- Configuration file setup
- Training with curriculum learning
- Monitoring and checkpointing
- Hyperparameter tuning
- Common issues and solutions

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- Prepared audio dataset in HuggingFace format

## Step-by-Step Training

### 1. Prepare Your Dataset

Your dataset should be in HuggingFace Datasets format with these columns:
- `audio`: Audio file (WAV, MP3, etc.)
- `text` or `transcription`: Ground truth transcription

**Option A: Use HuggingFace Dataset**

```python
from datasets import load_dataset

dataset = load_dataset("facebook/multilingual_librispeech", "french")
# Dataset has columns: audio, text, speaker_id, chapter_id, etc.
```

**Option B: Load from Local Files**

```python
from datasets import Dataset, Audio

# Create dataset from your files
data = {
    "audio": ["path/to/audio1.wav", "path/to/audio2.wav"],
    "text": ["transcription 1", "transcription 2"]
}

dataset = Dataset.from_dict(data)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset.save_to_disk("./my_dataset")
```

**Option C: Use Intelligent Segmentation**

For long audio files, use our segmentation script:

```bash
python scripts/intelligent_segmentation.py \
    --dataset facebook/multilingual_librispeech \
    --language french \
    --output ./data/mls_french_segmented \
    --max-duration 10.0 \
    --min-duration 1.0
```

### 2. Create Configuration File

Create `configs/my_model.yaml`:

```yaml
# Dataset configuration
dataset:
  name: "facebook/multilingual_librispeech"  # or path to local dataset
  language: "french"
  train_split: "train"
  test_split: "test"
  text_column: "text"  # Column name for transcriptions

# Model configuration
model:
  name: "UsefulSensors/moonshine-tiny"
  cache_dir: null  # Optional: custom cache directory

# Training configuration
training:
  output_dir: "./results-moonshine-fr"
  num_train_epochs: 3
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  gradient_accumulation_steps: 1

  learning_rate: 5e-5
  warmup_steps: 500

  logging_steps: 50
  eval_steps: 500
  save_steps: 1000
  save_total_limit: 3

  evaluation_strategy: "steps"
  save_strategy: "steps"
  load_best_model_at_end: true
  metric_for_best_model: "wer"
  greater_is_better: false

# Optimizer configuration
optimizer:
  type: "schedulefree_adamw"  # or "adamw"
  betas: [0.9, 0.999]
  epsilon: 1e-8
  weight_decay: 0.01

# Data processing
data:
  max_duration: 20.0  # Maximum audio duration in seconds
  min_duration: 0.5   # Minimum audio duration
  preprocessing_num_workers: 4

# Curriculum learning (optional)
curriculum:
  enabled: false
  stages:
    - duration: 2000  # steps
      max_audio_length: 5.0
      description: "Stage 1: Short clips"

    - duration: 3000
      max_audio_length: 10.0
      description: "Stage 2: Medium clips"

    - duration: 3000
      max_audio_length: 20.0
      description: "Stage 3: Full length"

# Mixed precision training
fp16: true

# Evaluation
evaluation:
  num_beams: 5
  max_new_tokens: 150
```

### 3. Start Training

```bash
# Basic training
python train.py --config configs/my_model.yaml

# Resume from checkpoint
python train.py --config configs/my_model.yaml --resume results-moonshine-fr/checkpoint-1000

# Override config parameters
python train.py --config configs/my_model.yaml --learning-rate 3e-5 --batch-size 32
```

### 4. Monitor Training

**TensorBoard:**

```bash
tensorboard --logdir results-moonshine-fr/runs
```

Visit http://localhost:6006 to see:
- Training/validation loss
- WER over time
- Learning rate schedule
- Gradient norms

**Console Output:**

```
Step 100/8000 | Loss: 2.345 | WER: 45.2% | LR: 4.5e-5 | Time: 1.2s/step
Step 200/8000 | Loss: 2.123 | WER: 42.1% | LR: 4.8e-5 | Time: 1.1s/step
```

### 5. Checkpointing

Checkpoints are automatically saved to `output_dir/checkpoint-{step}`:

```
results-moonshine-fr/
├── checkpoint-1000/
│   ├── model.safetensors
│   ├── config.json
│   ├── generation_config.json
│   ├── preprocessor_config.json
│   ├── trainer_state.json
│   └── training_args.bin
├── checkpoint-2000/
├── checkpoint-best/  # Best checkpoint based on WER
└── runs/  # TensorBoard logs
```

## Training Strategies

### Curriculum Learning

Train with progressively longer audio clips for better convergence:

```yaml
curriculum:
  enabled: true
  stages:
    # Stage 1: Easy (short clips)
    - duration: 2000  # 2000 steps
      max_audio_length: 5.0
      description: "Short and clear audio"

    # Stage 2: Medium
    - duration: 3000
      max_audio_length: 10.0
      description: "Medium length audio"

    # Stage 3: Hard (full length)
    - duration: 3000
      max_audio_length: 20.0
      description: "Full dataset"
```

**Benefits:**
- Faster initial convergence
- Better final performance
- More stable training

### Schedule-Free Optimization

Use schedule-free AdamW (no learning rate scheduling needed):

```yaml
optimizer:
  type: "schedulefree_adamw"
  learning_rate: 5e-5  # Fixed learning rate
  warmup_steps: 500
```

**Benefits:**
- No need to tune LR schedules
- Automatic adaptation
- State-of-the-art performance

### Mixed Precision Training

Enable FP16 for faster training and lower memory:

```yaml
fp16: true
```

**Benefits:**
- 2x faster training
- 50% less memory
- Same final performance

### Gradient Accumulation

For effective larger batch sizes with limited memory:

```yaml
per_device_train_batch_size: 8
gradient_accumulation_steps: 4
# Effective batch size = 8 * 4 = 32
```

## Hyperparameter Tuning

### Learning Rate

**Default:** `5e-5`

- Too high: Training unstable, loss oscillates
- Too low: Slow convergence, may not reach optimal performance

**Recommended range:** `3e-5` to `1e-4`

```yaml
learning_rate: 5e-5
warmup_steps: 500  # Gradual warmup helps stability
```

### Batch Size

**Default:** `16`

- Larger: Faster training, more stable gradients, needs more memory
- Smaller: Less memory, more gradient noise

**Recommended:**
- 8-16 for 8GB GPU
- 16-32 for 16GB GPU
- 32-64 for 24GB+ GPU

### Number of Epochs

**Default:** `3`

- Watch validation WER
- Stop when WER plateaus
- Use early stopping

```yaml
num_train_epochs: 3
load_best_model_at_end: true
```

### Audio Duration

```yaml
data:
  max_duration: 20.0  # seconds
  min_duration: 0.5
```

- Filter very short clips (< 0.5s)
- Limit long clips to reduce memory
- Sweet spot: 5-10 seconds

## Common Training Issues

### Issue 1: Out of Memory (OOM)

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
```yaml
# Reduce batch size
per_device_train_batch_size: 8  # from 16

# Enable gradient accumulation
gradient_accumulation_steps: 2

# Reduce max audio duration
max_duration: 15.0  # from 20.0

# Enable gradient checkpointing (if supported)
gradient_checkpointing: true
```

### Issue 2: Training Not Converging

**Symptoms:** Loss stays high, WER doesn't improve

**Solutions:**
1. Check your dataset (correct transcriptions?)
2. Enable curriculum learning
3. Increase warmup steps
4. Try different learning rate

```yaml
curriculum:
  enabled: true

warmup_steps: 1000  # from 500

learning_rate: 3e-5  # from 5e-5
```

### Issue 3: Overfitting

**Symptoms:** Training loss decreases but validation loss increases

**Solutions:**
```yaml
# Add weight decay
weight_decay: 0.01

# Reduce epochs
num_train_epochs: 2

# Use more data augmentation (if available)
```

### Issue 4: Slow Training

**Symptoms:** Very slow steps/sec

**Solutions:**
1. Enable FP16 mixed precision
2. Increase batch size
3. Use fewer data preprocessing workers if CPU bottleneck

```yaml
fp16: true
per_device_train_batch_size: 32
preprocessing_num_workers: 2  # from 4
```

### Issue 5: NaN Loss

**Symptoms:** Loss becomes NaN during training

**Solutions:**
```yaml
# Reduce learning rate
learning_rate: 3e-5

# Increase warmup
warmup_steps: 1000

# Disable FP16 temporarily
fp16: false

# Check for corrupted audio files in dataset
```

## Advanced Techniques

### Data Augmentation

Add noise, speed perturbation, etc. (requires custom data collator):

```python
# In your training script
from moonshine_ft.data_loader import MoonshineDataCollatorWithAugmentation

data_collator = MoonshineDataCollatorWithAugmentation(
    processor=processor,
    add_noise=True,
    speed_perturb=True
)
```

### Multi-GPU Training

```bash
# Using accelerate
accelerate config  # Configure multi-GPU setup
accelerate launch train.py --config configs/my_model.yaml

# Using torchrun
torchrun --nproc_per_node=4 train.py --config configs/my_model.yaml
```

### Resume Training

```bash
# Resume from specific checkpoint
python train.py --config configs/my_model.yaml --resume results-moonshine-fr/checkpoint-2000

# Resume from last checkpoint (auto-detect)
python train.py --config configs/my_model.yaml --resume results-moonshine-fr
```

## Best Practices

### 1. Start Small
- Train on subset first (1000 examples)
- Verify training loop works
- Then scale to full dataset

### 2. Monitor Closely
- Watch both train and validation loss
- Check WER on validation set
- Look for overfitting signs

### 3. Save Checkpoints
```yaml
save_steps: 500  # Save frequently at first
save_total_limit: 5  # Keep only last 5 checkpoints
```

### 4. Use Validation Set
```yaml
evaluation_strategy: "steps"
eval_steps: 500
load_best_model_at_end: true
```

### 5. Log Everything
```yaml
logging_steps: 50
logging_dir: "runs"  # TensorBoard logs
```

## Training Time Estimates

Based on MLS French dataset (231 hours):

| Configuration | Time per Epoch | Total Time (3 epochs) |
|--------------|----------------|----------------------|
| 1x GPU (16GB), BS=16 | ~8 hours | ~24 hours |
| 1x GPU (24GB), BS=32 | ~5 hours | ~15 hours |
| 4x GPU (16GB), BS=16 each | ~2 hours | ~6 hours |

*Times approximate, depends on hardware and dataset*

## After Training

### Evaluate Model

```bash
python scripts/evaluate.py \
    --model results-moonshine-fr/checkpoint-best \
    --dataset facebook/multilingual_librispeech \
    --language french \
    --split test
```

### Run Inference

```bash
python scripts/inference.py \
    --model results-moonshine-fr/checkpoint-best \
    --audio sample.wav
```

### Export for Deployment

```bash
python scripts/convert_for_deployment.py \
    --model results-moonshine-fr/checkpoint-best \
    --output moonshine-fr-deployment
```

## Next Steps

- [Evaluate your model](./INFERENCE_GUIDE.md)
- [Deploy to production](./DEPLOYMENT_GUIDE.md)
- [Try live transcription](./LIVE_MODE_GUIDE.md)
- [Optimize with ONNX](./ONNX_MODE_GUIDE.md)

---

**Need help?** Open an issue on GitHub or check the [troubleshooting section](#common-training-issues).
