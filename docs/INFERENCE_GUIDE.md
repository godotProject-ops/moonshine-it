## Publication-Ready Scripts for Moonshine ASR Fine-Tuning

This directory contains three publication-ready scripts for using, evaluating, and deploying fine-tuned Moonshine ASR models.

## Scripts Overview

### 1. `inference.py` - Model Inference

Run transcriptions on audio files with the fine-tuned model.

**Features:**
- Single file or batch directory processing
- GPU acceleration with FP16 support
- Customizable generation parameters
- JSON output for programmatic use
- Real-time factor (RTF) measurements

**Basic Usage:**
```bash
# Single audio file
python inference.py --model ./results-moonshine-fr-no-curriculum/final --audio sample.wav

# Directory of files
python inference.py --model ./model --audio ./test_audio/ --output results.json

# GPU with FP16 (2x faster)
python inference.py --model ./model --audio audio.wav --device cuda --fp16
```

**Advanced Usage:**
```bash
# Custom generation parameters
python inference.py --model ./model --audio audio.wav \
    --num-beams 5 \
    --repetition-penalty 1.3 \
    --no-repeat-ngram-size 2 \
    --max-new-tokens 50
```

**Output Format:**
```json
[
  {
    "file": "sample.wav",
    "text": "Transcribed text here",
    "audio_duration": 5.23,
    "inference_time": 0.15,
    "rtf": 0.029
  }
]
```

---

### 2. `evaluate.py` - Model Evaluation

Compute Word Error Rate (WER), Character Error Rate (CER), and other metrics on test datasets.

**Features:**
- WER, CER, substitutions, deletions, insertions
- Support for HuggingFace datasets and local datasets
- GPU acceleration with FP16
- Detailed per-sample predictions (optional)
- Real-time factor measurements

**Basic Usage:**
```bash
# Evaluate on local dataset
python evaluate.py \
    --model ./results-moonshine-fr-no-curriculum/final \
    --dataset ./data/mls_french_split \
    --split test

# Save detailed results
python evaluate.py --model ./model --dataset ./data/test \
    --output evaluation_results.json \
    --save-predictions
```

**Custom Datasets:**
```bash
# Specify column names
python evaluate.py --model ./model --dataset ./data/test \
    --audio-column audio \
    --text-column transcript
```

**Output Example:**
```
EVALUATION RESULTS
============================================================
Samples: 3,188

Accuracy Metrics:
  WER: 21.88%
  CER: 8.45%

Error Breakdown:
  Substitutions: 1,234
  Deletions: 234
  Insertions: 123
  Correct words: 15,678

Performance:
  Total audio duration: 3845.2s
  Total inference time: 287.3s
  Real-time factor: 0.07x
```

---

### 3. `convert_for_deployment.py` - Complete Conversion Pipeline

All-in-one conversion script for production deployment. Creates deployment-ready models compatible with Moonshine C++, ONNX Runtime, and mobile/embedded devices.

**Pipeline Steps:**
1. **Tokenizer Extension** - Extend vocabulary from 32000 to 32768 tokens (for C++ compatibility)
2. **Embedding Resize** - Resize model embeddings to match extended tokenizer
3. **ONNX Export** - Convert to ONNX format (encoder + decoder + decoder_with_past)
4. **Merged Decoder** - Create merged decoder for KV cache efficiency
5. **Tokenizer Binarization** - Convert tokenizer.json → tokenizer.bin (smaller, faster)
6. **ORT Conversion** - Convert to ONNX Runtime optimized format (10-20% faster)

**Basic Usage:**
```bash
# Full pipeline with defaults
python convert_for_deployment.py \
    --model ./results-moonshine-fr-no-curriculum/final \
    --output ./deployment/moonshine-fr-v1
```

**Advanced Usage:**
```bash
# Skip tokenizer extension (if already 32768 tokens)
python convert_for_deployment.py \
    --model ./model --output ./deployment \
    --skip-tokenizer-extension

# Platform-specific optimization (ARM devices, Raspberry Pi, etc.)
python convert_for_deployment.py \
    --model ./model --output ./deployment \
    --target-platform arm

# Skip specific steps
python convert_for_deployment.py \
    --model ./model --output ./deployment \
    --skip-merged-decoder \
    --skip-ort-conversion
```

**Output Structure:**
```
deployment/moonshine-fr-v1/
├── tokenizer_extended/       # Extended tokenizer (32768 tokens)
├── model_resized/            # Model with resized embeddings
├── onnx/                     # ONNX models
│   ├── encoder_model.onnx
│   ├── decoder_model.onnx
│   └── decoder_with_past_model.onnx
├── onnx_merged/              # Merged decoder (C++ compatible)
│   ├── encoder_model.onnx
│   ├── decoder_model_merged.onnx
│   ├── tokenizer.bin         # Binary tokenizer
│   └── config files
└── ort/                      # ORT optimized (fastest)
    ├── encoder_model.ort
    ├── decoder_model_merged.ort
    ├── tokenizer.bin
    └── config files
```

**Use Cases:**
- **Python Production**: Use `onnx_merged/` or `ort/` for fastest inference
- **C++ Integration**: Use `ort/` with Moonshine C++ library
- **Mobile/Embedded**: Use `ort/` with platform-specific optimization (`--target-platform arm`)

---

## Complete Workflow Example

Here's a complete end-to-end workflow for fine-tuning and deployment:

### 1. Train Model
```bash
uv run python train.py \
    --config moonshine_ft/configs/mls_french_no_curriculum.yaml \
    --no-curriculum
```

### 2. Evaluate Model
```bash
# Quick evaluation on validation set
python evaluate.py \
    --model ./results-moonshine-fr-no-curriculum/checkpoint-5000 \
    --dataset ./data/mls_french_split \
    --split test \
    --max-samples 500

# Full evaluation
python evaluate.py \
    --model ./results-moonshine-fr-no-curriculum/final \
    --dataset ./data/mls_french_split \
    --split test \
    --output ./evaluation_results.json \
    --save-predictions
```

### 3. Test Inference
```bash
# Test on sample files
python inference.py \
    --model ./results-moonshine-fr-no-curriculum/final \
    --audio ./test_samples/ \
    --output ./sample_transcriptions.json
```

### 4. Convert for Deployment
```bash
# Create production-ready models
python convert_for_deployment.py \
    --model ./results-moonshine-fr-no-curriculum/final \
    --output ./deployment/moonshine-fr-v1
```

### 5. Deploy

**Option A: Python Production (ONNX Runtime)**
```python
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor

model = ORTModelForSpeechSeq2Seq.from_pretrained(
    "./deployment/moonshine-fr-v1/ort"
)
processor = AutoProcessor.from_pretrained(
    "./deployment/moonshine-fr-v1/ort"
)
# Use as normal HuggingFace model
```

**Option B: C++ Integration**
Use the models in `deployment/moonshine-fr-v1/ort/` with:
- [Moonshine C++ library](https://github.com/usefulsensors/moonshine)
- ONNX Runtime C++ API

**Option C: Mobile (iOS/Android)**
Use ONNX Runtime Mobile with the ORT models.

---

## Performance Benchmarks

Based on RTX 2060 (6GB) with our fine-tuned French model:

| Configuration | RTF (Real-Time Factor) | Speed vs Real-Time |
|---------------|------------------------|-------------------|
| PyTorch FP32 | 0.15x | 6.7x faster |
| PyTorch FP16 | 0.08x | 12.5x faster |
| ONNX | 0.06x | 16.7x faster |
| ORT (optimized) | 0.05x | 20x faster |

**RTF = Inference Time / Audio Duration**
- RTF < 1.0 means faster than real-time
- RTF = 0.05 means 20x faster than real-time

---

## Requirements

Install dependencies:
```bash
# Base requirements (for inference and evaluation)
uv pip install torch torchaudio transformers datasets jiwer tqdm

# ONNX/ORT conversion requirements
uv pip install optimum onnx onnxruntime schedulefree

# For ORT format conversion
uv pip install onnxruntime-tools
```

---

## Citation

If you use these scripts in your research, please cite:

```bibtex
@misc{moonshine-french-finetuning,
  title={Fine-Tuning Moonshine ASR for French with Intelligent Segmentation},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/moonshine-french}
}
```

---

## Troubleshooting

### Out of Memory during Evaluation
```bash
# Use FP16 to reduce memory
python evaluate.py --model ./model --dataset ./data \
    --device cuda --fp16

# Or limit samples
python evaluate.py --model ./model --dataset ./data \
    --max-samples 1000
```

### ONNX Export Fails
```bash
# Check model vocab size matches tokenizer
python -c "
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained('./model')
processor = AutoProcessor.from_pretrained('./model')
print(f'Model: {model.config.vocab_size}')
print(f'Tokenizer: {processor.tokenizer.vocab_size}')
"

# If mismatch, run conversion with embedding resize
python convert_for_deployment.py --model ./model --output ./deploy
```

### ORT Conversion Fails
```bash
# Skip ORT and use ONNX directly
python convert_for_deployment.py --model ./model --output ./deploy \
    --skip-ort-conversion
```

---

## Additional Tools

The `src/tools/` directory contains individual utility scripts:

- `extend_tokenizer_vocab.py` - Extend tokenizer to 32768 tokens
- `resize_model_embeddings.py` - Resize model embeddings
- `create_merged_decoder.py` - Create merged decoder ONNX
- `convert_tokenizer_to_bin.py` - Convert tokenizer to binary
- `convert_to_ort.py` - Convert ONNX to ORT format

These are called automatically by `convert_for_deployment.py`, but can be used independently for fine-grained control.

---

## License

These scripts are provided under the same license as the Moonshine model (MIT License).
