# 🌙 Moonshine ASR Fine-Tuning (Italian Edition)

This repository is an optimized workspace for training, evaluating, and deploying a custom **Italian Speech-to-Text (ASR)** model using the Moonshine AI architecture.

By leveraging the monolingual approach outlined in the *"Flavors of Moonshine"* architecture, this pipeline helps you fine-tune a tiny, laser-focused **27-million parameter model (`moonshine-tiny-it`)** that provides ultra-low latency (<200ms processing windows) directly on edge devices like Raspberry Pis, Macs, and smartphones.

---

## 🚀 Quick Start Pipeline

We use [Task](https://taskfile.dev/) to automate the entire environment setup, dataset preparation, and training pipeline.

```bash
# 1. Bootstrap the project (Creates virtual environment & installs dependencies)
task setup

# 2. Download a tiny mini dataset (200 Italian samples) for rapid testing
task dataset-mini

# 3. Generate the mini configuration pointing to the local dataset
task config-mini

# 4. Run a rapid "smoke test" training loop using the mini dataset
task train-mini

# 5. Download and segment the full production-scale Italian MLS dataset
task segment

# 6. Generate the full production configuration
task config

# 7. Launch full production fine-tuning over the complete dataset
task train

# 8. Convert trained weights to ONNX for deployment
task export-mini

# 9. Nuke everything (venv, datasets, checkpoints)
task clean
```

---

## 📁 Project Structure

```
.
├── .venv/                       # Virtual environment (Apple Silicon PyTorch)
├── configs/                     # YAML training configurations
│   ├── my_italian_model.yaml   # Production config (full dataset)
│   └── my_italian_model_mini.yaml # Mini config (quick test)
├── data/                        # Datasets (gitignored)
│   ├── italian_mini/            # 200-sample mini dataset
│   └── mls_italian_segmented/  # Full segmented MLS dataset
├── results-moonshine-it/       # Production training outputs
├── results-moonshone-it-mini/  # Mini training outputs
├── scripts/                     # Utility scripts
├── train.py                     # Main training script
├── Taskfile.yaml                # Task automation definitions
└── requirements.txt             # Python dependencies
```

---

## 🧪 Quick Commands

| Command | What it does |
|---------|-------------|
| `task setup` | Create venv, install torch + deps |
| `task dataset-mini` | Download 200 Italian samples |
| `task config-mini` | Generate mini YAML config |
| `task train-mini` | Train on mini dataset (5 epochs) |
| `task segment` | Download & segment full MLS (heavy) |
| `task config` | Generate production YAML config |
| `task train` | Full production training (3 epochs) |
| `task export-mini` | Export mini-trained model to ONNX |
| `task clean` | Remove everything |

---

## ⚙️ Custom Training

```bash
# Train with a custom config
python train.py --config configs/my_custom_config.yaml

# Resume from a checkpoint
python train.py --config configs/my_config.yaml --resume results-moonshine-it/checkpoint-1000

# Test mode (uses only 500 samples)
python train.py --config configs/my_config.yaml --test-mode
```