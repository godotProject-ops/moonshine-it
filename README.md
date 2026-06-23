# 🌙 Moonshine ASR Fine-Tuning (Italian Edition)

This repository is an optimized workspace for training, evaluating, and deploying a custom **Italian Speech-to-Text (ASR)** model using the Moonshine AI architecture.

By leveraging the monolingual approach outlined in the *"Flavors of Moonshine"* architecture, this pipeline helps you fine-tune a tiny, laser-focused **27-million parameter model (`moonshine-tiny-it`)** that provides ultra-low latency (<200ms processing windows) directly on edge devices like Raspberry Pis, Macs, and smartphones.

---

## 🚀 Quick Start

### Prerequisiti

- **Python 3.10+**
- **[Task](https://taskfile.dev/)** — `brew install go-task` (macOS) oppure `snap install task` (Linux)
- ~12 GB di spazio su disco per il dataset completo

### 1. Ambiente e dipendenze

```bash
task setup
```

Crea un virtual environment (`.venv/`) e installa PyTorch con accelerazione MPS (Apple Silicon) o CUDA.

### 2. Mini test (200 campioni, ~2 minuti)

Prova l'intera pipeline con un dataset microscopico per verificare che tutto funzioni:

```bash
task dataset-mini    # Scarica 200 campioni italiani
task train-mini      # Allena per 3 epoche su 200 campioni
```

Il modello viene salvato in `results-moonshine-it-phoneme/final/`.

### 3. Training completo (produzione)

Per allenare sul dataset integrale **Multilingual LibriSpeech (Italian)**:

```bash
task segment         # Scarica e segmenta il dataset (~7 GB, richiede tempo)
task train           # Allena moonshine-base per 3 epoche
```

### 4. Deploy (ONNX)

```bash
task export-mini     # Converte i pesi del modello mini in ONNX
task export          # Converte i pesi del modello full in ONNX
```

### Reset

```bash
task clean           # Rimuove venv, dataset, e tutti i risultati
```

---

## 📊 Confronto: Mini vs Produzione

| Caratteristica | `task train-mini` | `task train` |
|---|---|---|
| **Modello** | `moonshine-tiny` (27M params) | `moonshine-base` (87M params) |
| **Dataset** | 200 campioni (MLS dev) | ~300h MLS italiano segmentato |
| **Tempo stimato** | ~2 minuti | ~ore/giorni |
| **Batch** | 4 | 8 (con grad_accum 4 → effettivo 32) |
| **Gradient checkpointing** | ❌ No | ✅ Sì |
| **Mixed precision** | `bf16` + `tf32` | `bf16` + `tf32` |
| **Dataloader workers** | — | 8 |
| **Output** | `results-moonshine-it-phoneme/final/` | `results-moonshine-it/final/` |
| **Scopo** | Smoke test / sviluppo | Training production |
| **Config** | `configs/my_italian_model_mini.yaml` | `configs/my_italian_model.yaml` |

---

## 📖 Comandi

| Comando | Cosa fa |
|---------|---------|
| `task setup` | Prepara l'ambiente virtuale |
| `task dataset-mini` | Scarica 200 campioni italiani per test |
| `task config-mini` | Genera config per mini dataset |
| `task train-mini` | Allena su 200 campioni (~2 min) |
| `task segment` | Scarica e segmenta MLS italiano (~7 GB) |
| `task config` | Genera config per produzione |
| `task train` | Allena su dataset completo (3 epoche) |
| `task export-mini` | Esporta modello mini in ONNX |
| `task export` | Esporta modello full in ONNX |
| `task clean` | Cancella tutto (venv, dati, risultati) |

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

## ⚙️ Custom Training

```bash
# Train with a custom config
python train.py --config configs/my_custom_config.yaml

# Resume from a checkpoint
python train.py --config configs/my_config.yaml --resume results-moonshine-it/checkpoint-1000

# Test mode (uses only 500 samples)
python train.py --config configs/my_config.yaml --test-mode
```