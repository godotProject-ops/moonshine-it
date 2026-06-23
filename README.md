# 🌙 Moonshine ASR Fine-Tuning (Italian Edition)

This repository is an optimized workspace for training, evaluating, and deploying a custom **Italian Speech-to-Text (ASR)** model using the Moonshine AI architecture. 

By leveraging the monolingual approach outlined in the *"Flavors of Moonshine"* architecture, this pipeline helps you fine-tune a tiny, laser-focused **27-million parameter model (`moonshine-tiny-it`)** that provides ultra-low latency (<200ms processing windows) directly on edge devices like Raspberry Pis, Macs, and smartphones.

---

## 🚀 Quick Start Pipeline

We use [Task](https://taskfile.dev/) to automate the entire environment setup, dataset preparation, and training pipeline. You do not need to manually configure virtual environments or install packages step-by-step.

```bash
# 1. Bootstrap the project (Creates virtual environment & fixes missing torch dependencies)
task setup

# 2. Build the fine-tuning configuration file
task config

# 3. Download, dynamically segment the Italian dataset, and launch training
task train