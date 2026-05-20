# Unsloth Fine-Tuning Notebook

This repository contains a notebook workflow for fine-tuning large language models with [Unsloth](https://github.com/unslothai/unsloth). It is intended for experimenting with efficient supervised fine-tuning using LoRA or QLoRA-style adapters.

## Overview

The project is organized around an interactive notebook, `unsloth_finetuning.ipynb`, that can be used to:

- Load a supported base language model.
- Prepare an instruction or chat-style dataset.
- Configure efficient fine-tuning settings.
- Train adapter weights with Unsloth.
- Save, export, or push the fine-tuned model artifacts.

## Requirements

Use a Python environment with GPU support when possible. A CUDA-capable GPU is strongly recommended for practical training speed.

Common dependencies include:

- Python 3.10+
- PyTorch
- Transformers
- Datasets
- TRL
- PEFT
- Accelerate
- Unsloth
- Jupyter Notebook or JupyterLab

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install torch transformers datasets trl peft accelerate jupyter
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

Depending on your CUDA, PyTorch, and platform versions, you may need to adjust the installation commands. Check the official Unsloth documentation for the most current install instructions.

## Usage

Start Jupyter:

```bash
jupyter notebook
```

Open `unsloth_finetuning.ipynb` and run the cells in order.

Typical workflow:

1. Select a base model.
2. Load and inspect the dataset.
3. Format prompts or chat messages.
4. Configure LoRA, sequence length, batch size, and training arguments.
5. Run training.
6. Test inference with the fine-tuned adapter.
7. Save or export the final model.

## Dataset

Update the notebook with the dataset you want to fine-tune on. For best results, verify that each training example has a consistent structure, such as:

```json
{
  "instruction": "Explain what LoRA is.",
  "input": "",
  "output": "LoRA is a parameter-efficient fine-tuning method..."
}
```

or a chat format:

```json
{
  "messages": [
    { "role": "user", "content": "Explain what LoRA is." },
    { "role": "assistant", "content": "LoRA is a parameter-efficient fine-tuning method..." }
  ]
}
```

## Outputs

Training may produce artifacts such as:

- LoRA adapter weights
- Tokenizer files
- Training checkpoints
- Merged model files
- GGUF or other export formats, if configured

Large generated artifacts should usually be excluded from Git and stored in a model registry or release asset.

## Notes

- Review model licenses before fine-tuning or publishing derived models.
- Review dataset licenses and privacy requirements before training.
- Track important training settings, including base model, dataset version, sequence length, learning rate, batch size, and number of epochs.
