# Whisper Fine Tune a Model

This repository contains instructions and scripts for fine-tuning a Whisper model.

## Installation

To get started, you need to upgrade `pip` and install the necessary Python packages. Run the following commands:

```bash
# Upgrade pip
pip install --upgrade pip

# Upgrade necessary packages
pip install --upgrade datasets[audio] transformers accelerate evaluate jiwer tensorboard gradio
