# TEOH masters thesis project

## Current Version Stack
- Python 3.13
- PyTorch: 2.8.0+cu128 (compiled with CUDA 12.8)
- NVIDIA Driver Version: 530 or 535
- CUDA Toolkit: 13.0

## Setup (Debian)

- Use pyenv to install python 3.13
```bash
pyenv shell 3.13
python -m venv env
source env/bin/activate
```

- Install dependencies
```bash
pip install -r requirements.txt
```

## Setup (Windows)

- Use pyenv to install python 3.13.1 (Tcl dependency issue with 3.13.0)
```bash
pyenv shell 3.13
python -m venv env
.\env\Scripts\Activate
```

- Install dependencies
```bash
pip install -r requirements.txt
```

## Environment Setup

```bash
source env/bin/activate
```

## Run Model Training and Evaluation

```bash
python run.py
```

## Tensorboard to check result

```
tensorboard --logdir=outputs
```