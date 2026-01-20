# VITS Bangla TTS Training Pipeline

This repository provides a pipeline for training a Bangla Text-to-Speech (TTS) model using the VITS architecture and the Coqui TTS library.

## Project Structure

- `git_clone.py` *(now integrated in trainer.py)*: Clones the Coqui TTS repository and installs dependencies.
- `import_lib.py`: Contains all necessary imports for the training pipeline.
- `dataset.py`: Downloads the Bangla TTS dataset from Kaggle and exposes the dataset path.
- `dataset_config.py`: Prepares dataset configuration, splits, and formatting for training.
- `config.py`: Sets up the model and training configuration.
- `trainer.py`: The main script that runs all steps in order and starts the training process.

## Setup Instructions

### 1. Clone and Install Dependencies

The first step is handled automatically by `trainer.py`, but you can run these manually if needed:

```sh
# Clone the Coqui TTS repository
!git clone https://github.com/idiap/coqui-ai-TTS
# Change directory
%cd coqui-ai-TTS
# Install the package in editable mode
!pip install -e .
# Install torchcodec
!pip install torchcodec
```

### 2. Dataset Download

The dataset is downloaded automatically using KaggleHub in `dataset.py`. Make sure you have Kaggle API credentials set up if required.

### 3. Training

To start training, simply run:

```sh
python trainer.py
```

This will:
- Clone and install dependencies
- Download and prepare the dataset
- Set up the model and training configuration
- Start the training loop

## File Details

### `trainer.py`
- Orchestrates the entire pipeline.
- Runs shell commands to clone and install dependencies.
- Imports all modules and runs the training loop.

### `import_lib.py`
- Contains all Python imports required for the pipeline.

### `dataset.py`
- Downloads the dataset from Kaggle using KaggleHub.
- Exposes a function `get_base_path()` to provide the dataset path.

### `dataset_config.py`
- Uses the dataset path to set up training and evaluation splits.
- Defines a formatter for the dataset.
- Exposes variables for use in training.

### `config.py`
- Sets up audio and character configurations for the VITS model.
- Prepares the main `VitsConfig` object for training.

## Notes
- All steps are automated in `trainer.py`.
- You can modify `config.py` and `dataset_config.py` to adjust training parameters or dataset splits.
- Make sure you have Python, pip, and all required system dependencies installed.

## License
This project is for research and educational purposes. Please check the licenses of the Coqui TTS library and the dataset for usage restrictions.
