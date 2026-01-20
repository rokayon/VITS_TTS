
# Step 1: Run git_clone.py logic (clone repo and install dependencies)
import subprocess
import sys

def run_shell(cmd):
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(1)


run_shell("git clone https://github.com/idiap/coqui-ai-TTS")
import os
os.chdir("coqui-ai-TTS")
run_shell("pip install -e .")
run_shell("pip install torchcodec")



# Step 2: Import all libraries (import_lib.py)
import numpy as np
import pandas as pd
import os
import torch
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.shared_configs import BaseDatasetConfig, BaseAudioConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.speakers import SpeakerManager
from TTS.config import load_config
from trainer import Trainer, TrainerArgs
import os
from dataset import get_base_path
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
import kagglehub
from pathlib import Path
# !nvidia-smi
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import torch
# Trainer: Where the ✨️ happens.
# TrainingArgs: Defines the set of arguments of the Trainer.
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsAudioConfig
# BaseDatasetConfig: defines name, formatter and path of the dataset.


from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from TTS.tts.configs.shared_configs import BaseDatasetConfig,BaseAudioConfig,CharactersConfig
#from TTS.configs import BaseDatasetConfig,BaseAudioConfig,CharactersConfig#GlowTTSConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.speakers import SpeakerManager

from TTS.config import load_config

from trainer import Trainer, TrainerArgs

# Step 3: Download dataset (dataset.py)
from dataset import get_base_path
base_path = get_base_path()

# Step 4: Prepare dataset config and samples (dataset_config.py)
from dataset_config import output_path, male, pretrained_path, dataset_config, train_samples, eval_samples

# Step 5: Prepare model config (config.py)
from config import config

# Step 6: Training logic
ap = AudioProcessor.init_from_config(config)
ap.resample
tokenizer, config = TTSTokenizer.init_from_config(config)
model = Vits(config, ap, tokenizer, speaker_manager=None)


model

import torch

# Clear CUDA cache to free up memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

trainer = Trainer(
    TrainerArgs(continue_path=pretrained_path),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples
)
%%time
trainer.fit()