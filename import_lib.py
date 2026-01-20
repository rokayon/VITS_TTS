# !nvidia-smi
import numpy as np 
import pandas as pd 
import os
import torch
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsAudioConfig

from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from TTS.tts.configs.shared_configs import BaseDatasetConfig,BaseAudioConfig,CharactersConfig

from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.speakers import SpeakerManager

from TTS.config import load_config

from trainer import Trainer, TrainerArgs
