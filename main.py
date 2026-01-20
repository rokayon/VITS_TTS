# Example main script to orchestrate the workflow
# Adjust paths and flags as needed
import os
from data.download_data import mobassir_comprehensive_bangla_tts_path, base_path
from data.prepare_dataset import get_dataset_config, formatter
from TTS.tts.datasets import load_tts_samples
from model.config import get_audio_config, get_characters_config, get_vits_config
from model.build_model import build_audio_processor, build_tokenizer, build_vits_model
from train.train import train_model
from inference.synthesize import synthesize

# Data preparation
male = False
output_path = "content/tts_model"
dataset_config, meta_file, root_path = get_dataset_config(base_path, male=male)
train_samples, eval_samples = load_tts_samples(dataset_config, formatter=formatter, eval_split=True)

# Model config
audio_config = get_audio_config()
characters_config = get_characters_config(male=male)
config = get_vits_config(audio_config, characters_config, dataset_config, output_path)

# Build model
ap = build_audio_processor(config)
tokenizer, config = build_tokenizer(config)
model = build_vits_model(config, ap, tokenizer)

# Train
train_model(config, output_path, model, train_samples, eval_samples)

# Inference
model_dir = output_path + "/vits_4_nov-January-18-2026_12+29PM-12b399fc"
config_dir = model_dir
output_dir = "/content/working/output"
text = "আমরা আপনাদের জন্য নিয়ে এলাম ফ্রি এবং ওপেনসোর্স বাংলা ফন্টের সবচেয়ে বড় সংগ্রহশালা, অক্ষর ৫২! "
synthesize(model_dir, config_dir, output_dir, text)
