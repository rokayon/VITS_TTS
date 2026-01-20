import os
from pathlib import Path
from TTS.tts.configs.shared_configs import BaseDatasetConfig

def get_dataset_config(base_path, male=False):
    if male:
        meta_file = base_path / "iitm_bangla_tts" / "comprehensive_bangla_tts" / "male" / "mono" / "metadata_male.txt"
        root_path = base_path / "iitm_bangla_tts" / "comprehensive_bangla_tts" / "male" / "mono"
    else:
        meta_file = base_path / "iitm_bangla_tts" / "comprehensive_bangla_tts" / "female" / "mono" / "metadata_female.txt"
        root_path = base_path / "iitm_bangla_tts" / "comprehensive_bangla_tts" / "female" / "mono"
    return BaseDatasetConfig(meta_file_train=meta_file, path=os.path.join(root_path, "")), meta_file, root_path

def formatter(root_path, meta_file, **kwargs):
    txt_file = meta_file
    items = []
    speaker_name = "ljspeech"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wav", cols[0] + ".wav")
            try:
                text = cols[1]
            except:
                print("not found")
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items
