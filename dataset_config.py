import os
from dataset import get_base_path
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples

base_path = get_base_path()

output_path = "/tts_train_dir"
if not os.path.exists(output_path):
    os.makedirs(output_path)

male = False
pretrained = False


# Get dataset path from environment variable (passed from notebook)
mobassir_comprehensive_bangla_tts_path = os.environ.get('DATASET_PATH')
male = os.environ.get('MALE_DATASET', 'false').lower() == 'true'


pretrained_path = ''
if(pretrained):
    pretrained_path = ''
if(male):
    meta_file = base_path / "iitm_bangla_tts" / "comprehensive_bangla_tts" / "male" / "mono" / "metadata_male.txt"
    root_path = base_path / "iitm_bangla_tts" / "comprehensive_bangla_tts" / "male" / "mono"
else:
    meta_file = base_path / "iitm_bangla_tts" / "comprehensive_bangla_tts" / "female" / "mono" / "metadata_female.txt"
    root_path = base_path / "iitm_bangla_tts" / "comprehensive_bangla_tts" / "female" / "mono"



def formatter(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes the LJSpeech meta data file to TTS format
    https://keithito.com/LJ-Speech-Dataset/"""
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


dataset_config = BaseDatasetConfig(
     meta_file_train=meta_file,
    path=os.path.join(root_path, "")
)

dataset_config

# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config,formatter=formatter, eval_split=True)
print(len(train_samples),len(eval_samples))
train_samples[0]