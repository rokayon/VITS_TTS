from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

def build_audio_processor(config):
    return AudioProcessor.init_from_config(config)

def build_tokenizer(config):
    tokenizer, config = TTSTokenizer.init_from_config(config)
    return tokenizer, config

def build_vits_model(config, ap, tokenizer):
    return Vits(config, ap, tokenizer, speaker_manager=None)
