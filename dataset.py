
# dataset.py: Handles dataset download and exposes base_path
import kagglehub
from pathlib import Path

def get_base_path():
	mobassir_comprehensive_bangla_tts_path = kagglehub.dataset_download('mobassir/comprehensive-bangla-tts')
	print('Data source import complete.')
	return Path(mobassir_comprehensive_bangla_tts_path)