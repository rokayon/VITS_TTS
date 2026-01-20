import kagglehub
from pathlib import Path

# Download dataset
mobassir_comprehensive_bangla_tts_path = kagglehub.dataset_download('mobassir/comprehensive-bangla-tts')
print('Data source import complete.')

# Set base path for dataset
base_path = Path(mobassir_comprehensive_bangla_tts_path)
