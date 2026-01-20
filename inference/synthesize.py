import os
import glob
import time
from TTS.utils.synthesizer import Synthesizer
from IPython.display import Audio, display
from utils.audio_enhance import enhance_audio

def synthesize(model_dir, config_dir, output_dir, text):
    os.makedirs(output_dir, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
    configs = sorted(glob.glob(os.path.join(config_dir, "*.json")))
    print("Found checkpoints:", ckpts)
    print("Found configs:", configs)
    if not ckpts or not configs:
        raise RuntimeError("❌ No trained model or config found!")
    best_ckpt = [c for c in ckpts if "best_model" in c]
    model_path = best_ckpt[0] if best_ckpt else ckpts[-1]
    config_path = configs[0]
    print(f"\n✅ Loading model: {model_path}")
    print(f"✅ Loading config: {config_path}")
    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        use_cuda=True
    )
    print(f"\n🗣 Generating speech for: {text}")
    start_time = time.time()
    wav = synthesizer.tts(text)
    wav = enhance_audio(wav)
    generation_time = time.time() - start_time
    output_file = os.path.join(output_dir, "bengali_glowtts.wav")
    synthesizer.save_wav(wav, output_file)
    duration = len(wav) / synthesizer.output_sample_rate
    print(f"✅ Saved to: {output_file}")
    print(f"⏱️ Generation time: {generation_time:.3f} sec")
    print(f"🎧 Audio duration: {duration:.2f} sec")
    print(f"⚡ RTF: {generation_time / duration:.2f}x")
    display(Audio(output_file))
