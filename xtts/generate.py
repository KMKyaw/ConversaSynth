import torch
from TTS.api import TTS
from datasets import load_dataset

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Load TinyStories dataset
dataset = load_dataset("roneneldan/TinyStories")

# Extract up to 50 rows from the text column and concatenate them
texts = dataset['train']['text'][:200]
combined_text = " ".join(texts)

# Limit to 3000 characters
max_characters = 300000
text = combined_text[:max_characters]

# Run TTS
tts.tts_to_file(text=text, speaker_wav="../raw/41/David.wav", language="en", file_path="David.wav")
