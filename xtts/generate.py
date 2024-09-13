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
texts = dataset['train']['text'][:40]
combined_text = " ".join(texts)

# Limit characters
max_characters = 200000
text = combined_text[:max_characters]

print("Starting to generate audio...")
# Run TTS
#tts.tts_to_file(text=text, speaker_wav="../voices/David.wav", language="en", file_path="extended/David_train.wav")
#tts.tts_to_file(text=text, speaker_wav="../voices/Alice.wav", language="en", file_path="extended/Alice_train.wav")
#tts.tts_to_file(text=text, speaker_wav="../voices/Ben.wav", language="en", file_path="extended/Ben_train.wav")
#tts.tts_to_file(text=text, speaker_wav="../voices/Cathy.wav", language="en", file_path="extended/Cathy_train.wav")
#tts.tts_to_file(text=text, speaker_wav="../voices/Eva.wav", language="en", file_path="extended/Eva_train.wav")
#tts.tts_to_file(text=text, speaker_wav="../voices/Frank.wav", language="en", file_path="extended/Frank_train.wav")
tts.tts_to_file(text=text, speaker_wav="../voices/Grace.wav", language="en", file_path="extended/Grace_train.wav")
#tts.tts_to_file(text=text, speaker_wav="../voices/Henry.wav", language="en", file_path="extended/Henry_train.wav")
#tts.tts_to_file(text=text, speaker_wav="../voices/Isabella.wav", language="en", file_path="extended/Isabella_train.wav")
