import json
import os
from tqdm import tqdm
import torch
from TTS.api import TTS
import re
import time
def remove_bounded_words(text):
    # Pattern to match text within *, (), [], {}
    pattern = r'[\*\(\[\{][^*\(\[\{\]\}\)]*[\*\)\]\}]'
    return re.sub(pattern, '', text).strip()

# Function to process the JSON file and generate audio for each dialogue
def generate_dialogue_audio(dialogue_folder, raw_folder, output_folder):
    # Get device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Init TTS model
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # Iterate over all JSON files in the dialogue folder
    for json_filename in os.listdir(dialogue_folder):
        if json_filename.endswith('.json'):
            dialogue_path = os.path.join(dialogue_folder, json_filename)
            
            # Extract dialogue index from the filename
            dialogue_index = os.path.splitext(json_filename)[0]
            
            # Read dialogues from JSON file
            with open(dialogue_path, 'r') as f:
                dialogues = [json.loads(line) for line in f]
            
            # Counter for the overall order of dialogues for the current JSON file
            overall_counter = 0

            # Iterate over each dialogue
            for dialogue in tqdm(dialogues, desc=f"Processing {json_filename}"):
                name = dialogue['name']
                text = dialogue['dialogue']
                text = remove_bounded_words(text)
                # Construct the path to the speaker's audio file
                speaker_wav = os.path.join(raw_folder, f"{name}.wav")
                
                # Ensure the speaker's audio file exists
                if not os.path.exists(speaker_wav):
                    print(f"Warning: {speaker_wav} does not exist. Skipping...")
                    continue
                
                # Construct the path for the output file
                output_wav_folder = os.path.join(output_folder, dialogue_index)
                output_wav = os.path.join(output_wav_folder, f"{name}_{overall_counter}.wav")
                
                # Ensure the output directory exists
                os.makedirs(output_wav_folder, exist_ok=True)
                print(text)
                print(speaker_wav)
                # Generate TTS audio
                tts.tts_to_file(text=text, speaker_wav=speaker_wav, language="en", file_path=output_wav)
                
                # Increment the overall counter
                overall_counter += 1

start_time = time.time()
# Define paths
dialogue_folder = './comparison/llama3'  # Folder containing the JSON files
raw_folder = './voices'  # Folder containing the .wav files for each speaker
output_folder = './llama3/output'  # Folder to save the generated audio files

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Generate audio for each dialogue
generate_dialogue_audio(dialogue_folder, raw_folder, output_folder)
# Record the end time
end_time = time.time()

# Calculate the duration
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")
