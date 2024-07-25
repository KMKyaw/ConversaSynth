import os
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import json
import re
from personas import personas  # Importing the personas

def remove_bounded_words(text):
    return re.sub(r'\*[^*]*\*', '', text).strip()

def process_json_file(file_path):
    with open(file_path, 'r') as file:
        dialogues = file.readlines()

    character_dialogues = {}
    for line in dialogues:
        data = json.loads(line.strip())
        prompt = data['dialogue']
        prompt = remove_bounded_words(prompt)
        name = data['name']
        
        if name not in character_dialogues:
            character_dialogues[name] = []
        character_dialogues[name].append(prompt)
    
    return character_dialogues
file_counter = 0
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

# Convert personas list to a dictionary for easy access
persona_dict = {persona.name: persona for persona in personas}

json_directory = './dialogues'
output_directory = './raw'
print("Starting to generate audio...")

# Iterate through each JSON file in the directory
for json_file in os.listdir(json_directory):
    file_counter += 1
    print(f"####Currently processing {json_file}####")
    if json_file.endswith('.json'):
        file_path = os.path.join(json_directory, json_file)
        character_dialogues = process_json_file(file_path)
        
        # Create output directory for the current JSON file
        json_name = os.path.splitext(json_file)[0]
        json_output_directory = os.path.join(output_directory, json_name)
        os.makedirs(json_output_directory, exist_ok=True)
        
        # Generate audio for each character in the JSON file
        for name, dialogues in character_dialogues.items():
            print(f"Generating audio for {name} in file {json_file}")
            
            # Retrieve the style from the persona
            description = persona_dict[name].style if name in persona_dict else "Default style if name not found"
            
            input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
            
            # Join all dialogues into one prompt
            full_prompt = " ".join(dialogues)
            prompt_input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
            
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, max_new_tokens=4000)
            audio_arr = generation.cpu().numpy().squeeze()
            
            output_file_path = os.path.join(json_output_directory, f"{name}.wav")
            sf.write(output_file_path, audio_arr, model.config.sampling_rate)
print("Audio generation completed.")
print(f"Total folder generated : {file_counter}")
