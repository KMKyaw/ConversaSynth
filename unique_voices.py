import os
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import re
from personas import personas  # Importing the personas
from datasets import load_dataset  # Importing the datasets module from Hugging Face

def remove_bounded_words(text):
    return re.sub(r'\*[^*]*\*', '', text).strip()

file_counter = 0
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

# Convert personas list to a dictionary for easy access
persona_dict = {persona.name: persona for persona in personas}

output_directory = './voices'
os.makedirs(output_directory, exist_ok=True)
print("Starting to generate audio...")

full_prompt = "That was the prospect a week ago. But another blow which might well have proved final was yet to fall upon us. The King of the Belgians had called upon us to come to his aid. Had not this Ruler and his Government severed themselves from the Allies, who rescued their country from extinction in the late war, and had they not sought refuge in what was proved to be a fatal neutrality, the French and British Armies might well at the outset have saved not only Belgium but perhaps even Poland. Yet at the last moment, when Belgium was already invaded, King Leopold called upon us to come to his aid, and even at the last moment we came. He and his brave, efficient Army, nearly half a million strong, guarded our left flank and thus kept open our only line of retreat to the sea. Suddenly, without prior consultation, with the least possible notice, without the advice of his Ministers and upon his own personal act, he sent a plenipotentiary to the German Command, surrendered his Army, and exposed our whole flank and means of retreat."

# Iterate through each persona
for persona_name, persona in persona_dict.items():
    file_counter += 1
    print(f"#### Currently processing {persona_name} ####")
    
    # Retrieve the style from the persona
    description = persona.style if persona_name in persona_dict else "Default style if name not found"
    
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    
    # Prepare the prompt
    prompt_input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
    
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, max_new_tokens=4000)
    audio_arr = generation.cpu().numpy().squeeze()
    
    output_file_path = os.path.join(output_directory, f"{persona_name}.wav")
    sf.write(output_file_path, audio_arr, model.config.sampling_rate)
    
print("Audio generation completed.")
print(f"Total personas processed: {file_counter}")
