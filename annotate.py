import os
import pandas as pd
import soundfile as sf

# Path to the directory containing the dialogue folders
dialogues_path = './llama3/output'

# List to store annotation data
annotations = []

# Iterate over each dialogue folder
for dialogue_folder in os.listdir(dialogues_path):
    dialogue_path = os.path.join(dialogues_path, dialogue_folder)
    
    if os.path.isdir(dialogue_path):
        # Sort files to ensure correct order processing
        files = sorted([f for f in os.listdir(dialogue_path) if f.endswith('.wav')], key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        # Variable to keep track of the end time of the previous segment
        prev_end_time = 0
        
        for file_name in files:
            # Extract speaker and order from the filename
            speaker = file_name.split('_')[0]
            order = int(file_name.split('_')[1].split('.')[0])
            
            # Get the duration of the audio file
            file_path = os.path.join(dialogue_path, file_name)
            try:
                audio, sr = sf.read(file_path)
                duration = len(audio) / sr
                
                # Calculate start and end times
                start_time = prev_end_time
                end_time = start_time + duration
                
                # Append the annotation
                annotations.append([f'conversation{dialogue_folder}.wav', start_time, end_time, speaker])
                
                # Update prev_end_time
                prev_end_time = end_time
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Convert the annotations to a DataFrame and save to CSV
annotations_df = pd.DataFrame(annotations, columns=['filename', 'start', 'end', 'speaker'])
annotations_df.to_csv('annotations_llama3.csv', index=False)

print('Annotations saved to annotations.csv')
