import json
import os
import re

def transform_json_to_text(json_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for json_file in os.listdir(json_folder):
        if json_file.endswith(".json"):
            with open(os.path.join(json_folder, json_file), 'r') as file:
                lines = file.readlines()
                
            # Prepare the output text
            output_text = ""
            for line in lines:
                dialogue = json.loads(line.strip())
                name = dialogue["name"]
                text = dialogue["dialogue"]
                # Remove words between * *
                text = re.sub(r"\*[^*]*\*", "", text)
                output_text += f"{name}|{text}\n"
            
            # Write the output text to a file
            output_file_name = os.path.splitext(json_file)[0] + ".txt"
            with open(os.path.join(output_folder, output_file_name), 'w') as output_file:
                output_file.write(output_text)

# Specify the folders
json_folder = "./dialogues"
output_folder = "./dialogues_text"

# Transform JSON files to formatted text files
transform_json_to_text(json_folder, output_folder)
