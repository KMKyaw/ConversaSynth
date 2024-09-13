import os
import json

# Path to the folder containing JSON files
folder_path = './comparison/llama3/'

# List of allowed personas
personas_list = ["Alice", "Ben", "Cathy", "David", "Eva", "Frank", "Grace", "Henry", "Isabella"]
# Function to clean JSON files and remove unwanted files
def clean_and_filter_json_files(folder_path, personas_list):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Initialize a flag to check if the file contains any valid personas
            valid_persona_found = False
            
            # Filter out lines where dialogue is empty and check for valid personas
            cleaned_lines = []
            for line in lines:
                try:
                    json_obj = json.loads(line)
                    if json_obj.get("name") in personas_list:
                        valid_persona_found = True
                        if json_obj.get("dialogue"):
                            cleaned_lines.append(line)
                except json.JSONDecodeError:
                    continue
            
            # If no valid persona is found, delete the file
            if not valid_persona_found:
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            else:
                # Otherwise, write the cleaned lines back to the file
                with open(file_path, 'w') as file:
                    file.writelines(cleaned_lines)
                print(f"Cleaned file: {file_path}")

# Run the function
clean_and_filter_json_files(folder_path, personas_list)

