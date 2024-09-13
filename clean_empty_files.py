import os

def remove_empty_json_files(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    for file in files:
        # Construct the full file path
        file_path = os.path.join(folder_path, file)
        
        # Check if the file is a JSON file
        if file.endswith('.json'):
            with open(file_path, 'r') as f:
                content = f.read().strip()
                # Check if the file is empty
                if not content:
                    os.remove(file_path)
                    print(f"Removed empty JSON file: {file_path}")

# Replace 'your_folder_path' with the path to your folder
remove_empty_json_files('./comparison/llama3')
