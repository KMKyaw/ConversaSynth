import json
import re

def remove_bounded_words(text):
    # Use regular expression to find words bounded by non-alphabetic characters
    return re.sub(r'[\*\[\]].*?[\*\[\]]', '', text)

# Load the JSON data from the file
with open('./dialogues/31.json', 'r') as file:
    dialogues = file.readlines()

# Process each line in the file
for line in dialogues:
    data = json.loads(line.strip())
    name = data['name']
    dialogue = data['dialogue']
    cleaned_dialogue = remove_bounded_words(dialogue)
    print(f"Name: {name}, Dialogue: {cleaned_dialogue.strip()}")
