# prompt_generator.py
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")
import random
from personas import personas
import json
import re
import argparse
import os
import time

def extract_conversation(text):
    start_token = "[CONV_BEGIN]"
    end_token = "[CONV_END]"
    
    start_index = text.find(start_token) + len(start_token)
    end_index = text.find(end_token)
    
    if start_index == -1 or end_index == -1:
        return ""
    
    return text[start_index:end_index].strip()

def extract_conversation_to_jsonl(conversation):
    dialogue_list = []
    pattern = re.compile(r"\[(\w+)\]\s(.*?)\s*(?=\[\w+\]|$)")
    matches = pattern.findall(conversation)
    
    for name, dialogue in matches:
        dialogue_obj = {
            "name": name,
            "dialogue": dialogue.strip()
        }
        dialogue_list.append(json.dumps(dialogue_obj))
    
    return dialogue_list

# Function to generate and save conversation
def generate_and_save_conversation(file_number, folder):
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Randomly pick a number between 2 and 5
    num_personas = random.randint(2, 5)

    # Randomly select the chosen number of personas from the list
    selected_personas = random.sample(personas, num_personas)

    # Construct the prompt
    prompt = ""
    for persona in selected_personas:
        prompt += f"\n{persona.name}:\nCharacteristics: {', '.join(persona.characteristics)}\nPersonality: {persona.personality}\n"

    prompt += f""" They are all sitting around a table, having a lively and engaging conversation. Always place the whole story inside [CONV_BEGIN] and [CONV_END]. The order of the personas doesn't have to be in sequential order it could be random. When referring to each character, please put their name in square brackets. Do not use any shortened form or contraction. Follow the format of the following example: 
Example 1:
[CONV_BEGIN]

[{selected_personas[0].name}] I believe there is a lot to be discussed. 
[{selected_personas[1].name}] I agree!

[CONV_END]

Example 2:
[CONV_BEGIN]

[{selected_personas[0].name}] Sometimes, I think about my life being good.
[{selected_personas[1].name}] That is great! I envy you.

[CONV_END]

IMPORTANT : MAKE SURE THAT EVERYONE PARTICIPATES IN THE CONVERSATION
"""

    # Get response from LLM
    response = llm.invoke(prompt)
    #print(response)
    # Extract conversation
    conversation = extract_conversation(response)
    #print(conversation)
    # Convert to JSONL
    jsonl_conversation = extract_conversation_to_jsonl(conversation)
    #print(jsonl_conversation)
    # Save to file in the specified folder
    filename = os.path.join(folder, f"{file_number}.json")
    with open(filename, 'w') as file:
        for line in jsonl_conversation:
            file.write(line + "\n")

    print(f"Conversation saved to {filename}")

# Main function to parse arguments and generate conversations
def main():
    parser = argparse.ArgumentParser(description="Generate and save conversations.")
    parser.add_argument("num_conversations", type=int, help="Number of conversations to generate")
    parser.add_argument("--folder", type=str, default="dialogues", help="Folder to save the conversations")
    args = parser.parse_args()
    start_time = time.time()
    for i in range(1, args.num_conversations + 1):
        generate_and_save_conversation(i, args.folder)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
if __name__ == "__main__":
    main()

