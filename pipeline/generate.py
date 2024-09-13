import argparse
from langchain_community.llms import Ollama
from  personas import personas

llm = Ollama(model='llama2')
parser = argparse.ArgumentParser(description="Generate conversations.")
parser.add_argument("--n", type=int, default=1, help="Number of conversations to generate")
parser.add_argument("--o", type=str, default="dialogues", help="Folder to save the conversations")
parser.add_argument("--min", type=int, default=2, help="Minimum number of personas participating the conversation")
parser.add_argument("--max", type=int, help="Maximum number of personas participating the conversation")
args = parser.parse_args()
