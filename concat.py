import os
from pydub import AudioSegment

# Define the main directory containing the subfolders
main_directory = './llama3/output'
# Define the output directory to save the combined audio files
output_directory = './llama3/audio'
os.makedirs(output_directory, exist_ok=True)

# Iterate over all subdirectories in the main directory
for subdir in os.listdir(main_directory):
    subdir_path = os.path.join(main_directory, subdir)
    if os.path.isdir(subdir_path):
        # Create a list to store tuples of (filename, number)
        files_with_numbers = []

        # Iterate over all files in the subdirectory
        for filename in os.listdir(subdir_path):
            if filename.endswith('.wav'):
                # Extract the number part from the filename
                number_part = int(filename.split('_')[1].split('.')[0])
                files_with_numbers.append((filename, number_part))

        # Sort the list of tuples by the number part
        files_with_numbers.sort(key=lambda x: x[1])

        # Initialize an empty AudioSegment
        combined_audio = AudioSegment.empty()

        # Concatenate the audio files in the sorted order
        for filename, _ in files_with_numbers:
            audio_path = os.path.join(subdir_path, filename)
            audio = AudioSegment.from_wav(audio_path)
            combined_audio += audio

        # Export the combined audio to a new file with the name of the subdirectory
        output_path = os.path.join(output_directory, f'{subdir}.wav')
        combined_audio.export(output_path, format='wav')

        print(f"Combined audio for {subdir} saved to {output_path}")

print("All audio combinations completed.")
