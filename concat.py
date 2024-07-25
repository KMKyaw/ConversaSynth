import os
from pydub import AudioSegment

# Define the directory containing the .wav files
directory = '/home/km_uot/kmk/ConversaSynth/output/129'
# Create a list to store tuples of (filename, number)
files_with_numbers = []

# Iterate over all files in the directory
for filename in os.listdir(directory):
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
    audio_path = os.path.join(directory, filename)
    audio = AudioSegment.from_wav(audio_path)
    combined_audio += audio

# Export the combined audio to a new file
output_path =  './combined_audio.wav'
combined_audio.export(output_path, format='wav')

print(f"Combined audio saved to {output_path}")
