import os
import random
from pydub import AudioSegment
from pydub.effects import normalize

# Define the input directory containing the .wav files
input_directory = './combined_outputs'
# Path to the noise file
noise_file_path = './NoisyClass.wav'
# Define the output directory to save the combined audio files
output_directory = './combined_outputs_w_noises'
os.makedirs(output_directory, exist_ok=True)

# Load the noise file
noise = AudioSegment.from_wav(noise_file_path)

# Iterate over all .wav files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.wav'):
        audio_file_path = os.path.join(input_directory, filename)

        # Load the audio file
        audio = AudioSegment.from_wav(audio_file_path)

        # Ensure the noise file is long enough; if not, loop it
        if len(noise) < len(audio):
            repeat_count = (len(audio) // len(noise)) + 1
            noise = noise * repeat_count

        # Trim the noise to match the length of the audio file
        noise_segment = noise[:len(audio)]

        # Apply effects to make the background noise more realistic
        # Normalize noise to ensure consistent volume
        noise_segment = normalize(noise_segment)
        # Reduce noise volume with a random value between 7 and 15
        noise_reduction = random.randint(7, 15)
        noise_segment = noise_segment - noise_reduction

        # Overlay the noise onto the original audio
        combined_audio = audio.overlay(noise_segment)

        # Export the combined audio
        output_path = os.path.join(output_directory, filename)
        combined_audio.export(output_path, format='wav')

        print(f"Combined audio with noise saved to {output_path}")

print("All audio files processed.")
