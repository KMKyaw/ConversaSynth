import os
import soundfile as sf

# Path to the folder containing wav files
wav_folder = "./llama3/audio"

# Initialize total duration in seconds
total_duration_seconds = 0.0

# Iterate over all files in the folder
for filename in os.listdir(wav_folder):
    if filename.endswith(".wav"):
        filepath = os.path.join(wav_folder, filename)
        # Read the audio file
        data, samplerate = sf.read(filepath)
        # Calculate the duration and add to the total duration
        total_duration_seconds += len(data) / samplerate

# Convert total duration to minutes and hours
total_duration_minutes = total_duration_seconds / 60
total_duration_hours = total_duration_minutes / 60

# Print the total duration
print(f"The cumulative length of all wav files is {total_duration_seconds:.2f} seconds.")
print(f"Which is approximately {total_duration_minutes:.2f} minutes.")
print(f"Or {total_duration_hours:.2f} hours.")
