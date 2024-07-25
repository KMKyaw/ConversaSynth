from pydub import AudioSegment
from pydub.effects import normalize

# Paths to your files
audio_file_path = './combined_audio.wav'
noise_file_path = './NoisyClass.wav'

# Load the audio file and the noise file
audio = AudioSegment.from_wav(audio_file_path)
noise = AudioSegment.from_wav(noise_file_path)

# Ensure the noise file is long enough; if not, loop it
if len(noise) < len(audio):
    repeat_count = (len(audio) // len(noise)) + 1
    noise = noise * repeat_count

# Trim the noise to match the length of the audio file
noise = noise[:len(audio)]

# Apply effects to make the background noise more realistic
# Normalize noise to ensure consistent volume
noise = normalize(noise)
# Reduce noise volume (optional)
noise = noise -  13

# Apply reverb to simulate distance (optional)
# This requires additional libraries like pydub-ffmpeg if using pydub alone
# You can use external tools or libraries for more advanced effects

# Overlay the noise onto the original audio
combined_audio = audio.overlay(noise)

# Export the combined audio
output_path = './combined_audio_with_noise.wav'
combined_audio.export(output_path, format='wav')

print(f"Combined audio with realistic noise saved to {output_path}")
