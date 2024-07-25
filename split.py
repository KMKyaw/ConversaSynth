import whisper
from pydub import AudioSegment
import os

def transcribe_audio(audio_path):
    model = whisper.load_model("base")  # Choose the model size as needed
    result = model.transcribe(audio_path, word_timestamps=True)
    return result['segments']

def split_audio(audio_path, segments, output_dir):
    audio = AudioSegment.from_wav(audio_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, segment in enumerate(segments):
        start_time = segment['start'] * 1000  # Convert to milliseconds
        end_time = segment['end'] * 1000
        segment_audio = audio[start_time:end_time]
        segment_audio.export(os.path.join(output_dir, f"segment_{i}.wav"), format="wav")

def load_transcript(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    transcript = []
    for line in lines:
        if "|" in line:
            speaker, dialogue = line.strip().split("|", 1)
            transcript.append({"speaker": speaker, "dialogue": dialogue})
    return transcript

def match_transcript_with_segments(transcript, segments):
    # Create a mapping from dialogue to segment index
    transcript_mapping = []
    for entry in transcript:
        for segment in segments:
            if entry["dialogue"].strip() in segment['text']:
                transcript_mapping.append({"speaker": entry["speaker"], "segment_index": segments.index(segment)})
                break
    return transcript_mapping

audio_path = "./raw/41/Ben.wav"
output_dir = "audio_segments"
transcript_file_path = "41.txt"

# Transcribe the audio
segments = transcribe_audio(audio_path)

# Load and match transcript
transcript = load_transcript(transcript_file_path)
transcript_mapping = match_transcript_with_segments(transcript, segments)

# Split the audio based on matched transcript segments
split_audio(audio_path, [segments[i] for i in transcript_mapping], output_dir)
