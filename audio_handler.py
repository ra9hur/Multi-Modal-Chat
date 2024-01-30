import torch
from transformers import pipeline
import librosa
import io


def convert_bytes_to_array(audio_bytes):
    audio_array, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    return audio_array


def transcribe_audio(audio_bytes):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        device=device,
    )

    audio_array = convert_bytes_to_array(audio_bytes)
    prediction = pipe(audio_array, batch_size=1)["text"]
    return prediction
