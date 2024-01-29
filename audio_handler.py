import torch
from transformers import pipeline
from datasets import load_dataset

def transcribe_audio(audio_bytes):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    chunk_length_s=30,
    device=device,
    )

    prediction = pipe(sample.copy(), batch_size=8)["text"]
