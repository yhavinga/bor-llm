from dotenv import load_dotenv
import os
from huggingface_hub import login
from datasets import load_dataset
from transformers import pipeline
import torch
from evaluate import load
sari = load("sari")

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HF_AUTH_TOKEN")
login(HUGGINGFACE_TOKEN)

ds = load_dataset("UWV/Leesplank_NL_wikipedia_simplifications_preprocessed", split="train", streaming=True)
ds = ds.shuffle(seed=42)

random_row = [i for i in ds.take(1)][0]
print(random_row)

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
)

query = f"""Vereenvoudig een Nederlandse alinea tot één duidelijke en aantrekelijke tekst geschikt voor volwassenen die Nederlands als tweede taal spreken. Gebruik woorden uit de basiswoordenlijkst Amsterdamse kleuters. Behoud directe citaten en leg culturele verwijzingen, uitdrukkingen en technische termen natuurlijk uit in de tekst. Pas de volgorde van informatie aan voor eenvoud en leesbaarheid. De alinea: {random_row["prompt"]}"""
messages = [
    {"role": "user", "content": query},
]
print(f"input: {messages[0]}")
print(f"expected output: {random_row['result']}")

outputs = pipe(
    messages,
    max_new_tokens=500,
)
response=outputs[0]["generated_text"][-1]["content"]
print(response)

sources=[random_row["prompt"]]
predictions=[response]
references=[[random_row["result"]]]
sari_score = sari.compute(sources=sources, predictions=predictions, references=references)
print(sari_score)
