# You can also use llama.cpp
# ./build/bin/llama-cli -m ./outputs/bor-openhermes-dutch/converted.gguf -p "Je bent een geweldige Nederlandse AI assistent" -cnv --color --mlock --gpu-layers 32  --prio-batch 2 --threads 6
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
from colorama import init, Fore

init()  # Initialize colorama for Windows compatibility


def load_model(model_path):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    return model, tokenizer


def evaluate_model(model, tokenizer, test_prompts, model_path):
    print(f"\nEvaluating model: {model_path}")
    for prompt in test_prompts:
        chat = [
            {
                "role": "system",
                "content": "Je bent een behulpzame AI-assistent die vragen beantwoordt in het Nederlands.",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ]

        formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        print(f"\nPrompt: {Fore.CYAN}{prompt}{Fore.RESET}")
        print(f"{Fore.GREEN}Response: ", end="", flush=True)

        tokens = model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            streamer=TextStreamer(tokenizer, skip_prompt=True),
        )
        print(f"{Fore.RESET}\n" + "-" * 50)


def main(model_path="outputs/bor-openhermes-dutch"):
    test_prompts = [
        """Geef een samenvatting van de volgende tekst in een paar zinnen:
De Deense politie onderzoekt meldingen over onbekende drones die boven de haven van Køge zijn gezien. De politie kreeg vrijdagavond rond 22.30 uur een melding over twintig grotere drones, staat in een persbericht.
Agenten die naar de haven gingen, telden vier drones. Aan het begin van de nacht verdwenen die met hoge snelheid boven zee. De politie zegt contact te hebben gehad met meerdere instanties, maar niet te weten waar de drones vandaan kwamen of wat ze boven de haven deden.
Køge ligt 40 kilometer ten zuiden van de Deense hoofdstad Kopenhagen, aan de Oostzeekust. De afgelopen weken zijn problemen vastgesteld aan onderzeese elektriciteits- en communicatiekabels in de Oostzee, mogelijk veroorzaakt door sabotage. De NAVO gaat daarom haar militaire aanwezigheid op de Oostzee uitbreiden.
Het Deense leger gebruikt de haven van Køge voor militaire transporten. In december keerde materieel dat gebruikt was tijdens een NAVO-missie in Letland via de haven van Køge terug in Denemarken.
In december waren Amerikanen in de ban van mysterieuze drones die door burgers werden gezien in de staten New Jersey en New York. Ze zouden boven belangrijke infrastructuur zweven, zoals waterreservoirs, elektriciteitsleidingen, treinstations, politiebureaus en militaire installaties. Dat zorgde voor talrijke complottheorieën.
Het Witte Huis zei dat er geen reden was tot zorg en dat de dronevluchten "volledig legaal en rechtmatig" waren.""",
        "Toon aan n-e wortel uit a^m = a^m/n",
        "Vertel de geschiedenis van de gemeente Baarn alsof Harry Mulisch het geschreven heeft.",
        "Leg uit hoe fotosynthese werkt.",
        "Wat is de verwachtingswaarde? Geef de definitie en een voorbeeld.",
        "Wat zijn de belangrijkste gebeurtenissen in de Nederlandse geschiedenis?",
        "Los deze vergelijking op: 2x + 5 = 15",
    ]

    model, tokenizer = load_model(model_path)

    print("\nStreaming model responses...")
    print("-" * 50)

    evaluate_model(model, tokenizer, test_prompts, model_path)


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/bor-openhermes-dutch"
    main(model_path)
