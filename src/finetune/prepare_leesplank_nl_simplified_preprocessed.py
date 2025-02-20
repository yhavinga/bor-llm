from datasets import load_dataset, interleave_datasets
import random

SIMPLIFICATION_PROMPTS = [
    "Vereenvoudig deze tekst:",
    "Kun je deze tekst makkelijker maken?",
    "Herschrijf dit in eenvoudigere taal:",
    "Maak deze tekst begrijpelijker:",
    "Schrijf dit in simpelere woorden:",
    "Vertaal dit naar makkelijkere taal:",
    "Maak dit leesbaar voor een breder publiek:",
    "Herschrijf dit zodat iedereen het kan begrijpen:",
    "Maak deze tekst toegankelijker:",
    "Schrijf dit in klare taal:",
    "Vereenvoudig dit stuk tekst:",
    "Kun je dit makkelijker uitleggen?",
    "Maak dit stuk tekst eenvoudiger:",
    "Herschrijf dit in begrijpelijke taal:",
    "Zet dit om naar eenvoudige taal:",
    "Maak dit makkelijker te lezen:",
    "Schrijf dit in simpele taal:",
    "Vertaal dit naar begrijpelijke woorden:",
    "Maak dit tekststuk eenvoudiger:",
    "Herschrijf dit in alledaagse taal:"
]

def format_chat_leesplank(examples, seed=42):
    random.seed(seed)
    formatted_messages = [
        [
            {"role": "user", "content": f"{'Vereenvoudig: ' if (random.random() < 0.2 or len(prompt) < 100) else random.choice(SIMPLIFICATION_PROMPTS)} {prompt}"},
            {"role": "assistant", "content": result},
        ]
        for prompt, result in zip(examples["prompt"], examples["result"])
    ]
    return {"messages": formatted_messages}

def main(seed=42, ordered_ratio=0.7):
    # Load dataset (already sorted by Levenshtein difficulty)
    dataset = load_dataset(
        "UWV/Leesplank_NL_wikipedia_simplifications_preprocessed", 
        split="train"
    )
    n = len(dataset)

    # Randomly choose which examples go in ordered vs shuffled groups
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    # Split indices into ordered and shuffled sets
    cutoff = int(n * ordered_ratio)
    ordered_indices = sorted(indices[:cutoff])  # preserve ascending order
    shuffled_indices = indices[cutoff:]         # will be randomized

    # Create two subsets
    ds_ordered = dataset.select(ordered_indices)
    ds_shuffled = dataset.select(shuffled_indices).shuffle(seed=seed)

    # Interleave with probability-based sampling
    interleaved_dataset = interleave_datasets(
        [ds_ordered, ds_shuffled],
        probabilities=[ordered_ratio, 1 - ordered_ratio],
        seed=seed,
        stopping_strategy="all_exhausted"
    )
    
    # Format to chatml
    formatted_dataset = interleaved_dataset.map(
        lambda x: format_chat_leesplank(x, seed),
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=None,
    )
    
    # Save to Hub
    formatted_dataset.push_to_hub(
        "yhavinga/Leesplank_NL_wikipedia_simplifications_preprocessed_chatml_format",
        private=True,
        commit_message="Add ChatML formatted Dutch text simplification dataset"
    )
    
    print(f"Dataset size: {len(formatted_dataset)}")
    # Print example
    print("\nExample conversation:")
    print(formatted_dataset[0]["messages"])
    
    return formatted_dataset

if __name__ == "__main__":
    main(ordered_ratio=0.7)