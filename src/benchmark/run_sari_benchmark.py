import argparse
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from evaluate import load
import pandas as pd
import torch
from tqdm import tqdm

def load_model_generator(model_path):
    """Initialize and return the model pipeline and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    generator = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        torch_dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True,
        model_kwargs={"attn_implementation": "flash_attention_2"}
    )
    return generator, tokenizer

def simplify_text(generator, tokenizer, text):
    """Generate simplified text using the model with proper EOS token stopping"""
    chat = [
        {"role": "user", "content": "Vereenvoudig: " + text},
    ]
    
    response = generator(
        chat,
        do_sample=False,
        num_beams=5,
        num_return_sequences=1,
        max_new_tokens=2048,
        length_penalty=1.0,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id  # Stop at EOS token
    )

    generated_text = response[0]["generated_text"]
    
    if isinstance(generated_text, list):
        for message in generated_text:
            if message.get("role") == "assistant":
                return message.get("content", "")
    return generated_text

def run_sari_benchmark(generator, tokenizer, num_samples=100):
    """Run SARI benchmark on a sample of the dataset"""
    # Load dataset and metric
    ds = load_dataset("UWV/Leesplank_NL_wikipedia_simplifications_preprocessed", split="test")
    ds = ds.shuffle(seed=42)
    sari = load("sari")
    
    sources = []
    predictions = []
    references = []
    scores = []
    
    # Run evaluation
    print(f"\nEvaluating {num_samples} samples...")
    for row in tqdm(ds.take(num_samples), total=num_samples):
        # Get prediction
        simplified = simplify_text(generator, tokenizer, row["prompt"])
        
        # Store results
        sources.append(row["prompt"])
        predictions.append(simplified)
        references.append([row["result"]])
        
        # Calculate individual SARI score
        score = sari.compute(
            sources=[row["prompt"]], 
            predictions=[simplified], 
            references=[[row["result"]]]
        )
        scores.append(score["sari"])
    
    # Calculate average SARI score
    avg_sari = sum(scores) / len(scores)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        "Original": sources,
        "Model Output": predictions,
        "Reference": [ref[0] for ref in references],
        "SARI Score": scores
    })
    
    return avg_sari, results_df

def get_model_name_from_path(model_path):
    """Extract a clean model name from the model path for filename use"""
    # Extract the last part of the path (after the last slash)
    model_name = model_path.split("/")[-1]
    # Remove any special characters that might cause issues in filenames
    model_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in model_name)
    return model_name

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run SARI benchmark on a text simplification model')
    parser.add_argument('model_path', type=str, help='Path or name of the model on HuggingFace')
    parser.add_argument('--num_samples', type=int, default=10, 
                      help='Number of samples to evaluate (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                      help='Path to save detailed results CSV (default: {model_name}_{num_samples}_results.csv)')
    args = parser.parse_args()

    # Generate default output filename if not provided
    if args.output is None:
        model_name = get_model_name_from_path(args.model_path)
        args.output = f"{model_name}_{args.num_samples}_results.csv"

    try:
        # Import needed at the top of the file
        from transformers import AutoTokenizer
        
        # Load model and tokenizer
        print(f"Loading model: {args.model_path}")
        generator, tokenizer = load_model_generator(args.model_path)
        print("Model loaded successfully!")
        print(f"Using EOS token: {tokenizer.eos_token}")

        # Run benchmark
        avg_score, detailed_results = run_sari_benchmark(generator, tokenizer, args.num_samples)

        # Print results
        print("\n" + "="*50)
        print(f"Benchmark Results for {args.model_path}")
        print("="*50)
        print(f"Average SARI Score: {avg_score:.2f}")
        print(f"\nSaving detailed results to {args.output}")
        
        # Save detailed results
        detailed_results.to_csv(args.output, index=False)
        print(f"Results saved successfully to {args.output}")

    except Exception as e:
        print(f"\nError during benchmark: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())