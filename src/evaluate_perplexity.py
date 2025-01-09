import numpy as np
import torch
from datasets import load_dataset
from more_itertools import chunked
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import dispatch_model
import os
import time
import psutil
import torch.cuda
import json

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4

def calculate_word_level_perplexity(model, tokenizer, context_length,text, device='cuda'):
    """
    Perplexity = exp(CE) = exp(-1/N ∑ log P(w_i))
    
    For subword tokens within a word:
    P(word) ≈ 1/k ∑ log P(t_i|context) 
    where k is number of subword tokens in the word

    Computes word-level perplexity by:

      1) Splitting the input text on whitespace to define word boundaries.
      2) Tokenizing the entire text in one pass to get subword tokens and offsets.
      3) Computing subword log-probabilities from the model outputs.
      4) Summing subword log-probabilities for each whitespace-based word.
      5) Computing the exponential of the average negative log-prob across words.
    """
    # Split the text into "words" by whitespace.
    # This is our tokenizer-independent boundary.
    words = text.strip().split()
    if not words:
        return float('nan')
    
    # Tokenize while returning offsets so we can map subword tokens back to the original text.
    encoded = tokenizer(
        text,
        max_length=context_length,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    
    # Move all tensors to device
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    offset_mapping = encoded.offset_mapping.to(device)
    offsets = encoded.offset_mapping[0].tolist()
    
    # Forward pass through the model.
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits.to(device)  # shape: [batch_size=1, seq_length, vocab_size]
    
    # Convert logits to log-probabilities of the *next* token.
    # We shift by one so token[i+1] is conditioned on token[i].
    # We'll discard the last position's log-prob because there's no next token for it.
    next_token_logits = logits[:, :-1, :]
    shifted_input_ids = input_ids[:, 1:].to(device)
    
    # log P(t_i+1|context) = log_softmax(logits_i)
    log_probs = torch.log_softmax(next_token_logits, dim=-1)  # [1, seq_length-1, vocab_size]
    seq_token_log_probs = log_probs[0, torch.arange(shifted_input_ids.size(1)).to(device), shifted_input_ids[0]]
    # seq_token_log_probs[i] is log P( token[i+1] | prefix up to token[i] )
    
    # Map each token to its corresponding "word" based on offset spans
    # and sum log-probs for all tokens in a word.
    # Each element in offsets is (start_char, end_char) in the original text.
    # We'll track which word each token belongs to by comparing offsets.
    word_log_sums = []
    current_word_idx = 0
    current_word_offset_start = 0
    word_char_positions = []
    
    # Precompute the char spans of each whitespace-based word
    # so we know which offsets fall inside which word.
    # We'll do a cumulative approach so we get a (start, end) for each word in text.
    char_index = 0
    for w in words:
        start_pos = char_index
        end_pos = char_index + len(w)
        word_char_positions.append((start_pos, end_pos))
        # +1 for the space or newline after the word
        char_index = end_pos + 1
    
    # Accumulate log probabilities for each word
    # word_sums[i] = ∑ log P(t_j|context) for all tokens j in word i
    # word_counts[i] = number of tokens in word i, for averaging
    word_sums = [0.0] * len(words)
    word_counts = [0] * len(words)  # how many subwords belong to each word
    
    # The last token in seq_token_log_probs doesn't have a next token offset, so we iterate carefully
    # offsets has length = seq_length. seq_token_log_probs has length = seq_length - 1.
    # We'll map offsets[:seq_length-1] to seq_token_log_probs.
    for i, (start_char, end_char) in enumerate(offsets[:-1]):
        # Find which word index corresponds to this subword offset.
        # We'll keep an index "word_idx" that walks forward in word_char_positions.
        # We assume offsets come in ascending order.
        # We find the word whose [start_pos, end_pos) covers the center of this token's position.
        
        token_center = (start_char + end_char) // 2
        
        # Move current_word_idx forward if necessary
        while current_word_idx < len(word_char_positions):
            w_start, w_end = word_char_positions[current_word_idx]
            if token_center >= w_start and token_center < w_end:
                # This subword belongs to the current word
                word_sums[current_word_idx] += seq_token_log_probs[i].item()
                word_counts[current_word_idx] += 1
                break
            elif token_center >= w_end:
                current_word_idx += 1
            else:
                break
    
    # Now compute average log-probs across words, ignoring words that had no tokens
    valid_word_log_probs = []
    for i in range(len(words)):
        if word_counts[i] > 0:
            # Average subword log-prob for this word
            avg_word_log_prob = word_sums[i] / word_counts[i]
            valid_word_log_probs.append(avg_word_log_prob)
    
    if not valid_word_log_probs:
        return float('nan')
    
    # Word-level perplexity: exponent of the negative average log-prob across words
    mean_log_prob = torch.tensor(sum(valid_word_log_probs) / len(valid_word_log_probs), device=device)
    
    if torch.isinf(mean_log_prob) or torch.isnan(mean_log_prob):
        return float('nan')
    
    word_level_ppl = torch.exp(-mean_log_prob.clone().detach()).item()
    
    if word_level_ppl > 1e6 or word_level_ppl < 1:
        print(f"Warning: Unusual perplexity value: {word_level_ppl}")
    
    # Add stability checks
    if torch.isinf(mean_log_prob) or torch.isnan(mean_log_prob):
        return float('nan')
    
    # Final perplexity calculation:
    # CE = -1/N ∑ valid_word_log_probs
    # PPL = exp(CE) = exp(-mean(valid_word_log_probs))
    word_level_ppl = torch.exp(-torch.tensor(mean_log_prob)).item()
    
    # Sanity check on final perplexity
    if word_level_ppl > 1e6 or word_level_ppl < 1:
        print(f"Warning: Unusual perplexity value: {word_level_ppl}")
    
    return word_level_ppl


def calculate_cumulative_word_bits(model, tokenizer, context_length, text, device='cuda'):
    """
    Calculates Cumulative Word-Level Bits (CWB).
    
    For each word j:
    bits(word_j) = -log_2(P(word_j)) 
                 = -log_2(∏_{k=1}^{M_j} P(t_{j,k}))
                 = ∑_{k=1}^{M_j} -log_2(P(t_{j,k}))
    
    Final metric:
    CWB = (1/W) ∑_{j=1}^W bits(word_j)
    
    This metric:
    1. Naturally penalizes inefficient tokenization (more tokens = more bits)
    2. Returns interpretable values (average bits needed per word)
    3. Avoids numerical instability from exponentiation
    """
    loss_fct = CrossEntropyLoss(reduction="none")

    words = text.strip().split()
    if not words:
        return float('nan')
    
    encoded = tokenizer(
        text,
        max_length=context_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False
    )

    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()
    
    # Token-level bits (negative log2 probabilities)
    token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    token_loss = token_loss.view(shift_labels.size()) * shift_attention_mask
    
    # Convert from natural log to log2
    token_bits = token_loss * torch.log2(torch.exp(torch.tensor(1.0)))
    
    # Map tokens to words and sum bits
    word_bits = torch.zeros(len(words), device=device)
    current_word_index = 0
    char_index = 0

    token_word_map = []
    for i, token_id in enumerate(encoded.input_ids[0]):
        token = tokenizer.convert_ids_to_tokens(token_id.item())
        token_len = len(token)
        if token_len == 0:
            token_word_map.append(-1)
            continue

        while current_word_index < len(words):
            word = words[current_word_index]
            word_len = len(word)
            if char_index + token_len <= char_index + word_len:
                token_word_map.append(current_word_index)
                if char_index + token_len >= char_index + word_len:
                    current_word_index += 1
                break
            else:
                char_index += (word_len + 1)
                current_word_index += 1

    # Accumulate bits per word
    for i, word_idx in enumerate(token_word_map[:token_bits.size(1)]):
        if word_idx >= 0 and word_idx < len(word_bits):
            word_bits[word_idx] += token_bits[0, i]

    # Filter out words with no tokens
    valid_indices = (word_bits != 0)
    word_bits = word_bits[valid_indices]
    
    if len(word_bits) == 0:
        return float('nan')
    
    # Calculate mean bits per word
    mean_bits = word_bits.mean().item()
    
    return mean_bits


def calculate_token_level_perplexity(model, tokenizer, texts, context_length, stride=1024, device='cuda'):
    """Calculate token-level perplexity and performance metrics for a batch of texts."""
    loss_fct = CrossEntropyLoss(reduction="none")
    token_ppls = []
    total_tokens = 0
    total_loss = 0
    
    encodings = tokenizer(texts, add_special_tokens=False, padding=True,
                         truncation=True, max_length=context_length,
                         return_tensors='pt', return_attention_mask=True,
                         return_overflowing_tokens=True, stride=stride).to(device)
    
    performance_metrics = {
        'total_tokens': 0,
        'inference_time': 0,
        'peak_memory': 0,
        'tokens_per_second': 0
    }
    
    start_time = time.time()
    initial_memory = torch.cuda.memory_allocated() if device == 'cuda' else psutil.Process().memory_info().rss

    for i in range(0, encodings.input_ids.size(0), BATCH_SIZE):
        batch_start = time.time()
        batch_input_ids = encodings.input_ids[i:i+BATCH_SIZE]
        batch_attention_mask = encodings.attention_mask[i:i+BATCH_SIZE]
        target_ids = batch_input_ids.clone()

        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        shift_attention_mask = batch_attention_mask[..., 1:].contiguous()

        loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
        
        if torch.isnan(loss).any():
            print(f"WARNING: NaN losses detected in batch {i//BATCH_SIZE}")
        
        loss = loss * shift_attention_mask
        
        seq_lengths = shift_attention_mask.sum(1)
        seq_loss = loss.sum(1) / seq_lengths
        seq_perplexity = torch.exp(seq_loss)
        
        total_tokens += seq_lengths.sum().item()
        total_loss += loss.sum().item()
        
        # Track and report high perplexities
        batch_max = seq_perplexity.max().item()
        if batch_max > 100:
            high_ppl_idx = torch.where(seq_perplexity > 100)[0]
            for idx in high_ppl_idx:
                print(f"\nHigh perplexity ({seq_perplexity[idx]:.2f}) detected:")
                print(f"Sequence length: {seq_lengths[idx]}")
                print(f"Average token loss: {seq_loss[idx]:.4f}")
        
        token_ppls.extend(seq_perplexity.tolist())

        # Update performance metrics
        batch_tokens = seq_lengths.sum().item()
        batch_time = time.time() - batch_start
        current_memory = torch.cuda.memory_allocated() if device == 'cuda' else psutil.Process().memory_info().rss
        
        performance_metrics['total_tokens'] += batch_tokens
        performance_metrics['inference_time'] += batch_time
        performance_metrics['peak_memory'] = max(performance_metrics['peak_memory'], current_memory - initial_memory)
        
        del outputs
        torch.cuda.empty_cache()

    performance_metrics['tokens_per_second'] = (
        performance_metrics['total_tokens'] / performance_metrics['inference_time']
    )
    
    return token_ppls, total_tokens, total_loss, performance_metrics


def calculate_perplexities(model, tokenizer, dataset, context_length, stride=None):
    """Calculate both token and word-level perplexity."""
    if stride is None or stride >= context_length:
        stride = context_length // 2

    token_ppls = []
    word_ppls = []
    cwb_ppls = []
    total_tokens = 0
    total_loss = 0
    extremes = {'min_ppl': float('inf'), 'max_ppl': 0}

    total_batches = len(dataset) // BATCH_SIZE
    pbar = tqdm(chunked(dataset, BATCH_SIZE), total=total_batches, 
                desc="Processing batches")
    for batch_idx, rows in enumerate(pbar):
        texts = [row['text'] for row in rows]
        
        # Calculate metrics for each text in batch
        for text in texts:
            word_ppl = calculate_word_level_perplexity(model, tokenizer, context_length, text, DEVICE)
            if not np.isnan(word_ppl):
                word_ppls.append(word_ppl)
            
            cwb_ppl = calculate_cumulative_word_bits(model, tokenizer, context_length, text, DEVICE)
            if not np.isnan(cwb_ppl):
                cwb_ppls.append(cwb_ppl)
        
        # Calculate token-level perplexity for the batch
        batch_token_ppls, batch_tokens, batch_loss, batch_perf = calculate_token_level_perplexity(
            model, tokenizer, texts, context_length, stride, DEVICE)
        
        token_ppls.extend(batch_token_ppls)
        total_tokens += batch_tokens
        total_loss += batch_loss
        
        # Update extremes
        if batch_token_ppls:
            batch_min = min(batch_token_ppls)
            batch_max = max(batch_token_ppls)
            extremes['min_ppl'] = min(extremes['min_ppl'], batch_min)
            extremes['max_ppl'] = max(extremes['max_ppl'], batch_max)

    # Final statistics
    print("\nEvaluation Statistics:")
    
    # Token-level metrics
    valid_token_ppls = [x for x in token_ppls if not (np.isnan(x) or np.isinf(x))]
    if valid_token_ppls:
        print("\nToken-level metrics:")
        print(f"Total tokens processed: {total_tokens:,}")
        print(f"Average loss per token: {total_loss/total_tokens:.4f}")
        print(f"Token perplexity range: {min(valid_token_ppls):.2f} - {max(valid_token_ppls):.2f}")
        print(f"Token perplexity distribution:")
        token_percentiles = np.percentile(valid_token_ppls, [25, 50, 75])
        print(f"25th percentile: {token_percentiles[0]:.2f}")
        print(f"Median: {token_percentiles[1]:.2f}")
        print(f"75th percentile: {token_percentiles[2]:.2f}")
    else:
        print("\nNo valid token perplexities found")
    
    # Word-level metrics
    valid_word_ppls = [x for x in word_ppls if not (np.isnan(x) or np.isinf(x))]
    if valid_word_ppls:
        print("\nWord-level metrics:")
        word_percentiles = np.percentile(valid_word_ppls, [25, 50, 75])
        print(f"Word perplexity distribution:")
        print(f"25th percentile: {word_percentiles[0]:.2f}")
        print(f"Median: {word_percentiles[1]:.2f}")
        print(f"75th percentile: {word_percentiles[2]:.2f}")
        print(f"Mean word perplexity: {np.mean(valid_word_ppls):.2f}")
    else:
        print("\nNo valid word perplexities found")

    # CWB-level metrics
    valid_cwb_ppls = [x for x in cwb_ppls if not (np.isnan(x) or np.isinf(x))]
    if valid_cwb_ppls:
        print("\nCumulative Word Bits metrics:")
        cwb_percentiles = np.percentile(valid_cwb_ppls, [25, 50, 75])
        print(f"Cumulative Word Bits distribution:")
        print(f"25th percentile: {cwb_percentiles[0]:.2f}")
        print(f"Median: {cwb_percentiles[1]:.2f}")
        print(f"75th percentile: {cwb_percentiles[2]:.2f}")
        print(f"Mean Cumulative Word Bits: {np.mean(valid_cwb_ppls):.2f}")
    else:
        print("\nNo valid Cumulative Word Bits found")

    # Add performance metrics to final statistics
    print("\nPerformance Metrics:")
    print(f"Average throughput: {batch_perf['total_tokens']/batch_perf['inference_time']:.2f} tokens/second")
    print(f"Peak memory usage: {batch_perf['peak_memory']/1024**2:.2f} MB")
    print(f"Memory efficiency: {batch_perf['total_tokens']/(batch_perf['peak_memory']/1024**2):.2f} tokens/MB")

    return {
        'token_perplexities': valid_token_ppls,
        'mean_token_perplexity': np.mean(valid_token_ppls) if valid_token_ppls else float('nan'),
        'word_perplexities': valid_word_ppls,
        'mean_word_perplexity': np.mean(valid_word_ppls) if valid_word_ppls else float('nan'),
        'cumulative_word_bits': valid_cwb_ppls,
        'mean_cumulative_word_bits': np.mean(valid_cwb_ppls) if valid_cwb_ppls else float('nan'),
        'performance': {
            'throughput': batch_perf['total_tokens']/batch_perf['inference_time'],
            'peak_memory_mb': batch_perf['peak_memory']/1024**2,
            'tokens_per_mb': batch_perf['total_tokens']/(batch_perf['peak_memory']/1024**2)
        }
    }

def main():
    print("Loading datasets...")
    datasets = {
        'fineweb_edu_4': load_dataset('yhavinga/fineweb_edu_score_gt_4_test', split='test', num_proc=16),
        'mc4_nl': load_dataset('yhavinga/mc4_nl_cleaned', 'tiny', split='validation[:1000]', num_proc=16),
        'culturax_nl': load_dataset('yhavinga/culturax_dutch_test', split='train', num_proc=16),
    }

    # select max 10 batches for quick testing
    datasets['mc4_nl'] = datasets['mc4_nl'].select(range(40))
    datasets['culturax_nl'] = datasets['culturax_nl'].select(range(40))
    datasets['fineweb_edu_4'] = datasets['fineweb_edu_4'].select(range(40))

    models = {
        # 'gpt-neo-125M-dutch': {"path": "yhavinga/gpt-neo-125M-dutch", "context_length": 512},
        # 'gpt2-medium-dutch': {"path": "yhavinga/gpt2-medium-dutch", "context_length": 512},
        # 'gpt2-large-dutch': {"path": "yhavinga/gpt2-large-dutch", "context_length": 512},
        'Bor-1B': {"path": "yhavinga/Bor-1B", "context_length": 4096},
        # 'gpt-neo-1.3B-dutch': {"path": "yhavinga/gpt-neo-1.3B-dutch", "context_length": 512},
        # 'Llama-3.2-1B': {"path": "meta-llama/Llama-3.2-1B", "context_length": 4096},
        # 'Phi-3.5-mini-instruct': {"path": "microsoft/Phi-3.5-mini-instruct", "context_length": 4096},
        'Fietje-2': {"path": "BramVanroy/fietje-2", "context_length": 2048},
    }

    model_kwargs = {
        'device_map': "auto",
        'torch_dtype': torch.bfloat16,
        'trust_remote_code': True,
        'use_flash_attention_2': False
    }

    for model_name, model_info in models.items():
        # Enable flash attention for non-GPT models
        model_kwargs['use_flash_attention_2'] = 'gpt' not in model_name.lower()
        
        safe_model_name = model_name.replace('/', '_')
        output_file = f'{safe_model_name}_evaluation_results.json'
        
        if os.path.exists(output_file):
            print(f"\nSkipping {model_name} - results already exist in {output_file}")
            continue

        print(f"\nLoading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_info["path"], **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_info["path"])
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        
        model.eval()

        results = {}
        for dataset_name, dataset in datasets.items():
            print(f"\nEvaluating {model_name} on {dataset_name}...")
            results[dataset_name] = calculate_perplexities(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                context_length=model_info["context_length"]
            )
            
            # Clean up GPU memory after each dataset
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Save results for this model to a JSON file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        # More thorough cleanup between models
        del model
        del tokenizer
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()

if __name__ == '__main__':
    main() 