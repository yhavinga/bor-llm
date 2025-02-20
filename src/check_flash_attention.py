import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def run_model_profile(model_id, attention_impl="flash_attention_2"):
    torch.cuda.reset_peak_memory_stats()  # Reset memory stats before run
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    long_text = "This is a test sentence. " * 2048
    inputs = tokenizer(
        long_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to("cuda")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attention_impl,
        use_cache=False,
    ).cuda()

    # Warm up
    with torch.no_grad():
        _ = model(**inputs)

    # Profile
    with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
        start_time = time.perf_counter()
        output = model(**inputs)
        loss = output.logits[:, -1, :].float().mean()
        loss.backward()
        end_time = time.perf_counter()

    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    
    print(f"\nAttention Implementation: {attention_impl}")
    print(f"Total execution time: {end_time - start_time:.3f} seconds")
    print(f"Peak GPU memory usage: {max_memory:.2f} MB")
    print("\nTop 20 CUDA operations by time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return prof, max_memory

def main():
    model_id = "yhavinga/Bor-1b"
    implementations = ["eager", "sdpa"] # , "flash_attention_2"]
    
    # Prime model load with one run
    print("\nPriming run...")
    run_model_profile(model_id, implementations[0])
    torch.cuda.empty_cache()
    
    # Store results
    results = {impl: {
        "cuda_times": [], 
        "cpu_times": [],
        "memory_usage": []
    } for impl in implementations}
    
    # Run each implementation 3 times
    for impl in implementations:
        print(f"\nTesting {impl}...")
        for run in range(3):
            print(f"Run {run + 1}/3")
            prof, max_memory = run_model_profile(model_id, attention_impl=impl)
            
            # Extract times - safely handle None values
            cuda_time = sum(event.cuda_time_total or 0 for event in prof.function_events)
            cpu_time = sum(event.cpu_time_total or 0 for event in prof.function_events)
            
            results[impl]["cuda_times"].append(cuda_time / 1000)  # Convert to ms
            results[impl]["cpu_times"].append(cpu_time / 1000)  # Convert to ms
            results[impl]["memory_usage"].append(max_memory)  # In MB
            
            torch.cuda.empty_cache()
    
    # Print results table
    print("\nResults:")
    print("-" * 100)
    print(f"{'Implementation':<20} {'CUDA Time (ms)':<25} {'CPU Time (ms)':<25} {'Memory (MB)':<25}")
    print("-" * 100)
    
    for impl in implementations:
        cuda_times = results[impl]["cuda_times"]
        cpu_times = results[impl]["cpu_times"]
        memory_usage = results[impl]["memory_usage"]
        
        cuda_mean = sum(cuda_times) / len(cuda_times)
        cuda_std = (sum((x - cuda_mean) ** 2 for x in cuda_times) / len(cuda_times)) ** 0.5
        
        cpu_mean = sum(cpu_times) / len(cpu_times)
        cpu_std = (sum((x - cpu_mean) ** 2 for x in cpu_times) / len(cpu_times)) ** 0.5
        
        mem_mean = sum(memory_usage) / len(memory_usage)
        mem_std = (sum((x - mem_mean) ** 2 for x in memory_usage) / len(memory_usage)) ** 0.5
        
        print(f"{impl:<20} {cuda_mean:>8.2f} ± {cuda_std:<12.2f} {cpu_mean:>8.2f} ± {cpu_std:<12.2f} {mem_mean:>8.2f} ± {mem_std:<12.2f}")
    print("-" * 100)

if __name__ == "__main__":
    main()
