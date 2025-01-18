import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_perplexity import calculate_entropy_metrics
import evaluate


@pytest.fixture
def model_and_tokenizer():
    model_name = "yhavinga/gpt-neo-125M-dutch"
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float32
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def test_word_level_perplexity_dutch(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    perplexity_metric = evaluate.load("perplexity")
    bpw_metric = evaluate.load("bits_per_word.py", module_type="metric")

    text = """Wil je een beukennootje?"""
    #     De Nederlandse taal is rijk aan geschiedenis en cultuur.
    # Het wordt gesproken door ongeveer 24 miljoen mensen wereldwijd,
    # voornamelijk in Nederland en België. De taal staat bekend om zijn
    # lange samengestelde woorden en karakteristieke uitdrukkingen."""

    # Calculate custom metrics
    token_ppl, num_tokens, bits_per_word, num_words = calculate_entropy_metrics(
        model,
        tokenizer,
        context_length=512,
        text=text,
        device="cuda" if torch.cuda.is_available() else "cpu",
        add_start_token=False,
    )

    # Calculate HF perplexity
    evaluate_perplexity = perplexity_metric.compute(
        predictions=[text], model_id=model.config._name_or_path, add_start_token=False
    )
    evaluate_ppl = evaluate_perplexity["perplexities"][0]

    bpw_results = bpw_metric.compute(
        predictions=[text],
        model_id=model.config._name_or_path,
        add_start_token=False,
        # batch_size=1
    )
    evaluate_bpw = bpw_results["bits_per_word_scores"][0]

    # Basic sanity checks
    assert not torch.isnan(torch.tensor(token_ppl))
    assert not torch.isnan(torch.tensor(bits_per_word))
    assert not torch.isnan(torch.tensor(evaluate_bpw))
    assert token_ppl > 1.0
    assert num_tokens > 0
    assert num_words > 0
    assert isinstance(num_tokens, int)
    assert isinstance(num_words, int)

    # Model-specific expectations for Dutch text
    assert token_ppl < 1000  # Reasonable upper bound for coherent Dutch text
    assert 0 < bits_per_word < 20  # Reasonable range for bits per word
    assert 0 < evaluate_bpw < 20  # Same range for HF implementation

    # Compare metrics - allow for some numerical differences
    assert (
        abs(token_ppl - evaluate_ppl) < token_ppl * 0.001
    )  # Within 0.1% relative difference
    assert (
        abs(bits_per_word - evaluate_bpw) < bits_per_word * 0.001
    )  # Within 0.1% relative difference
