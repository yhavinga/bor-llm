import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_perplexity import calculate_word_level_perplexity_v2


@pytest.fixture
def model_and_tokenizer():
    model_name = "yhavinga/gpt-neo-125M-dutch"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def test_word_level_perplexity_dutch(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    text = """De Nederlandse taal is rijk aan geschiedenis en cultuur.
Het wordt gesproken door ongeveer 24 miljoen mensen wereldwijd,
voornamelijk in Nederland en België. De taal staat bekend om zijn
lange samengestelde woorden en karakteristieke uitdrukkingen."""

    token_ppl, num_tokens, mean_bits, num_words = calculate_word_level_perplexity_v2(
        model, tokenizer, context_length=512, text=text
    )

    # Basic sanity checks
    assert not torch.isnan(torch.tensor(token_ppl))
    assert not torch.isnan(torch.tensor(mean_bits))
    assert token_ppl > 1.0
    assert num_tokens > 0
    assert num_words > 0
    assert isinstance(num_tokens, int)
    assert isinstance(num_words, int)

    # Model-specific expectations for Dutch text
    assert token_ppl < 1000  # Reasonable upper bound for coherent Dutch text
    assert 0 < mean_bits < 20  # Reasonable range for bits per word in natural language
