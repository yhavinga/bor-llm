# Copyright 2024 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bits Per Word (BPW) Metric."""

import datasets
import evaluate
import numpy as np
import torch
from evaluate import logging
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

_CITATION = """\
@article{chip2019evaluation,
    author = {Huyen, Chip},
    title = {Evaluation Metrics for Language Modeling},
    journal = {The Gradient},
    year = {2019},
    url = {https://thegradient.pub/understanding-evaluation-metrics-for-language-models/}
}
"""

_DESCRIPTION = """\
Bits Per Word (BPW) measures the average number of bits needed to encode each word in a text sequence
using a language model. It is calculated by converting the negative log-likelihood to base 2 (bits) 
and normalizing by the number of words. Lower BPW indicates better compression/prediction of the text.
"""

_KWARGS_DESCRIPTION = """\
Args:
    model_id (str): model used for calculating BPW
        NOTE: BPW can only be calculated for causal language models.
    predictions (list of str): input text, each separate text snippet is one list entry.
    batch_size (int): the batch size to run texts through the model. Defaults to 16.
    add_start_token (bool): whether to add the start token to the texts. Defaults to True.
    device (str): device to run on, defaults to 'cuda' when available

Returns:
    bits_per_word: dictionary containing the BPW scores for each input text and the mean BPW.
    If a text exceeds the model's max length, it is truncated.

Examples:
    >>> bpw = evaluate.load("bits_per_word", module_type="metric")
    >>> texts = ["The cat sat on the mat.", "How are you today?"]
    >>> results = bpw.compute(model_id='gpt2', predictions=texts)
    >>> print(results.keys())
    dict_keys(['bits_per_word_scores', 'mean_bits_per_word'])
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BitsPerWord(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                }
            ),
            reference_urls=[],
        )

    def _compute(
        self,
        predictions,
        model_id,
        batch_size: int = 16,
        add_start_token: bool = True,
        device=None,
        max_length=None,
    ):
        if device is not None:
            assert device in [
                "gpu",
                "cpu",
                "cuda",
            ], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(
                tokenizer.special_tokens_map_extended.values()
            )
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 1)
            ), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        bpw_scores = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)
                ).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [
                        torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(
                            device
                        ),
                        attn_mask,
                    ],
                    dim=1,
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            # For causal language modeling, at each position we predict the next token
            # So we need to align the input and target sequences:
            #   Input:  [BOS, t1, t2, t3]  -> predict -> [t1, t2, t3, EOS]
            #   Therefore:
            #   Logits: [BOS, t1, t2, t3] -> remove last  -> [BOS, t1, t2]    (input)
            #   Labels: [BOS, t1, t2, t3] -> remove first -> [t1,  t2, t3]    (target)

            shift_logits = out_logits[
                ..., :-1, :
            ].contiguous()  # Remove last position  [batch_size, seq_len-1, vocab_size]
            targets = labels[
                ..., 1:
            ].contiguous()  # Remove first position [batch_size, seq_len-1]
            shift_attention_mask_batch = attn_mask[
                ..., 1:
            ].contiguous()  # Match target shape [batch_size, seq_len-1]

            # transpose(1,2) converts from [batch_size, seq_len-1, vocab_size] to [batch_size, vocab_size, seq_len-1]
            # This is required because CrossEntropyLoss expects logits in shape (N, C, L) where:
            # N = batch size, C = number of classes (vocab_size), L = sequence length
            token_losses = (
                loss_fct(
                    shift_logits.transpose(1, 2), targets
                )  # [batch_size, seq_len-1]
                * shift_attention_mask_batch
            )
            # sum(1) sums along sequence length dimension, giving loss per sequence in batch
            neg_log_likelihood = token_losses.sum(1)  # [batch_size]

            # Convert to bits and normalize by number of words
            neg_log_likelihood_bits = neg_log_likelihood / torch.log(torch.tensor(2.0))

            # Count words for each sequence in batch
            word_counts = torch.tensor(
                [
                    len(
                        tokenizer.decode(
                            seq[mask.bool()], skip_special_tokens=True
                        ).split()
                    )
                    for seq, mask in zip(encoded_batch, attn_mask)
                ]
            ).to(device)

            bits_per_word_batch = neg_log_likelihood_bits / word_counts
            bpw_scores += bits_per_word_batch.tolist()

        return {
            "bits_per_word_scores": bpw_scores,
            "mean_bits_per_word": np.mean(bpw_scores),
        }
