from copy import deepcopy
from abc import ABC, abstractmethod

IGNORE_TOKEN_ID = -100


class ConversationTokenizer(ABC):
    def __init__(self, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.system_header = self.get_system_header()
        self.assistant_header = self.get_assistant_header()
        self.user_header = self.get_user_header()
        self.assistant_token_length = len(
            tokenizer.encode(self.assistant_header, add_special_tokens=False)
        )
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    @abstractmethod
    def get_system_header(self):
        """Return the system header specific to this tokenizes chat template."""
        pass

    @abstractmethod
    def get_user_header(self):
        """Return the user header specific to this tokenizes chat template."""
        pass

    @abstractmethod
    def get_assistant_header(self):
        """Return the assistant header specific to this tokenizes chat template."""
        pass

    @abstractmethod
    def format_turn_text(self, turn):
        """Return the formatted turn text specific to this tokenizes chat template."""
        pass

    def tokenize_conversation(self, conversation):
        tokenized_example = deepcopy(
            {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }
        )

        for i, turn in enumerate(conversation):
            role, turn_text = self.format_turn_text(turn)
            ids = self.tokenizer.encode(turn_text, add_special_tokens=False)
            if i == 0 and self.tokenizer.bos_token_id is not None:
                ids = [
                    self.tokenizer.bos_token_id
                ] + ids  # Add BOS token to the first turn
            if role in ["system", "human", "user"]:
                labels = [IGNORE_TOKEN_ID] * len(ids)
            else:
                labels = [IGNORE_TOKEN_ID] * self.assistant_token_length + ids[
                    self.assistant_token_length:
                ]

            total_ids_length = len(ids) + len(tokenized_example["input_ids"])
            if total_ids_length > self.block_size:
                print(
                    f"\033[34mSTOPPING EXAMPLE DUE TO LENGTH {total_ids_length}\033[0m"
                )
                break
            tokenized_example["input_ids"] += ids
            tokenized_example["attention_mask"] += [1] * len(ids)
            tokenized_example["labels"] += labels

        return tokenized_example


class Llama3ConversationTokenizer(ConversationTokenizer):

    def get_system_header(self):
        return "<|start_header_id|>system<|end_header_id|>\n\n"

    def get_user_header(self):
        return "<|start_header_id|>user<|end_header_id|>\n\n"

    def get_assistant_header(self):
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"

    def format_turn_text(self, turn):
        # Format a single text, according to the format bos_token header_start role header_end newline newline content ltrimmed eot_id
        role = turn["from"]
        value = turn["value"]
        if role == "system":
            role_text = self.system_header
        elif role in ["human", "user"]:
            role_text = self.user_header
        elif role in ["gpt", "assistant"]:
            role_text = self.assistant_header
        else:
            raise ValueError(f"Unknown role {role}")
        turn_text = f"{role_text}{value.strip()}<|eot_id|>"
        return role, turn_text



class ZephyrConversationTokenizer(ConversationTokenizer):

    def get_system_header(self):
        return "<|system|>\n"

    def get_user_header(self):
        return "<|user|>\n"

    def get_assistant_header(self):
        return "<|assistant|>\n"

    def format_turn_text(self, turn) -> tuple:
        # Format a single text, according to the format <s><|user|>\n inputs eos <|assistant|>\n answer eos
        role = turn["from"]
        value = turn["value"]
        if role == "system":
            role_text = self.system_header
        elif role in ["human", "user"]:
            role_text = self.user_header
        elif role in ["gpt", "assistant"]:
            role_text = self.assistant_header
        else:
            raise ValueError(f"Unknown role {role}")
        turn_text = f"{role_text}{value}</s>"
        return role, turn_text


class Qwen2ConversationTokenizer(ConversationTokenizer):
    """
    {% for message in messages %}
        {% if loop.first and messages[0]['role'] != 'system' %}
            {{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}
        {% endif %}
        {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
    {% endfor %}
    {% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    """

    def get_system_header(self):
        return f"system\n"

    def get_user_header(self):
        return "user\n"

    def get_assistant_header(self):
        return "assistant\n"

    def format_turn_text(self, turn):
        role = turn["from"]
        value = turn["value"]
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"

        if role == "system":
            role_text = self.get_system_header()
        elif role in ["human", "user"]:
            role_text = self.get_user_header()
        elif role in ["gpt", "assistant"]:
            role_text = self.get_assistant_header()
        else:
            raise ValueError(f"Unknown role {role}")

        turn_text = f"{im_start}{role_text}{value.strip()}{im_end}\n"
        return role, turn_text


class ConversationTokenizerFactory:
    @staticmethod
    def create(tokenizer, block_size):
        if tokenizer.vocab_size == 32000 and tokenizer.bos_token_id == 1:
            return ZephyrConversationTokenizer(tokenizer, block_size)
        elif tokenizer.vocab_size == 128000 and tokenizer.bos_token_id == 128000:
            return Llama3ConversationTokenizer(tokenizer, block_size)
        elif tokenizer.vocab_size == 151643 and tokenizer.bos_token_id is None:  # Assuming specific ids for Qwen2
            return Qwen2ConversationTokenizer(tokenizer, block_size)
        elif tokenizer.vocab_size == 131072 and tokenizer.bos_token_id == 1:  # mistral small
            return ZephyrConversationTokenizer(tokenizer, block_size)
        else:
            raise ValueError(f"Cannot choose ConversationTokenizer for tokenizer {tokenizer}")