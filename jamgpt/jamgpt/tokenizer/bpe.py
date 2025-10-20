import os
import copy
from functools import lru_cache

import pickle
import rustbpe
import tiktoken
import torch
from jamgpt.common import get_base_dir

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>",  # user messages
    "<|user_end|>",
    "<|assistant_start|>",  # assistant messages
    "<|assistant_end|>",
    "<|python_start|>",  # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>",  # python REPL outputs back to assistant
    "<|output_end|>",
]

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(self, encoding: tiktoken.Encoding, bos_token: str = "<|bos|>"):
        self.encoding = encoding
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, iterator: iter, vocab_size: int):
        bpe = rustbpe.Tokenizer()
        bpe.train_from_iterator(
            iterator, vocab_size - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN
        )
        mergeable_ranks = {bytes(k): v for k, v in bpe.get_mergeable_ranks()}

        encoding = tiktoken.Encoding(
            name="custom-bpe",
            pat_str=bpe.get_pattern(),
            mergeable_ranks=mergeable_ranks,
            special_tokens={
                token: len(mergeable_ranks) + i
                for i, token in enumerate(SPECIAL_TOKENS)
            },
        )
        return cls(encoding, bos_token="<|bos|>")

    @classmethod
    def from_directory(cls, dir_path: str):
        with open(os.path.join(dir_path, "tokenizer.pkl"), "rb") as f:
            encoding = pickle.load(f)
        return cls(encoding, bos_token="<|bos|>")

    @classmethod
    def from_pretrained(cls, model_name: str):
        encoding = tiktoken.get_encoding(model_name)
        return cls(encoding, bos_token="<|endoftext|>")

    def get_vocab_size(self):
        return self.encoding.n_vocab

    def get_special_tokens(self):
        return self.encoding.special_tokens_set

    def id_to_token(self, token_id: int) -> str:
        return self.encoding.decode([token_id])

    @lru_cache(maxsize=32)
    def encode_special(self, text: str) -> int:
        return self.encoding.encode_single_token(text)

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def encode_batch(
        self, text: list[str], prepend=None, append=None, num_threads=8
    ) -> list[list[int]]:
        token_batches = self.encoding.encode_ordinary_batch(
            text, num_threads=num_threads
        )
        prepend_id = (
            prepend
            if isinstance(prepend, int)
            else self.encode_special(prepend) if prepend is not None else None
        )
        append_id = (
            append
            if isinstance(append, int)
            else self.encode_special(append) if append is not None else None
        )

        if prepend_id is not None:
            token_batches = [[prepend_id] + seq for seq in token_batches]
        if append_id is not None:
            token_batches = [seq + [append_id] for seq in token_batches]

        return token_batches

    def __call__(self, text: str) -> list[int]:
        return self.encode_batch([text])

    def decode(self, token_ids: list[int]) -> str:
        return self.encoding.decode(token_ids)

    def save_to_directory(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "tokenizer.pkl"), "wb") as f:
            pickle.dump(self.encoding, f)
        print(f"Tokenizer saved to {dir_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """
        # ids, masks that we will return and a helper function to help build them up.
        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation)  # avoid mutating the original
            messages = conversation["messages"]
            assert (
                messages[1]["role"] == "user"
            ), "System message must be followed by a user message"
            messages[1]["content"] = (
                messages[0]["content"] + "\n\n" + messages[1]["content"]
            )
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special(
            "<|user_start|>"
        ), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special(
            "<|assistant_start|>"
        ), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special(
            "<|python_start|>"
        ), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special(
            "<|output_start|>"
        ), self.encode_special("<|output_end|>")

        # now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # some sanity checking here around assumptions, to prevent footguns
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert (
                message["role"] == must_be_from
            ), f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(
                    content, str
                ), "User messages are simply expected to be strings"
                value_ids = self.encode_batch([content])[0]
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = self.encode_batch([content])[0]
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode_batch([part["text"]])[0]
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation)  # avoid mutating the original
        messages = conversation["messages"]
        assert (
            messages[-1]["role"] == "assistant"
        ), "Last message must be from the Assistant"
        messages.pop()  # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids


def get_tokenizer():

    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    print(f"Loading tokenizer from {tokenizer_dir}...")
    return BPETokenizer.from_directory(tokenizer_dir)


def get_token_bytes(device="cpu"):
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(
        token_bytes_path
    ), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
