"""
Dataset utilities for LLM training.

Includes:
- PreTokenizedDataset for pre-tokenized data loading
- TinyShakespeareDataset for small-scale testing
- Utility functions for device selection and masking
"""

from __future__ import annotations
from typing import Any
from collections.abc import Iterable
from dataclasses import dataclass
from transformers import AutoTokenizer
from itertools import cycle
from datasets import load_from_disk

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from pathlib import Path

import os
import requests
import random

PATH = Path("./llamadata")


def get_tokenizer(self, tokenizer: AutoTokenizer | str = None):
    self.tokenizer = tokenizer
    if tokenizer is None:
        tokenizer = "NeelNanda/gpt-neox-tokenizer-digits"
    if isinstance(tokenizer, str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, cache_dir=self.cache_dir, use_fast=True
        )


class PreTokenizedDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str = "JeanKaddour/minipile",
        tokenizer: AutoTokenizer = None,
        split: str = "train",
        path: Path = PATH,
        microbatch_size: int = 32,
        min_length: int = 512,
        max_length: int = 2048,
        cache_dir=None,
    ):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.length = len(self.tokenizer)
        self.PAD = tokenizer.pad_token_id

        self.microbatch_size = microbatch_size
        self.vocab_size = len(tokenizer)
        self.min_length = min_length
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name

        if split == "train":
            self.ds = load_from_disk("./data/train")
        else:
            self.ds = load_from_disk("./data/val")
        self.toks_cycle = cycle(self.ds)

    def __iter__(self) -> torch.Tensor:
        while True:
            x = next(self.toks_cycle)["input_ids"]
            x = torch.tensor(x, dtype=torch.long)
            x = F.pad(x, (0, self.max_length - x.shape[0]), "constant", value=self.PAD)
            yield x[:-1][: self.max_length], x[1:][: self.max_length]


class TinyShakespeareDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer = None,
        path: Path = PATH,
        min_length: int = 512,
        max_length: int = 512,
        cache_dir=None,
    ):
        assert tokenizer is not None, "must pass tokenizer"
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.length = len(self.tokenizer)
        self.PAD = tokenizer.pad_token_id

        self.vocab_size = len(tokenizer)
        self.min_length = min_length
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.dataset_name = "tinyshakespeare"

        self.data_path = path.joinpath(self.dataset_name + ".txt")

        try:
            with open(self.data_path) as f:
                self.data = torch.Tensor(tokenizer.encode(f.read())).long()
        except Exception:
            self.download_data()
            with open(self.data_path) as f:
                self.data = torch.Tensor(tokenizer.encode(f.read())).long()

        self.length = len(self.data)

    def __iter__(self) -> torch.Tensor:
        while True:
            idx = random.randint(0, (self.length - self.max_length - 1))
            x = self.data[idx : idx + self.max_length + 1]
            yield x[:-1][: self.max_length], x[1:][: self.max_length]

    def download_data(self):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        if response.status_code == 200:
            with open(self.data_path, "w", encoding="utf-8") as file:
                file.write(response.text)
        else:
            raise Exception(f"Failed to download data. Status code: {response.status_code}")


def auto_accelerator(device: str | None = None) -> torch.device:
    """
    Automatically selects and returns a torch device.

    If a device is specified, it validates and returns the specified device.
    If no device is specified, it checks for available devices in the order
    of CUDA, MPS (Apple Silicon GPUs), and defaults to CPU if none are available.

    Args:
        device: The name of the device to use. Can be 'cpu', 'cuda', 'mps', or None.

    Returns:
        torch.device: The selected torch device.

    Raises:
        AssertionError: If the device passed is not None, 'cpu', 'cuda', or 'mps'.
    """
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    assert device is None, "Pass Valid Device"
    accelerator = "cpu"
    if torch.cuda.is_available():
        accelerator = "cuda"
    if torch.backends.mps.is_built():
        accelerator = "mps"
    return torch.device(accelerator)


def build_mask(seq_len, sliding_window_attention=False, window_size=1):
    """Build causal attention mask with optional sliding window."""
    mask = torch.full((seq_len, seq_len), float("-inf"))

    assert window_size != 0, "window_size cannot be 0"
    if not sliding_window_attention:
        window_size = seq_len

    row_indices = torch.arange(seq_len).unsqueeze(-1)
    col_indices = torch.arange(seq_len)
    distance = row_indices - col_indices

    mask[(distance >= 0) & (distance <= (window_size - 1))] = 0

    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


@dataclass
class BetterCycle:
    """
    A data class that implements a better cycle iterator over any iterable.
    It cycles through the iterable indefinitely and tracks cycle count.

    Attributes:
        iterable: The iterable to cycle through.
        idx: The current cycle index (how many times the iterable has been cycled through).
        _iterator: The iterator generated from the iterable.
    """

    iterable: Iterable
    idx: int = 0
    _iterator: Iterable = None

    def __iter__(self) -> BetterCycle:
        return self

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.iterable)

        try:
            return next(self._iterator)
        except StopIteration:
            self.idx += 1
            self._iterator = iter(self.iterable)
            return next(self._iterator)
