import os
from rich import print
from pathlib import Path
from transformers import AutoTokenizer
import datasets
from datasets import load_dataset, DownloadMode

class DatasetPreprocessor:
    def __init__(
        self,
        dataset_name: str = "JeanKaddour/minipile",
        tokenizer_name: str = "NeelNanda/gpt-neox-tokenizer-digits",
        min_length: int = 512,
        max_length: int = 2049,
        splits: list[str] | None = None,
        revision: str | None = None,
        num_proc: int | None = None,
        output_dir: Path = Path("./data"),
        hf_cache: Path = Path("./hf_cache"), 
    ):
        if splits is None:
            splits = ["train", "test"]
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.min_length = min_length
        self.max_length = max_length
        self.splits = splits
        self.output_dir = output_dir
        self.hf_cache = hf_cache
        self.revision = revision

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.padding_side = "right"
        self.PAD = self.tokenizer.pad_token_id
        self.length = self.tokenizer.vocab_size

        self.num_proc = num_proc if num_proc else max(os.cpu_count() - 1, 1)

    def load_and_preprocess_dataset(self, split, remove_columns=None):
        if remove_columns is None:
            remove_columns = ["text"]
        dataset = load_dataset(
            self.dataset_name,
            split=split,
            download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
            revision=self.revision,
            cache_dir=self.hf_cache,
        )

        tokenized = (
            dataset.map(
                self.tokenize_fn,
                batched=True,
                remove_columns=remove_columns,
                num_proc=self.num_proc,
            )
            .shuffle(seed=31415)
            .filter(self.filter_fn, num_proc=self.num_proc)
        )

        return tokenized

    def tokenize_fn(self, x):
        return self.tokenizer(x["text"], max_length=self.max_length, truncation=True)

    def filter_fn(self, x):
        return len(x["input_ids"]) >= self.min_length

    def save_pre_tokenized_dataset(self, dataset, split):
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        fpath = "tinystories"
        print(self.output_dir.joinpath(fpath).joinpath(split))
        if split == "train":
            out = "./data/train"
            dataset.save_to_disk(out)
        else:
            out = "./data/val"
            dataset.save_to_disk(out)

        print(f"Dataset saved to {self.output_dir}")

    def process_and_save(self, remove_columns=None):
        if remove_columns is None:
            remove_columns = ["text"]
        for split in self.splits:
            tokenized_dataset = self.load_and_preprocess_dataset(
                split, remove_columns=remove_columns
            )
            self.save_pre_tokenized_dataset(tokenized_dataset, split)
            print(f"{split} Processing and saving completed.")


def tinystories(
    dataset="roneneldan/TinyStories",
    tokenizer="microsoft/phi-2",
):
    print(f"Pretokenizing {dataset=} with {tokenizer=}")
    preprocessor = DatasetPreprocessor(
        dataset_name=dataset,
        tokenizer_name=tokenizer,
        splits=["train", "validation"],
        hf_cache=Path("./hf_cache"),
    )
    preprocessor.process_and_save()


def minipile(
    dataset="JeanKaddour/minipile",
    tokenizer="EleutherAI/gpt-neo-125m",
):
    print(f"Pretokenizing {dataset=} with {tokenizer=}")
    preprocessor = DatasetPreprocessor(
        dataset_name=dataset,
        tokenizer_name=tokenizer,
        splits=["train", "validation"],
        hf_cache=Path("./hf_cache"),
    )
    preprocessor.process_and_save()


# datasets = {"openhermes": prepare_openhermes_2_5, "tinystories": tinystories, "minipile": minipile}


if __name__ == "__main__":
    # minipile()
    tinystories()
    # print(datasets.__version__)