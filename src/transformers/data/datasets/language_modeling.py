import logging
import os
import pickle
import time

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from ...tokenization_utils import PreTrainedTokenizer
from constants import LABEL_SPECIAL_TOKENS

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str,
                 block_size: int, replace_separator: str = None):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = []
            for line in f.read().splitlines():
                if (len(line) > 0 and not line.isspace()):
                    if replace_separator:
                        assert replace_separator in line
                        line = line.replace(replace_separator, tokenizer._sep_token)
                    lines.append(line)

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)


        # For WholeWordMasking: Hardcoded for Roberta and XLNet
        assert type(tokenizer).__name__ in ["RobertaTokenizer", "XLNetTokenizer"]

        list_word_starts = []
        for ids_ in batch_encoding["input_ids"]:
            word_starts = []
            first_non_special_encountered = False
            for id_ in ids_:
                token = tokenizer.convert_ids_to_tokens(id_)[0]
                word_start = True
                start_special = (str(token).startswith('Ġ') if type(tokenizer).__name__ == "RobertaTokenizer" \
                                 else str(token).startswith('▁'))
                if not start_special and id_ not in tokenizer.all_special_ids:
                    if type(tokenizer).__name__ == "RobertaTokenizer" and first_non_special_encountered:
                        word_start = False
                    if type(tokenizer).__name__ != "RobertaTokenizer":
                        word_start = False
                word_starts.append(word_start)

                # Don't consider the first non-special token to be non-word-start
                if not first_non_special_encountered and not id_ in tokenizer.all_special_ids:
                    first_non_special_encountered = True
            list_word_starts.append(word_starts)

        label_token_ids = []
        for token in LABEL_SPECIAL_TOKENS:
            id_ = tokenizer.encode(token, add_special_tokens=False)
            assert len(id_) == 1
            label_token_ids.append(id_[0])

        for input_ids in batch_encoding["input_ids"]:
            assert sum([input_id in label_token_ids for input_id in input_ids]) <= 1

        self.examples = [{"input_ids": input_ids, "word_starts": word_starts}
                          for input_ids, word_starts in zip(batch_encoding["input_ids"], list_word_starts)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return {key: torch.tensor(value, dtype=torch.long) for key, value in self.examples[i].items()}
