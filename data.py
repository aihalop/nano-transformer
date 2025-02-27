import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
import datasets

from torchtext import vocab
import os

seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class Multi30k(object):
    def __init__(self, batch_size=2, max_token_length=1000):
        self._max_token_length = max_token_length
        dataset_path="./dataset"
        if os.path.exists(dataset_path):
            dataset = datasets.load_from_disk(dataset_path)
        else:
            dataset = datasets.load_dataset("bentrevett/multi30k")
            dataset.save_to_disk(dataset_path)

        print(dataset)
        train_data, valid_data, test_data = (
            dataset["train"],
            dataset["validation"],
            dataset["test"],
        )

        en_nlp = spacy.load("en_core_web_sm")
        de_nlp = spacy.load("de_core_news_sm")

        tokenize = lambda data, en_nlp, de_nlp, max_token_length, sos_token, eos_token: {
            "en_tokens": [sos_token] + [token.text.lower() for token in en_nlp.tokenizer(data["en"])][:max_token_length] + [eos_token],
            "de_tokens": [sos_token] + [token.text.lower() for token in de_nlp.tokenizer(data["de"])][:max_token_length] + [eos_token]
        }
        
        min_freq = 3
        unk_token = "<unk>"
        pad_token = "<pad>"
        sos_token = "<sos>"
        eos_token = "<eos>"

        self._special_tokens = [
            unk_token,
            pad_token,
            sos_token,
            eos_token,
        ]

        fn_kwargs = {
            "en_nlp": en_nlp,
            "de_nlp": de_nlp,
            "max_token_length": max_token_length,
            "sos_token": sos_token,
            "eos_token": eos_token,
        }

        train_data = train_data.map(tokenize, fn_kwargs=fn_kwargs)
        valid_data = valid_data.map(tokenize, fn_kwargs=fn_kwargs)
        test_data = test_data.map(tokenize, fn_kwargs=fn_kwargs)


        en_vocab = vocab.build_vocab_from_iterator(
            train_data["en_tokens"],
            min_freq=min_freq,
            specials=self._special_tokens,
        )

        de_vocab = vocab.build_vocab_from_iterator(
            train_data["de_tokens"],
            min_freq=min_freq,
            specials=self._special_tokens,
        )
        self._en_vocab = en_vocab
        self._de_vocab = de_vocab
        self._en_vocab_size = len(en_vocab)
        self._de_vocab_size = len(de_vocab)

        assert en_vocab[unk_token] == de_vocab[unk_token]
        assert en_vocab[pad_token] == de_vocab[pad_token]

        unk_index = en_vocab[unk_token]
        pad_index = en_vocab[pad_token]
        self._pad_index = pad_index
        en_vocab.set_default_index(unk_index)
        de_vocab.set_default_index(unk_index)

        self._special_idx = list(map(lambda token: en_vocab[token], self._special_tokens))

        numericalize = lambda data, en_vocab, de_vocab: {
            "en_ids": en_vocab.lookup_indices(data["en_tokens"]),
            "de_ids": de_vocab.lookup_indices(data["de_tokens"])
        }
        fn_kwargs = {"en_vocab": en_vocab, "de_vocab": de_vocab}

        train_data = train_data.map(numericalize, fn_kwargs=fn_kwargs)
        valid_data = valid_data.map(numericalize, fn_kwargs=fn_kwargs)
        test_data = test_data.map(numericalize, fn_kwargs=fn_kwargs)

        data_type = "torch"
        format_columns = ["en_ids", "de_ids"]
        train_data = train_data.with_format(
            type=data_type, columns=format_columns, output_all_columns=True
        )
        valid_data = valid_data.with_format(
            type=data_type, columns=format_columns, output_all_columns=True,
        )
        test_data = test_data.with_format(
            type=data_type, columns=format_columns, output_all_columns=True,
        )

        def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
            def collate_fn(batch):
                batch_en_ids = [example["en_ids"] for example in batch]
                batch_de_ids = [example["de_ids"] for example in batch]
                batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
                batch_de_ids = nn.utils.rnn.pad_sequence(batch_de_ids, padding_value=pad_index)
                batch = {
                    "en_ids": batch_en_ids.T,
                    "de_ids": batch_de_ids.T,
                }
                return batch

            data_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                shuffle=shuffle,
            )
            return data_loader


        self._train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
        self._valid_data_loader = get_data_loader(valid_data, batch_size, pad_index, shuffle=True)
        self._test_data_loader = get_data_loader(test_data, batch_size, pad_index, shuffle=True)

        
    def train_data(self):
        return self._train_data_loader

    def valid_data(self):
        return self._valid_data_loader

    def test_data(self):
        return self._test_data_loader

    def en_vocab_size(self):
        return self._en_vocab_size

    def de_vocab_size(self):
        return self._de_vocab_size

    def de_vocab(self):
        return self._de_vocab

    def en_vocab(self):
        return self._en_vocab

    def pad_index(self):
        return self._pad_index

    def max_token_length(self):
        return self._max_token_length

    def special_tokens(self):
        return self._special_tokens

    def spacial_idx(self):
        return self._special_idx

    def __str__(self):
        return "Multi30k"

if __name__=="__main__":
    batch_size = 4
    dataset = Multi30k(batch_size)
    print("en_vocab_size: ", dataset.en_vocab_size())
    print("de_vocab_size: ", dataset.de_vocab_size())
    for i, item in enumerate(dataset.train_data()):
        # print("item: ", item, len(item))
        print("item['de_ids']: ", item['de_ids'].T, item['de_ids'].shape)
        print("item['en_ids']: ", item['en_ids'].T, item['en_ids'].shape)
        if i == 0: break
