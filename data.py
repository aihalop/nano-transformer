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


def dataset_multi30k(batch_size=2):
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

    tokenize = lambda data, en_nlp, de_nlp, max_length, sos_token, eos_token: {
        "en_tokens": [sos_token] + [token.text.lower() for token in en_nlp.tokenizer(data["en"])][:max_length] + [eos_token],
        "de_tokens": [sos_token] + [token.text.lower() for token in de_nlp.tokenizer(data["de"])][:max_length] + [eos_token]
    }

    max_length = 1_000
    sos_token = "<sos>"
    eos_token = "<eos>"

    fn_kwargs = {
        "en_nlp": en_nlp,
        "de_nlp": de_nlp,
        "max_length": max_length,
        "sos_token": sos_token,
        "eos_token": eos_token,
    }

    train_data = train_data.map(tokenize, fn_kwargs=fn_kwargs)
    valid_data = valid_data.map(tokenize, fn_kwargs=fn_kwargs)
    test_data = test_data.map(tokenize, fn_kwargs=fn_kwargs)

    min_freq = 2
    unk_token = "<unk>"
    pad_token = "<pad>"

    special_tokens = [
        unk_token,
        pad_token,
        sos_token,
        eos_token,
    ]

    en_vocab = vocab.build_vocab_from_iterator(
        train_data["en_tokens"],
        min_freq=min_freq,
        specials=special_tokens,
    )

    de_vocab = vocab.build_vocab_from_iterator(
        train_data["de_tokens"],
        min_freq=min_freq,
        specials=special_tokens,
    )

    assert en_vocab[unk_token] == de_vocab[unk_token]
    assert en_vocab[pad_token] == de_vocab[pad_token]

    unk_index = en_vocab[unk_token]
    pad_index = en_vocab[pad_token]
    en_vocab.set_default_index(unk_index)
    de_vocab.set_default_index(unk_index)


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


    train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
    valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
    test_data_loader = get_data_loader(test_data, batch_size, pad_index)

    print("train_data_loader: ", len(train_data_loader))
    print("valid_data_loader: ", len(valid_data_loader))
    print("test_data_loader: ", len(test_data_loader))

    return train_data_loader, valid_data_loader, test_data_loader


if __name__=="__main__":
    train_data_loader, valid_data_loader, test_data_loader = dataset_multi30k()
    for item in train_data_loader:
        print("item: ", item, len(item))
