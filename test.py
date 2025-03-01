# Copyright (c) 2025 Jin Cao
# SPDX-License-Identifier: MIT
#
# A Translator using the transformer.
#
# Author: Jin Cao <aihalop@gmail.com>

import torch
import torch.nn.functional as F
import math
import random
import os

from data import Multi30k, de_tokenize
from model import Transformer


device = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def translate(model, de_sentence, de_vocab, en_vocab,
              sos_idx, eos_idx, max_token_length):
    model.eval()

    src = torch.tensor(
        [de_vocab[token] for token in de_tokenize(de_sentence)]
    ).unsqueeze(0).to(device)
    translated_idx = [sos_idx]
    for i in range(max_token_length):
        trg = torch.tensor([translated_idx]).to(device)
        predict = model(src, trg)
        next_token_idx = int(predict[0,-1,:].argmax(dim=-1).cpu())
        translated_idx.append(next_token_idx)

        if next_token_idx == eos_idx:
            break

    # print("translated_idx: ", translated_idx)
    return " ".join(en_vocab.lookup_tokens(translated_idx[1:-1]))


if __name__=="__main__":
    model_file = "model.pth"
    batch_size = 1
    dataset = Multi30k(batch_size)
    num_de_embeddings = dataset.de_vocab_size()
    num_en_embeddings = dataset.en_vocab_size()
    embedding_dim = 64
    num_heads = 8
    num_layers = 6
    padding_idx = dataset.pad_index()
    max_token_length = 200

    model = Transformer(num_layers, num_de_embeddings, num_en_embeddings, embedding_dim,
                        padding_idx, max_token_length, num_heads, dataset.en_vocab_size())
    model.to(device)

    if os.path.exists(model_file):
        print(f"\nLoad a trained model parameters from {model_file}\n")
        model.load_state_dict(torch.load(model_file))

    print("\nTranslate sentences from German to English.\n")
    model.eval()
    de_vocab = dataset.de_vocab()
    en_vocab = dataset.en_vocab()
    special_tokens = dataset.special_tokens()
    special_idx = dataset.spacial_idx()
    sos_idx = dataset.sos_idx()
    eos_idx = dataset.eos_idx()

    for batch in dataset.test_data():
        for de_sentence, en_sentence in zip(batch["de"], batch["en"]):
            print("German sentence: ", de_sentence)
            print("English sentence: ", en_sentence)
            en_sentence = translate(model, de_sentence, de_vocab, en_vocab,
                      sos_idx, eos_idx, max_token_length)
            print("Translation: ", en_sentence)
            print("---")
