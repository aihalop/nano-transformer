# Copyright (c) 2025 Jin Cao
# SPDX-License-Identifier: MIT
#
# Train the Transformer as a seq2seq machine translator.
#
# Author: Jin Cao <aihalop@gmail.com>


import torch
import torch.nn.functional as F
import math
import random

from data import Multi30k, de_tokenize
from model import Transformer

import config

# Repeatability
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


device = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def train(model, optimizer, loss_function, train_data):
    model.train()
    size = len(train_data.dataset)
    loss = None
    for count, item in enumerate(train_data):
        src = item['de_ids'].to(device)
        trg = item['en_ids'].to(device)

        # TODO(Jin Cao): Try not shifted right. If we put the whole
        # length tokens as input data, and it will not only train the
        # last token, but the tokens at every positions as well, then
        # why should we shift right the output?
        shifted_right_trg = trg[:, :-1]
        ground_truth = trg[:, 1:].contiguous()
        optimizer.zero_grad()
        predict = model(src, shifted_right_trg)
        predict = predict.transpose(-2, -1).contiguous() # transpose for the input of CrossEntropyLoss.

        loss = loss_function(predict, ground_truth)

        loss.backward()
        optimizer.step()

        if count % 10 == 0:
            loss, current = loss.item(), (count + 1) * len(src)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss


def validate(model, validate_data, loss_function):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, items in enumerate(validate_data):
            src = items['de_ids'].to(device)
            trg = items['en_ids'].to(device)
            predict = model(src, trg[:, :-1])
            predict = predict.transpose(-2, -1)
            loss = loss_function(predict, trg[:, 1:].long())
            total_loss += loss.item()

    return total_loss / len(validate_data)


if __name__=="__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-d', '--data', type=str, help='data path')
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('--load_trained', action='store_true', default=False)
    parser.add_argument('--batch', type=int, default=50)
    args = parser.parse_args()

    num_epochs = args.epoch
    batch_size = args.batch
    dataset = Multi30k(batch_size)
    num_de_embeddings = dataset.de_vocab_size()
    num_en_embeddings = dataset.en_vocab_size()
    padding_idx = dataset.pad_index()

    model = Transformer(config.num_layers, num_de_embeddings, num_en_embeddings,
                        config.embedding_dim, padding_idx, config.max_token_length,
                        config.num_heads, dataset.en_vocab_size())
    model.to(device)

    if args.load_trained and os.path.exists(config.model_file):
        print(f"\nLoad a trained model parameters from {config.model_file}\n")
        model.load_state_dict(torch.load(model_file))


    print(f"\nTrain the transformer model with dataset {dataset}.\n")
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = torch.optim.Adam(model.parameters())
    print(f"{num_epochs} epochs to train")
    for i in range(num_epochs):
        loss = train(model, optimizer, loss_function, dataset.train_data())
        validate_loss = validate(model, dataset.valid_data(), loss_function)
        print(f"epoch {i}, validate_loss: {validate_loss}")

        torch.save(model.state_dict(), "model.pth")

    print("save the model to model.pth")
