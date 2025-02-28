from data import Multi30k, de_tokenize
import torch
import torch.nn.functional as F
import math
import random

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


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, max_token_length):
        super().__init__()
        self.input_embedding = \
            torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.position_encoding = \
            torch.nn.Embedding(max_token_length, embedding_dim)

    def forward(self, x):
        positions = torch.arange(0, x.shape[1]).repeat(x.shape[0], 1).to(device)
        return self.position_encoding(positions) + self.input_embedding(x)


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        dk = K.size(-2)
        # TODO(Jin Cao): mulit-head attention

        if mask is None:
            mask = (torch.zeros(Q.shape[-2], K.shape[-2]) == 1).to(device)

        return F.softmax(
            (Q @ K.transpose(-2, -1)).masked_fill(mask, -1e10) \
            / math.sqrt(dk), dim=-1
        ) @ V


class FeedForward(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 4 * embedding_dim),
            torch.nn.ReLU(), # TODO(Jin Cao): try Tanh
            torch.nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.f(x)


class SelfAttention(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.projector_q = torch.nn.Linear(embedding_dim, embedding_dim)
        self.projector_k = torch.nn.Linear(embedding_dim, embedding_dim)
        self.projector_v = torch.nn.Linear(embedding_dim, embedding_dim)
        self.attention = Attention()

    def forward(self, x, mask=None):
        return self.attention(
            self.projector_q(x), self.projector_k(x), self.projector_v(x),
            mask
        )


class EncoderDecoderAttention(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.projector_q = torch.nn.Linear(embedding_dim, embedding_dim)
        self.projector_k = torch.nn.Linear(embedding_dim, embedding_dim)
        self.projector_v = torch.nn.Linear(embedding_dim, embedding_dim)
        self.attenion = Attention()

    def forward(self, y, x):
        return self.attenion(
            self.projector_q(y),
            self.projector_k(x),
            self.projector_v(x)
        )


class EncoderBlock(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx,
                 max_token_length, num_heads):
        super().__init__()
        self._padding_idx = padding_idx
        self._embedding_dim = embedding_dim
        self.self_attention = SelfAttention(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim)

    def forward(self, x, padding_mask=None):
        attention = F.layer_norm(x + self.self_attention(x, padding_mask), x.shape)
        output = F.layer_norm(attention + self.feed_forward(attention), attention.shape)
        return output


class Encoder(torch.nn.Module):
    def __init__(self, num_layers, num_embeddings, embedding_dim,
                 padding_idx, max_token_length, num_heads):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [EncoderBlock(num_embeddings, embedding_dim, padding_idx,
                          max_token_length, num_heads)
             for i in range(num_layers)]
        )

    def forward(self, x, padding_mask=None):
        for block in self.blocks:
            x = block(x, padding_mask)
        return x


class DecoderBlock(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.self_attention = SelfAttention(embedding_dim)
        self.encoder_decoder_attention = EncoderDecoderAttention(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim)

    def forward(self, y, x):
        # [* 0 0]
        # [* * 0]
        # [* * *]
        mask = (torch.tril(torch.ones(y.shape[-2], y.shape[-2])) == 0).to(device)
        self_attention = F.layer_norm(y + self.self_attention(y, mask), y.shape)
        encode_decode_attention = F.layer_norm(
            self_attention + self.encoder_decoder_attention(self_attention, x),
            self_attention.shape
        )
        output = F.layer_norm(
            encode_decode_attention + self.feed_forward(encode_decode_attention),
            encode_decode_attention.shape
        )

        return output


class Decoder(torch.nn.Module):
    def __init__(self, num_layers, num_embeddings, embedding_dim):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.blocks = torch.nn.ModuleList(
            [DecoderBlock(num_embeddings, embedding_dim) for i in range(num_layers)]
        )

    def forward(self, y, x):
        y = self.dropout(y)
        for block in self.blocks:
            y = block(y, x)

        return y


class Transformer(torch.nn.Module):
    def __init__(self, num_layers, num_input_embeddings, num_output_embeddings,
                 embedding_dim, padding_idx, max_token_length, num_heads, en_vocab_size):
        super().__init__()
        self._padding_idx = padding_idx
        self._embedding_dim = embedding_dim
        self._num_heads = num_heads
        # Since the input and output vocabularies are different, we
        # have to use two embeddings respectively.
        self.input_embedding = Embedding(
            num_input_embeddings, embedding_dim, padding_idx, max_token_length)
        self.output_embedding = Embedding(
            num_output_embeddings, embedding_dim, padding_idx, max_token_length)

        self.encoder = Encoder(num_layers, num_input_embeddings, embedding_dim,
                               padding_idx, max_token_length, num_heads)

        self.decoder = Decoder(num_layers, num_output_embeddings, embedding_dim)
        self.linear = torch.nn.Linear(embedding_dim, en_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)


    def forward(self, x, y):
        '''x, input; y, output'''
        padding_mask = (x == self._padding_idx).unsqueeze(-2)
        x = F.dropout(self.input_embedding(x))
        y = self.output_embedding(y)
        encode = self.encoder(x, padding_mask)
        output = self.decoder(y, encode)
        return self.linear(output)


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
    import os
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-d', '--data', type=str, help='data path')
    parser.add_argument('-e', '--epoch', type=int, default=1)
    parser.add_argument('--load_trained', action='store_true', default=False)
    parser.add_argument('--predict', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--batch', type=int, default=50)
    args = parser.parse_args()

    model_file = "model.pth"
    num_epochs = args.epoch
    batch_size = args.batch
    dataset = Multi30k(batch_size)
    num_de_embeddings = dataset.de_vocab_size()
    num_en_embeddings = dataset.en_vocab_size()
    embedding_dim = 32
    num_heads = 8
    num_layers = 6
    padding_idx = dataset.pad_index()
    max_token_length = 200

    model = Transformer(num_layers, num_de_embeddings, num_en_embeddings, embedding_dim,
                        padding_idx, max_token_length, num_heads, dataset.en_vocab_size())
    model.to(device)

    if args.load_trained and os.path.exists(model_file):
        print(f"\nLoad a trained model parameters from {model_file}\n")
        model.load_state_dict(torch.load(model_file))

    if args.train:
        print(f"\nTrain the transformer model with dataset {dataset}.\n")
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=padding_idx, reduction='sum')
        optimizer = torch.optim.Adam(model.parameters())
        for i in range(num_epochs):
            loss = train(model, optimizer, loss_function, dataset.train_data())
            validate_loss = validate(model, dataset.valid_data(), loss_function)
            print(f"epoch {i}, validate_loss: {validate_loss}")

        torch.save(model.state_dict(), "model.pth")
        print("save the model to model.pth")


    print("\nTranslate sentences from German to English.\n")
    model.eval()
    de_vocab = dataset.de_vocab()
    en_vocab = dataset.en_vocab()
    special_tokens = dataset.special_tokens()
    special_idx = dataset.spacial_idx()
    sos_idx = dataset.sos_idx()
    eos_idx = dataset.eos_idx()

    for batch in dataset.test_data():
        for de_sentence in batch["de"]:
            print("German sentence: ", de_sentence)
            en_sentence = translate(model, de_sentence, de_vocab, en_vocab,
                      sos_idx, eos_idx, max_token_length)
            print("English sentence: ", en_sentence)
            print("---")
