from data import Multi30k
# from model import Transformer
import torch
import torch.nn.functional as F
import math


device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, max_token_length):
        super().__init__()
        self.input_embedding = \
            torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        self.position_encoding = \
            torch.nn.Embedding(max_token_length, embedding_dim)

    def forward(self, x):
        positions = torch.zeros(x.shape, dtype=torch.int).to(device)
        positions[:, :] = torch.range(0, x.shape[1] - 1)
        return self.position_encoding(positions) + self.input_embedding(x)


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        dk = K.size(-2)
        # TODO(Jin Cao): Mask (opt)
        # TODO(Jin Cao): mulit-head attention
        return F.softmax(Q @ K.transpose(-2, -1) / math.sqrt(dk)) @ V


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

    def forward(self, x):
        return self.attention(
            self.projector_q(x),
            self.projector_k(x),
            self.projector_v(x),
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
        self._embedding_dim = embedding_dim
        self.self_attention = SelfAttention(embedding_dim)
        # self.transform_QKV = torch.nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.feed_forward = FeedForward(embedding_dim)

    def forward(self, x):
        # Q, K, V = self.transform_QKV(x).split(self._embedding_dim, dim=-1)
        # attention = self.attention(Q, K, V)
        attention = F.layer_norm(x + self.self_attention(x), x.shape)
        output = F.layer_norm(attention + self.feed_forward(attention), attention.shape)
        return output


class Encoder(torch.nn.Module):
    def __init__(self, num_layers, num_embeddings, embedding_dim,
                 padding_idx, max_token_length, num_heads):
        super().__init__()
        self.encoder = torch.nn.Sequential(*(
            EncoderBlock(num_embeddings, embedding_dim, padding_idx,
                         max_token_length, num_heads)
            for i in range(num_layers)
        ))

    def forward(self, x):
        return self.encoder(x)


class DecoderBlock(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.self_attention = SelfAttention(embedding_dim)
        self.encoder_decoder_attention = EncoderDecoderAttention(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim)

    def forward(self, y, x):
        self_attention = F.layer_norm(y + self.self_attention(y), y.shape)
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
        self.blocks = torch.nn.ModuleList(
            [DecoderBlock(num_embeddings, embedding_dim) for i in range(num_layers)]
        )

    def forward(self, y, x):
        for block in self.blocks:
            y = block(y, x)

        return y


class Transformer(torch.nn.Module):
    def __init__(self, num_layers, num_embeddings, embedding_dim, padding_idx,
                 max_token_length, num_heads, en_vocab_size):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_heads = num_heads
        self.embedding = Embedding(
            num_embeddings, embedding_dim, padding_idx, max_token_length)

        self.encoder = Encoder(num_layers, num_embeddings, embedding_dim,
                               padding_idx, max_token_length, num_heads)

        self.decoder = Decoder(2, num_embeddings, embedding_dim)
        self.linear = torch.nn.Linear(embedding_dim, en_vocab_size)


    def forward(self, x, y):
        '''x, input; y, output'''
        x = self.embedding(x)
        y = self.embedding(y)
        encode = self.encoder(x)
        output = self.decoder(y, encode)
        return self.linear(output)




def train(model, optimizer, loss_function, train_data):
    size = len(train_data.dataset)
    print(f"size: {size}")
    loss = None
    for count, item in enumerate(train_data):
        # print("item: ", item, len(item), type(item))
        src = item['de_ids'].to(device)
        trg = item['en_ids'].to(device)

        # TODO(Jin Cao): Try not shifted right. If we put the whole length
        # tokens as input data, and it will not only train the last token,
        # but the tokens at every positions as well, then why should we
        # shifted right the output?
        shifted_right_trg = trg[:, :-1]
        predict = model(src, shifted_right_trg)
        predict = predict.transpose(-2, -1)
        # print("output >> ", predict, predict.shape)
        loss = loss_function(predict, trg[:,1:].long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if count % 10 == 0:
            loss, current = loss.item(), (count + 1) * len(src)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss


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
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--batch', type=int, default=50)
    args = parser.parse_args()

    model_file = "model.pth"
    num_epochs = args.epoch
    batch_size = args.batch
    dataset = Multi30k(batch_size)
    num_embeddings = dataset.de_vocab_size()
    embedding_dim = 512
    num_heads = 8
    num_layers = 6
    padding_idx = dataset.pad_index()
    max_token_length = dataset.max_token_length()

    model = Transformer(num_layers, num_embeddings, embedding_dim,
                        padding_idx, max_token_length, num_heads, dataset.en_vocab_size())
    model.to(device)
    if args.load_trained and os.path.exists(model_file):
        print(f"load a trained model parameters from {model_file}")
        model.load_state_dict(torch.load(model_file))

    if args.train:
        model.train()
        optimizer = torch.optim.Adam(model.parameters())
        loss_function = torch.nn.CrossEntropyLoss()

        # train
        for i in range(num_epochs):
            loss = train(model, optimizer, loss_function, dataset.train_data())
            print(f"epoch {i}, loss: {loss}")

        torch.save(model.state_dict(), "model.pth")
        print("save the model to model.pth")


    model.eval()
    item = next(iter(dataset.test_data()))
    src = item['de_ids'].to(device)
    trg = item['en_ids'].to(device)
    predict = model(src,  trg[:, :-1])
    result = torch.argmax(F.softmax(predict, dim=2), dim=2)
    print("predict: ", result, result.shape)
    print("trg: ", trg, trg.shape)
