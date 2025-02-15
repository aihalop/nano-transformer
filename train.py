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



batch_size = 10
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
optimizer = torch.optim.Adam(model.parameters())
loss_function = torch.nn.CrossEntropyLoss()

for count, item in enumerate(dataset.train_data()):
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
    print("loss: ", loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if count == 1000: break



# # train
# while True:
#     if count == 5: break
#     count += 1


print([x for x in range(5)])
