from data import Multi30k
# from model import Transformer
import torch


def position_encoding(pos):
    return None


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, max_token_length):
        super().__init__()
        self.input_embedding = \
            torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        self.position_encoding = \
            torch.nn.Embedding(max_token_length, embedding_dim)

    def forward(self, x):
        position = torch.zeros(x.shape, dtype=torch.int)
        position[:, :] = torch.range(0, x.shape[1] - 1)
        return self.position_encoding(position) + self.input_embedding(x)


class Transformer(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, max_token_length):
        super().__init__()
        self.embedding = Embedding(
            num_embeddings, embedding_dim, padding_idx, max_token_length)


    def forward(self, x):
        print(self.embedding(x), self.embedding(x).shape)
        return 0.0



batch_size = 3
dataset = Multi30k(batch_size)
num_embeddings = dataset.de_vocab_size()
embedding_dim = 128
padding_idx = dataset.pad_index()
max_token_length = dataset.max_token_length()
print("padding_idx: ", padding_idx)
model = Transformer(num_embeddings, embedding_dim, padding_idx, max_token_length)
for count, item in enumerate(dataset.train_data()):
    print("item: ", item, len(item), type(item))
    src = item['de_ids']
    trg = item['en_ids']
    model(src)

    if count == 5: break



# # train
# while True:
#     if count == 5: break
#     count += 1
