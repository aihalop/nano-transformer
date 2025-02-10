from data import dataset_multi30k
from model import Transformer

batch_size = 8
train_data_loader, valid_data_loader, test_data_loader = dataset_multi30k(batch_size)

model = Transformer()

# train
while True:
    for item in train_data_loader:
        print("item: ", item, len(item), type(item))
        src = item['de_ids']
        trg = item['en_ids']
        model(src)
