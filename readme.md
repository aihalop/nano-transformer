# Transformer in Pytorch

The purpose of this project is to study the Transformer described in
the paper [1]. So the Transformer as a language translator is
implemented as close as possible to the original paper without tweaks
and tricks.

You may find it helpful if you are looking for an simple
implementation of the Transformer with updated pytorch libraries.

High performance GPU is not necessary. You could play this transformer
using a CUDA-Enabled Nvidia GPU with only 2G memory.

Issues and pull requests are welcome!

## Prerequisites

```bash
pip install -r requirements.txt

# Install spaCy models
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm

# Or you could install a downloaded spaCy model.
pip install <local_path>/en_core_web_sm-3.8.0.tar.gz
pip install <local_path>/de_core_news_sm-3.8.0.tar.gz
# Or install from the release url
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz
pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0.tar.gz
```

## Usage
### Train

It loads Multi30K dataset, and train the transformer as a
German-English translator and saves the trained model to "model.pth"

```bash
python3 train.py
```

### Test

It translates all German sentences in the test set of Multi30K into
English using the model loaded from "model.pth" if it exists;
otherwise, an untrained model will be used.

```bash
python3 translate.py
```

# References

  * [1] Vaswani, A. et al. Attention is All you Need.
  * [2] https://github.com/bentrevett/pytorch-seq2seq/
  * [3] https://github.com/jadore801120/attention-is-all-you-need-pytorch.git
