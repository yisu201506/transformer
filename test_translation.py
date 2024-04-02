import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from transformer import Transformer
import time


"""
To install spacy languages do:
python -m spacy download en
python -m spacy download de
"""

spacy_ger = spacy.load("de")
spacy_eng = spacy.load("en")
# spacy_zh = spacy.load("zh_core_web_md")

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

# Setup the training phase
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training hyperparameters
num_epochs = 10
learning_rate = 3e-4
batch_size = 32

# Model hyperparameters
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embed_size = 512
num_heads = 8
num_layers = 3
dropout = 0.1
max_len = 100
forward_expansion = 4
src_pad_idx = german.vocab.stoi["<pad>"]
trg_pad_idx = 0

model = Transformer(
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    trg_pad_idx,
    embed_size=embed_size,
    num_layers=num_layers,
    heads=num_heads,
    forward_expansion=forward_expansion,
    dropout=dropout,
    max_len=max_len,
    device=device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

load_model = True
if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)

# sentence = 'ein pferd geht unter einer brücke neben einem boot.'
sentence = 'Ich würde gerne einen Apfel zum Abendessen essen'
model.eval()
with torch.no_grad():
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=100
    )

print(f'Translated example sentence \n {translated_sentence}')