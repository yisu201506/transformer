import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()

        self.dim_head = embed_size // heads
        self.heads = heads
        self.embed_size = embed_size

        assert self.embed_size == self.heads * self.dim_head, "embed_size should be divisible by heads"

        self.values = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.keys = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.queries = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.fc_out = nn.Linear(self.heads * self.dim_head, embed_size)

    def forward(self, query, key, value, mask):
        N = query.shape[0]
        query_len, value_len, key_len = query.shape[1], value.shape[1], key.shape[1]

        query = torch.reshape(query, [N, query_len, self.heads, self.dim_head])
        key = torch.reshape(key, [N, key_len, self.heads, self.dim_head])
        value = torch.reshape(value, [N, value_len, self.heads, self.dim_head])

        query = self.queries(query)
        key = self.keys(key)
        value = self.values(value)

        energy = torch.einsum('nqhd, nkhd -> nhqk', [query, key])
        # query [N, query_len, self.heads, self.dim_head]
        # key   [N, key_len, self.heads, self.dim_head]
        # after einsum, out [N, self.heads, query_len, key_len]

        if mask is not None:
            energy = (energy / self.dim_head ** 0.5).masked_fill(mask == 0, float('-1e20'))

        attention = nn.Softmax(dim=3)(energy)

        out = torch.einsum('nhqv, nvhd -> nqhd', [attention, value]).reshape(
            N, query_len, self.embed_size
        )
        # attention [N, self.heads, query_len, key_len]
        # value     [N, value_len, self.heads, self.dim_head]
        # value_len == key_len
        # after einsum, out  [N, query_len, self.heads, self.dim_head]

        out = self.fc_out(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            dropout,
            forward_expansion,
    ):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        attention = self.attention(query, key, value, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion
                ) for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            dropout,
            forward_expansion,
    ):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.transformer = TransformerBlock(
            embed_size,
            heads,
            dropout,
            forward_expansion,
        )

    def forward(self, query, key, value, src_mask, trg_mask):
        out = self.attention(query, query, query, trg_mask)
        out = self.dropout(self.norm(out + query))
        out = self.transformer(out, key, value, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
    ):
        super(Decoder, self).__init__()
        # self.embed_size = embed_size
        # self.heads = heads
        self.device = device

        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size, heads, dropout, forward_expansion
                ) for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        # trg_mask = torch.tril(torch.ones(seq_length, seq_length)).to(self.device)

        for layer in self.layers:
            out = layer(out, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(out)

        return out


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_size=256,
            num_layers=6,
            heads=8,
            forward_expansion=4,
            dropout=0.0,
            max_len=100,
            device='cpu',
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_len,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_len,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src == self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask shape [N, 1, 1, src_len ]
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)

        return out


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device = 'cpu'
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    print(f'shape of x is {x.shape}')
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    print(f'shape of trg is {trg.shape}')
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10

    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
    out = model(x, trg[:, :-1])

    # print(nn.Softmax(dim=-1)(out))
    print(out.shape)







