import torch
from torch import nn
from torch.nn import functional as F

import math

def main():
    # Example usage of the ScaledDotProductAttention function
    q = torch.rand(1, 5, 512)  # (batch_size, seq_len, d_k)
    k = torch.rand(1, 5, 512)  # (batch_size, seq_len, d_k)
    v = torch.rand(1, 5, 512)  # (batch_size, seq_len, d_v)
    scr_input_ids = torch.randint(0, 128, (1, 5)).type(torch.int64)  # (batch_size, seq_len)
    print("The shape of scr Input:", scr_input_ids.shape)
    
    tgt_input_ids = torch.randint(0, 128, (1, 5)).type(torch.int64)  # (batch_size, seq_len)
    print("The shape of scr Input:", tgt_input_ids.shape)

    transformer = Transformer(src_vocab_size=128, tgt_vocab_size=128, d_model=512, num_layers=6, num_heads=8)
    output = transformer(scr_input_ids, tgt_input_ids)
    print("The shape of output:", output.shape)  # (batch_size, seq_len, tgt_vocab_size)


def ScaledDotProductAttention(q: torch.tensor, k: torch.tensor, v: torch.tensor,  mask=None):
  d_k = q.shape[-1]
  scores = q @ k.transpose(-2, -1) / (d_k ** (1/2))
  if mask is not None:
    scores = scores.masked_fill(mask == 0, float('-inf'))
  attention = torch.softmax(scores, dim=-1)
  return attention @ v, attention

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int, num_heads: int):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads

    assert d_model % num_heads == 0, "The dimension of model can not be divided by the dimention of k"
    self.d_k = d_model // num_heads

    self.linear_q = nn.Linear(in_features=d_model, out_features=d_model)
    self.linear_k = nn.Linear(in_features=d_model, out_features=d_model)
    self.linear_v = nn.Linear(in_features=d_model, out_features=d_model)
    self.out_projection = nn.Linear(in_features=d_model, out_features=d_model)


  def forward(self, q: torch.tensor, k: torch.tensor, v: torch.tensor, mask=None):
    B, S, _ = q.shape
    q = self.linear_q(q) # (Batch, Seq_length, d_model), adding linear layer to q, k, v before splitting
    q = q.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    k = self.linear_k(k)
    k = k.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
    
    v = self.linear_v(v)
    v = v.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
    
    attn_output, _ = ScaledDotProductAttention(q, k, v, mask) # (Batch, num_heads, Seq_lenth, d_k)

    B, _, S, _ = attn_output.shape
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_model)

    return self.out_projection(attn_output)

class PostionWiseForward(nn.Module):
  def __init__(self, d_model, d_ff = 2048):
    super().__init__()
    self.d_model = d_model
    self.ff = nn.Sequential(
        nn.Linear(in_features=d_model, out_features=d_ff),
        nn.ReLU(),
        nn.Linear(in_features=d_ff, out_features=d_model)
    )

  def forward(self, x):
    return self.ff(x)

class PostionalEncoding(nn.Module):
  def __init__(self, d_model: int, max_lenth: int = 5000):
    super().__init__()
    pe = torch.zeros(max_lenth, d_model)
    pos = torch.arange(0, max_lenth).unsqueeze(dim=1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1000) /d_model))

    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)

    pe = pe.unsqueeze(dim=0) # To match the shape of q, k, v -> (Batch, Seq_Len, d_model)
    self.register_buffer('pe', pe) # Static table, no grad desc

  def forward(self, x):
    seq_len = x.shape[1]
    return x + self.pe[:, :seq_len]

class EncoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, dropout=0.1):
    super().__init__()
    self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    self.norm1 = nn.LayerNorm(d_model)
    self.ff = PostionWiseForward(d_model=d_model, d_ff=2048)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask=None):
    attention = self.self_attention(q=x, k=x, v=x, mask=mask)
    x = self.norm1(x + self.dropout(attention))
    ff = self.ff(x)
    x = self.norm2(x + self.dropout(ff))
    return x

class Encoder(nn.Module):
  def __init__(self, num_vocabs: int, d_model: int, num_heads: int = 8, num_layers: int = 6):
    super().__init__()
    self.embedding = nn.Embedding(num_vocabs, d_model)
    self.positionencoding = PostionalEncoding(d_model=d_model)
    self.encoderlayers = nn.ModuleList([EncoderLayer(d_model=d_model, num_heads=num_heads) for _ in range(num_layers)])
    self.norm = nn.LayerNorm(d_model)

  def forward(self, input_ids, mask=None):
    x = self.embedding(input_ids)
    x = self.positionencoding(x)
    for encoderlayer in self.encoderlayers:
      x = encoderlayer(x, mask)
    return self.norm(x)

class DecoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
    super().__init__()
    self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    self.cross_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    self.norm1 = nn.LayerNorm(d_model)
    self.ff = PostionWiseForward(d_model=d_model, d_ff=d_ff)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
    attention = self.self_attention(q=x, k=x, v=x, mask=src_mask)
    x = self.norm1(x + self.dropout(attention))
    cross_attention = self.cross_attention(q=x, k=encoder_out, v=encoder_out, mask=tgt_mask)
    x = self.norm1(x + self.dropout(cross_attention))
    ff = self.ff(x)
    x = self.norm2(x + self.dropout(ff))
    return x

class Decoder(nn.Module):
  def __init__(self, num_vocabs: int, d_model: int, num_heads: int = 8, num_layers: int = 6):
    super().__init__()
    self.embedding = nn.Embedding(num_vocabs, d_model)
    self.positionencoding = PostionalEncoding(d_model=d_model)
    self.decorderlayers = nn.ModuleList([DecoderLayer(d_model=d_model, num_heads=num_heads) for _ in range(num_layers)])
    self.norm = nn.LayerNorm(d_model)

  def forward(self, target, encoder_out, scr_mask=None, tgt_mask=None):
    x = self.embedding(target)
    x = self.positionencoding(x)
    for decorderlayer in self.decorderlayers:
      x = decorderlayer(x, encoder_out, scr_mask, tgt_mask)
    return self.norm(x)
  
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, num_heads=8):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model=d_model, num_layers=num_layers, num_heads=num_heads)
        self.decoder = Decoder(tgt_vocab_size, d_model=d_model, num_layers=num_layers, num_heads=num_heads)
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.out(dec_out)
  
if __name__ == "__main__":
    main()