import torch
from torch import nn
import math

class InputEmbedding(nn.Module):
  def __init__(self, d_model: int, vocab_size: int):
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    return self.embedding(x) * math.sqrt(self.d_model) # (Batch, Seq_len, d_model)
  

class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, seq_len: int , droupout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.droupout = nn.Dropout(droupout)

    # Creat a matrix of shape (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model)

    # Creat a vector of shape (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(dim=1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1000) /d_model))

    # Apply the sin to even positions
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(dim=0) # (1, seq_len, d_model)
    self.register_buffer('pe', pe) # Static table, no grad desc

  def forward(self, x):
    x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch_size, seq_len, d_model)
    return self.droupout(x)
  
class LayerNormalization(nn.Module):
  def __init__(self, eps: float = 10**-6) -> None:
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(1)) # Mutiplied
    self.bias = nn.Parameter(torch.zeros(1)) # Added

  def forward(self, x):
    x = x.type(torch.float32) # (Batch, seq_len, d_model)
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return self.alpha * (x - mean) / (std + self.eps) + self.bias #(Batch, seq_len, d_model)
  
class FeedForwardBlock(nn.Module):
  def __init__(self, d_model, d_ff: int, dropout: float) -> None:
    super().__init__()
    self.linear_1 = nn.Linear(in_features=d_model, out_features=d_ff) # W1 and B1
    self.droupout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(in_features=d_ff, out_features=d_model) # W2 and B2

  def forward(self, x):
    # (Batch, Seq_Len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
    return self.linear_2(self.droupout(torch.relu(self.linear_1(x))))
  
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads

    assert d_model % num_heads == 0, "The dimension of model can not be divided by the dimention of k"
    self.d_k = d_model // num_heads

    self.w_q = nn.Linear(in_features=d_model, out_features=d_model)
    self.w_k = nn.Linear(in_features=d_model, out_features=d_model)
    self.w_v = nn.Linear(in_features=d_model, out_features=d_model)

    self.w_o = nn.Linear(in_features=d_model, out_features=d_model)
    self.dropout = nn.Dropout(dropout)

  @staticmethod
  def attention(query, key, value, mask, dropout: nn.Dropout):
    d_k = query.shape[-1]

    # (Batch, h, seq_len, d_k) --> (Batch, h, seq_len, seq_len)
    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
      mask = mask
      attention_scores.masked_fill_(mask == 0, -1e9)
    attention_scores = attention_scores.softmax(dim=-1) # (Batch, h, seq_len, seq_len)
    if dropout is not None:
      attention_scores = dropout(attention_scores)
    return (attention_scores @ value), attention_scores


  def forward(self, q: torch.tensor, k: torch.tensor, v: torch.tensor, mask=None):
    quary = self.w_q(q) # (Batch, seq_len, d_model)
    key = self.w_k(k) # (Batch, seq_len, d_model)
    value = self.w_v(v) # (Batch, seq_len, d_model)

    # (Batch, seq_len, d_model) --> (Batch, Seq_len, h, d_k) --> (Batch, Seq_len, d_k, h)
    quary = quary.view(quary.shape[0], quary.shape[1], self.num_heads, self.d_k).transpose(1, 2)
    key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
    value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

    x, self.attention_scores = MultiHeadAttention.attention(quary, key, value, mask, self.dropout)

    # (Batch,h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, h, d_model)
    x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)

    return self.w_o(x)
  
class ResidualConnection(nn.Module):
  def __init__(self, dropout: float) -> None:
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNormalization()

  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x))) # (Batch, seq_len, d_model)
  
class EncoderBlock(nn.Module):
  def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
    super().__init__()
    self.self_attention_block = self_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

  def forward(self, x, src_mask):
    x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
    x = self.residual_connection[1](x, self.feed_forward_block)
    return x
  
class Encoder(nn.Module):
  def __init__(self, layers: nn.Module):
    super().__init__()
    self.layers = layers
    self.norm = LayerNormalization()

  def forward(self, x, mask):
    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)
  
class DecoderBlock(nn.Module):
  def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
    super().__init__()
    self.self_attention_block = self_attention_block
    self.cross_attention_block = cross_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

  def forward(self, x, encoder_out, src_mask, tgt_mask):
    x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
    x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_out, encoder_out, src_mask))
    x = self.residual_connection[2](x, self.feed_forward_block)
    return x
  
class Decoder(nn.Module):
  def __init__(self, layers: nn.Module):
    super().__init__()
    self.layers = layers
    self.norm = LayerNormalization()

  def forward(self, x, encoder_out, src_mask, tgt_mask):
    for layer in self.layers:
      x = layer(x, encoder_out, src_mask, tgt_mask)
    return self.norm(x)
  
class ProjectionLayer(nn.Module):
  def __init__(self, d_model, vocab_size) -> None:
    super().__init__()
    self.proj = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
    return torch.log_softmax(self.proj(x), dim=-1)
  
class Transformer(nn.Module):
  def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, scr_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.src_pos = scr_pos
    self.tgt_pos = tgt_pos
    self.projection_layer = projection_layer

  def encode(self, src, src_mask):
    src = self.src_embed(src)
    src = self.src_pos(src)
    return self.encoder(src, src_mask)

  def decode(self, encoder_out, src_mask, tgt, tgt_mask):
    tgt = self.tgt_embed(tgt)
    tgt = self.tgt_pos(tgt)
    return self.decoder(tgt, encoder_out, src_mask, tgt_mask)

  def project(self, x):
    return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
  # Create the embedding layers
  src_embed = InputEmbedding(d_model, src_vocab_size)
  tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

  # Create the postional encoding layers
  scr_pos = PositionalEncoding(d_model, src_seq_len, dropout)
  tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

  # Create the encoder blocks
  encoder_blocks = []
  for _ in range(N):
    encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
    encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
    encoder_blocks.append(encoder_block)

  # Create the decoder blocks
  decoder_blocks = []
  for _ in range(N):
      decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
      decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
      feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
      decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
      decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
  encoder = Encoder(nn.ModuleList(encoder_blocks))
  decoder = Decoder(nn.ModuleList(decoder_blocks))

  # Create the transformer
  transformer = Transformer(encoder, decoder, src_embed, tgt_embed, scr_pos, tgt_pos, ProjectionLayer(d_model, tgt_vocab_size))

  # Initialize the parameters
  for p in transformer.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)

  return transformer