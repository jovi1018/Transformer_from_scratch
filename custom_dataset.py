import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

from model import build_transformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

def causal_mask(size):
  mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
  return mask == 0

class BilingualDataset(Dataset):
  def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
    super().__init__()
    self.dataset = dataset
    self.tokenizer_src = tokenizer_src
    self.tokenizer_tgt = tokenizer_tgt
    self.src_lang = src_lang
    self.tgt_lang = tgt_lang
    self.seq_len = seq_len
    

    self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
    self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
    self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    src_target_pair = self.dataset[index]
    src_text = src_target_pair[self.src_lang]
    tgt_text = src_target_pair[self.tgt_lang]

    enc_input_tokens = self.tokenizer_src.encode(src_text).ids
    dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

    enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
    dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

    if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
      raise ValueError("Sentence is too long")

    enc_input = torch.cat(
        [
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ]
    )

    dec_input = torch.cat(
        [
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ]
    )

    label = torch.cat(
        [
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ]
    )

    assert enc_input.size(0) == self.seq_len
    assert dec_input.size(0) == self.seq_len
    assert label.size(0) == self.seq_len

    return {
        "encoder_input": enc_input,
        "decoder_input": dec_input,
        # For model from scratch
        "encoder_mask": (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
        "decoder_mask": (dec_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(dec_input.size(0)),
        "label": label,
        "src_text": src_text,
        "tgt_text": tgt_text
    }
  
def get_all_sentences(dataset, lang):
  for item in dataset:
    yield item[lang]

def get_or_build_tokenizer(dataset, lang):
  tokenizer_path = Path(f"tokenizer_{lang}.json")
  if not Path.exists(tokenizer_path):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[MASK]", "[SOS]", "[EOS]"], min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
  else:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  return tokenizer

def get_data(config):
  ds_raw = load_dataset("zetavg/coct-en-zh-tw-translations-twp-300k", split='train')

  Tokenizer_src = get_or_build_tokenizer(ds_raw, config['lang_src'])
  Tokenizer_tgt = get_or_build_tokenizer(ds_raw, config['lang_tgt'])

  train_ds_size = int(0.9 * len(ds_raw))
  val_data_size = len(ds_raw) - train_ds_size
  train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_data_size])

  train_ds = BilingualDataset(train_ds_raw, Tokenizer_src, Tokenizer_tgt, config['lang_src'], config['lang_tgt'], seq_len=config['seq_len'])
  val_ds = BilingualDataset(val_ds_raw, Tokenizer_src, Tokenizer_tgt, config['lang_src'], config['lang_tgt'], seq_len=config['seq_len'])

  max_len_scr = 0
  max_len_tgt = 0

  for item in ds_raw:
    src_ids = Tokenizer_src.encode(item[config['lang_src']]).ids
    tgt_ids = Tokenizer_tgt.encode(item[config['lang_tgt']]).ids
    max_len_scr = max(max_len_scr, len(src_ids))
    max_len_tgt = max(max_len_tgt, len(tgt_ids))

  print(f"Max length of source sentence: {max_len_scr}")
  print(f"Max length of target sentence: {max_len_tgt}")

  train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
  val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

  return train_dataloader, val_dataloader, Tokenizer_src, Tokenizer_tgt

def get_model(config, vocab_scr_len, vocab_tgt_len):
  model = build_transformer(vocab_scr_len, vocab_tgt_len, config['seq_len'], config['seq_len'], d_model=config['d_model'])
  return model