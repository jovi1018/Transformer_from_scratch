from custom_dataset import BilingualDataset

from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.data import random_split, DataLoader

from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="datasets")

def main():
  train_dataloader, val_dataloader, Tokenizer_src, Tokenizer_tgt = get_ds()
  print(f"The size of train dataset: {len(train_dataloader.dataset)}")
  print(f"The size of val dataset: {len(val_dataloader.dataset)}")
  print(f"The size of tokenizer src: {len(Tokenizer_src.get_vocab())}")
  print(f"The size of tokenizer tgt: {len(Tokenizer_tgt.get_vocab())}")

  print(f"Example of train dataset: {train_dataloader.dataset[0]}")

def get_all_sentences(ds, lang):
  for item in ds:
    yield item[lang]

def get_or_buid_tokenizer(ds, lang):
  Path("tokenizers").mkdir(parents=True, exist_ok=True)

  tokenizer_path = Path(f"tokenizers/{lang}_tokenizer.json")
  if not Path.exists(tokenizer_path):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[MASK]", "[SOS]", "[EOS]"], min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
  else:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  return tokenizer

def get_ds():
  ds_raw = load_dataset("zetavg/coct-en-zh-tw-translations-twp-300k", split='train')

  Tokenizer_src = get_or_buid_tokenizer(ds_raw, 'en')
  Tokenizer_tgt = get_or_buid_tokenizer(ds_raw, 'ch')  

  train_ds_size = int(0.9 * len(ds_raw))
  val_data_size = len(ds_raw) - train_ds_size
  train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_data_size])

  train_ds = BilingualDataset(train_ds_raw, Tokenizer_src, Tokenizer_tgt, 'en', 'ch', seq_len=128)
  val_ds = BilingualDataset(val_ds_raw, Tokenizer_src, Tokenizer_tgt, 'en', 'ch', seq_len=128)

  max_len_scr = 0
  max_len_tgt = 0

  for item in ds_raw:
    src_ids = Tokenizer_src.encode(item['en']).ids
    tgt_ids = Tokenizer_tgt.encode(item['ch']).ids
    max_len_scr = max(max_len_scr, len(src_ids))
    max_len_tgt = max(max_len_tgt, len(tgt_ids))

  print(f"Max length of source sentence: {max_len_scr}")
  print(f"Max length of target sentence: {max_len_tgt}")

  train_dataloader = DataLoader(train_ds, batch_size=8, shuffle=True)
  val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

  return train_dataloader, val_dataloader, Tokenizer_src, Tokenizer_tgt


if __name__ == "__main__":
    main()