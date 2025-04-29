from pathlib import Path

def get_config():
  return {
      "batch_size": 8,
      "num_epochs": 20,
      "lr": 10**-4,
      "seq_len": 128,
      "d_model": 512,
      "lang_src": "en",
      "lang_tgt": "ch",
      "model_folder": "weights",
      "model_basename": "tmodel_",
      "preload": None,
      "tokenizer_file": "tokenizer_{}.json",
      "experiment_name": "runs/tmodel"
  }

def get_weights_file_path(self, epoch: str):
  model_folder = f"{self.config['model_folder']}"
  model_basename = f"{self.config['model_basename']}"
  model_filename = f"{model_basename}{epoch}.pt"
  return str(Path('.') / model_folder /model_filename)