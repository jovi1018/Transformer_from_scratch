import torch
from torch import nn
from tqdm.auto import tqdm

from config import get_config, get_weights_file_path
from custom_dataset import get_data, causal_mask
from custom_dataset import get_model

from torch.utils.tensorboard import SummaryWriter
import warnings as warning
from pathlib import Path
from tqdm import tqdm

def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
  sos_idx = tokenizer_tgt.token_to_id("[SOS]")
  eos_idx = tokenizer_tgt.token_to_id("[EOS]")

  # Precompute the encoder output and reuse it for evey token we get from the decoder
  encoder_output = model.encode(source, source_mask)

  # Initialize the decoder input with the sos token
  decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

  while True:
    if decoder_input.size(1) == max_len:
      break

    # Build the target mask for the decoder
    decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

    # calculate the output of the ecoder
    out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

    # get the next token
    prob = model.project(out[:, -1])

    # Selecte the token with the max probability (because it is a greedy search)
    _, next_word = torch.max(prob, dim=1)
    decoder_input = torch.cat(
        [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
    )

    if next_word == eos_idx:
      break

  return decoder_input.squeeze(0)

def run_validation(model, validation_dataloader, tokenizer_tgt, max_len, device, print_msg, global_status, writer, num_examples=2):
  model.eval()
  count = 0

  console_width = 80

  with torch.inference_mode():
    for batch in validation_dataloader:
      count += 1
      encoder_input = batch["encoder_input"].to(device)
      encoder_mask = batch["encoder_mask"].to(device)

      assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
      model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

      source_text = batch["src_text"][0]
      target_text = batch["tgt_text"][0]
      model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

      # Print the source, target, and model output
      print_msg(f"-"*console_width)
      print_msg(f"Source: {source_text}")
      print_msg(f"Target: {target_text}")
      print_msg(f"Predicted: {model_out_text}")

      if count == num_examples:
        break

def train(config):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device {device}")

  Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

  train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_data(config)
  model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

  writer = SummaryWriter(config['experiment_name'])

  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

  initial_epoch = 0
  global_step = 0
  if config['preload']:
    model_filename = get_weights_file_path(config, config['preload'])
    print(f"Preloading model {model_filename}")
    state = torch.load(model_filename)
    initial_epoch = state['epoch'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step = state['global_step']

  loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

  for epoch in range(initial_epoch, config['num_epochs']):
    
    batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
    for batch in batch_iterator:
      model.train()

      encoder_input = batch["encoder_input"].to(device)
      decoder_input = batch["decoder_input"].to(device)
      encoder_mask = batch["encoder_mask"].to(device)
      decoder_mask = batch["decoder_mask"].to(device)

      encoder_output = model.encode(encoder_input, encoder_mask)
      decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
      proj_output = model.project(decoder_output)

      label = batch["label"].to(device)

      loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
      batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

      writer.add_scalar("train_loss", loss.item(), global_step)
      writer.flush()

      loss.backward()

      optimizer.step()
      optimizer.zero_grad(set_to_none=True)

      global_step += 1

    run_validation(model, val_dataloader, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }, model_filename)


if __name__ == "__main__":
  config = get_config()
  warning.filterwarnings("ignore", category=UserWarning, module="torch")
  train(config)