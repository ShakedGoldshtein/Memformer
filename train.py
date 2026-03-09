"""
Train transformer on WikiText-2: tokenizer, data, training loop.
"""
import math
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

from load_data import get_wikitext2
from model import TransformerLM


# Model name: base_model now; change to e.g. memristor_model for other variants
MODEL_NAME = "base_model"

# Vocabulary and simple tokenizer
PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"


def build_vocab(train_ds, max_vocab=15000):
    """Build vocabulary from train text."""
    from collections import Counter
    counter = Counter()
    for line in train_ds["text"]:
        if not line.strip():
            continue
        for w in line.strip().split():
            counter[w] += 1
    special = [PAD, UNK, EOS]
    vocab_list = special + [w for w, _ in counter.most_common(max_vocab - len(special))]
    return {w: i for i, w in enumerate(vocab_list)}


def tokenize(text, vocab):
    """Convert a text line to a list of token ids."""
    return [vocab.get(w, vocab[UNK]) for w in text.strip().split() if text.strip()]


def get_all_tokens(ds, vocab):
    """Return a single list of all token ids (concatenating all lines)."""
    out = []
    for line in ds["text"]:
        if not line.strip():
            continue
        out.extend(tokenize(line, vocab))
        out.append(vocab[EOS])
    return out


class LMDataset(Dataset):
    """Dataset that yields blocks of length block_size (input + next token)."""
    def __init__(self, token_ids, block_size):
        self.block_size = block_size
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.n = max(0, len(self.data) - block_size)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        chunk = self.data[i : i + self.block_size + 1]
        x = chunk[:-1]   # input
        y = chunk[1:]    # target (next token)
        return x, y


def get_dataloaders(batch_size=32, block_size=128, max_vocab=15000):
    """Load data, build vocabulary, return train, validation, and test DataLoaders."""
    train_ds = get_wikitext2("train")
    val_ds = get_wikitext2("validation")
    test_ds = get_wikitext2("test")
    vocab = build_vocab(train_ds, max_vocab=max_vocab)
    vocab_size = len(vocab)
    pad_idx = vocab[PAD]

    train_tokens = get_all_tokens(train_ds, vocab)
    val_tokens = get_all_tokens(val_ds, vocab)
    test_tokens = get_all_tokens(test_ds, vocab)

    train_dataset = LMDataset(train_tokens, block_size)
    val_dataset = LMDataset(val_tokens, block_size)
    test_dataset = LMDataset(test_tokens, block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    return train_loader, val_loader, test_loader, vocab_size, pad_idx


def train_epoch(model, loader, optimizer, device, causal_mask, pad_idx):
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, mask=causal_mask)
        # logits: (B, T, V), y: (B, T)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=pad_idx,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean().item()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        wandb.log({"train_loss": loss.item(), "train_accuracy": acc})
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")
    return total_loss / n if n else 0.0


@torch.no_grad()
def evaluate(model, loader, device, causal_mask, pad_idx):
    """Single pass: returns (perplexity, accuracy, avg_loss)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    n_tokens = 0
    n_samples = 0
    for x, y in tqdm(loader, desc="Evaluating", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x, mask=causal_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=pad_idx,
        )
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=-1)
        total_correct += (pred == y).sum().item()
        n_tokens += y.numel()
        n_samples += x.size(0)
    if n_samples == 0:
        return float("inf"), 0.0, float("inf")
    avg_loss = total_loss / n_samples
    ppl = math.exp(avg_loss)
    acc = total_correct / n_tokens
    return ppl, acc, avg_loss


def main():
    # Load wandb API key from project file if not set
    if not os.environ.get("WANDB_API_KEY"):
        key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".wandb_key")
        if os.path.isfile(key_file):
            with open(key_file) as f:
                os.environ["WANDB_API_KEY"] = f.read().strip()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    batch_size = 64
    block_size = 128
    max_vocab = 15000
    d_model = 256
    n_layers = 4
    n_heads = 8
    d_ff = 1024
    lr = 3e-4
    weight_decay =  0.1
    epochs = 3

    print("Loading data....")
    train_loader, val_loader, test_loader, vocab_size, pad_idx = get_dataloaders(
        batch_size=batch_size,
        block_size=block_size,
        max_vocab=max_vocab,
    )
    print(f"Vocab size: {vocab_size}")

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=block_size,
        dropout=0.3,
        pad_idx=pad_idx,
    ).to(device)
    causal_mask = model.get_causal_mask(block_size, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    wandb.init(
        project="Memristor-LM",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "block_size": block_size,
            "max_vocab": max_vocab,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "dropout": 0.2,
            "dataset": "WikiText-2",
        },
    )

    # Early stopping: stop if val_loss doesn't improve for `patience` epochs
    patience = 2
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, causal_mask, pad_idx)
        ppl, val_acc, val_loss = evaluate(model, val_loader, device, causal_mask, pad_idx)
        wandb.log({
            "train_loss_epoch": train_loss,
            "perplexity": ppl,
            "val_accuracy": val_acc,
            "val_loss": val_loss,
        })
        print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_perplexity={ppl:.2f}  val_accuracy={val_acc:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping: no improvement for {patience} epochs.")
                break

    # Restore best model and save in base_model_weights/<run_id>/
    if best_state is not None:
        model.load_state_dict(best_state)
    weights_dir = f"{MODEL_NAME}_weights"
    run_name = wandb.run.id if wandb.run is not None else "manual_run"
    run_dir = os.path.join(weights_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    weights_path = os.path.join(run_dir, "model_weights.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")

    # Final evaluation on test set
    test_ppl, test_acc, test_loss = evaluate(model, test_loader, device, causal_mask, pad_idx)
    wandb.log({
        "test_perplexity": test_ppl,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
    })
    print(f"Test  perplexity={test_ppl:.2f}  accuracy={test_acc:.4f}  loss={test_loss:.4f}")

    wandb.finish()
    print("Training done.")


if __name__ == "__main__":
    main()
