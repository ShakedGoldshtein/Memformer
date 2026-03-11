"""
Train memristor-patched transformer on WikiText-2.
"""
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from memristor_model import get_memristor_model
from train import get_dataloaders, evaluate

MODEL_NAME = "memristor_model"


def train_epoch_memristor(
    model,
    loader,
    optimizer,
    shadow_mgr,
    device,
    causal_mask,
    pad_idx,
    val_loader=None,
    eval_every_steps=None,
    global_step=None,
    early_state=None,
    patience_evals=None,
):
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y in pbar:
        if global_step is not None:
            global_step[0] += 1
            step = global_step[0]
        else:
            step = None

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x, mask=causal_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=pad_idx,
        )
        loss.backward()

        # Move gradients from device weights to shadow weights, then step optimizer
        shadow_mgr.copy_grads_from_device()
        optimizer.step()
        shadow_mgr.optimizer_steps += 1
        shadow_mgr.sync()

        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean().item()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        log_dict = {
            "train_loss": loss.item(),
            "train_accuracy": acc,
            "write_efficiency": shadow_mgr.write_efficiency,
        }
        if step is not None:
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)

        if (
            val_loader is not None
            and eval_every_steps is not None
            and global_step is not None
            and step is not None
            and step % eval_every_steps == 0
        ):
            ppl, val_acc, val_loss = evaluate(model, val_loader, device, causal_mask, pad_idx)
            wandb.log(
                {
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "perplexity": ppl,
                },
                step=step,
            )
            if early_state is not None and patience_evals is not None:
                if val_loss < early_state["best_val_loss"]:
                    early_state["best_val_loss"] = val_loss
                    early_state["best_state"] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    early_state["bad_evals"] = 0
                else:
                    early_state["bad_evals"] += 1
                    if early_state["bad_evals"] >= patience_evals:
                        early_state["stop"] = True
                        print(f"Early stopping: no val_loss improvement in {patience_evals} evaluations.")
                        break
            model.train()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{acc:.4f}",
            weff=f"{shadow_mgr.write_efficiency:.4f}",
        )

    return total_loss / n if n else 0.0


def main():
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
    n_heads = 4
    d_ff = 1024
    lr = 3e-4
    weight_decay = 0.1
    epochs = 3

    print("Loading data....")
    train_loader, val_loader, test_loader, vocab_size, pad_idx = get_dataloaders(
        batch_size=batch_size,
        block_size=block_size,
        max_vocab=max_vocab,
    )
    print(f"Vocab size: {vocab_size}")

    print("Building memristor-patched model...")
    model, shadow_mgr = get_memristor_model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=block_size,
        dropout=0.3,
        pad_idx=pad_idx,
    )
    model = model.to(device)
    causal_mask = model.get_causal_mask(block_size, device)
    optimizer = torch.optim.AdamW(shadow_mgr.shadow_params, lr=lr, weight_decay=weight_decay)

    wandb.init(
        project="Memristor-LM",
        config={
            "model": "memristor",
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
            "dropout": 0.3,
            "dataset": "WikiText-2",
        },
    )

    eval_every_steps = 1000
    global_step = [0]
    early_state = {
        "best_val_loss": float("inf"),
        "best_state": None,
        "bad_evals": 0,
        "stop": False,
    }
    patience_evals = 5

    for epoch in range(epochs):
        train_loss = train_epoch_memristor(
            model,
            train_loader,
            optimizer,
            shadow_mgr,
            device,
            causal_mask,
            pad_idx,
            val_loader=val_loader,
            eval_every_steps=eval_every_steps,
            global_step=global_step,
            early_state=early_state,
            patience_evals=patience_evals,
        )
        if early_state["stop"]:
            break

        ppl, val_acc, val_loss = evaluate(model, val_loader, device, causal_mask, pad_idx)
        wandb.log({
            "train_loss_epoch": train_loss,
            "perplexity": ppl,
            "val_accuracy": val_acc,
            "val_loss": val_loss,
        }, step=global_step[0])
        print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_perplexity={ppl:.2f}  val_accuracy={val_acc:.4f}  val_loss={val_loss:.4f}")

        if val_loss < early_state["best_val_loss"]:
            early_state["best_val_loss"] = val_loss
            early_state["best_state"] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            early_state["bad_evals"] = 0
        else:
            early_state["bad_evals"] += 1
            if early_state["bad_evals"] >= patience_evals:
                early_state["stop"] = True
                print(f"Early stopping: no val_loss improvement in {patience_evals} evaluations.")
                break

    if early_state["best_state"] is not None:
        model.load_state_dict(early_state["best_state"])
    weights_dir = f"{MODEL_NAME}_weights"
    run_name = wandb.run.id if wandb.run is not None else "manual_run"
    run_dir = os.path.join(weights_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    weights_path = os.path.join(run_dir, "model_weights.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")

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
