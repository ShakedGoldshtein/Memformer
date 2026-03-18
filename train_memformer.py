"""
Train memristor-patched transformer on WikiText-2.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from memformer_model import get_memformer_model
from train import get_dataloaders, evaluate

MODEL_NAME = "memformer_model"

import memtorch.mn as memtorch_mn


# Cache of per-crossbar physical conductance bounds used to clamp stochastic updates efficiently.
_CROSSBAR_BOUNDS_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


@torch.no_grad()
def _prepare_stochastic_update_cache(model: torch.nn.Module) -> None:
    """
    Prepare per-crossbar (per-cell) conductance bounds once.

    This extracts Ron/Roff from each Python device object and converts them into
    tensor bounds g_min=1/Roff and g_max=1/Ron matching the conductance matrix shape.
    """
    _CROSSBAR_BOUNDS_CACHE.clear()
    for module in model.modules():
        if not isinstance(module, memtorch_mn.Linear):
            continue
        for cb in module.crossbars:
            devices_flat = np.array(cb.devices).flatten()
            r_off = np.array([d.r_off for d in devices_flat]).reshape(cb.conductance_matrix.shape)
            r_on = np.array([d.r_on for d in devices_flat]).reshape(cb.conductance_matrix.shape)
            device = cb.conductance_matrix.device
            dtype = cb.conductance_matrix.dtype
            g_min = torch.from_numpy(1.0 / r_off).to(device=device, dtype=dtype)
            g_max = torch.from_numpy(1.0 / r_on).to(device=device, dtype=dtype)
            _CROSSBAR_BOUNDS_CACHE[id(cb)] = (g_min, g_max)


@torch.no_grad()
def _stochastic_programming_step(
    model: torch.nn.Module,
    lr: float,
    n_conductance_states: int | None,
    writing_noise_std: float,
) -> None:
    """
    Apply Babu-style stochastic updates directly to each crossbar conductance matrix.

    We treat the dense gradient on conductance (dL/dg) as producing a desired update Δg = -lr * dL/dg.
    Then we perform a probabilistic state transition with step size Δg_min:
      p = |Δg| / Δg_min (clipped to [0, 1])
      if U(0,1) < p: g ← g + sign(Δg) * Δg_min

    Additive write noise is applied to Δg before sampling and programming.
    """
    if lr <= 0:
        return
    for module in model.modules():
        if not isinstance(module, memtorch_mn.Linear):
            continue
        for cb in module.crossbars:
            g = cb.conductance_matrix
            if g.grad is None:
                continue

            # Desired continuous update in conductance space
            delta = -float(lr) * g.grad

            # Additive write noise on the update (programming variability)
            if writing_noise_std and writing_noise_std > 0.0:
                delta = delta + torch.randn_like(delta) * float(writing_noise_std)

            # Determine Δg_min as one discrete state step (if finite states are modeled)
            cb_id = id(cb)
            bounds = _CROSSBAR_BOUNDS_CACHE.get(cb_id)
            if bounds is None:
                continue
            g_min, g_max = bounds
            if n_conductance_states is not None and int(n_conductance_states) > 1:
                delta_min = (g_max - g_min) / float(int(n_conductance_states) - 1)
            else:
                # Fallback: smallest meaningful step as a tiny fraction of dynamic range
                delta_min = 1e-3 * (g_max - g_min)

            # Probability of a state transition
            p = (delta.abs() / (delta_min + 1e-12)).clamp_(0.0, 1.0)
            u = torch.rand_like(p)
            mask = u < p

            # Program one-step updates where sampled
            step = torch.sign(delta) * delta_min
            g.add_(step * mask.to(g.dtype))

            # Clamp per-cell to physical limits and sync crossbar
            g.copy_(torch.max(torch.min(g, g_max), g_min))
            cb.update(from_devices=False)


@torch.no_grad()
def _update_duty_cycled_readout_masks(model: torch.nn.Module) -> None:
    """
    After a write event, latch a duty-cycled single-ended readout state.

    For DoubleColumn weights (G+ and G- crossbars), we compute:
      p = G_pos / (G_pos + G_neg)
    Then we sample a Bernoulli mask per cell:
      with probability p: keep G_pos active, clamp G_neg to g_min
      with probability 1-p: keep G_neg active, clamp G_pos to g_min

    The "clamped" side is only disabled for readout between writes; the true
    conductance state is preserved by restoring conductances after each forward.
    """
    eps = 1e-12
    for module in model.modules():
        if not isinstance(module, memtorch_mn.Linear):
            continue

        cbs = list(getattr(module, "crossbars", []))
        # DoubleColumn uses pairs (pos, neg) in order.
        for i in range(0, len(cbs) - 1, 2):
            cb_pos = cbs[i]
            cb_neg = cbs[i + 1]
            g_pos = cb_pos.conductance_matrix
            g_neg = cb_neg.conductance_matrix

            p = (g_pos / (g_pos + g_neg + eps)).clamp_(0.0, 1.0)
            keep_pos = torch.rand_like(p) < p
            keep_neg = ~keep_pos

            bounds_pos = _CROSSBAR_BOUNDS_CACHE.get(id(cb_pos))
            bounds_neg = _CROSSBAR_BOUNDS_CACHE.get(id(cb_neg))
            if bounds_pos is None or bounds_neg is None:
                continue
            g_min_pos, _ = bounds_pos
            g_min_neg, _ = bounds_neg

            # These tensors are consumed by NoisyMemristorLayer in memformer_model.py
            cb_pos._duty_keep = keep_pos
            cb_neg._duty_keep = keep_neg
            cb_pos._duty_gmin = g_min_pos
            cb_neg._duty_gmin = g_min_neg


@torch.no_grad()
def _enable_conductance_grads(model: torch.nn.Module) -> None:
    """
    Enable autograd on crossbar conductance matrices so we can compute dL/dg.

    This is required for "no shadow" stochastic programming, because MemTorch Linear weights
    are not standard trainable parameters in the patched model.
    """
    for module in model.modules():
        if not isinstance(module, memtorch_mn.Linear):
            continue
        for cb in module.crossbars:
            cb.conductance_matrix.requires_grad_(True)
            if cb.conductance_matrix.grad is not None:
                cb.conductance_matrix.grad.zero_()


def train_epoch_memformer(
    model,
    loader,
    optimizer,
    device,
    causal_mask,
    pad_idx,
    lr,
    n_conductance_states,
    writing_noise_std,
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

        # Enable gradients on conductance matrices so backward computes dL/dg
        _enable_conductance_grads(model)

        logits = model(x, mask=causal_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=pad_idx,
        )
        loss.backward()

        # Update regular (digital) parameters (e.g., embeddings, layer norms, etc.)
        optimizer.step()

        # Apply stochastic programming update to memristor conductances
        _stochastic_programming_step(
            model=model,
            lr=float(lr),
            n_conductance_states=n_conductance_states,
            writing_noise_std=float(writing_noise_std),
        )

        # Latch duty-cycled G+/G- readout state until the next write.
        _update_duty_cycled_readout_masks(model)

        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean().item()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        log_dict = {
            "train_loss": loss.item(),
            "train_accuracy": acc,
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

    # Default hyperparameters (can be overridden by W&B sweeps)
    default_config = dict(
        epochs=3,
        batch_size=64,
        block_size=128,
        max_vocab=15000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
        learning_rate=3e-4,
        weight_decay=0.1,
        dropout=0.3,
        dataset="WikiText-2",
        ADC_resolution=8,
        n_conductance_states=None,
        noise_scale=0.0,
        t_min=1e-3,
        non_linearity=1.0,
        device_variation=0.0,
        # Additive write noise (standard deviation) applied to stochastic updates in conductance space.
        writing_noise_std=0.0,
    )

    wandb.init(
        project="Memristor-LM",
        config=default_config,
    )
    cfg = wandb.config

    batch_size = cfg.batch_size
    block_size = cfg.block_size
    max_vocab = cfg.max_vocab
    d_model = cfg.d_model
    n_layers = cfg.n_layers
    n_heads = cfg.n_heads
    d_ff = cfg.d_ff
    lr = cfg.learning_rate
    weight_decay = cfg.weight_decay
    epochs = cfg.epochs

    adc_resolution = cfg.ADC_resolution
    n_conductance_states = cfg.n_conductance_states
    noise_scale = cfg.noise_scale
    non_linearity = cfg.non_linearity
    device_variation = cfg.device_variation
    read_noise_std = 0.1 * float(noise_scale)
    writing_noise_std = float(getattr(cfg, "writing_noise_std", 0.0))

    print("Loading data....")
    train_loader, val_loader, test_loader, vocab_size, pad_idx = get_dataloaders(
        batch_size=batch_size,
        block_size=block_size,
        max_vocab=max_vocab,
    )
    print(f"Vocab size: {vocab_size}")

    print("Building memristor-patched model (no shadow)...")
    model = get_memformer_model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=block_size,
        dropout=0.3,
        pad_idx=pad_idx,
        memristor_model_params=None,
        patch_kwargs={"ADC_resolution": adc_resolution},
        n_conductance_states=n_conductance_states,
        read_noise_std=read_noise_std,
        non_linearity=non_linearity,
        device_variation=device_variation,
    )
    model = model.to(device)
    # Precompute per-crossbar physical bounds once (Ron/Roff -> conductance limits)
    _prepare_stochastic_update_cache(model)
    causal_mask = model.get_causal_mask(block_size, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    eval_every_steps = 1000
    global_step = [0]
    early_state = {
        "best_val_loss": float("inf"),
        "best_state": None,
        "bad_evals": 0,
        "stop": False,
    }
    patience_evals = 3

    for epoch in range(epochs):
        train_loss = train_epoch_memformer(
            model,
            train_loader,
            optimizer,
            device,
            causal_mask,
            pad_idx,
            lr=lr,
            n_conductance_states=n_conductance_states,
            writing_noise_std=writing_noise_std,
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

