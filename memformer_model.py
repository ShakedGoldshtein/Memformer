"""
Memformer (memristor-only) model builder.

Wraps the base `TransformerLM` and patches it with MemTorch so that linear
layers are implemented as memristive crossbars (VTEAM device model by default).

This module is self-contained and does NOT use any digital shadow weights:
gradients are computed with PyTorch autograd and optimizers update the memristor
layer parameters directly.
"""
import torch
import torch.nn as nn
from model import TransformerLM
from memtorch.mn.Module import patch_model
from memtorch.bh import Scheme
from memtorch.bh.memristor.VTEAM import VTEAM
from memtorch.bh.nonideality.NonIdeality import NonIdeality, apply_nonidealities
import memtorch.mn as memtorch_mn


# VTEAM: r_off/r_on=100, realistic thresholds, time_series_resolution for faster sim (ohms, s, V)
DEFAULT_MEMRISTOR_PARAMS = {
    "time_series_resolution": 1e-6,
    "r_off": 100_000,
    "r_on": 1_000,
    "d": 3e-9,
    "k_on": -10,
    "k_off": 5e-4,
    "alpha_on": 3,
    "alpha_off": 3,
    "v_on": -0.3,
    "v_off": 0.3,
    "x_on": 0,
    "x_off": 3e-9,
}

DEFAULT_PATCH_KWARGS = {
    "ADC_resolution": 8,
    "quant_method": "linear",
    "ADC_overflow_rate": 0.0,
    "verbose": False,
    "scheme": Scheme.DoubleColumn,
}


class NoisyMemristorLayer(nn.Module):
    def __init__(self, layer: nn.Module, read_noise_std: float) -> None:
        super().__init__()
        self.layer = layer
        self.read_noise_std = read_noise_std

    def forward(self, *args, **kwargs):
        # Apply duty-cycled single-ended readout (if enabled by the trainer).
        # We temporarily modify each crossbar's conductance matrix for the duration
        # of this forward pass, then restore the original conductances so the
        # underlying device state is preserved.
        saved = None
        if isinstance(self.layer, memtorch_mn.Linear):
            saved = []
            for cb in getattr(self.layer, "crossbars", []):
                g = cb.conductance_matrix
                g_saved = g.detach().clone()
                saved.append((cb, g_saved))
                keep = getattr(cb, "_duty_keep", None)
                g_min = getattr(cb, "_duty_gmin", None)
                if keep is not None and g_min is not None:
                    cb.conductance_matrix.copy_(torch.where(keep, g_saved, g_min))

        y = self.layer(*args, **kwargs)

        # Restore original conductances.
        if saved is not None:
            for cb, g_saved in saved:
                cb.conductance_matrix.copy_(g_saved)

        if self.read_noise_std > 0.0 and isinstance(y, torch.Tensor):
            # Read noise is modeled as additive output noise (e.g., ADC/sensing noise).
            noise = torch.randn_like(y) * self.read_noise_std
            y = y + noise
        return y


def _wrap_memristor_linear_layers(module: nn.Module, read_noise_std: float) -> None:
    for name, child in list(module.named_children()):
        _wrap_memristor_linear_layers(child, read_noise_std)
        if isinstance(child, memtorch_mn.Linear):
            setattr(module, name, NoisyMemristorLayer(child, read_noise_std))


def get_memformer_model(
    vocab_size,
    d_model=256,
    n_layers=4,
    n_heads=4,
    d_ff=1024,
    max_seq_len=256,
    dropout=0.2,
    pad_idx=0,
    checkpoint_path=None,
    memristor_model=VTEAM,
    memristor_model_params=None,
    patch_kwargs=None,
    n_conductance_states=None,
    read_noise_std: float = 0.0,
    non_linearity=None,
    device_variation=None,
):
    """
    Build TransformerLM, optionally load weights, then patch it with MemTorch
    so weights are represented as memristors (VTEAM by default).

    Returns:
        model: patched memristor model (forward(ids, mask=None), get_causal_mask).
    """
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        pad_idx=pad_idx,
    )
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)

    params = memristor_model_params if memristor_model_params is not None else DEFAULT_MEMRISTOR_PARAMS.copy()
    kwargs = {**DEFAULT_PATCH_KWARGS, **(patch_kwargs or {})}

    patched = patch_model(
        model,
        memristor_model,
        params,
        **kwargs,
    )

    if n_conductance_states is not None:
        patched = apply_nonidealities(
            patched,
            [NonIdeality.FiniteConductanceStates],
            conductance_states=n_conductance_states,
        )

    if non_linearity is not None:
        patched = apply_nonidealities(
            patched,
            [NonIdeality.NonLinear],
            sweep_duration=1.0,
            sweep_voltage_signal_amplitude=float(non_linearity),
            sweep_voltage_signal_frequency=1.0,
        )

    if device_variation is not None and device_variation > 0.0:
        p = float(device_variation) / 3.0
        patched = apply_nonidealities(
            patched,
            [NonIdeality.DeviceFaults],
            lrs_proportion=p,
            hrs_proportion=p,
            electroform_proportion=p,
        )

    if read_noise_std > 0.0:
        # Read noise is injected in the forward path to emulate ADC / sensing noise.
        _wrap_memristor_linear_layers(patched, read_noise_std)

    # Ensure MemTorch layers use crossbar-based computation (not the legacy digital matmul path).
    for m in patched.modules():
        if isinstance(m, memtorch_mn.Linear):
            m.forward_legacy_enabled = False

    return patched

