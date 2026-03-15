"""
Memristor variant: wraps the base TransformerLM and patches it with MemTorch
so that linear layers use memristive crossbars (VTEAM device model).

Parameter choices (for academic justification): r_off/r_on ratio 100 (common in RRAM);
realistic voltage thresholds (v_off, v_on); ADC 8-bit; optional finite conductance states.

This module also defines a shadow-weight manager used to implement a hybrid
analog–digital update scheme: optimizer updates high-precision shadow weights,
and device (memristor) weights are updated only when a state-aware threshold is
exceeded, with optional quantization to a finite number of conductance states.
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
        y = self.layer(*args, **kwargs)
        if self.read_noise_std > 0.0 and isinstance(y, torch.Tensor):
            noise = torch.randn_like(y) * self.read_noise_std * y.abs()
            y = y + noise
        return y


def _wrap_memristor_linear_layers(module: nn.Module, read_noise_std: float) -> None:
    for name, child in list(module.named_children()):
        _wrap_memristor_linear_layers(child, read_noise_std)
        if isinstance(child, memtorch_mn.Linear):
            setattr(module, name, NoisyMemristorLayer(child, read_noise_std))


class MemristorShadowManager:
    """
    Manages shadow (digital) weights for a memristor-patched model.

    - shadow_params: high-precision weights updated by the optimizer.
    - device_params: actual memristor-simulated weights inside the model.
    - copy_grads_from_device: copies gradients from device params to shadows.
    - sync: state-aware, thresholded, quantized update from shadows to device.
    """

    def __init__(
        self,
        model: nn.Module,
        n_conductance_states: int | None = None,
        t_min: float = 1e-3,
        t_max: float = 1e-2,
        write_noise_std: float = 0.0,
    ) -> None:
        self.model = model
        self.n_conductance_states = n_conductance_states
        self.t_min = t_min
        self.t_max = t_max
        self.write_noise_std = write_noise_std

        self._pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
        shadows = []
        # Create a shadow parameter for each trainable device parameter
        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            shadow = nn.Parameter(p.detach().clone().to(p.device).float())
            self._pairs.append((p, shadow))
            shadows.append(shadow)

        self.shadow_params = nn.ParameterList(shadows)
        self.optimizer_steps: int = 0
        self.device_updates: int = 0

    @torch.no_grad()
    def copy_grads_from_device(self) -> None:
        """
        Copy gradients accumulated on device parameters to the corresponding
        shadow parameters so that the optimizer can update the shadows.
        """
        for device_param, shadow_param in self._pairs:
            if device_param.grad is None:
                shadow_param.grad = None
            else:
                if shadow_param.grad is None:
                    shadow_param.grad = device_param.grad.detach().clone()
                else:
                    shadow_param.grad.copy_(device_param.grad.detach())
            # clear device gradients; only shadows are optimized
            device_param.grad = None

    @torch.no_grad()
    def sync(self) -> None:
        """
        Apply state-aware thresholding and quantized programming from
        shadow weights to device weights.

        For simplicity, the device weights are treated in their current
        numeric domain (post-mapping). Threshold is adaptive based on the
        magnitude of the current device weight: higher near saturation,
        lower near the centre.
        """
        updated_elems = 0

        for device_param, shadow_param in self._pairs:
            w_dev = device_param.data
            w_sh = shadow_param.data
            delta = w_sh - w_dev

            if self.n_conductance_states is None:
                # Simple global threshold
                mask = delta.abs() > self.t_min
                num = int(mask.sum().item())
                if num == 0:
                    continue
                w_target = w_sh[mask]
                if self.write_noise_std > 0.0:
                    noise = torch.randn_like(w_target) * self.write_noise_std * w_target.abs()
                    w_target = w_target + noise
                w_dev[mask] = w_target
                updated_elems += num
            else:
                # Adaptive threshold: higher near |w_dev| max, lower near centre
                max_abs = float(w_dev.abs().max().item())
                if max_abs == 0.0:
                    thresh = torch.full_like(w_dev, self.t_min)
                else:
                    s_norm = w_dev.abs() / max_abs  # in [0, 1]
                    thresh = self.t_min + (self.t_max - self.t_min) * (s_norm**2)

                mask = delta.abs() > thresh
                num = int(mask.sum().item())
                if num == 0:
                    continue

                # Target values from shadow
                w_target = w_sh[mask]
                if self.write_noise_std > 0.0:
                    noise = torch.randn_like(w_target) * self.write_noise_std * w_target.abs()
                    w_target = w_target + noise

                # Uniform quantization between current min/max device weights
                w_min = float(w_dev.min().item())
                w_max = float(w_dev.max().item())
                if w_max == w_min:
                    w_q = torch.full_like(w_target, w_min)
                else:
                    levels = max(int(self.n_conductance_states), 1)
                    step = (w_max - w_min) / max(levels - 1, 1)
                    k = torch.round((w_target - w_min) / step).clamp(0, levels - 1)
                    w_q = w_min + k * step

                w_dev[mask] = w_q
                updated_elems += num

        self.device_updates += updated_elems

    @property
    def write_efficiency(self) -> float:
        if self.optimizer_steps == 0:
            return 0.0
        return float(self.device_updates) / float(self.optimizer_steps)


def get_memristor_model(
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
        shadow_mgr: MemristorShadowManager instance managing shadow weights and
                    thresholded, quantized programming of device weights.

    If n_conductance_states is set (e.g. 16 or 32 for ~4–5 bit), applies finite
    conductance states nonideality after patching and uses this number of
    states for the shadow manager's quantization.
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
        _wrap_memristor_linear_layers(patched, read_noise_std)

    shadow_mgr = MemristorShadowManager(
        patched,
        n_conductance_states=n_conductance_states,
    )

    return patched, shadow_mgr

