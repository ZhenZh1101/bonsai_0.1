"""Optimizers and the one-curve S0 -> S1 -> S2 schedule (plan section 8).

Default (V1 / nanochat): Muon on every 2-D matrix inside the blocks (layers.N.self_attn.{q,k,v,o}_proj,
layers.N.mlp.{gate,up,down}_proj), LR = matrix_lr * (768 / d)^0.5, momentum warmup 0.85 -> 0.95 over 300 steps,
no weight decay; AdamW (betas 0.9 / 0.95, eps 1e-8, no weight decay) on the tied embedding and, as its own group,
on every RMSNorm weight (input / post-attention / final norm and the per-head q_norm / k_norm).

Norm-weight LR (`norm_lr`, default 1e-3): Adam moves every parameter by about lr per step whatever its gradient
scale, and an RMSNorm weight is a gain that starts at 1, so lr is its relative change per step. 1e-3 is the LR
that AdamW-trained Qwen / Llama-style models give their norm gains (the Qwen-style alternative below uses 1.2e-3
at d640 for everything), lets a gain move by O(1) over the run, and keeps the q/k-norm gains, which multiply the
attention logits (logit ~ g_q * g_k) and are the only way those logits can grow without a softcap (plan section 7),
from jumping. It is not width-scaled (a gain has no fan-in).

Qwen-style alternative (`optimizer="adamw"`, plan section 8 table): AdamW on all parameters, betas (0.9, 0.95),
eps 1e-8, weight decay 0.1 on 2-D weights only (incl. the embedding), peak LR 1.5e-3 at d576 / 1.2e-3 at d640,
linear in between and ~1/d outside.

One run, one LR curve (`RunPlan`): S0 = the first s0_frac of the steps (hard switch to S1), S2 = the last s2_frac
(LR decays linearly to 0 over it), a 50/50 S1/S2 transition of `transition_steps` right before S2; warmup from
step 0 (S0 is mostly at peak). `schedule="cosine"` replaces the constant + linear decay by first-generation Qwen's
cosine to 10% of the peak (data phases unchanged).

LR and Muon momentum live in the optimizer state as arrays, so the schedule changes them every step without
recompiling the compiled update.
"""

import math
from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

DEFAULT_TC = {
    "optimizer": "muon",     # muon (Muon + AdamW) | adamw (Qwen style)
    "matrix_lr": 0.02,       # Muon LR at width 768, scaled by (768/d)^0.5
    "embed_lr": 1e-3,        # AdamW, tied embedding (lm_head too if untied)
    "norm_lr": 1e-3,         # AdamW, every RMSNorm weight (see the module docstring)
    "muon_momentum": 0.95,
    "adamw_lr": None,        # Qwen-style peak LR; None = by width (adamw_peak_lr)
    "weight_decay": 0.1,     # Qwen-style, 2-D weights only
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_eps": 1e-8,
}


class Muon(optim.Optimizer):
    """Nesterov momentum + Newton-Schulz orthogonalisation (Keller Jordan / nanochat), as pagoda.

    Momentum is part of the state (so the warmup does not recompile); no weight decay."""

    def __init__(self, learning_rate, momentum=0.95, ns_steps=5):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.state["momentum"] = mx.array(momentum, dtype=mx.float32)
        self.ns_steps = ns_steps

    def init_single(self, parameter, state):
        state["v"] = mx.zeros_like(parameter)

    @staticmethod
    def newton_schulz(X, steps):
        a, b, c = (3.4445, -4.7750, 2.0315)
        transpose = X.shape[0] > X.shape[1]
        if transpose:
            X = X.T
        X = X / (mx.linalg.norm(X) + 1e-7)
        for _ in range(steps):
            A = X @ X.T
            B = mx.addmm(b * A, A, A, beta=1.0, alpha=c)
            X = mx.addmm(a * X, B, X, beta=1.0, alpha=1.0)
        return X.T if transpose else X

    def apply_single(self, gradient, parameter, state):
        mu = self.state["momentum"]
        v = mu * state["v"] + (1 - mu) * gradient
        state["v"] = v
        update = self.newton_schulz((1 - mu) * gradient + mu * v, self.ns_steps)  # Nesterov
        scale = max(1.0, parameter.shape[0] / parameter.shape[1]) ** 0.5
        return parameter - (self.learning_rate * scale).astype(parameter.dtype) * update


def is_muon_param(path, p):
    return path.startswith("layers.") and p.ndim == 2


def is_norm_param(path, p):
    return p.ndim == 1  # the only 1-D parameters are RMSNorm weights


def adamw_peak_lr(d):
    """Plan section 8: 1.5e-3 at d576, 1.2e-3 at d640, linear in between, ~1/d outside."""
    if d <= 576:
        return 1.5e-3 * 576 / d
    if d >= 640:
        return 1.2e-3 * 640 / d
    return 1.5e-3 + (d - 576) * (1.2e-3 - 1.5e-3) / 64


def resolve_tc(tc, d_model):
    """DEFAULT_TC updated with `tc`, plus the resolved per-group peak LRs (saved in checkpoints)."""
    tc = dict(tc or {})
    if "embed_lr" not in tc and "embed_lr_tied" in tc:  # pagoda's key (a port of pagoda's sft.py)
        tc["embed_lr"] = tc["embed_lr_tied"]
    tc = {**DEFAULT_TC, **{k: v for k, v in tc.items() if v is not None}}
    if tc["optimizer"] not in ("muon", "adamw"):
        raise ValueError(f"unknown optimizer {tc['optimizer']!r}")
    if tc["optimizer"] == "muon":
        tc["groups"] = {"muon": tc["matrix_lr"] * (768 / d_model) ** 0.5, "norm": tc["norm_lr"],
                        "embed": tc["embed_lr"]}
    else:
        lr = tc["adamw_lr"] or adamw_peak_lr(d_model)
        tc["adamw_lr"] = lr
        tc["groups"] = {"decay": lr, "no_decay": lr}
    return tc


def scale_tc(tc, frac):
    """The same optimizer at `frac` x every peak LR (SFT / distillation: 'LR = 0.15 x the pretraining peak')."""
    out = {**tc, "groups": {k: v * frac for k, v in tc["groups"].items()}}
    for k in ("matrix_lr", "embed_lr", "norm_lr", "adamw_lr"):
        if out.get(k) is not None:
            out[k] = out[k] * frac
    return out


def build_optimizer(model, tc=None):
    """Returns (MultiOptimizer, {group: peak lr}) in the order of opt.optimizers. `tc`: see DEFAULT_TC; a dict
    already resolved by resolve_tc / scale_tc is used as is (its "groups")."""
    tc = tc if tc and "groups" in tc else resolve_tc(tc, model.cfg.d_model)
    groups = dict(tc["groups"])
    betas, eps = (tc["adam_beta1"], tc["adam_beta2"]), tc["adam_eps"]

    def adamw(lr, wd):
        return optim.AdamW(learning_rate=lr, betas=betas, eps=eps, weight_decay=wd, bias_correction=True)

    if tc["optimizer"] == "muon":
        parts = [Muon(groups["muon"], momentum=tc["muon_momentum"]), adamw(groups["norm"], 0.0),
                 adamw(groups["embed"], 0.0)]
        opt = optim.MultiOptimizer(parts, [is_muon_param, is_norm_param])
    else:
        parts = [adamw(groups["decay"], tc["weight_decay"]), adamw(groups["no_decay"], 0.0)]
        opt = optim.MultiOptimizer(parts, [lambda path, p: p.ndim == 2])
    opt.init(model.trainable_parameters())
    return opt, groups


def param_groups(model, opt, groups):
    """{group: [parameter paths]} as the MultiOptimizer routes them (tests, logging)."""
    out = {g: [] for g in groups}
    names = list(groups)
    for path, p in tree_flatten(model.trainable_parameters()):
        for i, fn in enumerate(opt.filters):
            if fn(path, p):
                out[names[i]].append(path)
                break
    return out


def set_lr(opt, groups, multiplier):
    for o, base in zip(opt.optimizers, groups.values()):
        o.learning_rate = base * multiplier


def set_muon_momentum(opt, value):
    for o in opt.optimizers:
        if isinstance(o, Muon):
            o.state["momentum"] = mx.array(value, dtype=mx.float32)


def muon_momentum(step, warmup=300, start=0.85, end=0.95):
    """nanochat's momentum warmup 0.85 -> 0.95 over the first 300 steps."""
    frac = min(step / warmup, 1.0) if warmup else 1.0
    return (1 - frac) * start + frac * end


@dataclass(frozen=True)
class RunPlan:
    """Step boundaries of one pretraining run: [0, s0_end) S0, [s0_end, transition) S1, [transition, decay) 50/50
    S1/S2, [decay, total) S2 + LR decay. Without S2 (s2_frac 0) transition == decay == total."""

    total: int
    s0_end: int
    transition: int
    decay: int
    warmup: int = 300
    schedule: str = "wsd"

    @classmethod
    def build(cls, total, s0_frac=0.05, s2_frac=0.25, transition_steps=200, warmup=300, schedule="wsd"):
        if schedule not in ("wsd", "cosine"):
            raise ValueError(f"unknown schedule {schedule!r}")
        s0_end = int(round(total * s0_frac))
        decay = total - int(round(total * s2_frac))
        if not 0 <= s0_end <= decay <= total:
            raise ValueError(f"S0 ({s0_frac}) and S2 ({s2_frac}) do not fit in {total} steps")
        transition = max(s0_end, decay - transition_steps) if decay < total else total
        return cls(total, s0_end, transition, decay, warmup, schedule)

    def phase(self, step):
        if step < self.s0_end:
            return "s0"
        if step >= self.decay:
            return "s2"
        return "transition" if step >= self.transition else "s1"

    def lr(self, step):
        """LR multiplier (x the group's peak) for optimizer step `step` (0-based)."""
        warm = (step + 1) / self.warmup if step < self.warmup else 1.0
        if self.schedule == "cosine":
            p = min(1.0, max(0, step - self.warmup) / max(1, self.total - self.warmup))
            return min(warm, 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
        if step < self.decay:
            return warm
        return min(warm, max(0.0, (self.total - step) / max(1, self.total - self.decay)))

    def phase_tokens(self, tokens_per_step, s2_share=0.5):
        """Tokens each data phase consumes over the whole run, {"s0", "s1", "s2"}; the 50/50 transition counts
        in S1 and S2 (`s2_share` of its rows come from S2)."""
        mix = (self.decay - self.transition) * tokens_per_step
        return {"s0": self.s0_end * tokens_per_step,
                "s1": (self.transition - self.s0_end) * tokens_per_step + (1 - s2_share) * mix,
                "s2": (self.total - self.decay) * tokens_per_step + s2_share * mix}

    def to_dict(self):
        return asdict(self)

    def describe(self):
        return (f"{self.total} steps: S0 [0, {self.s0_end}), S1 [{self.s0_end}, {self.transition}), "
                f"S1/S2 50/50 [{self.transition}, {self.decay}), S2 + decay [{self.decay}, {self.total}); "
                f"warmup {self.warmup}, {self.schedule}")


def linear_warmup_decay(step, total_steps, warmup_steps):
    """SFT schedule: linear warmup then linear decay to 0."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def global_grad_norm(grads):
    return mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for _, g in tree_flatten(grads)))


def clip_by_global_norm(grads, max_norm):
    norm = global_grad_norm(grads)
    scale = mx.minimum(1.0, max_norm / (norm + 1e-6))
    return tree_map(lambda g: g * scale, grads), norm


def round_steps(tokens, tokens_per_step):
    return max(1, int(math.ceil(tokens / tokens_per_step)))
