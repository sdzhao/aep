"""Reverse-mode cotangent pass: f(x), tangent derivatives J T and C = J J^T.

J has 3 rows, so it is propagated as 3 cotangents per cell, written as explicit tensor ops so
one ordinary backward pass differentiates the whole loss (no jacrev, no double backward).
"""
import torch
from torch import nn

ROUTES = ("auto", "explicit", "factorized")


def select_route(d, h, route="auto"):
    """'explicit' (J = U1 W1 formed) iff d <= h under 'auto'; otherwise 'factorized'
    (M = W1 W1^T, J not formed). Forming C costs ~3hd per cell explicitly vs ~3h^2 factorized."""
    if route not in ROUTES:
        raise ValueError(f"unknown jacobian_route {route!r}")
    if route == "auto":
        return "explicit" if d <= h else "factorized"
    return route


def _forward(net, x):
    """Forward pass; returns (Y, Linear layers, SiLU derivatives after each hidden Linear)."""
    layers = net.net if hasattr(net, "net") else net
    linears, dsig = [], []
    H = x
    for layer in layers:
        if isinstance(layer, nn.Linear):
            H = layer(H)
            linears.append(layer)
        elif isinstance(layer, nn.SiLU):
            s = torch.sigmoid(H)
            dsig.append(s * (1 + H * (1 - s)))  # silu'(z) from the cached sigmoid
            H = H * s
        else:
            raise TypeError(f"unsupported layer {type(layer).__name__}")
    if len(dsig) != len(linears) - 1:
        raise TypeError("expected alternating Linear and SiLU layers")
    return H, linears, dsig


def cotangents(net, x):
    """Returns (Y (B, 3), U1 (B, 3, h), W1 (h, d)) with J_i = U1_i W1.

    U_L = W_L * silu'(z_{L-1}), U_l = (U_{l+1} W_l) * silu'(z_{l-1}): one batched GEMM per layer.
    """
    Y, linears, dsig = _forward(net, x)
    U = torch.eye(Y.shape[1], dtype=Y.dtype).expand(len(x), -1, -1)
    for layer, ds in zip(linears[:0:-1], dsig[::-1]):
        U = (U @ layer.weight) * ds[:, None, :]
    return Y, U, linears[0].weight


def jacobian(net, x):
    """Explicit J (B, 3, d) = U1 W1 (tests and diagnostics)."""
    _, U, W1 = cotangents(net, x)
    return U @ W1


def jacobian_terms(net, x, T, route="auto"):
    """Returns (Y (B, 3), m = J T (B, 3, 2), C = J J^T (B, 3, 3)); T (B, d, 2) tangent bases.

    explicit:   J = U1 W1, m = J T, C = J J^T.
    factorized: M = W1 W1^T (once per step), P = W1 T, m = U1 P, C = U1 M U1^T.
    """
    Y, U, W1 = cotangents(net, x)
    if select_route(W1.shape[1], W1.shape[0], route) == "explicit":
        J = U @ W1
        return Y, J @ T, J @ J.transpose(1, 2)
    M = W1 @ W1.T
    return Y, U @ (W1 @ T), U @ M @ U.transpose(1, 2)
