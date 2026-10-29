"""CUDA implementation of the frozen DPCA and R3 detection statistics."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping
import numpy as np


def cuda_r3_predictions(iq: np.ndarray, physics: Mapping[str, Any], r3: Mapping[str, Any]) -> dict[str, Any]:
    """Compute B3 and both R3 scores in complex128 on CUDA, preserving median semantics."""
    import torch
    from radardistill.d2a_baselines import dpca_time_shift_s
    values = np.asarray(iq)
    if values.ndim != 4 or values.shape[1] < 2 or not np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError('IQ must be a finite complex batch')
    if not torch.cuda.is_available():
        raise RuntimeError('R6-A CUDA statistics require CUDA')
    opts = r3['detection']
    t1, t2 = opts['t1_global_robust_background'], opts['t2_local_cfar_like']
    if any(x['doppler_boundary'] != 'circular' or x['range_boundary'] != 'clipped' for x in (t1, t2)):
        raise ValueError('unsupported boundary configuration')
    n, r = values.shape[2:]
    prf = float(physics['radar']['prf_hz'])
    x = torch.as_tensor(values, dtype=torch.complex128, device='cuda:0')
    phase = torch.exp(2j * torch.pi * torch.fft.fftfreq(n, device=x.device, dtype=torch.float64) * (dpca_time_shift_s(dict(physics)) * prf))
    residual = x[:, 0] - torch.fft.ifft(torch.fft.fft(x[:, 1], dim=1) * phase[None, :, None], dim=1)
    power = torch.abs(torch.fft.fftshift(torch.fft.fft(residual, dim=1), dim=1)) ** 2
    flat = power.reshape(len(values), -1).argmax(dim=1)
    backgrounds = []
    dr, rr = int(t1['peak_guard_doppler_radius_cells']), int(t1['peak_guard_range_radius_cells'])
    for i in range(len(values)):
        d, gate = int(flat[i] // r), int(flat[i] % r)
        mask = torch.ones((n, r), device=x.device, dtype=torch.bool)
        ds = (torch.arange(-dr, dr + 1, device=x.device) + d) % n
        rs = torch.arange(max(0, gate - rr), min(r, gate + rr + 1), device=x.device)
        mask[ds[:, None], rs[None, :]] = False
        backgrounds.append(torch.quantile(power[i][mask], .5))
    background = torch.stack(backgrounds)
    if torch.any(background <= 0):
        raise ValueError('nonpositive global background')
    shifts = []
    dw, rw = int(t2['doppler_window_width_cells']), int(t2['range_window_width_cells'])
    dg, rg = int(t2['doppler_guard_width_cells']), int(t2['range_guard_width_cells'])
    if any(w <= 0 or w % 2 == 0 for w in (dw, rw, dg, rg)) or dg >= dw or rg >= rw:
        raise ValueError('invalid local CFAR window')
    for d in range(-dw // 2 + 1, dw // 2 + 1):
        for offset in range(-rw // 2 + 1, rw // 2 + 1):
            if abs(d) <= dg // 2 and abs(offset) <= rg // 2:
                continue
            shifted = torch.roll(power, shifts=(-d, -offset), dims=(1, 2))
            if offset > 0:
                shifted[:, :, -offset:] = torch.nan
            elif offset < 0:
                shifted[:, :, :-offset] = torch.nan
            shifts.append(shifted)
    references = torch.stack(shifts, dim=-1)
    counts = torch.isfinite(references).sum(dim=-1)
    if torch.any(counts < int(t2['min_reference_cells'])):
        raise ValueError('insufficient local reference cells')
    local = torch.nanquantile(references, .5, dim=-1)
    if torch.any(~torch.isfinite(local)) or torch.any(local <= 0):
        raise ValueError('invalid local background')
    maps = {'B3': power, 'R3_T1': power / background[:, None, None], 'R3_T2': power / local}
    frequency_axis = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / prf))
    result = {}
    for name, score_map in maps.items():
        peaks = score_map.reshape(len(values), -1).argmax(dim=1)
        score = score_map.reshape(len(values), -1).gather(1, peaks[:, None]).squeeze(1).cpu().numpy()
        index = peaks.cpu().numpy()
        result[name] = SimpleNamespace(score=score, range_cell=(index % r).astype(float), doppler_hz=frequency_axis[index // r])
    return result
