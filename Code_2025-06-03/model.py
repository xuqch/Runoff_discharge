from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class EALSTMCell(nn.Module):
    def __init__(self, dyn_dim: int, hidden_dim: int):
        super().__init__()
        self.lin_x = nn.Linear(dyn_dim, 3 * hidden_dim, bias=True)
        self.lin_h = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)

        with torch.no_grad():
            self.lin_x.bias[:hidden_dim].fill_(1.0)

    def forward(
        self,
        x_t: Optional[torch.Tensor],
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        i_gate: torch.Tensor,
        gates_x_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if gates_x_t is None:
            if x_t is None:
                raise ValueError("Either x_t or gates_x_t must be provided.")
            gates_x_t = self.lin_x(x_t)
        gates = gates_x_t + self.lin_h(h_prev)
        f_t, o_t, g_t = gates.chunk(3, dim=-1)
        f_t = torch.sigmoid(f_t)
        o_t = torch.sigmoid(o_t)
        g_t = torch.tanh(g_t)
        c_t = f_t * c_prev + i_gate * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class EALSTM(nn.Module):
    def __init__(
        self,
        dyn_dim: int,
        stat_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        precompute_inputs: bool = True,
        precompute_time_chunk: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.precompute_inputs = bool(precompute_inputs)
        self.precompute_time_chunk = int(precompute_time_chunk)
        self.static_gate = nn.Linear(stat_dim, hidden_dim, bias=True)
        self.cell = EALSTMCell(dyn_dim=dyn_dim, hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _precompute_lin_x(self, x_dyn: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.precompute_inputs:
            return None

        chunk = int(self.precompute_time_chunk)
        if chunk <= 0 or chunk >= x_dyn.shape[1]:
            return self.cell.lin_x(x_dyn)

        pieces: List[torch.Tensor] = []
        for start in range(0, x_dyn.shape[1], chunk):
            end = min(start + chunk, x_dyn.shape[1])
            pieces.append(self.cell.lin_x(x_dyn[:, start:end, :]))
        return torch.cat(pieces, dim=1)

    def forward(
        self,
        x_dyn: torch.Tensor,
        x_stat: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
        return_sequence: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x_dyn.shape
        i_gate = torch.sigmoid(self.static_gate(x_stat))
        gates_x_all = self._precompute_lin_x(x_dyn)

        if state is None:
            h = x_dyn.new_zeros(batch_size, self.hidden_dim)
            c = x_dyn.new_zeros(batch_size, self.hidden_dim)
        else:
            h, c = state

        hs: Optional[List[torch.Tensor]] = [] if return_sequence else None
        h_last: Optional[torch.Tensor] = None
        for t in range(seq_len):
            gates_x_t = gates_x_all[:, t, :] if gates_x_all is not None else None
            x_t = None if gates_x_t is not None else x_dyn[:, t, :]
            h, c = self.cell(x_t, h, c, i_gate, gates_x_t=gates_x_t)
            h_drop = self.dropout(h)
            if return_sequence:
                hs.append(h_drop)
            h_last = h_drop

        assert h_last is not None
        out = torch.stack(hs, dim=1) if return_sequence else h_last
        if return_state:
            return out, (h, c)
        return out, None


class PixelWiseRunoffGenerator(nn.Module):
    def __init__(
        self,
        dyn_dim: int,
        stat_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.4,
        head_hidden: int = 32,
        precompute_inputs: bool = True,
        precompute_time_chunk: int = 0,
    ):
        super().__init__()
        self.ealstm = EALSTM(
            dyn_dim=dyn_dim,
            stat_dim=stat_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            precompute_inputs=precompute_inputs,
            precompute_time_chunk=precompute_time_chunk,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        x_dyn: torch.Tensor,
        x_stat: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        h_last, out_state = self.ealstm(
            x_dyn,
            x_stat,
            state=state,
            return_state=return_state,
            return_sequence=False,
        )
        return self.head(h_last), out_state

    def forward_sequence(
        self,
        x_dyn: torch.Tensor,
        x_stat: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        h_seq, out_state = self.ealstm(
            x_dyn,
            x_stat,
            state=state,
            return_state=return_state,
            return_sequence=True,
        )
        return self.head(h_seq), out_state

class DifferentiableRoutingLayer(nn.Module):
    def __init__(self, init_velocity_mps: float = 1.0):
        super().__init__()
        self.velocity_param = nn.Parameter(torch.tensor(float(init_velocity_mps)))

    def forward(
            self,
            runoff_m3s: torch.Tensor,
            dist_map_m: torch.Tensor,
            max_lag: int = 60,
            sigma: float = 0.5,
            pixel_chunk: int = 200_000,
    ) -> torch.Tensor:
        """Route pixel runoff time series to outlet discharge.

        Parameters
        ----------
        runoff_m3s : torch.Tensor
            Shape (P, N). N is the number of target days in the chosen date range.
        dist_map_m : torch.Tensor
            Shape (P,)
        Returns
        -------
        torch.Tensor
            Shape (N, 1)
        """
        p_count, n_count = runoff_m3s.shape
        device = runoff_m3s.device
        dtype = runoff_m3s.dtype

        velocity = F.softplus(self.velocity_param) + 0.1
        lags = (dist_map_m / velocity) / 86400.0
        lag_slots = torch.arange(max_lag, device=device, dtype=dtype)
        weights = torch.exp(-((lag_slots[None, :] - lags[:, None]) ** 2) / (2.0 * sigma ** 2 + 1e-6))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        runoff_np = runoff_m3s.transpose(0, 1).contiguous()
        routed = torch.zeros(n_count, max_lag, device=device, dtype=dtype)
        for s in range(0, p_count, pixel_chunk):
            e = min(p_count, s + pixel_chunk)
            routed = routed + runoff_np[:, s:e] @ weights[s:e, :]

        q = torch.zeros(n_count, device=device, dtype=dtype)
        for k in range(max_lag):
            flow_k = routed[:, k]
            if k == 0:
                q = q + flow_k
            else:
                q[k:] = q[k:] + flow_k[:-k]
        return q.unsqueeze(-1)


class PhysicallyGuidedRoutingLayer(nn.Module):
    """
    Physically guided differentiable routing with:
      1) pixel-dependent channel velocity parameterization
      2) distance-dependent sigma
      3) separated hillslope lag + channel lag
      4) adaptive max_lag
      5) regularization/constraints for stable weak supervision

    Inputs
    ------
    runoff_m3s   : [P, T]
    dist_map_m   : [P]
    pixel_static : [P, S]

    Outputs
    -------
    q_out        : [T, 1]
    aux          : dict of routing diagnostics
    """

    def __init__(
            self,
            static_dim: int,
            hidden_dim: int = 32,
            vmin_mps: float = 0.3,
            vmax_mps: float = 3.5,
            hillslope_max_days: float = 3.0,
            sigma_min_days: float = 0.35,
            sigma_max_days: float = 8.0,
            max_lag_limit: int = 120,
            kernel_trunc_k: float = 3.0,
    ):
        super().__init__()

        self.vmin_mps = float(vmin_mps)
        self.vmax_mps = float(vmax_mps)
        self.hillslope_max_days = float(hillslope_max_days)
        self.sigma_min_days = float(sigma_min_days)
        self.sigma_max_days = float(sigma_max_days)
        self.max_lag_limit = int(max_lag_limit)
        self.kernel_trunc_k = float(kernel_trunc_k)

        # channel velocity head: uses static embedding + normalized distance
        self.velocity_head = nn.Sequential(
            nn.Linear(static_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # hillslope lag head: only uses static embedding
        self.hillslope_head = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # sigma scale head: static-dependent multiplicative factor
        self.sigma_scale_head = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # optional global residual parameters for initialization flexibility
        self.global_velocity_bias = nn.Parameter(torch.tensor(0.0))
        self.global_hillslope_bias = nn.Parameter(torch.tensor(0.0))
        self.global_sigma_bias = nn.Parameter(torch.tensor(0.0))

    def _normalize_dist(self, dist_map_m: torch.Tensor) -> torch.Tensor:
        # shape: [P]
        d = dist_map_m
        d_mean = d.mean()
        d_std = d.std().clamp_min(1e-6)
        return (d - d_mean) / d_std

    def _predict_velocity_mps(
            self,
            pixel_static: torch.Tensor,
            dist_map_m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict pixel-dependent channel velocity in m/s.
        Range constrained to [vmin_mps, vmax_mps].
        """
        d_norm = self._normalize_dist(dist_map_m).unsqueeze(-1)  # [P, 1]
        x = torch.cat([pixel_static, d_norm], dim=-1)
        raw = self.velocity_head(x).squeeze(-1) + self.global_velocity_bias
        gate = torch.sigmoid(raw)
        velocity = self.vmin_mps + (self.vmax_mps - self.vmin_mps) * gate
        '''把速度严格压在 [0.3, 3.5] m/s。原因是弱监督下如果不加边界，routing 会很容易跑出极端速度，去补偿 generator 误差。'''
        return velocity  # [P]

    def _predict_hillslope_lag_days(
            self,
            pixel_static: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict hillslope concentration lag in days.
        Range constrained to [0, hillslope_max_days].
        """
        raw = self.hillslope_head(pixel_static).squeeze(-1) + self.global_hillslope_bias
        lag = self.hillslope_max_days * torch.sigmoid(raw)
        return lag  # [P]

    def _predict_sigma_days(
            self,
            pixel_static: torch.Tensor,
            channel_lag_days: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sigma increases with travel time.
        sigma = sigma_min + positive_scale * sqrt(channel_lag)
        then clipped softly to sigma_max_days.
        """
        raw_scale = self.sigma_scale_head(pixel_static).squeeze(-1) + self.global_sigma_bias
        positive_scale = F.softplus(raw_scale) + 1e-4

        sigma = self.sigma_min_days + positive_scale * torch.sqrt(channel_lag_days.clamp_min(1e-6))
        sigma = torch.clamp(sigma, min=self.sigma_min_days, max=self.sigma_max_days)
        return sigma  # [P]

    def _build_kernel(
            self,
            total_lag_days: torch.Tensor,
            sigma_days: torch.Tensor,
            max_lag: int,
            dtype: torch.dtype,
            device: torch.device,
    ) -> torch.Tensor:
        """
        Gaussian-like lag kernel centered at total_lag_days with width sigma_days.
        Kernel is mass-normalized for each pixel.
        Returns [P, L]
        """
        lag_slots = torch.arange(max_lag, device=device, dtype=dtype)[None, :]  # [1, L]
        mu = total_lag_days[:, None]  # [P, 1]
        sig = sigma_days[:, None].clamp_min(1e-4)  # [P, 1]

        weights = torch.exp(-0.5 * ((lag_slots - mu) / sig) ** 2)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return weights  # [P, L]

    def _adaptive_max_lag(
            self,
            total_lag_days: torch.Tensor,
            sigma_days: torch.Tensor,
            runoff_len: int,
            user_max_lag: Optional[int] = None,
    ) -> int:
        """
        Adaptive lag window:
            max_lag ~ max(total_lag + k*sigma) + margin
        then clipped by max_lag_limit and runoff_len.
        """
        if user_max_lag is not None and user_max_lag > 0:
            return int(min(user_max_lag, self.max_lag_limit, max(2, runoff_len)))

        effective_tail = total_lag_days + self.kernel_trunc_k * sigma_days
        max_needed = int(torch.ceil(effective_tail.max()).item()) + 2
        max_needed = max(2, max_needed)
        max_needed = min(max_needed, self.max_lag_limit, runoff_len)
        return max_needed

    def forward(
            self,
            runoff_m3s: torch.Tensor,  # [P, T]
            dist_map_m: torch.Tensor,  # [P]
            pixel_static: torch.Tensor,  # [P, S]
            max_lag: Optional[int] = None,
            pixel_chunk: int = 200_000,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Route pixel runoff time series to outlet discharge.
        """
        p_count, t_count = runoff_m3s.shape
        device = runoff_m3s.device
        dtype = runoff_m3s.dtype

        # ---- 1) channel velocity ----
        '''每个像元都有自己的传播速度：
        利用 AlphaEarth embedding 学习静态地表/下垫面差异
        加入 dist 的归一化项，让网络感知像元在流域中的相对位置'''
        velocity_mps = self._predict_velocity_mps(pixel_static, dist_map_m)  # [P]

        # ---- 2) channel lag ----
        channel_lag_days = (dist_map_m / velocity_mps.clamp_min(1e-6)) / 86400.0  # [P]

        # ---- 3) hillslope lag ----
        hillslope_lag_days = self._predict_hillslope_lag_days(pixel_static)  # [P]

        # ---- 4) total lag ----
        total_lag_days = hillslope_lag_days + channel_lag_days  # [P]

        # ---- 5) sigma depending on channel travel time ----
        '''传播距离越远，响应核越宽；不同静态 embedding 还可调节这个扩散程度。'''
        sigma_days = self._predict_sigma_days(pixel_static, channel_lag_days)  # [P]

        # ---- 6) adaptive max lag ----
        lag_len = self._adaptive_max_lag(total_lag_days, sigma_days, runoff_len=t_count, user_max_lag=max_lag)

        # ---- 7) kernel ----
        weights = self._build_kernel(
            total_lag_days=total_lag_days,
            sigma_days=sigma_days,
            max_lag=lag_len,
            dtype=dtype,
            device=device,
        )  # [P, L]

        # ---- 8) pixel aggregation ----
        # runoff_m3s: [P, T] -> [T, P]
        runoff_tp = runoff_m3s.transpose(0, 1).contiguous()  # [T, P]

        routed = torch.zeros(t_count, lag_len, device=device, dtype=dtype)  # [T, L]
        for s in range(0, p_count, pixel_chunk):
            e = min(p_count, s + pixel_chunk)
            routed = routed + runoff_tp[:, s:e] @ weights[s:e, :]  # [T, L]

        # ---- 9) causal shift-and-sum ----
        q = torch.zeros(t_count, device=device, dtype=dtype)
        for k in range(lag_len):
            flow_k = routed[:, k]
            if k == 0:
                q = q + flow_k
            else:
                q[k:] = q[k:] + flow_k[:-k]

        aux = {
            "velocity_mps": velocity_mps,
            "channel_lag_days": channel_lag_days,
            "hillslope_lag_days": hillslope_lag_days,
            "total_lag_days": total_lag_days,
            "sigma_days": sigma_days,
            "lag_len": torch.tensor(float(lag_len), device=device, dtype=dtype),
            "kernel_mass_error": (weights.sum(dim=1) - 1.0).abs().mean(),
        }
        return q.unsqueeze(-1), aux

    def regularization(
            self,
            dist_map_m: torch.Tensor,
            aux: Dict[str, torch.Tensor],
            runoff_m3s: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Routing constraints / regularization terms.
        Returns scalar losses in a dict.
        """

        velocity = aux["velocity_mps"]  # [P]
        sigma = aux["sigma_days"]  # [P]
        hill = aux["hillslope_lag_days"]  # [P]
        channel = aux["channel_lag_days"]  # [P]

        losses: Dict[str, torch.Tensor] = {}

        # 1) mass conservation: kernel per-pixel should sum to 1
        losses["route_mass"] = aux["kernel_mass_error"]

        # 2) sigma should generally increase with distance
        # sort by distance and penalize negative sigma differences
        d_sorted, idx = torch.sort(dist_map_m)
        sigma_sorted = sigma[idx]
        sigma_diff = sigma_sorted[1:] - sigma_sorted[:-1]
        losses["route_sigma_monotonic"] = F.relu(-sigma_diff).mean()

        # 3) hillslope lag should not dominate channel lag for far pixels
        # allow local hillslope delay, but discourage absurdly large hillslope ratios
        ratio = hill / (channel.clamp_min(0.25))
        losses["route_hillslope_ratio"] = F.relu(ratio - 1.0).mean()

        # 4) avoid overly noisy pixel-to-pixel velocity variations
        # use distance-sorted roughness as a weak smoothness prior
        v_sorted = velocity[idx]
        v_diff = v_sorted[1:] - v_sorted[:-1]
        losses["route_velocity_smooth"] = (v_diff ** 2).mean()

        # 5) avoid excessively wide kernels everywhere
        losses["route_sigma_width"] = sigma.mean()

        # 6) optional storage-like damping regularization:
        # keep very large velocity + very large sigma combinations under control
        losses["route_dispersion_coupling"] = (velocity * sigma).mean()

        # 7) optional runoff-weighted effective lag regularization
        if runoff_m3s is not None:
            runoff_weight = runoff_m3s.mean(dim=1).detach()  # [P]
            runoff_weight = runoff_weight / (runoff_weight.sum() + 1e-8)
            eff_lag = (runoff_weight * (hill + channel)).sum()
            losses["route_effective_lag"] = eff_lag

        return losses

class DifferentiableMuskingumRoutingLayer(nn.Module):
    """
    Differentiable Muskingum routing.

    Pixel runoff is routed with a Muskingum linear reservoir:

        O_t = C0 I_t + C1 I_{t-1} + C2 O_{t-1}

    The coefficients are parameterized from learnable velocity and X heads. K is
    tied to travel time and softly lower-bounded so the daily-step coefficients
    stay non-negative. The recurrence is represented as a finite impulse
    response kernel, which keeps the efficient pixel aggregation pattern used by
    the previous routing layer while remaining fully differentiable.
    """

    def __init__(
        self,
        static_dim: int,
        hidden_dim: int = 32,
        vmin_mps: float = 0.3,
        vmax_mps: float = 3.5,
        x_max: float = 0.49,
        dt_days: float = 1.0,
        storage_factor_min: float = 0.5,
        storage_factor_max: float = 2.0,
        max_lag_limit: int = 120,
        kernel_tail_eps: float = 1e-4,
    ):
        super().__init__()
        self.vmin_mps = float(vmin_mps)
        self.vmax_mps = float(vmax_mps)
        self.x_max = float(x_max)
        self.dt_days = float(dt_days)
        self.storage_factor_min = float(storage_factor_min)
        self.storage_factor_max = float(storage_factor_max)
        self.max_lag_limit = int(max_lag_limit)
        self.kernel_tail_eps = float(kernel_tail_eps)

        self.velocity_head = nn.Sequential(
            nn.Linear(static_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.storage_scale_head = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.x_head = nn.Sequential(
            nn.Linear(static_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.global_velocity_bias = nn.Parameter(torch.tensor(0.0))
        self.global_storage_bias = nn.Parameter(torch.tensor(0.0))
        self.global_x_bias = nn.Parameter(torch.tensor(0.0))

    def _normalize_dist(self, dist_map_m: torch.Tensor) -> torch.Tensor:
        d_mean = dist_map_m.mean()
        d_std = dist_map_m.std(unbiased=False).clamp_min(1e-6)
        return (dist_map_m - d_mean) / d_std

    @staticmethod
    def _smooth_lower_bound(values: torch.Tensor, lower: torch.Tensor, beta: float = 5.0) -> torch.Tensor:
        return lower + F.softplus((values - lower) * beta) / beta

    def _predict_velocity_mps(self, pixel_static: torch.Tensor, dist_map_m: torch.Tensor) -> torch.Tensor:
        d_norm = self._normalize_dist(dist_map_m).unsqueeze(-1)
        x = torch.cat([pixel_static, d_norm], dim=-1)
        raw = self.velocity_head(x).squeeze(-1) + self.global_velocity_bias
        gate = torch.sigmoid(raw)
        return self.vmin_mps + (self.vmax_mps - self.vmin_mps) * gate

    def _predict_storage_days(
        self,
        pixel_static: torch.Tensor,
        dist_map_m: torch.Tensor,
        velocity_mps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hydraulic_lag_days = (dist_map_m / velocity_mps.clamp_min(1e-6)) / 86400.0
        raw_factor = self.storage_scale_head(pixel_static).squeeze(-1) + self.global_storage_bias
        factor_gate = torch.sigmoid(raw_factor)
        storage_factor = self.storage_factor_min + (
            self.storage_factor_max - self.storage_factor_min
        ) * factor_gate

        k_raw = hydraulic_lag_days * storage_factor
        k_min = hydraulic_lag_days.new_tensor(0.5 * self.dt_days + 1e-4)
        k_days = self._smooth_lower_bound(k_raw, k_min)
        return hydraulic_lag_days, k_days, storage_factor

    def _predict_muskingum_x(
        self,
        pixel_static: torch.Tensor,
        dist_map_m: torch.Tensor,
        k_days: torch.Tensor,
    ) -> torch.Tensor:
        d_norm = self._normalize_dist(dist_map_m).unsqueeze(-1)
        x_in = torch.cat([pixel_static, d_norm], dim=-1)
        raw = self.x_head(x_in).squeeze(-1) + self.global_x_bias

        half_dt_over_k = 0.5 * self.dt_days / k_days.clamp_min(1e-6)
        stable_cap = torch.minimum(half_dt_over_k, 1.0 - half_dt_over_k)
        stable_cap = torch.clamp(stable_cap, min=0.0, max=self.x_max)
        return stable_cap * torch.sigmoid(raw)

    def _muskingum_coefficients(
        self,
        k_days: torch.Tensor,
        x_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dt = k_days.new_tensor(self.dt_days)
        half_dt = 0.5 * dt
        denom = (k_days * (1.0 - x_weight) + half_dt).clamp_min(1e-8)

        c0 = (half_dt - k_days * x_weight) / denom
        c1 = (half_dt + k_days * x_weight) / denom
        c2 = (k_days * (1.0 - x_weight) - half_dt) / denom

        coeffs = torch.stack([c0, c1, c2], dim=0).clamp_min(0.0)
        coeffs = coeffs / (coeffs.sum(dim=0, keepdim=True) + 1e-8)
        return coeffs[0], coeffs[1], coeffs[2]

    def _adaptive_max_lag(
        self,
        c2: torch.Tensor,
        runoff_len: int,
        user_max_lag: Optional[int] = None,
    ) -> int:
        if user_max_lag is not None and user_max_lag > 0:
            max_allowed = min(self.max_lag_limit, max(2, runoff_len))
            return int(max(2, min(user_max_lag, max_allowed)))

        c2_safe = c2.detach().clamp(0.0, 1.0 - 1e-6)
        eps = torch.full_like(c2_safe, self.kernel_tail_eps)
        tail_steps = torch.log(eps) / torch.log(c2_safe.clamp_min(1e-6))
        tail_steps = torch.where(c2_safe <= 1e-6, torch.ones_like(tail_steps), tail_steps)
        max_needed = int(torch.ceil(tail_steps.max()).item()) + 2
        return int(min(max(max_needed, 2), self.max_lag_limit, max(2, runoff_len)))

    def _build_muskingum_kernel(
        self,
        c0: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        max_lag: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_values: List[torch.Tensor] = [c0]
        if max_lag > 1:
            h_prev = c1 + c2 * c0
            h_values.append(h_prev)
            for _ in range(2, max_lag):
                h_prev = c2 * h_prev
                h_values.append(h_prev)

        weights = torch.stack(h_values, dim=1)
        raw_mass = weights.sum(dim=1)
        weights = weights / (raw_mass[:, None] + 1e-8)
        return weights, raw_mass

    @staticmethod
    def _kernel_moments(weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lag_len = weights.shape[1]
        lag_slots = torch.arange(lag_len, device=weights.device, dtype=weights.dtype)[None, :]
        mean_lag = (weights * lag_slots).sum(dim=1)
        variance = (weights * (lag_slots - mean_lag[:, None]) ** 2).sum(dim=1)
        width = torch.sqrt(variance.clamp_min(1e-8))
        return mean_lag, width

    @staticmethod
    def _causal_shift_sum(routed: torch.Tensor) -> torch.Tensor:
        t_count, lag_len = routed.shape
        q = torch.zeros(t_count, device=routed.device, dtype=routed.dtype)
        for k in range(lag_len):
            flow_k = routed[:, k]
            if k == 0:
                q = q + flow_k
            else:
                q[k:] = q[k:] + flow_k[:-k]
        return q

    def forward(
        self,
        runoff_m3s: torch.Tensor,
        dist_map_m: torch.Tensor,
        pixel_static: torch.Tensor,
        max_lag: Optional[int] = None,
        pixel_chunk: int = 200_000,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        p_count, t_count = runoff_m3s.shape
        device = runoff_m3s.device
        dtype = runoff_m3s.dtype

        velocity_mps = self._predict_velocity_mps(pixel_static, dist_map_m)
        hydraulic_lag_days, k_days, storage_factor = self._predict_storage_days(
            pixel_static=pixel_static,
            dist_map_m=dist_map_m,
            velocity_mps=velocity_mps,
        )
        muskingum_x = self._predict_muskingum_x(pixel_static, dist_map_m, k_days)
        c0, c1, c2 = self._muskingum_coefficients(k_days, muskingum_x)

        lag_len = self._adaptive_max_lag(c2=c2, runoff_len=t_count, user_max_lag=max_lag)
        weights, raw_mass = self._build_muskingum_kernel(c0, c1, c2, max_lag=lag_len)
        response_mean_lag, response_width = self._kernel_moments(weights)

        runoff_tp = runoff_m3s.transpose(0, 1).contiguous()
        routed = torch.zeros(t_count, lag_len, device=device, dtype=dtype)
        for start in range(0, p_count, pixel_chunk):
            end = min(p_count, start + pixel_chunk)
            routed = routed + runoff_tp[:, start:end] @ weights[start:end, :]

        q = self._causal_shift_sum(routed)
        aux = {
            "velocity_mps": velocity_mps,
            "channel_lag_days": hydraulic_lag_days,
            "hillslope_lag_days": torch.zeros_like(k_days),
            "total_lag_days": response_mean_lag,
            "sigma_days": response_width,
            "muskingum_k_days": k_days,
            "muskingum_x": muskingum_x,
            "muskingum_c0": c0,
            "muskingum_c1": c1,
            "muskingum_c2": c2,
            "muskingum_storage_factor": storage_factor,
            "lag_len": torch.tensor(float(lag_len), device=device, dtype=dtype),
            "kernel_mass_error": (1.0 - raw_mass).abs().mean(),
        }
        return q.unsqueeze(-1), aux

    def regularization(
        self,
        dist_map_m: torch.Tensor,
        aux: Dict[str, torch.Tensor],
        runoff_m3s: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        velocity = aux["velocity_mps"]
        response_width = aux["sigma_days"]
        effective_lag = aux["total_lag_days"]
        muskingum_x = aux["muskingum_x"]

        zero = dist_map_m.new_zeros(())
        losses: Dict[str, torch.Tensor] = {"route_mass": aux["kernel_mass_error"]}

        if dist_map_m.numel() > 1:
            _, idx = torch.sort(dist_map_m)

            width_sorted = response_width[idx]
            losses["route_sigma_monotonic"] = F.relu(-(width_sorted[1:] - width_sorted[:-1])).mean()

            v_sorted = velocity[idx]
            losses["route_velocity_smooth"] = ((v_sorted[1:] - v_sorted[:-1]) ** 2).mean()
        else:
            losses["route_sigma_monotonic"] = zero
            losses["route_velocity_smooth"] = zero

        losses["route_hillslope_ratio"] = F.relu(muskingum_x - 0.35).mean()
        losses["route_sigma_width"] = response_width.mean()
        losses["route_dispersion_coupling"] = (muskingum_x * response_width).mean()

        if runoff_m3s is not None:
            runoff_weight = runoff_m3s.mean(dim=1).detach()
            runoff_weight = runoff_weight / (runoff_weight.sum() + 1e-8)
            losses["route_effective_lag"] = (runoff_weight * effective_lag).sum()
        else:
            losses["route_effective_lag"] = effective_lag.mean()

        return losses


class HydroAIBasin(nn.Module):
    """Basin-batch wrapper.

    Workflow:
      1. block mode: collate_fn flattens every basin item from (P, N, L, D) to (P*N, L, D)
      2. basin mode: collate_fn passes (P, T, D) base sequences plus target_idx
      3. generator returns pixel runoff and routing converts basin-wise runoff into outlet discharge
    """

    def __init__(
        self,
        dims: Dict[str, int],
        hidden_dim: int = 128,
        dropout: float = 0.4,
        precompute_inputs: bool = True,
        precompute_time_chunk: int = 0,
    ):
        super().__init__()
        self.generator = PixelWiseRunoffGenerator(
            dyn_dim=dims["dyn"],
            stat_dim=dims["stat"],
            hidden_dim=hidden_dim,
            dropout=dropout,
            precompute_inputs=precompute_inputs,
            precompute_time_chunk=precompute_time_chunk,
        )
        self.routing = DifferentiableMuskingumRoutingLayer(
            static_dim=dims["stat"],
            hidden_dim=32,
            vmin_mps=0.3,
            vmax_mps=3.5,
            x_max=0.49,
            dt_days=1.0,
            storage_factor_min=0.5,
            storage_factor_max=2.0,
            max_lag_limit=120,
            kernel_tail_eps=1e-4,
        )

    @staticmethod
    def mmday_to_m3s(runoff_mm: torch.Tensor, area_m2: torch.Tensor) -> torch.Tensor:
        return runoff_mm * (area_m2[:, None] / 1000.0 / 86400.0)

    @staticmethod
    def apply_fraction_coefficient(runoff: torch.Tensor, fraction: torch.Tensor) -> torch.Tensor:
        return runoff * torch.clamp(fraction.reshape(-1), min=0.0)[:, None]

    def _generator_forward_tensor(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        return self.generator(x_dyn, x_stat, return_state=False)[0]

    def _generator_forward_sequence_tensor(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        return self.generator.forward_sequence(x_dyn, x_stat, return_state=False)[0]

    def generate_runoff(
        self,
        x_dyn: torch.Tensor,
        x_stat: torch.Tensor,
        generator_chunk_size: Optional[int] = None,
        clear_cache: bool = False,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        model_device = next(self.parameters()).device
        total = x_dyn.shape[0]
        chunk_size = total if not generator_chunk_size or generator_chunk_size <= 0 else int(generator_chunk_size)
        chunk_outputs: List[torch.Tensor] = []

        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size)
            x_dyn_chunk = x_dyn[start:end].to(model_device, non_blocking=True)
            x_stat_chunk = x_stat[start:end].to(model_device, non_blocking=True)

            if self.training and use_checkpoint:
                runoff_chunk = checkpoint(
                    self._generator_forward_tensor,
                    x_dyn_chunk,
                    x_stat_chunk,
                    use_reentrant=False,
                )
            else:
                runoff_chunk = self._generator_forward_tensor(x_dyn_chunk, x_stat_chunk)

            chunk_outputs.append(runoff_chunk.squeeze(-1))
            del x_dyn_chunk, x_stat_chunk, runoff_chunk

            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(chunk_outputs, dim=0)

    def generate_runoff_sequence(
        self,
        x_dyn: torch.Tensor,
        x_stat: torch.Tensor,
        generator_chunk_size: Optional[int] = None,
        clear_cache: bool = False,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        model_device = next(self.parameters()).device
        total = x_dyn.shape[0]
        chunk_size = total if not generator_chunk_size or generator_chunk_size <= 0 else int(generator_chunk_size)
        chunk_outputs: List[torch.Tensor] = []

        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size)
            x_dyn_chunk = x_dyn[start:end].to(model_device, non_blocking=True)
            x_stat_chunk = x_stat[start:end].to(model_device, non_blocking=True)

            if self.training and use_checkpoint:
                runoff_chunk = checkpoint(
                    self._generator_forward_sequence_tensor,
                    x_dyn_chunk,
                    x_stat_chunk,
                    use_reentrant=False,
                )
            else:
                runoff_chunk = self._generator_forward_sequence_tensor(x_dyn_chunk, x_stat_chunk)

            chunk_outputs.append(runoff_chunk.squeeze(-1))
            del x_dyn_chunk, x_stat_chunk, runoff_chunk

            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(chunk_outputs, dim=0)

    def forward_grid(
        self,
        x_dyn: torch.Tensor,
        x_stat: torch.Tensor,
        generator_chunk_size: Optional[int] = None,
        clear_cache: bool = False,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        return self.generate_runoff(
            x_dyn=x_dyn,
            x_stat=x_stat,
            generator_chunk_size=generator_chunk_size,
            clear_cache=clear_cache,
            use_checkpoint=use_checkpoint,
        )

    def forward(
        self,
        x_dyn_flat: Optional[torch.Tensor],
        x_stat_flat: Optional[torch.Tensor],
        basin_meta: Optional[List[Dict]] = None,
        max_lag: int = 60,
        generator_chunk_size: Optional[int] = None,
        clear_cache: bool = False,
        use_checkpoint: bool = False,
        is_basin: bool = True,
    ):
        if not is_basin:
            return self.forward_grid(
                x_dyn=x_dyn_flat,
                x_stat=x_stat_flat,
                generator_chunk_size=generator_chunk_size,
                clear_cache=clear_cache,
                use_checkpoint=use_checkpoint,
            )

        if basin_meta is None:
            raise ValueError("basin_meta is required when is_basin=True.")

        model_device = next(self.parameters()).device
        basin_runoff_raw: List[torch.Tensor] = []
        basin_runoff_mm: List[torch.Tensor] = []
        basin_runoff_m3s: List[torch.Tensor] = []
        basin_q_pred: List[torch.Tensor] = []
        basin_ids: List[str] = []
        basin_route_aux: List[Dict[str, torch.Tensor]] = []

        basin_stateful = bool(basin_meta) and ("x_dyn_base" in basin_meta[0])

        for meta in basin_meta:
            if basin_stateful:
                x_dyn_base = meta["x_dyn_base"]
                x_stat_base = meta["x_stat_base"]
                target_idx = meta["target_idx"].to(model_device, non_blocking=True).long()
                p_count = int(x_dyn_base.shape[0])
                n_count = int(target_idx.numel())
                runoff_basin_full = self.generate_runoff_sequence(
                    x_dyn=x_dyn_base,
                    x_stat=x_stat_base,
                    generator_chunk_size=generator_chunk_size,
                    clear_cache=False,
                    use_checkpoint=use_checkpoint,
                )
                runoff_basin_raw = runoff_basin_full.index_select(dim=1, index=target_idx)
                pixel_static = x_stat_base.to(model_device, non_blocking=True)
            else:
                if x_dyn_flat is None or x_stat_flat is None:
                    raise ValueError("x_dyn_flat and x_stat_flat are required for block-mode basin forward.")
                start = int(meta["pn_start"])
                end = int(meta["pn_end"])
                p_count = int(meta["p_count"])
                n_count = int(meta["n_count"])

                runoff_basin_flat = self.generate_runoff(
                    x_dyn=x_dyn_flat[start:end],
                    x_stat=x_stat_flat[start:end],
                    generator_chunk_size=generator_chunk_size,
                    clear_cache=False,
                    use_checkpoint=use_checkpoint,
                )

                runoff_basin_raw = runoff_basin_flat.reshape(p_count, n_count)
                pixel_static = x_stat_flat[start:end].reshape(p_count, n_count, -1)[:, 0, :].to(model_device, non_blocking=True)

            dist_m = meta["dist_m"].to(model_device, non_blocking=True)
            area_m2 = meta["area_m2"].to(model_device, non_blocking=True)
            fraction = meta["fraction"].to(model_device, non_blocking=True)

            runoff_m3s = self.mmday_to_m3s(runoff_basin_raw, area_m2)
            runoff_m3s_scaled = self.apply_fraction_coefficient(runoff_m3s, fraction)
            runoff_mm_scaled = self.apply_fraction_coefficient(runoff_basin_raw, fraction)
            q_pred, route_aux = self.routing(
                runoff_m3s=runoff_m3s_scaled,
                dist_map_m=dist_m,
                pixel_static=pixel_static,
                max_lag=max_lag,
            )

            basin_ids.append(meta["basin_id"])
            basin_runoff_raw.append(runoff_basin_raw)
            basin_runoff_mm.append(runoff_mm_scaled)
            basin_runoff_m3s.append(runoff_m3s_scaled)
            basin_q_pred.append(q_pred)
            basin_route_aux.append(route_aux)

            if basin_stateful:
                del x_dyn_base, x_stat_base, target_idx, runoff_basin_full
            else:
                del runoff_basin_flat
            del runoff_m3s, runoff_m3s_scaled, runoff_mm_scaled
            del pixel_static, dist_m, area_m2, fraction
            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "basin_ids": basin_ids,
            "runoff": basin_runoff_raw,
            "runoff_mm": basin_runoff_mm,
            "runoff_m3s": basin_runoff_m3s,
            "q_pred": basin_q_pred,
            "route_aux": basin_route_aux,
        }
