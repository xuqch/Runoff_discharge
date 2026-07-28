from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_mean_abs(x: torch.Tensor, eps: float) -> torch.Tensor:
    denom = torch.abs(torch.mean(x))
    alt = torch.mean(torch.abs(x))
    return torch.where(denom > eps, denom, alt + eps)


def _connected_zero_from_q_pred(q_pred_list: List[torch.Tensor], fallback_device) -> torch.Tensor:
    for q_pred in q_pred_list:
        if torch.is_tensor(q_pred):
            return q_pred.reshape(-1).sum() * 0.0
    return torch.zeros((), device=fallback_device, requires_grad=True)


class MFMLoss(nn.Module):
    def __init__(
        self,
        p: float = 1.0,
        bins_suse: int = 10,
        bins_phi: int = 10,
        phase_penalty_scaling: float = 4.0,
        phase: bool = True,
        eps: float = 1e-6,
        soft_hist_sigma_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.p = float(p)
        self.bins_suse = int(bins_suse)
        self.bins_phi = int(bins_phi)
        self.phase_penalty_scaling = float(phase_penalty_scaling)
        self.phase = bool(phase)
        self.eps = float(eps)
        self.soft_hist_sigma_scale = float(soft_hist_sigma_scale)

    def _soft_hist(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = torch.clamp(edges[1:] - edges[:-1], min=self.eps)
        sigma = torch.clamp(widths.mean() * self.soft_hist_sigma_scale, min=self.eps)
        weights = torch.exp(-0.5 * ((x[:, None] - centers[None, :]) / sigma) ** 2)
        return weights.sum(dim=0)

    def _entropy(self, p: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(p, min=self.eps)
        return -torch.sum(p * torch.log(p))

    def _phi_component(self, sim: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        bin_min = torch.minimum(torch.min(sim), torch.min(obs))
        bin_max = torch.maximum(torch.max(sim), torch.max(obs))
        if torch.abs(bin_max - bin_min) < self.eps:
            return sim.new_tensor(1.0)
        edges = torch.linspace(bin_min, bin_max, self.bins_phi + 1, device=sim.device, dtype=sim.dtype)
        hist_sim = self._soft_hist(sim, edges)
        hist_obs = self._soft_hist(obs, edges)
        return torch.sum(torch.minimum(hist_sim, hist_obs)) / (torch.sum(hist_obs) + self.eps)

    def _suse_component(self, sim: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        min_val = torch.minimum(torch.min(sim), torch.min(obs))
        max_val = torch.maximum(torch.max(sim), torch.max(obs))
        if torch.abs(max_val - min_val) < self.eps:
            return sim.new_tensor(0.0)

        edges_scaled = torch.linspace(min_val, max_val, self.bins_suse + 1, device=sim.device, dtype=sim.dtype)
        hist_sim_s = self._soft_hist(sim, edges_scaled)
        hist_obs_s = self._soft_hist(obs, edges_scaled)
        p_sim_s = hist_sim_s / (torch.sum(hist_sim_s) + self.eps)
        p_obs_s = hist_obs_s / (torch.sum(hist_obs_s) + self.eps)
        hs = torch.abs(self._entropy(p_sim_s) - self._entropy(p_obs_s))

        def unscaled_entropy(x: torch.Tensor) -> torch.Tensor:
            x_min = torch.min(x)
            x_max = torch.max(x)
            if torch.abs(x_max - x_min) < self.eps:
                return x.new_tensor(0.0)
            edges = torch.linspace(x_min, x_max, self.bins_suse + 1, device=x.device, dtype=x.dtype)
            hist = self._soft_hist(x, edges)
            p = hist / (torch.sum(hist) + self.eps)
            return self._entropy(p)

        hu = torch.abs(unscaled_entropy(sim) - unscaled_entropy(obs))
        return torch.maximum(hs, hu)

    def _fft_phase_component(self, sim: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        n = obs.numel()
        if n < 3:
            return sim.new_tensor(0.0)

        fft_obs = torch.fft.fft(obs)
        fft_sim = torch.fft.fft(sim)
        half = n // 2
        if half < 1:
            return sim.new_tensor(0.0)

        amps = torch.abs(fft_obs[1:half + 1])
        dom_rel = int(torch.argmax(amps).item())
        if n > 365:
            dom_rel = max(dom_rel, 33)
        dom_idx = dom_rel + 1

        phase_obs = torch.angle(fft_obs[dom_idx])
        phase_sim = torch.angle(fft_sim[dom_idx])
        phase_diff = phase_sim - phase_obs
        return torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))

    def _mfm_1d(self, sim: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        finite_mask = torch.isfinite(sim) & torch.isfinite(obs)
        sim = sim[finite_mask]
        obs = obs[finite_mask]

        if sim.numel() < 3 or obs.numel() < 3:
            return sim.new_tensor(float("nan"))

        denom = _safe_mean_abs(obs, self.eps)
        nmaep = torch.mean(torch.abs(sim - obs) ** self.p) ** (1.0 / self.p) / denom

        if self.phase:
            phase_diff = self._fft_phase_component(sim, obs)
            normalized_error = torch.cos(phase_diff / self.phase_penalty_scaling) * torch.exp(-nmaep)
        else:
            normalized_error = torch.exp(-nmaep)

        variability_capture = torch.exp(-self._suse_component(sim, obs))
        distribution_similarity = self._phi_component(sim, obs)

        return 1.0 - torch.sqrt(
            (
                (1.0 - normalized_error) ** 2
                + (1.0 - variability_capture) ** 2
                + (1.0 - distribution_similarity) ** 2
            ) / 3.0
        )

    def forward(
        self,
        q_pred_list: List[torch.Tensor],
        basin_meta: List[Dict],
        q_mean_global: torch.Tensor,
        q_std_global: torch.Tensor,
    ) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        for q_pred, meta in zip(q_pred_list, basin_meta):
            device = q_pred.device
            q_valid = meta["q_valid"].to(device, non_blocking=True).reshape(-1).bool()
            if int(q_valid.sum().item()) < 3:
                continue

            q_true_raw = meta["q_true"].to(device, non_blocking=True).reshape(-1)
            q_pred_raw = q_pred.reshape(-1)
            mfm = self._mfm_1d(q_pred_raw[q_valid], q_true_raw[q_valid])
            if torch.isnan(mfm):
                continue
            losses.append(1.0 - mfm)

        if not losses:
            return _connected_zero_from_q_pred(q_pred_list, q_mean_global.device)
        return torch.stack(losses).mean()


class NSELoss(nn.Module):
    """ealstm-style weighted MSE on globally normalized targets."""

    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        q_pred_list: List[torch.Tensor],
        basin_meta: List[Dict],
        q_mean_global: torch.Tensor,
        q_std_global: torch.Tensor,
    ) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        for q_pred, meta in zip(q_pred_list, basin_meta):
            device = q_pred.device
            q_valid = meta["q_valid"].to(device, non_blocking=True).bool().reshape(-1)
            if int(q_valid.sum().item()) == 0:
                continue

            q_std_loss = meta["q_std_loss"].to(device, non_blocking=True).reshape(-1)[0]
            obs = meta["q_true"].to(device, non_blocking=True).reshape(-1)[q_valid]
            sim = q_pred.reshape(-1)[q_valid]

            q90 = torch.quantile(obs.detach(), 0.90)
            k = 5
            alpha = 1
            scaled = (obs.detach() - q90) / (q_std_loss + self.eps)
            high_weight = 1.0 + alpha * torch.sigmoid(k * scaled)
            losses.append(torch.mean(high_weight * ((sim - obs) / (q_std_loss + self.eps)) ** 2))
            del q_valid, q_std_loss, q90, high_weight, obs, sim
        if not losses:
            return _connected_zero_from_q_pred(q_pred_list, q_mean_global.device)
        return torch.stack(losses).mean()


class PeakFlowLoss(nn.Module):
    """Bounded high-flow loss using basin-scale normalization and Smooth L1."""

    def __init__(
        self,
        quantile: float = 0.9,
        weight: float = 1.0,
        eps: float = 0.5,
        min_valid_count: int = 100,
        min_peak_count: int = 10,
        huber_beta: float = 1.0,
    ):
        super().__init__()
        self.quantile = float(quantile)
        self.weight = float(weight)
        self.eps = float(eps)
        self.min_valid_count = max(1, int(min_valid_count))
        self.min_peak_count = max(1, int(min_peak_count))
        self.huber_beta = float(huber_beta)

    def _scale(self, obs: torch.Tensor, meta: Dict, device: torch.device) -> torch.Tensor:
        obs_scale = torch.quantile(torch.abs(obs.detach()), 0.50).clamp_min(self.eps)
        scale = obs_scale
        q_std_value = meta.get("q_std_loss")
        if q_std_value is not None:
            q_std_tensor = torch.as_tensor(q_std_value, device=device, dtype=obs.dtype).reshape(-1)
            q_std_tensor = q_std_tensor[torch.isfinite(q_std_tensor)]
            if q_std_tensor.numel() > 0:
                q_std_scalar = torch.abs(q_std_tensor[0])
                if float(q_std_scalar.detach().cpu()) > self.eps:
                    scale = torch.maximum(obs_scale, q_std_scalar)
        return scale.clamp_min(self.eps)

    def forward(self, q_pred_list: List[torch.Tensor], basin_meta: List[Dict]) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        for q_pred, meta in zip(q_pred_list, basin_meta):
            device = q_pred.device
            q_valid = meta["q_valid"].to(device, non_blocking=True).reshape(-1).bool()
            if int(q_valid.sum().item()) < self.min_valid_count:
                continue

            obs = meta["q_true"].to(device, non_blocking=True).reshape(-1)[q_valid]
            sim = q_pred.reshape(-1)[q_valid]
            finite_mask = torch.isfinite(obs) & torch.isfinite(sim)
            obs = obs[finite_mask]
            sim = sim[finite_mask]
            if obs.numel() < self.min_valid_count:
                continue

            threshold = torch.quantile(obs.detach(), self.quantile)
            peak_mask = obs.detach() >= threshold
            if int(peak_mask.sum().item()) < self.min_peak_count:
                continue

            obs_peak = obs[peak_mask]
            sim_peak = sim[peak_mask]
            scale = self._scale(obs, meta, device)
            normalized_error = (sim_peak - obs_peak) / scale
            huber = F.smooth_l1_loss(
                normalized_error,
                torch.zeros_like(normalized_error),
                beta=self.huber_beta,
                reduction="none",
            )
            bounded_error = -torch.expm1(-huber)
            bounded_error = bounded_error.clamp(max=1.0 - torch.finfo(bounded_error.dtype).eps)
            losses.append(self.weight * bounded_error.mean())

        if not losses:
            device = q_pred_list[0].device if q_pred_list else "cpu"
            return _connected_zero_from_q_pred(q_pred_list, device)
        return torch.stack(losses).mean()



class MassBalanceLoss(nn.Module):
    """Multi-scale weak water-balance regularizer for routed runoff outputs."""

    def __init__(
            self,
            tol_frac: float = 0.10,
            annual_tol_frac: float = 0.02,
            min_year_days: int = 300,
            water_year_start_month: int = 10,
            w_pixel: float = 0.20,
            w_basin_daily: float = 0.30,
            w_basin_annual: float = 0.50,
            eps: float = 1e-6,
    ):
        super().__init__()
        self.tol_frac = float(tol_frac)
        self.annual_tol_frac = float(annual_tol_frac)
        self.min_year_days = int(min_year_days)
        self.water_year_start_month = int(water_year_start_month)
        self.w_pixel = float(w_pixel)
        self.w_basin_daily = float(w_basin_daily)
        self.w_basin_annual = float(w_basin_annual)
        self.eps = float(eps)

    def _relative_excess(self, actual: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
        return F.relu(actual - limit) / (limit + self.eps)

    @staticmethod
    def _finite_mean(values: torch.Tensor, default: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return default
        finite_mask = torch.isfinite(values)
        if not torch.any(finite_mask):
            return default
        return values[finite_mask].mean()

    def _infer_water_year(self, meta: Dict) -> np.ndarray | None:
        if meta.get('target_years') is not None:
            years = meta['target_years']
            if torch.is_tensor(years):
                return years.detach().cpu().numpy().reshape(-1).astype(np.int32)
            return np.asarray(years).reshape(-1).astype(np.int32)

        if meta.get('target_dates') is None:
            return None

        dates_ns = np.asarray(meta['target_dates']).reshape(-1).astype(np.int64)
        dates = dates_ns.astype('datetime64[ns]')
        month_index = dates.astype('datetime64[M]')
        year = month_index.astype('datetime64[Y]').astype(np.int32) + 1970
        month = (month_index.astype(np.int32) % 12) + 1
        return (year + (month >= self.water_year_start_month)).astype(np.int32)

    def forward(self, runoff_list: List[torch.Tensor], basin_meta: List[Dict]) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        for runoff_mm, meta in zip(runoff_list, basin_meta):
            device = runoff_mm.device
            precip_mm = meta['precip_mm'].to(device, non_blocking=True)
            precip_eff = precip_mm
            zero = runoff_mm.new_zeros(())

            pixel_valid = torch.isfinite(runoff_mm) & torch.isfinite(precip_eff)
            if torch.any(pixel_valid):
                runoff_pixel = runoff_mm[pixel_valid]
                precip_pixel = precip_eff[pixel_valid]
                pixel_limit = (1.0 + self.tol_frac) * precip_pixel
                loss_pixel = self._relative_excess(runoff_pixel, pixel_limit).mean()
            else:
                loss_pixel = zero

            day_valid = torch.isfinite(runoff_mm) & torch.isfinite(precip_eff)
            runoff_basin = torch.where(day_valid, runoff_mm, zero).sum(dim=0)
            precip_basin = torch.where(day_valid, precip_eff, zero).sum(dim=0)
            valid_day_count = day_valid.sum(dim=0)
            basin_day_mask = valid_day_count > 0

            if torch.any(basin_day_mask):
                daily_limit = (1.0 + self.tol_frac) * precip_basin[basin_day_mask]
                daily_excess = self._relative_excess(runoff_basin[basin_day_mask], daily_limit)
                loss_basin_daily = self._finite_mean(daily_excess, zero)
            else:
                loss_basin_daily = zero

            years_np = self._infer_water_year(meta)
            annual_losses: List[torch.Tensor] = []
            if years_np is not None and years_np.size == runoff_basin.numel():
                years = torch.from_numpy(years_np).to(device=device, dtype=torch.long)
                for year in torch.unique(years, sorted=True):
                    year_mask = (years == year) & basin_day_mask
                    if int(year_mask.sum().item()) < self.min_year_days:
                        continue
                    p_year = precip_basin[year_mask].sum()
                    q_year = runoff_basin[year_mask].sum()
                    if (not torch.isfinite(p_year)) or (not torch.isfinite(q_year)) or p_year <= self.eps:
                        continue
                    annual_limit = (1.0 + self.annual_tol_frac) * p_year
                    loss_year = self._relative_excess(q_year, annual_limit)
                    if torch.isfinite(loss_year):
                        annual_losses.append(loss_year)

            if annual_losses:
                loss_basin_annual = torch.stack(annual_losses).mean()
            else:
                if torch.any(basin_day_mask):
                    precip_total = precip_basin[basin_day_mask].sum()
                    runoff_total = runoff_basin[basin_day_mask].sum()
                    if torch.isfinite(precip_total) and torch.isfinite(runoff_total) and precip_total > self.eps:
                        total_limit = (1.0 + self.annual_tol_frac) * precip_total
                        fallback_loss = self._relative_excess(runoff_total, total_limit)
                        loss_basin_annual = fallback_loss if torch.isfinite(fallback_loss) else zero
                    else:
                        loss_basin_annual = zero
                else:
                    loss_basin_annual = zero

            total_weight = self.w_pixel + self.w_basin_daily + self.w_basin_annual
            basin_loss = (
                                 self.w_pixel * loss_pixel
                                 + self.w_basin_daily * loss_basin_daily
                                 + self.w_basin_annual * loss_basin_annual
                         ) / max(total_weight, self.eps)
            if torch.isfinite(basin_loss):
                losses.append(basin_loss)

            del precip_mm, precip_eff
            del runoff_basin, precip_basin, valid_day_count, basin_day_mask
            del loss_pixel, loss_basin_daily, loss_basin_annual, basin_loss

        if len(losses) == 0:
            device = runoff_list[0].device if runoff_list else 'cpu'
            return torch.zeros((), device=device)
        return torch.stack(losses).mean()



class AnnualBudykoBalanceLoss(nn.Module):
    """Annual-scale Budyko water-balance regularizer at basin level."""

    def __init__(self,
                 budyko_alpha: float = 2.6,
                 min_days_per_year: int = 300,
                 eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.budyko_alpha = float(budyko_alpha)
        self.min_days_per_year = int(min_days_per_year)
        self.eps = float(eps)

    def _fu_runoff_coefficient(self, phi: torch.Tensor) -> torch.Tensor:
        alpha = max(self.budyko_alpha, 1.0 + self.eps)
        return torch.pow(1.0 + torch.pow(phi, alpha), 1.0 / alpha) - phi

    def forward(self, runoff_list: List[torch.Tensor], basin_meta: List[Dict]) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        for runoff_mm, meta in zip(runoff_list, basin_meta):
            device = runoff_mm.device
            target_years = meta.get("target_years")
            pet_mm = meta.get("pet_mm")
            if target_years is None or pet_mm is None:
                raise KeyError("Annual Budyko balance requires target_years and pet_mm in basin_meta.")

            years = target_years.to(device, non_blocking=True).reshape(-1).long()
            precip_mm = meta["precip_mm"].to(device, non_blocking=True)
            pet_mm = pet_mm.to(device, non_blocking=True)

            day_valid = torch.isfinite(runoff_mm) & torch.isfinite(precip_mm) & torch.isfinite(pet_mm)
            zero = runoff_mm.new_zeros(())
            valid_day_count = day_valid.sum(dim=0)
            basin_day_mask = valid_day_count > 0
            basin_precip = torch.where(day_valid, precip_mm, zero).sum(dim=0)
            basin_pet = torch.where(day_valid, pet_mm, zero).sum(dim=0)
            basin_runoff = torch.where(day_valid, runoff_mm, zero).sum(dim=0)

            for year in torch.unique(years, sorted=True):
                year_mask = (years == year) & basin_day_mask
                if int(year_mask.sum().item()) < self.min_days_per_year:
                    continue

                p_year = basin_precip[year_mask].sum()
                pet_year = basin_pet[year_mask].sum()
                q_year = basin_runoff[year_mask].sum()
                if (
                    (not torch.isfinite(p_year))
                    or (not torch.isfinite(pet_year))
                    or (not torch.isfinite(q_year))
                    or p_year <= self.eps
                ):
                    continue

                phi = pet_year / (p_year + self.eps)
                if not torch.isfinite(phi):
                    continue
                rc_target = self._fu_runoff_coefficient(torch.clamp(phi, min=0.0))
                rc_pred = q_year / (p_year + self.eps)
                if (not torch.isfinite(rc_target)) or (not torch.isfinite(rc_pred)):
                    continue
                overflow = F.relu(q_year - p_year) / (p_year + self.eps)
                loss_year = F.smooth_l1_loss(rc_pred, rc_target) + overflow
                if torch.isfinite(loss_year):
                    losses.append(loss_year)
            del precip_mm, pet_mm, years, basin_runoff, basin_precip, basin_pet, valid_day_count, basin_day_mask
        if not losses:
            device = runoff_list[0].device if runoff_list else "cpu"
            return torch.zeros((), device=device)
        return torch.stack(losses).mean()


def build_balance_loss(
    balance_loss: str = "budyko_annual",
    budyko_alpha: float = 2.6,
    budyko_min_days_per_year: int = 300,
) -> Optional[nn.Module]:
    balance_loss = str(balance_loss).lower()
    if balance_loss in ("none", "off"):
        return None
    if balance_loss in ("legacy", "daily", "precip_cap"):
        return MassBalanceLoss(tol_frac=0.10)
    if balance_loss in ("budyko_annual", "budyko", "annual"):
        return AnnualBudykoBalanceLoss(
            budyko_alpha=budyko_alpha,
            min_days_per_year=budyko_min_days_per_year,
        )
    raise ValueError(f"Unsupported balance_loss={balance_loss!r}")


class _BaseHydroLoss(nn.Module):
    def __init__(self, w_q: float = 1.0, w_balance: float = 0.0, balance_module: Optional[nn.Module] = None):
        super().__init__()
        self.w_q = float(w_q)
        self.w_balance = float(w_balance)
        self.balance_module = balance_module

    def _balance_term(self, outputs: Dict[str, List[torch.Tensor]], basin_meta: List[Dict], device: torch.device) -> torch.Tensor:
        if self.w_balance <= 0.0 or self.balance_module is None:
            return torch.zeros((), device=device)
        return self.balance_module(outputs["runoff"], basin_meta)

    @staticmethod
    def _package_metrics(total: torch.Tensor, loss_q: torch.Tensor, loss_mb: torch.Tensor, extra: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        metrics = {
            "loss_total": float(total.detach().cpu()),
            "loss_q": float(loss_q.detach().cpu()),
            "loss_mb": float(loss_mb.detach().cpu()),
        }
        if extra:
            metrics.update(extra)
        return metrics


class HydroLoss(_BaseHydroLoss):
    def __init__(
        self,
        w_q: float = 1.0,
        w_balance: float = 0.0,
        eps: float = 0.1,
        balance_loss: str = "budyko_annual",
        budyko_alpha: float = 2.6,
        budyko_min_days_per_year: int = 300,
    ):
        super().__init__(
            w_q=w_q,
            w_balance=w_balance,
            balance_module=build_balance_loss(
                balance_loss=balance_loss,
                budyko_alpha=budyko_alpha,
                budyko_min_days_per_year=budyko_min_days_per_year,
            ),
        )
        self.q_loss = NSELoss(eps=eps)

    def forward(
        self,
        outputs: Dict[str, List[torch.Tensor]],
        basin_meta: List[Dict],
        q_mean_global: torch.Tensor,
        q_std_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_q = self.q_loss(outputs["q_pred"], basin_meta, q_mean_global=q_mean_global, q_std_global=q_std_global)
        loss_mb = self._balance_term(outputs, basin_meta, loss_q.device)
        total = self.w_q * loss_q + self.w_balance * loss_mb
        if np.isnan(float(total.detach().cpu())):
            print(basin_meta[0]["basin_id"], 'loss_q:', loss_q, 'loss_mb:' ,loss_mb)
        return total, self._package_metrics(total, loss_q, loss_mb)


class HydroMFMLoss(_BaseHydroLoss):
    def __init__(
        self,
        w_q: float = 1.0,
        w_balance: float = 0.0,
        p: float = 1.0,
        bins_suse: int = 10,
        bins_phi: int = 10,
        phase_penalty_scaling: float = 4.0,
        phase: bool = True,
        eps: float = 1e-6,
        soft_hist_sigma_scale: float = 0.5,
        balance_loss: str = "budyko_annual",
        budyko_alpha: float = 2.6,
        budyko_min_days_per_year: int = 300,
        w_peak: float = 0.0,
        peak_quantile: float = 0.9,
        peak_weight: float = 1.0,
        peak_eps: float = 0.5,
        peak_min_valid_count: int = 100,
        peak_min_peak_count: int = 10,
        peak_huber_beta: float = 1.0,
    ) -> None:
        super().__init__(
            w_q=w_q,
            w_balance=w_balance,
            balance_module=build_balance_loss(
                balance_loss=balance_loss,
                budyko_alpha=budyko_alpha,
                budyko_min_days_per_year=budyko_min_days_per_year,
            ),
        )
        self.q_loss = MFMLoss(
            p=p,
            bins_suse=bins_suse,
            bins_phi=bins_phi,
            phase_penalty_scaling=phase_penalty_scaling,
            phase=phase,
            eps=eps,
            soft_hist_sigma_scale=soft_hist_sigma_scale,
        )
        self.w_peak = float(w_peak)
        self.peak_loss = PeakFlowLoss(
            quantile=peak_quantile,
            weight=peak_weight,
            eps=peak_eps,
            min_valid_count=peak_min_valid_count,
            min_peak_count=peak_min_peak_count,
            huber_beta=peak_huber_beta,
        )

    def forward(
        self,
        outputs: Dict[str, List[torch.Tensor]],
        basin_meta: List[Dict],
        q_mean_global: torch.Tensor,
        q_std_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_q = self.q_loss(outputs["q_pred"], basin_meta, q_mean_global=q_mean_global, q_std_global=q_std_global)
        loss_mb = self._balance_term(outputs, basin_meta, loss_q.device)
        if self.w_peak > 0.0:
            loss_peak = self.peak_loss(outputs["q_pred"], basin_meta)
        else:
            loss_peak = torch.zeros((), device=loss_q.device)

        total = self.w_q * loss_q + self.w_balance * loss_mb + self.w_peak * loss_peak
        return total, self._package_metrics(
            total,
            loss_q,
            loss_mb,
            extra={
                "loss_q_mfm": float(loss_q.detach().cpu()),
                "loss_peak": float(loss_peak.detach().cpu()),
            },
        )
